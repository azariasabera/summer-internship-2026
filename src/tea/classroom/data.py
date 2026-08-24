# src/tea/classroom/data.py

"""Loads per-video chunk audio + annotation CSVs, groups videos into
teachers, and yields Leave-One-Teacher-Out (or GroupKFold) train/test folds.

Expected layout:

```
audio_root/
    1B2251/chunk_s_0.wav, chunk_n_1.wav, ...
    1B2251_video2/...
csv_root/
    1B2251.csv
    1B2251_video2.csv
```

Each CSV needs at least `name` (chunk id) and `gt_label`. Optional:
`confidence` (1-3), `overlap` (child-speech, 0/1). Rows with `gt_label`
NaN (non-speech / unannotated chunks) are dropped for training use, but
kept available via `read_video_csv_raw` for `tea.classroom.fesc`'s
noise-pool extraction.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from datasets import Audio, Dataset
from omegaconf import DictConfig
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, LeaveOneGroupOut

from tea.utils.constants import CLASS_ORDER, LABEL2ID, SAMPLE_RATE
from tea.utils.io import get_teacher_id
from tea.utils.logging import get_logger

logger = get_logger(__name__)


def read_video_csv_raw(csv_root: str | Path, audio_root: str | Path, video_id: str, chunk_col: str = "name", label_col: str = "gt_label") -> pd.DataFrame | None:
    """Read one video's CSV AS-IS; no dropna, no exclusion checks.

    Rows with a NaN `gt_label` are exactly the non-speech / unannotated
    chunks, which `tea.classroom.fesc`'s noise-pool extraction wants. Used
    by both `build_full_df` (which drops the NaNs) and the noise extractor
    (which wants only the NaNs).
    """
    csv_path = Path(csv_root) / f"{video_id}.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)

    def _to_path(chunk):
        fname = chunk if str(chunk).endswith(".wav") else f"{chunk}.wav"
        return str(Path(audio_root) / video_id / fname)

    df["audio_path"] = df[chunk_col].apply(_to_path)
    df["video_id"] = video_id
    df["teacher_id"] = get_teacher_id(video_id)
    df["chunk_uid"] = video_id + "__" + df[chunk_col].astype(str)
    return df


def _load_one_video(csv_root, audio_root, video_id, chunk_col="name", label_col="gt_label") -> pd.DataFrame | None:
    """Load one video's CSV, drop un-annotated chunks, return dataframe or None if empty."""
    df = read_video_csv_raw(csv_root, audio_root, video_id, chunk_col, label_col)
    if df is None:
        logger.info("%s: no csv found, skipping", video_id)
        return None
    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    if len(df) == 0:
        logger.info("%s: 0 labeled chunks after dropping NaN %s, skipping", video_id, label_col)
        return None
    df["gt_label"] = df[label_col].astype(str).str.lower()
    return df


def build_full_df(
    audio_root: str | Path, csv_root: str | Path, excluded_videos: set[str], videos: list[str] | None = None,
    chunk_col: str = "name", label_col: str = "gt_label",
) -> pd.DataFrame:
    """Load every per-video CSV, drop un-annotated chunks and excluded videos, merge into one dataframe.

    Parameters
    ----------
    audio_root, csv_root:
        Directories of per-video chunk audio and annotation CSVs.
    excluded_videos:
        Video ids to always skip (e.g. `{"1B3261"}` -- too few annotated chunks).
    videos:
        Restrict to these video ids. Defaults to every `*.csv` found in `csv_root`.
    """
    if videos is None:
        videos = sorted(p.stem for p in Path(csv_root).glob("*.csv"))

    frames = []
    for video_id in videos:
        if video_id in excluded_videos:
            logger.info("%s: excluded, skipping", video_id)
            continue
        df = _load_one_video(csv_root, audio_root, video_id, chunk_col, label_col)
        if df is not None:
            frames.append(df)

    if not frames:
        raise ValueError("No videos loaded -- check audio_root / csv_root / videos list.")

    full = pd.concat(frames, ignore_index=True)
    full = full.loc[full["gt_label"].isin(CLASS_ORDER)].reset_index(drop=True)
    full["label"] = full["gt_label"].map(LABEL2ID).astype(int)

    if "confidence" not in full.columns:
        full["confidence"] = 3
    if "overlap" not in full.columns:
        full["overlap"] = 0
    full["is_augmented"] = False

    logger.info(
        "Loaded %d labeled chunks from %d videos, %d teacher groups.",
        len(full), full["video_id"].nunique(), full["teacher_id"].nunique(),
    )
    logger.info("Class distribution: %s", full.groupby("gt_label").size().to_dict())
    return full


def to_hf_dataset(df: pd.DataFrame) -> Dataset:
    """Convert a classroom dataframe to a HF `Dataset` with audio decoding, keeping optional weighting columns."""
    base_cols = ["audio_path", "label", "chunk_uid"]
    optional = ["confidence", "is_augmented", "loss_weight"]
    cols = base_cols + [c for c in optional if c in df.columns]
    ds = Dataset.from_pandas(df[cols].rename(columns={"audio_path": "audio"}))
    return ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))


def teacher_grouped_folds(df: pd.DataFrame, n_splits: str | int = "loto"):
    """Yield `(fold_name, held_out_teacher, train_df, test_df)` tuples.

    `train_df`/`test_df` are already fully assembled. A fold here is
    just a boolean filter on `teacher_id`.

    Parameters
    ----------
    df:
        Full classroom dataframe (see `build_full_df`).
    n_splits:
        `"loto"` for Leave-One-Teacher-Out (exhaustive, every teacher
        tested exactly once -- recommended given few teachers), or an int
        for GroupKFold with that many folds (use if LOTO is too slow).
    """
    groups = df["teacher_id"].values
    teachers = sorted(df["teacher_id"].unique())
    logger.info("%d teacher groups: %s", len(teachers), teachers)

    splitter = LeaveOneGroupOut() if n_splits == "loto" else GroupKFold(n_splits=min(int(n_splits), len(teachers)))

    for train_idx, test_idx in splitter.split(df, groups=groups):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        held_out = sorted(test_df["teacher_id"].unique())
        fold_name = "+".join(held_out)
        yield fold_name, (held_out[0] if len(held_out) == 1 else held_out), train_df, test_df


def internal_val_split(train_df: pd.DataFrame, val_frac: float = 0.0, seed: int = 42):
    """Optionally carve a video-grouped validation slice OUT OF TRAIN ONLY, for early stopping.

    - `val_frac=0` (default): train a fixed number of epochs, evaluate on
      test only at the end. Simple, no leakage.
    - `val_frac>0`: carve a small slice out of train, grouped by
      `video_id` so a video's chunks never straddle both sides, purely
      for early-stopping. Test is never touched by this.

    Parameters
    ----------
    train_df:
        This fold's training dataframe.
    val_frac:
        Fraction of train to carve out for validation. 0 disables this.
    seed:
        Random seed for the split.
    """
    if val_frac <= 0:
        return train_df, None
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_idx, val_idx = next(gss.split(train_df, groups=train_df["video_id"]))
    return train_df.iloc[tr_idx].reset_index(drop=True), train_df.iloc[val_idx].reset_index(drop=True)
