# src/tea/classroom/data.py

import glob
import os
import re

import pandas as pd
from pathlib import Path
from datasets import Audio, Dataset
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, LeaveOneGroupOut

from tea.utils.constants import CLASS_ORDER, LABEL2ID, SAMPLE_RATE
from tea.utils.logging import get_logger

_VIDEO_SUFFIX_RE = re.compile(r"_video\d+$")

logger = get_logger(__name__)


def get_teacher_id(video_id: str) -> str:
    """'1B2251' and '1B2251_video2' -> '1B2251'. Singletons map to
    themselves (e.g. '1B4262' -> '1B4262')."""
    return _VIDEO_SUFFIX_RE.sub("", video_id)


def read_video_csv_raw(csv_root: str | Path, audio_root: str | Path, video_id: str, 
                       chunk_col: str = "name", label_col: str = "gt_label") -> pd.DataFrame | None:
    """Read one video's CSV AS-IS; no dropna, no exclusion checks.
    
    Rows with a NaN `gt_label` are exactly the non-speech / unannotated
    chunks, which `tea.classroom.fesc`'s noise-pool extraction wants. Used
    by both `build_full_df` (which drops the NaNs) and the noise extractor
    (which wants only the NaNs).
    """
    csv_path = os.path.join(csv_root, f"{video_id}.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)

    def _to_path(chunk):
        fname = chunk if str(chunk).endswith(".wav") else f"{chunk}.wav"
        return os.path.join(audio_root, video_id, fname)

    df["audio_path"] = df[chunk_col].apply(_to_path)
    df["video_id"] = video_id
    df["teacher_id"] = get_teacher_id(video_id)
    df["chunk_uid"] = video_id + "__" + df[chunk_col].astype(str)
    return df


def _load_one_video(csv_root: str | Path, audio_root: str | Path, video_id: str, 
                    chunk_col: str = "name", label_col: str = "gt_label") -> pd.DataFrame | None:
    """Load one video's CSV, drop un-annotated chunks, return dataframe or None if empty.
    
    Parameters
    ----------
    csv_root: str | Path
        Directory containing per-video CSVs.
    audio_root: str | Path
        Directory containing per-video audio chunks.
    video_id: str
        Video ID to load, e.g. "1B2251".
    chunk_col: str
        Column name in the CSV that contains chunk names.
    label_col: str
        Column name in the CSV that contains ground truth labels.

    Returns
    -------
    pd.DataFrame | None
        DataFrame with labeled chunks, or None if no CSV found or no labeled chunks.
    """
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


def build_full_df(audio_root: str | Path, csv_root: str | Path,  excluded_videos: set[str],
                  videos: list[str] | None = None, chunk_col: str = "name", label_col: str = "gt_label") -> pd.DataFrame:
    """Load every per-video CSV, drop un-annotated chunks and excluded videos, merge into one dataframe.
    
    Parameters
    ----------
    audio_root, csv_root:
        Directories of per-video chunk audio and annotation CSVs.
    excluded_videos:
        Video ids to always skip e.g. `{"1B3261"}`.
    videos:
        Restrict to these video ids. Defaults to every `*.csv` found in `csv_root`.

    Returns
    -------
    full_df:
        A single dataframe with all labeled chunks from all videos, with columns:
        - audio_path: path to the chunk's .wav file
        - video_id: the video id (e.g. "1B2251")
        - teacher_id: the teacher id (e.g. "1B2251" -> "1B2251", "1B2251_video2" -> "1B2251")
        - chunk_uid: unique id for the chunk, e.g. "1B2251__chunk123"
        - gt_label: the ground truth label (lowercased string)
    """
    if videos is None:
        videos = sorted(
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(csv_root, "*.csv"))
        )

    frames = []
    for video_id in videos:
        if video_id in excluded_videos:
            logger.info("%s: excluded, skipping", video_id)
            continue
        df = _load_one_video(csv_root, audio_root, video_id, chunk_col, label_col)
        if df is not None:
            frames.append(df)

    if not frames:
        raise ValueError("No videos loaded — check audio_root / csv_root / videos list.")

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
    print(full.groupby("gt_label").size().to_dict())
    return full


def to_hf_dataset(df: pd.DataFrame) -> Dataset:
    """Convert a classroom dataframe to a HuggingFace Dataset, keeping only the columns needed for training."""
    base_cols = ["audio_path", "label", "chunk_uid"]
    optional = ["confidence", "is_augmented", "loss_weight"]
    cols = base_cols + [c for c in optional if c in df.columns]
    ds = Dataset.from_pandas(df[cols].rename(columns={"audio_path": "audio"}))
    return ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))


def teacher_grouped_folds(df: pd.DataFrame, n_splits="loto"):
    """Yield (fold_name, held_out_teachers, train_df, test_df) for each fold of teacher-grouped cross-validation.
    If n_splits is "loto", use Leave-One-Group-Out (one teacher held out at a time). If n_splits is an int, use GroupKFold with that
    many folds. The train_df and test_df are already fully assembled; a fold here is just a boolean filter on teacher_id.

    Parameters
    ----------
    df: pd.DataFrame
        Full classroom dataframe (see build_full_df).
    n_splits: str or int
        "loto" for Leave-One-Teacher-Out (exhaustive, every teacher tested exactly once), or an int for GroupKFold with that many folds (use if LOTO is too slow).

    Yields
    ------
    fold_name: str
        Name of the fold, e.g. "Teacher-1" or "Teacher1+Teacher2" if multiple teachers are held out.
    held_out_teachers: list[str]
        List of teacher IDs held out in this fold.
    train_df: pd.DataFrame
        Training dataframe for this fold.
    test_df: pd.DataFrame
        Testing dataframe for this fold.
    """

    groups = df["teacher_id"].values
    teachers = sorted(df["teacher_id"].unique())
    logger.info("%d teacher groups: %s", len(teachers), teachers)

    splitter = (
        LeaveOneGroupOut()
        if n_splits == "loto"
        else GroupKFold(n_splits=min(int(n_splits), len(teachers)))
    )

    for train_idx, test_idx in splitter.split(df, groups=groups):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        held_out = sorted(test_df["teacher_id"].unique())
        fold_name = "+".join(held_out)
        yield fold_name, (held_out[0] if len(held_out) == 1 else held_out), train_df, test_df


def internal_val_split(train_df: pd.DataFrame, val_frac: float = 0.0, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """For a given training dataframe, split off a validation set of size `val_frac` (fraction of the training set). 
    The split is done by video_id to avoid leakage.

    Parameters
    ----------
    train_df:
        The training dataframe to split.
    val_frac:
        Fraction of the training set to use as validation. If 0, no validation set is created.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    train_df:
        The training dataframe after splitting off the validation set.
    val_df:
        The validation dataframe, or None if val_frac is 0.  
    """
    if val_frac <= 0:
        return train_df, None
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_idx, val_idx = next(gss.split(train_df, groups=train_df["video_id"]))
    return train_df.iloc[tr_idx].reset_index(drop=True), train_df.iloc[val_idx].reset_index(drop=True)