# src/tea/features/master_table.py

"""Builds the wide, per-chunk feature table that `tea.confidence` and `tea.probes` both use."""

from __future__ import annotations

import glob
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from tea.features.acoustic import batch_extract_acoustic_features, clean_transcription, speech_rate_from_text
from tea.features.softmax import add_softmax_engineered_features
from tea.utils.io import get_teacher_id, load_nested_json
from tea.utils.logging import get_logger

logger = get_logger(__name__)

MTKD_CLASSES = ["neutral", "sadness", "happiness", "anger"]
SENTIMENT_CLASSES = ["neutral", "negative", "positive"]

EPS = 1e-8


# ---------------------------------------------------------------------------
# Annotation CSVs (kept all rows + is_annotated flag, as in conf_estim.txt)
# ---------------------------------------------------------------------------


def load_annotation_csvs_for_features(
    csv_root: str | Path, audio_root: str | Path, videos: list[str] | None = None,
    excluded_videos: list[str] | None = None, chunk_col: str = "name", label_col: str = "gt_label",
) -> pd.DataFrame:
    """Read every per-video CSV, keeping ALL rows (annotated + non-speech) tagged `is_annotated`.

    Dropping non-speech rows happens once, after the MTKD/sentiment merge
    (in `build_master_table`), so you can see exactly how many chunks were
    dropped and why, instead of losing them silently in two places.

    Parameters
    ----------
    csv_root, audio_root:
        Directories of per-video annotation CSVs and chunk audio.
    videos:
        Restrict to these video ids. Defaults to every `*.csv` found.
    excluded_videos:
        Video ids to skip entirely.
    """
    excluded_videos = set(excluded_videos or [])
    if videos is None:
        videos = sorted(Path(p).stem for p in glob.glob(os.path.join(csv_root, "*.csv")))

    frames = []
    for video_id in videos:
        if video_id in excluded_videos:
            logger.info("%s: excluded, skipping", video_id)
            continue
        csv_path = Path(csv_root) / f"{video_id}.csv"
        if not csv_path.exists():
            logger.info("%s: no csv found, skipping", video_id)
            continue

        df = pd.read_csv(csv_path)
        df = df.rename(columns={chunk_col: "chunk_id"})
        df["video_id"] = video_id
        df["teacher_id"] = get_teacher_id(video_id)
        df["audio_path"] = df["chunk_id"].apply(
            lambda c: os.path.join(audio_root, video_id, f"{c}.wav" if not str(c).endswith(".wav") else c)
        )
        df["is_annotated"] = df[label_col].notna()
        if "gt_label" in df.columns:
            df["gt_label"] = df["gt_label"].astype(str).str.lower().replace("nan", np.nan)
        frames.append(df)

    if not frames:
        raise ValueError("No per-video csvs loaded -- check csv_root / videos / excluded_videos.")

    full = pd.concat(frames, ignore_index=True)
    logger.info(
        "Loaded %d total chunks from %d videos (%d annotated), %d teacher groups.",
        len(full), full["video_id"].nunique(), full["is_annotated"].sum(), full["teacher_id"].nunique(),
    )
    return full


# ---------------------------------------------------------------------------
# Embeddings (wide format: embedding_0..embedding_N), only when requested
# ---------------------------------------------------------------------------


def _load_embeddings_wide(video_ids: list[str], embedding_root: str | Path) -> pd.DataFrame:
    """Load `pooled.npy`/`names.npy` per video into one wide `(video_id, chunk_id, embedding_0..N)` table."""
    frames = []
    for video_id in video_ids:
        video_dir = Path(embedding_root) / video_id
        pooled_path, names_path = video_dir / "pooled.npy", video_dir / "names.npy"
        if not pooled_path.exists() or not names_path.exists():
            logger.warning("%s: missing pooled.npy/names.npy under %s, skipping", video_id, embedding_root)
            continue

        pooled = np.asarray(np.load(pooled_path))
        names = np.load(names_path, allow_pickle=True)
        if pooled.ndim != 2 or len(pooled) != len(names):
            raise ValueError(f"{video_id}: pooled.npy/names.npy shape mismatch ({pooled.shape} vs {len(names)} names)")

        names = [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in names]
        names = [n[:-4] if n.endswith(".wav") else n for n in names]

        wide = pd.DataFrame(pooled, columns=[f"embedding_{i}" for i in range(pooled.shape[1])])
        wide.insert(0, "chunk_id", names)
        wide.insert(0, "video_id", video_id)
        frames.append(wide)

    embeddings = pd.concat(frames, ignore_index=True)
    n_dims = sum(1 for c in embeddings.columns if c.startswith("embedding_"))
    logger.info("Loaded embeddings: %d chunks x %d dimensions", len(embeddings), n_dims)
    return embeddings


# ---------------------------------------------------------------------------
# Master assembly
# ---------------------------------------------------------------------------


def build_master_table(
    csv_root: str | Path, audio_root: str | Path, mtkd_json_path: str | Path,
    sentiment_fi_json_path: str | Path, sentiment_en_json_path: str | Path | None = None,
    embedding_root: str | Path | None = None,
    videos: list[str] | None = None, excluded_videos: list[str] | None = None,
    compute_acoustic: bool = True, use_three_class: bool = False,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build the wide per-chunk feature table.

    Parameters
    ----------
    csv_root, audio_root:
        Directories of per-video annotation CSVs and chunk audio.
    mtkd_json_path:
        MTKD prediction JSON (`{video: {chunk: {class: prob}}}`).
    sentiment_fi_json_path, sentiment_en_json_path:
        Sentiment JSONs (same nested shape). EN is optional.
    embedding_root:
        If given, inner-joins 768-d pooled embeddings (`embedding_0..N`
        columns) from `tea.mtkd.embeddings`' output layout -- this is the
        `tea.probes.feature_fusion` use case. `None` (default) skips
        embeddings entirely, matching `tea.confidence`'s use case.
    videos, excluded_videos:
        Restrict/exclude specific video ids.
    compute_acoustic:
        If False, acoustic columns are all-NaN placeholders (faster, useful
        for quick iteration when acoustic features aren't needed yet).
    use_three_class:
        If True, reads MTKD as the 3-class sentiment representation
        (negative=sadness+anger, neutral=neutral, positive=happiness) and
        maps `gt_label` onto the same 3 classes, instead of the native 4-class output.

    Returns
    -------
    tuple
        `(df, mtkd_classes, sentiment_classes)`. `df` has one row per
        ANNOTATED chunk with columns: `video_id`, `chunk_id`, `teacher_id`,
        `gt_label`, `confidence`, `overlap`, `transcription`, `translation`,
        `duration_sec`, `mtkd_<class>...`, `text_fi_<class>...`,
        `text_en_<class>...` (if provided), `embedding_0..N` (if
        `embedding_root` given), `rms_mean`, `rms_std`, `f0_mean`, `f0_std`,
        `voiced_ratio`, `speech_rate`.
    """
    logger.info("Loading annotation CSVs...")
    ann = load_annotation_csvs_for_features(csv_root, audio_root, videos=videos, excluded_videos=excluded_videos)

    mtkd_classes = SENTIMENT_CLASSES if use_three_class else MTKD_CLASSES
    sentiment_classes = SENTIMENT_CLASSES

    logger.info("Loading MTKD predictions...")
    mtkd_df = _load_nested_json_df(mtkd_json_path, class_order=mtkd_classes)
    mtkd_df = mtkd_df.rename(columns={c: f"mtkd_{c}" for c in mtkd_classes})

    logger.info("Loading Finnish sentiment predictions...")
    sent_fi_df = _load_nested_json_df(sentiment_fi_json_path, class_order=sentiment_classes)
    sent_fi_df = sent_fi_df.rename(columns={c: f"text_fi_{c}" for c in sentiment_classes})

    df = ann.merge(mtkd_df, on=["video_id", "chunk_id"], how="left").merge(sent_fi_df, on=["video_id", "chunk_id"], how="left")

    if len(mtkd_classes) == 3:
        gt_map = {"neutral": "neutral", "sadness": "negative", "anger": "negative", "happiness": "positive"}
        df["gt_label"] = df["gt_label"].map(gt_map)

    if sentiment_en_json_path:
        logger.info("Loading English sentiment predictions...")
        sent_en_df = _load_nested_json_df(sentiment_en_json_path, class_order=sentiment_classes)
        sent_en_df = sent_en_df.rename(columns={c: f"text_en_{c}" for c in sentiment_classes})
        df = df.merge(sent_en_df, on=["video_id", "chunk_id"], how="left")

    n_total = len(df)
    df = df[df["is_annotated"]].reset_index(drop=True)
    logger.info("Dropped %d non-speech/unannotated chunks. %d annotated chunks remain.", n_total - len(df), len(df))

    mtkd_cols = [f"mtkd_{c}" for c in mtkd_classes]
    sent_fi_cols = [f"text_fi_{c}" for c in sentiment_classes]
    n_missing_mtkd = df[mtkd_cols].isna().any(axis=1).sum()
    n_missing_sent = df[sent_fi_cols].isna().any(axis=1).sum()
    if n_missing_mtkd or n_missing_sent:
        warnings.warn(
            f"{n_missing_mtkd} annotated chunks have no MTKD json match, {n_missing_sent} have no sentiment "
            f"json match -- these chunk_ids exist in the csv but not in the corresponding json. Dropping them."
        )
    df = df.dropna(subset=mtkd_cols + sent_fi_cols).reset_index(drop=True)
    logger.info("Merged dataframe: %d rows", len(df))

    if embedding_root is not None:
        logger.info("Loading pooled embeddings...")
        emb_df = _load_embeddings_wide(sorted(df["video_id"].unique()), embedding_root)
        before = len(df)
        df = df.merge(emb_df, on=["video_id", "chunk_id"], how="inner")
        if len(df) < before:
            logger.warning("Dropped %d chunks with no matching embedding.", before - len(df))

    if compute_acoustic:
        logger.info("Computing acoustic features...")
        acoustic = batch_extract_acoustic_features(df)
        df = pd.concat([df, acoustic], axis=1)
        logger.info("Computing speaking rate...")
        df["transcription_clean"] = df["transcription"].apply(clean_transcription)
        df["speech_rate"] = speech_rate_from_text(df["transcription_clean"], df["duration_sec"])
        before = len(df)
        df = df.dropna(subset=["rms_mean", "f0_mean"]).reset_index(drop=True)
        if len(df) < before:
            warnings.warn(f"Dropped {before - len(df)} chunks with unreadable audio (check audio_root path).")
    else:
        for c in ("rms_mean", "rms_std", "f0_mean", "f0_std", "voiced_ratio", "speech_rate"):
            df[c] = np.nan

    if embedding_root is not None or not compute_acoustic:
        # feature_fusion consumes raw class-name softmax columns too -- add those unless already three-class-mapped.
        if not use_three_class:
            df = add_softmax_engineered_features(df, class_cols=[f"mtkd_{c}" for c in mtkd_classes])

    return df, mtkd_classes, sentiment_classes


def _load_nested_json_df(path: str | Path, class_order: list[str]) -> pd.DataFrame:
    """`load_nested_json` + automatic 4-class -> 3-class MTKD collapse.

    Thin wrapper over `tea.utils.io.load_nested_json` (which returns the
    raw `{video: {chunk: {...}}}` dict) that flattens it into the
    `(video_id, chunk_id, *class_order)` shape this module's callers want.
    """
    raw = load_nested_json(path)
    rows = []
    json_classes = set()
    for video_id, chunks in raw.items():
        for chunk_id, scores in chunks.items():
            row = {"video_id": video_id, "chunk_id": chunk_id}
            row.update(scores)
            rows.append(row)
            json_classes.update(scores.keys())

    df = pd.DataFrame(rows)

    if set(json_classes) == set(MTKD_CLASSES) and class_order == SENTIMENT_CLASSES:
        df["negative"] = df["sadness"] + df["anger"]
        df["neutral"] = df["neutral"]
        df["positive"] = df["happiness"]

    missing = df[class_order].isna()
    if missing.any().any():
        n_bad = int(missing.any(axis=1).sum())
        warnings.warn(f"{n_bad} chunks in {path} are missing one or more of {class_order}; filling with 0.0.")
        df[class_order] = df[class_order].fillna(0.0)

    return df[["video_id", "chunk_id"] + class_order]


def rescale_confidence(conf_1_to_3: pd.Series) -> pd.Series:
    """Rescale the 1-3 annotation-confidence scale to [0, 1] for the reliability regression head."""
    return (conf_1_to_3 - 1.0) / 2.0


# ---------------------------------------------------------------------------
# Feature table (numeric, model-ready) -- tea.confidence's consumer
# ---------------------------------------------------------------------------


def per_teacher_zscore(df: pd.DataFrame, col: str, teacher_col: str = "teacher_id", min_samples: int = 5) -> pd.Series:
    """Z-score `col` within each teacher's own distribution.

    Falls back to a global z-score for teachers with too few samples
    (avoids unstable per-teacher stats on e.g. a teacher with 3 chunks).
    """
    global_mean, global_std = df[col].mean(), df[col].std() + EPS

    def _z(group):
        if len(group) < min_samples:
            return (group - global_mean) / global_std
        mu, sigma = group.mean(), group.std() + EPS
        return (group - mu) / sigma

    return df.groupby(teacher_col)[col].transform(_z)


def text_sentiment_agreement(fi_probs: np.ndarray, en_probs: np.ndarray) -> np.ndarray:
    """Scalar FI/EN sentiment disagreement (L1 distance, halved to [0, 1]). High = text signal less trustworthy."""
    return 0.5 * np.sum(np.abs(fi_probs - en_probs), axis=1)


def asr_confidence_gate(text_probs: np.ndarray, asr_confidence: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Soft-gate text sentiment toward uniform when ASR confidence is low, rather than hard-zeroing.

    Parameters
    ----------
    text_probs:
        `(N, C)` sentiment probabilities.
    asr_confidence:
        `(N,)` per-chunk ASR confidence.
    threshold:
        Confidence at/above which the text signal is fully trusted.
    """
    n_classes = text_probs.shape[1]
    uniform = np.full_like(text_probs, 1.0 / n_classes)
    weight = np.clip(asr_confidence / max(threshold, EPS), 0.0, 1.0)[:, None]
    return weight * text_probs + (1 - weight) * uniform


def build_feature_table(
    df: pd.DataFrame, mtkd_prob_cols: list[str], text_fi_cols: list[str], text_en_cols: list[str] | None = None,
    teacher_col: str = "teacher_id", use_mfcc: bool = False, mfcc_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Build the compact numeric feature table `tea.confidence`'s reliability nets consume.

    Parameters
    ----------
    df:
        Output of `build_master_table`.
    mtkd_prob_cols:
        Column names holding per-class MTKD probabilities.
    text_fi_cols, text_en_cols:
        Column names for FI (required) / EN (optional) sentiment probabilities.
    teacher_col:
        Used for per-teacher speech-rate normalization.
    use_mfcc, mfcc_cols:
        If `use_mfcc=True`, appends `mfcc_cols` from `df` as-is.

    Returns
    -------
    pd.DataFrame
        Numeric feature table, same row order as `df`, ready for
        `StandardScaler` + model input. Does NOT include the label.
    """
    from tea.features.softmax import mtkd_derived_features

    out = pd.DataFrame(index=df.index)

    mtkd_probs = df[mtkd_prob_cols].to_numpy()
    out = pd.concat([out, mtkd_derived_features(mtkd_probs, prefix="mtkd")], axis=1)

    out["log_rms_mean"] = np.log(df["rms_mean"].clip(lower=EPS))
    out["rms_std"] = df["rms_std"]
    out["log_duration"] = np.log(df["duration_sec"].clip(lower=EPS))
    out["f0_mean"] = df["f0_mean"]
    out["f0_std"] = df["f0_std"]
    out["voiced_ratio"] = df["voiced_ratio"]

    tmp = df[[teacher_col, "speech_rate"]].copy()
    out["speech_rate_z_per_teacher"] = per_teacher_zscore(tmp, "speech_rate", teacher_col)

    if use_mfcc and mfcc_cols:
        out = pd.concat([out, df[mfcc_cols].reset_index(drop=True)], axis=1)

    fi_probs = df[text_fi_cols].to_numpy()
    if "asr_confidence" in df.columns:
        fi_probs = asr_confidence_gate(fi_probs, df["asr_confidence"].to_numpy())
    for i, c in enumerate(text_fi_cols):
        out[f"text_fi_p{i}"] = fi_probs[:, i]

    if text_en_cols and all(col in df.columns for col in text_en_cols):
        en_probs = df[text_en_cols].to_numpy()
        out["text_fi_en_disagreement"] = text_sentiment_agreement(fi_probs, en_probs)

    return out
