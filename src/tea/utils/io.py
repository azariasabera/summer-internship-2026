# src/tea/utils/io.py

"""Common JSON / CSV loaders shared across analysis, confidence, and probes."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from tea.utils.logging import get_logger
from tea.utils.paths import ensure_dir, resolve

from omegaconf import DictConfig

logger = get_logger(__name__)


def get_teacher_id(video_id: str) -> str:
    """Strip a trailing `_videoN` suffix to get the teacher grouping id.

    Used for leave-one-teacher-out folds: recordings from the same teacher
    (e.g. `1B2251` and `1B2251_video2`) must land in the same fold.

    Parameters
    ----------
    video_id:
        Video identifier, e.g. `"1B2251_video2"`.

    Examples
    --------
    >>> get_teacher_id("1B2251_video2")
    '1B2251'
    >>> get_teacher_id("1B2251")
    '1B2251'
    """
    return re.sub(r"_video\d+$", "", video_id)


def load_annotation_csvs(annotation_root: str | Path, exclude: set[str] | None = None) -> pd.DataFrame:
    """Load and concatenate every per-video annotation CSV into one DataFrame.

    Parameters
    ----------
    annotation_root:
        Directory containing one CSV per video (`<video_id>.csv`).
    exclude:
        Video ids to skip (e.g. `constants.EXCLUDED_VIDEOS`).

    Returns
    -------
    pd.DataFrame
        Concatenated table with an added `video` column (the CSV's stem)
        and a `teacher` column (via `get_teacher_id`).
    """
    exclude = exclude or set()
    root = Path(annotation_root)

    frames = []
    for csv_path in sorted(root.glob("*.csv")):
        video_id = csv_path.stem
        if video_id in exclude:
            continue

        df = pd.read_csv(csv_path)
        df["video"] = video_id
        df["teacher"] = get_teacher_id(video_id)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No annotation CSVs found under {root}")

    return pd.concat(frames, ignore_index=True)


def load_nested_json(path: str | Path) -> dict:
    """Load a JSON file shaped `{video_id: {chunk_name: {...}}}`.

    This is the shape used by MTKD prediction JSONs and the FI/EN
    sentiment-probability JSONs.

    Parameters
    ----------
    path:
        Path to the JSON file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_embeddings(embedding_root: str | Path, video_ids: list[str] | None = None) -> dict[str, dict]:
    """Load pooled embeddings for every video under `embedding_root`.

    Expects the layout produced by `tea.mtkd.extract_embeddings`:
    `<embedding_root>/<video_id>/pooled.npy` (shape `[n_chunks, 768]`) plus
    `<embedding_root>/<video_id>/chunk_names.json` (list of chunk names,
    same order as the rows of `pooled.npy`).

    Parameters
    ----------
    embedding_root:
        Root directory containing one subfolder per video.
    video_ids:
        Restrict loading to these video ids. Defaults to every subfolder found.

    Returns
    -------
    dict
        `{video_id: {"embeddings": np.ndarray, "chunk_names": list[str]}}`.
    """
    root = Path(embedding_root)
    if video_ids is None:
        video_ids = sorted(p.name for p in root.iterdir() if p.is_dir())

    out: dict[str, dict] = {}
    for video_id in video_ids:
        video_dir = root / video_id
        pooled_path = video_dir / "pooled.npy"
        names_path = video_dir / "chunk_names.json"

        if not pooled_path.exists():
            continue

        embeddings = np.load(pooled_path)
        with open(names_path, "r", encoding="utf-8") as f:
            chunk_names = json.load(f)

        out[video_id] = {"embeddings": embeddings, "chunk_names": chunk_names}

    return out


def add_predictions_to_annotation_csvs(
    predictions_json_path: str | Path,
    annotation_root: str | Path,
    output_dir: str | Path,
) -> None:
    """Join an MTKD prediction JSON onto per-video annotation CSVs.

    For each video in the predictions JSON, reads the matching annotation
    CSV, and for every chunk (matched by the `name` column) adds
    `pred_label` and `pred_confidence` columns from the prediction JSON's
    `pred_label` / `score` fields. Chunks present in the CSV but missing
    from the predictions JSON are left with `pred_label = NaN` and are
    dropped in downstream analysis where a prediction is required.

    This is the previously-missing local `utils.py` function that
    `checking_pred.ipynb` imported — reconstructed from its usage in that
    notebook (`from utils import add_predictions_to_annotation_csvs`), since
    the original file was not part of the internship-dump upload.

    Parameters
    ----------
    predictions_json_path:
        Path to a prediction JSON shaped `{video_id: {chunk_name: {"pred_label": ..., "score": ...}}}`.
    annotation_root:
        Directory containing the per-video annotation CSVs to join onto.
    output_dir:
        Directory to write the joined CSVs to (one per video, same filename).
    """
    predictions = load_nested_json(predictions_json_path)
    annotation_root = Path(annotation_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for video_id, chunk_preds in predictions.items():
        csv_path = annotation_root / f"{video_id}.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        df["pred_label"] = df["name"].map(lambda n: chunk_preds.get(n, {}).get("pred_label"))
        df["pred_confidence"] = df["name"].map(lambda n: chunk_preds.get(n, {}).get("score"))

        df.to_csv(output_dir / f"{video_id}.csv", index=False)

def vad_json_to_annotation_csv(
    json_path: str | Path,
    output_csv: str | Path,
    chunk_col: str = "name",
    label_col: str = "gt_label",
) -> pd.DataFrame:
    """Convert a VAD segmenter JSON into a per-video annotation CSV.

    The JSON produced by ``tea.vad.Segmenter.chunk_vad`` looks like::

        {
          "audio_path": "...",
          "sr": 16000,
          "total_samples": ...,
          "segments": [
            {"start": 3520, "end": 140864, "type": "speech"},
            {"start": 140864, "end": 300864, "type": "non-speech"},
            ...
          ]
        }

    Writes a CSV using the original classroom annotation naming convention::

        name, duration_sec, start, end, type, gt_label, confidence, overlap

    Chunk names follow the original convention:

        speech     -> chunk_s_0, chunk_s_1, ...
        non-speech -> chunk_n_0, chunk_n_1, ...

    ``gt_label``, ``confidence`` and ``overlap`` are left as NaN.

    Parameters
    ----------
    json_path:
        Path to one VAD JSON file.
    output_csv:
        Destination CSV path (parent dirs are created if needed).

    Returns
    -------
    pd.DataFrame
        The table that was written.
    """
    json_path = Path(json_path)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    sr = int(meta.get("sr", 16000))

    rows = []
    for i, seg in enumerate(meta.get("segments", [])):
        start = int(seg["start"])
        end = int(seg["end"])
        segment_type = seg.get("type", "speech")

        # Original annotation convention:
        # speech     -> chunk_s_N
        # non-speech -> chunk_n_N
        prefix = "s" if segment_type == "speech" else "n"

        rows.append(
            {
                chunk_col: f"chunk_{prefix}_{i}",
                "duration_sec": (end - start) / sr,
                "start": start,
                "end": end,
                "type": segment_type,
                label_col: np.nan,
                "confidence": np.nan,
                "overlap": np.nan,
            }
        )

    df = pd.DataFrame(
        rows,
        columns=[
            chunk_col,
            "duration_sec",
            "start",
            "end",
            "type",
            label_col,
            "confidence",
            "overlap",
        ],
    )

    df.to_csv(output_csv, index=False)
    return df

_ANNOTATION_COLS = ("gt_label", "confidence", "overlap")

def merge_annotations(cfg: DictConfig) -> int:
    """Merge `gt_label`, `confidence`, `overlap` from prepared CSVs.

    Typical layout
    --------------
    - `paths.prepared_annotation_root` (e.g. `data/annotations`) holds the
      internship CSVs already labelled by you.
    - `paths.annotation_root` (e.g. `generated/annotations`) holds the
      CSVs produced by `tea chunk`.

    Matching is by the `name` column (chunk id).
    """
    prepared_root = resolve(cfg.paths.prepared_annotation_root)
    target_root = ensure_dir(resolve(cfg.paths.annotation_root))

    if not prepared_root.is_dir():
        logger.error("Prepared annotation root does not exist: %s", prepared_root)
        return 1

    prepared_files = sorted(prepared_root.glob("*.csv"))
    if not prepared_files:
        logger.error("No CSV files under %s", prepared_root)
        return 1

    n_ok = 0
    for prep_path in prepared_files:
        video_id = prep_path.stem
        target_path = target_root / f"{video_id}.csv"

        prep = pd.read_csv(prep_path)
        if "name" not in prep.columns:
            logger.warning("Skip %s: no 'name' column", prep_path.name)
            continue

        missing_cols = [c for c in _ANNOTATION_COLS if c not in prep.columns]
        if missing_cols:
            logger.warning(
                "Skip %s: prepared CSV missing columns %s",
                prep_path.name,
                missing_cols,
            )
            continue

        if target_path.exists():
            target = pd.read_csv(target_path)
            if "name" not in target.columns:
                logger.warning("Skip %s: target has no 'name' column", target_path.name)
                continue
            # Drop any previous label columns so the merge is clean.
            drop = [c for c in _ANNOTATION_COLS if c in target.columns]
            if drop:
                target = target.drop(columns=drop)
            merged = target.merge(
                prep[["name", *_ANNOTATION_COLS]],
                on="name",
                how="left",
            )
        else:
            # No VAD CSV yet, so copy prepared as the starting annotation table.
            logger.info(
                "No generated CSV for %s; copying prepared file into %s",
                video_id,
                target_root,
            )
            merged = prep.copy()

        merged.to_csv(target_path, index=False)
        n_labelled = int(merged["gt_label"].notna().sum()) if "gt_label" in merged.columns else 0
        logger.info(
            "  %s: %d / %d rows have gt_label -> %s",
            video_id,
            n_labelled,
            len(merged),
            target_path,
        )
        n_ok += 1

    logger.info("Merged annotations for %d video(s).", n_ok)
    return 0 if n_ok else 1

def merge_asr_annotations(cfg: DictConfig) -> int:
    """Merge `transcription` and `translation` from prepared CSVs.

    Typical layout
    --------------
    - `paths.prepared_annotation_root` (e.g. `data/annotations`) holds the
      prepared CSVs containing transcription and translation.
    - `paths.annotation_root` (e.g. `generated/annotations`) holds the
      CSVs produced by `tea chunk` (and optionally already ASR-filled).

    Matching is by the `name` column (chunk id).
    """
    prepared_root = resolve(cfg.paths.prepared_annotation_root)
    target_root = ensure_dir(resolve(cfg.paths.annotation_root))

    if not prepared_root.is_dir():
        logger.error("Prepared annotation root does not exist: %s", prepared_root)
        return 1

    prepared_files = sorted(prepared_root.glob("*.csv"))
    if not prepared_files:
        logger.error("No CSV files under %s", prepared_root)
        return 1

    asr_cols = ["transcription", "translation"]

    n_ok = 0
    for prep_path in prepared_files:
        video_id = prep_path.stem
        target_path = target_root / f"{video_id}.csv"

        prep = pd.read_csv(prep_path)
        if "name" not in prep.columns:
            logger.warning("Skip %s: no 'name' column", prep_path.name)
            continue

        missing_cols = [c for c in asr_cols if c not in prep.columns]
        if missing_cols:
            logger.warning("Skip %s: prepared CSV missing columns %s", prep_path.name, missing_cols)
            continue

        if target_path.exists():
            target = pd.read_csv(target_path)
            if "name" not in target.columns:
                logger.warning("Skip %s: target has no 'name' column", target_path.name)
                continue

            # Drop any previous transcription/translation columns so the merge
            # is clean and does not create _x/_y suffixes.
            drop = [c for c in asr_cols if c in target.columns]
            if drop:
                target = target.drop(columns=drop)

            merged = target.merge(prep[["name", *asr_cols]], on="name", how="left")
        else:
            # No generated CSV yet, so copy prepared as the starting annotation table.
            logger.info("No generated CSV for %s; copying prepared file into %s", video_id, target_root)
            merged = prep.copy()

        merged.to_csv(target_path, index=False)

        n_transcribed = (int(merged["transcription"].notna().sum()) if "transcription" in merged.columns else 0)
        n_translated = (int(merged["translation"].notna().sum()) if "translation" in merged.columns else 0)

        logger.info("  %s: %d / %d rows have transcription, %d / %d have translation -> %s",
                video_id, n_transcribed, len(merged), n_translated, len(merged), target_path)
        n_ok += 1

    logger.info("Merged ASR annotations for %d video(s).", n_ok)
    return 0 if n_ok else 1