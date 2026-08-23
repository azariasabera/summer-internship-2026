# src/tea/utils/io.py

"""Common JSON / CSV loaders shared across analysis, confidence, and probes."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


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
