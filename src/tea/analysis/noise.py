# src/tea/analysis/noise.py

"""Noise-condition prediction-distribution comparison (report Section 4.1, Table 21)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from tea.utils.constants import CLASS_ORDER
from tea.utils.io import load_nested_json
from tea.utils.logging import get_logger

logger = get_logger(__name__)


def class_distribution(prediction_json_path: str | Path, class_order: list[str] = CLASS_ORDER) -> dict[str, float]:
    """Predicted-label distribution (as fractions) across every chunk in a prediction JSON.

    Parameters
    ----------
    prediction_json_path:
        MTKD prediction JSON (`{video: {chunk: {class: prob}}}`).
    class_order:
        Which classes to report (and their order).
    """
    predictions = load_nested_json(prediction_json_path)

    counts = {c: 0 for c in class_order}
    total = 0
    for _video, chunks in predictions.items():
        for _chunk, class_probs in chunks.items():
            pred_label = max(class_probs, key=class_probs.get)
            if pred_label in counts:
                counts[pred_label] += 1
            total += 1

    if total == 0:
        raise ValueError(f"No chunks found in {prediction_json_path}")
    return {c: counts[c] / total for c in class_order}


def compare_conditions(condition_json_paths: dict[str, str | Path], class_order: list[str] = CLASS_ORDER) -> pd.DataFrame:
    """Build report Table 21: one row per noise condition, one column per class.

    Parameters
    ----------
    condition_json_paths:
        `{condition_name: prediction_json_path}`, e.g.
        `{"Original": ..., "DeepFilterNet 15 dB": ..., "Retrain 15-30 dB": ...}`.
    class_order:
        Which classes to report (and their order).
    """
    rows = []
    for condition, path in condition_json_paths.items():
        dist = class_distribution(path, class_order)
        rows.append({"condition": condition, **dist})
    df = pd.DataFrame(rows).set_index("condition")
    logger.info("\n%s", (df * 100).round(1).astype(str) + "%")
    return df


def non_speech_prediction_distribution(prediction_json_path: str | Path, annotation_root: str | Path, class_order: list[str] = CLASS_ORDER) -> dict[str, float]:
    """Sanity check (report Section 4.1): what does the model predict on chunks with NO ground-truth label (non-speech)?

    Expected: mostly neutral. A video whose non-speech regions get
    predicted as something else (e.g. `1B3261`'s flat microphone-noise
    segments getting predicted as anger) indicates an acoustic-artifact
    confound, not a genuine emotional signal.

    Parameters
    ----------
    prediction_json_path:
        MTKD prediction JSON.
    annotation_root:
        Directory of per-video annotation CSVs (to identify which chunks are non-speech).
    class_order:
        Which classes to report.
    """
    predictions = load_nested_json(prediction_json_path)
    counts = {c: 0 for c in class_order}
    total = 0

    for video_name, chunks in predictions.items():
        csv_path = Path(annotation_root) / f"{video_name}.csv"
        if not csv_path.exists():
            continue
        ann = pd.read_csv(csv_path)
        non_speech_chunks = set(ann.loc[ann["gt_label"].isna(), "name"])

        for chunk_name, class_probs in chunks.items():
            if chunk_name not in non_speech_chunks:
                continue
            pred_label = max(class_probs, key=class_probs.get)
            if pred_label in counts:
                counts[pred_label] += 1
            total += 1

    if total == 0:
        raise ValueError("No non-speech chunks found -- check annotation_root/prediction_json_path match.")
    return {c: counts[c] / total for c in class_order}


def noise_analysis_cli(cfg: DictConfig) -> int:
    """`tea noise-analysis` entry point.

    Requires `analysis.noise_conditions` -- a mapping of condition name to
    prediction JSON path, e.g. set via a YAML override file or:
    `tea noise-analysis +analysis.noise_conditions.Original=generated/predictions/baseline.json +analysis.noise_conditions."DeepFilterNet 15 dB"=generated/predictions/df15.json`
    """
    ac = cfg.analysis
    if not ac.get("noise_conditions"):
        logger.error("Set analysis.noise_conditions (mapping of condition name -> prediction json path)")
        return 2

    table = compare_conditions(dict(ac.noise_conditions))

    if ac.get("output_dir"):
        from tea.utils.paths import ensure_dir, resolve

        out_dir = ensure_dir(resolve(ac.output_dir))
        table.to_csv(out_dir / "noise_condition_distribution.csv")
        logger.info("Saved -> %s", out_dir / "noise_condition_distribution.csv")

    if ac.get("check_non_speech", False):
        for condition, path in dict(ac.noise_conditions).items():
            try:
                dist = non_speech_prediction_distribution(path, cfg.paths.annotation_root)
                logger.info("%s non-speech distribution: %s", condition, {k: round(v, 3) for k, v in dist.items()})
            except ValueError as e:
                logger.warning("%s: %s", condition, e)

    return 0
