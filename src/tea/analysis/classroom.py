# src/tea/analysis/classroom.py

"""Classroom evaluation: WAR/UAR/F1/confusion matrices (per-video and
overall), plus the accuracy-by-child-speech and accuracy-by-annotation-
confidence breakdowns (report Sections 2.1, 2.1.3, Tables 6-13). ~ checking_pred
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

from tea.utils.constants import CLASS_ORDER
from tea.utils.io import load_nested_json
from tea.utils.logging import get_logger

logger = get_logger(__name__)

THREE_CLASS_MAP = {"happiness": "positive", "anger": "negative", "sadness": "negative", "neutral": "neutral"}
THREE_CLASS_LABELS = ["negative", "neutral", "positive"]


def join_predictions(mtkd_json_path: str | Path, annotation_root: str | Path, excluded_videos: set[str] | None = None) -> pd.DataFrame:
    """Join an MTKD prediction JSON onto annotation CSVs, keeping only chunks with a ground-truth label.

    Parameters
    ----------
    mtkd_json_path:
        MTKD prediction JSON (`{video: {chunk: {class: prob}}}`).
    annotation_root:
        Directory of per-video annotation CSVs.
    excluded_videos:
        Video ids to skip.

    Returns
    -------
    pd.DataFrame
        One row per annotated chunk with `video`, `chunk`, `gt_label`,
        `pred_label`, `pred_confidence`, plus whatever other annotation
        columns existed (`confidence`, `overlap`, `start`, `end`, ...).
    """
    predictions = load_nested_json(mtkd_json_path)
    excluded_videos = excluded_videos or set()

    rows = []
    for video_name, chunks in predictions.items():
        if video_name in excluded_videos:
            continue

        csv_path = Path(annotation_root) / f"{video_name}.csv"
        if not csv_path.exists():
            logger.warning("%s: no annotation csv found, skipping", video_name)
            continue
        ann = pd.read_csv(csv_path)

        for chunk_name, class_probs in chunks.items():
            pred_label = max(class_probs, key=class_probs.get)
            pred_score = class_probs[pred_label]

            match = ann.loc[ann["name"] == chunk_name]
            if match.empty:
                continue
            row = match.iloc[0].to_dict()
            if pd.isna(row.get("gt_label")):
                continue

            row.update({"video": video_name, "chunk": chunk_name, "pred_label": pred_label, "pred_confidence": pred_score})
            rows.append(row)

    df = pd.DataFrame(rows)
    df["gt_label"] = df["gt_label"].astype(str).str.strip().str.lower()
    logger.info("Joined %d annotated chunks with predictions across %d videos.", len(df), df["video"].nunique())
    return df


def evaluate_predictions(gt: list, pred: list, labels: list[str]) -> dict:
    """WAR/UAR/F1/confusion matrix for one gt/pred pair, at whatever label granularity is given.

    Called with `CLASS_ORDER` for the native 4-class report (Tables 6-8),
    or with `THREE_CLASS_LABELS` after mapping via `to_three_class` for
    the polarity comparison.

    Parameters
    ----------
    gt, pred:
        Aligned label lists/arrays.
    labels:
        Canonical label order for the confusion matrix/report.
    """
    cm = confusion_matrix(gt, pred, labels=labels)
    return {
        "confusion_matrix": cm,
        "confusion_matrix_df": pd.DataFrame(cm, index=labels, columns=labels),
        "classification_report": classification_report(gt, pred, labels=labels, digits=4, zero_division=0),
        "war": accuracy_score(gt, pred),
        "uar": balanced_accuracy_score(gt, pred),
        "macro_f1": f1_score(gt, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(gt, pred, average="weighted", zero_division=0),
    }


def to_three_class(labels: pd.Series) -> pd.Series:
    """Map native 4-class emotion labels to 3-class sentiment (happiness->positive, anger/sadness->negative, neutral->neutral)."""
    return labels.map(THREE_CLASS_MAP)


def evaluate_per_video_and_overall(df: pd.DataFrame, labels: list[str] = CLASS_ORDER, gt_col: str = "gt_label", pred_col: str = "pred_label") -> tuple[dict[str, dict], dict]:
    """Run `evaluate_predictions` per video AND pooled overall (report Tables 6-8's structure).

    Parameters
    ----------
    df:
        Output of `join_predictions`.
    labels:
        Label order.
    gt_col, pred_col:
        Column names.
    """
    per_video = {}
    for video_name, group in df.groupby("video"):
        per_video[video_name] = evaluate_predictions(group[gt_col].tolist(), group[pred_col].tolist(), labels)
        logger.info(
            "%s: n=%d WAR=%.4f UAR=%.4f MacroF1=%.4f", video_name, len(group), per_video[video_name]["war"],
            per_video[video_name]["uar"], per_video[video_name]["macro_f1"],
        )

    overall = evaluate_predictions(df[gt_col].tolist(), df[pred_col].tolist(), labels)
    logger.info("OVERALL: n=%d WAR=%.4f UAR=%.4f MacroF1=%.4f WeightedF1=%.4f", len(df), overall["war"], overall["uar"], overall["macro_f1"], overall["weighted_f1"])
    return per_video, overall


def accuracy_by_child_speech(df: pd.DataFrame, gt_col: str = "gt_label", pred_col: str = "pred_label", child_col: str = "overlap") -> pd.DataFrame:
    """Accuracy split by whether child speech was present (report Table 12)."""
    df = df.copy()
    df["correct"] = df[gt_col] == df[pred_col]
    df["child_present"] = df[child_col] > 0

    rows = []
    for present, group in df.groupby("child_present"):
        rows.append({"child_speech_present": bool(present), "accuracy": group["correct"].mean(), "n_correct": int(group["correct"].sum()), "n_total": len(group)})
    return pd.DataFrame(rows)


def accuracy_by_confidence(df: pd.DataFrame, gt_col: str = "gt_label", pred_col: str = "pred_label", confidence_col: str = "confidence") -> pd.DataFrame:
    """Accuracy split by 1-3 annotation-confidence level (report Table 11), and further by class (report's per-class confidence breakdown)."""
    df = df.copy()
    df["correct"] = df[gt_col] == df[pred_col]

    overall_rows = []
    for conf, group in df.groupby(confidence_col):
        overall_rows.append({"confidence": conf, "accuracy": group["correct"].mean(), "n_correct": int(group["correct"].sum()), "n_total": len(group)})

    per_class_rows = []
    for label in df[gt_col].unique():
        label_df = df[df[gt_col] == label]
        for conf in (1.0, 2.0, 3.0):
            subset = label_df[label_df[confidence_col] == conf]
            if len(subset) == 0:
                continue
            per_class_rows.append({"gt_label": label, "confidence": conf, "accuracy": subset["correct"].mean(), "n_total": len(subset)})

    return pd.DataFrame(overall_rows), pd.DataFrame(per_class_rows)


def child_speech_by_predicted_emotion(df: pd.DataFrame, labels: list[str] = CLASS_ORDER, gt_col: str = "gt_label", pred_col: str = "pred_label", child_col: str = "overlap") -> pd.DataFrame:
    """Proportion of chunks with child speech present, grouped by GT label vs. by predicted label side by side (report Fig. 9)."""
    rows = []
    for label in labels:
        gt_subset = df[df[gt_col] == label]
        pred_subset = df[df[pred_col] == label]
        rows.append({
            "label": label,
            "gt_child_speech_proportion": gt_subset[child_col].mean() if len(gt_subset) else np.nan,
            "gt_n": len(gt_subset),
            "pred_child_speech_proportion": pred_subset[child_col].mean() if len(pred_subset) else np.nan,
            "pred_n": len(pred_subset),
        })
    return pd.DataFrame(rows)


def class_distribution(df: pd.DataFrame, labels: list[str] = CLASS_ORDER, gt_col: str = "gt_label", pred_col: str = "pred_label") -> pd.DataFrame:
    """GT vs. predicted label distribution, side by side (report Fig. 1 vs Fig. 4)."""
    gt_props = df[gt_col].value_counts(normalize=True).reindex(labels, fill_value=0)
    pred_props = df[pred_col].value_counts(normalize=True).reindex(labels, fill_value=0)
    return pd.DataFrame({"label": labels, "gt_proportion": gt_props.values, "pred_proportion": pred_props.values})


def evaluate_classroom_cli(cfg: DictConfig) -> int:
    """`tea evaluate-classroom` entry point.

    Requires `analysis.mtkd_json`. Optional `analysis.three_class`
    (also report the 3-class sentiment-level comparison, Tables in
    checking_pred.ipynb cells 11-12).
    """
    ac = cfg.analysis
    if not ac.get("mtkd_json"):
        logger.error("Set analysis.mtkd_json")
        return 2

    df = join_predictions(ac.mtkd_json, cfg.paths.annotation_root, excluded_videos=set(cfg.classroom.excluded_videos))

    logger.info("=== 4-class evaluation ===")
    per_video, overall = evaluate_per_video_and_overall(df, labels=CLASS_ORDER)

    if ac.get("three_class", False):
        logger.info("=== 3-class sentiment evaluation ===")
        df["gt_3class"] = to_three_class(df["gt_label"])
        df["pred_3class"] = to_three_class(df["pred_label"])
        evaluate_per_video_and_overall(df, labels=THREE_CLASS_LABELS, gt_col="gt_3class", pred_col="pred_3class")

    logger.info("=== Accuracy by child speech ===")
    logger.info("\n%s", accuracy_by_child_speech(df))

    logger.info("=== Accuracy by annotation confidence ===")
    overall_conf, per_class_conf = accuracy_by_confidence(df)
    logger.info("\n%s", overall_conf)

    logger.info("=== Child speech by predicted emotion ===")
    logger.info("\n%s", child_speech_by_predicted_emotion(df))

    if ac.get("output_dir"):
        from tea.utils.paths import ensure_dir, resolve

        out_dir = ensure_dir(resolve(ac.output_dir))
        df.to_csv(out_dir / "joined_predictions.csv", index=False)
        overall["confusion_matrix_df"].to_csv(out_dir / "overall_confusion_matrix.csv")
        logger.info("Saved -> %s", out_dir)

    return 0
