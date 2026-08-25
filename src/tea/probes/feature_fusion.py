# src/tea/probes/feature_fusion.py

"""Handcrafted-feature-fusion ablation: does adding acoustic/text/softmax-
derived features on top of the raw MTKD softmax (or the raw 768-d
embedding) actually help predict the ground-truth emotion label?

Answers the report's finding directly: adding handcrafted features barely
moves UAR/WAR either way, and using the raw 768-d embedding as the base
representation performs WORSE than using the 4 softmax scores alone.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from tea.utils.logging import get_logger

logger = get_logger(__name__)


def get_feature_groups(df: pd.DataFrame, mtkd_cols: list[str]) -> dict[str, list[str]]:
    """Bucket available columns into named feature groups for combinatorial ablation.

    Parameters
    ----------
    df:
        Output of `tea.features.build_master_table(embedding_root=...)`.
    mtkd_cols:
        The raw MTKD softmax column names (the mandatory base representation).
    """
    groups = {"softmax": list(mtkd_cols)}

    embedding_cols = [c for c in df.columns if c.startswith("embedding_")]
    if embedding_cols:
        groups["embedding"] = embedding_cols

    softmax_derived = [c for c in ("mtkd_entropy", "mtkd_margin", "mtkd_max_prob") if c in df.columns]
    if softmax_derived:
        groups["softmax_derived"] = softmax_derived

    acoustic = [c for c in ("rms_mean", "rms_std", "f0_mean", "f0_std", "voiced_ratio", "speech_rate") if c in df.columns]
    if acoustic:
        groups["acoustic"] = acoustic

    text_cols = [c for c in df.columns if c.startswith("text_fi_") or c.startswith("text_en_")]
    if text_cols:
        groups["text_sentiment"] = text_cols

    return groups


def get_feature_combinations(groups: dict[str, list[str]], base: str = "softmax") -> list[tuple[str, ...]]:
    """Every combination of optional groups added on top of the mandatory `base` group.

    `base="softmax"` (4 columns) or `base="embedding"` (768 columns) --
    the report's two base-representation conditions (Table 23).

    Returns
    -------
    list
        Tuples of group names, e.g. `("softmax",)`, `("softmax", "acoustic")`, ...
    """
    optional = [g for g in groups if g != base]
    combos = []
    for r in range(len(optional) + 1):
        for subset in itertools.combinations(optional, r):
            combos.append((base,) + subset)
    return combos


def get_folds(teacher_ids: np.ndarray):
    """Leave-one-teacher-out folds. Thin wrapper for readability at call sites."""
    return LeaveOneGroupOut().split(np.zeros(len(teacher_ids)), groups=teacher_ids)


def make_model(seed: int = 42) -> LogisticRegression:
    """Standard multinomial logistic regression used for every feature-set comparison, so only the features vary."""
    return LogisticRegression(max_iter=2000, random_state=seed, multi_class="multinomial")


def calculate_metrics(y_true, y_pred, n_classes: int) -> dict:
    """UAR/WAR + confusion matrix for one fold or one pooled OOF result."""
    labels = list(range(n_classes))
    return {
        "uar": recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "war": recall_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def run_feature_set(df: pd.DataFrame, feature_cols: list[str], label_col: str, teacher_col: str, n_classes: int, seed: int = 42) -> dict:
    """LOTO-evaluate one specific feature-column combination.

    Parameters
    ----------
    df:
        Full feature table.
    feature_cols:
        Which columns to use as model input for this run.
    label_col, teacher_col:
        Column names for the label and LOTO grouping.
    n_classes:
        Number of emotion classes.
    seed:
        Random seed passed to `make_model`.
    """
    X = df[feature_cols].to_numpy()
    y = df[label_col].to_numpy()
    teacher_ids = df[teacher_col].to_numpy()

    fold_metrics = []
    oof_true, oof_pred = [], []

    for train_idx, test_idx in get_folds(teacher_ids):
        scaler = StandardScaler().fit(X[train_idx])
        X_train, X_test = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])

        model = make_model(seed=seed)
        model.fit(X_train, y[train_idx])
        preds = model.predict(X_test)

        fold_metrics.append(calculate_metrics(y[test_idx], preds, n_classes))
        oof_true.extend(y[test_idx].tolist())
        oof_pred.extend(preds.tolist())

    pooled = calculate_metrics(np.array(oof_true), np.array(oof_pred), n_classes)
    return {
        "n_features": len(feature_cols),
        "mean_uar": float(np.mean([m["uar"] for m in fold_metrics])),
        "mean_war": float(np.mean([m["war"] for m in fold_metrics])),
        "pooled_uar": pooled["uar"],
        "pooled_war": pooled["war"],
        "pooled_confusion_matrix": pooled["confusion_matrix"],
    }


def run_experiment(df: pd.DataFrame, groups: dict[str, list[str]], label_col: str, teacher_col: str, n_classes: int, base: str = "softmax", seed: int = 42) -> pd.DataFrame:
    """Run every feature-group combination (report Table 23) and return a comparison table.

    Parameters
    ----------
    df:
        Output of `tea.features.build_master_table(embedding_root=...)`.
    groups:
        Output of `get_feature_groups`.
    label_col, teacher_col:
        Column names for label and LOTO grouping.
    n_classes:
        Number of emotion classes.
    base:
        `"softmax"` or `"embedding"` -- which mandatory base representation to build on.
    seed:
        Random seed.
    """
    combos = get_feature_combinations(groups, base=base)
    rows = []
    for combo in combos:
        feature_cols = [c for group in combo for c in groups[group]]
        combo_name = "+".join(combo)
        logger.info("Running feature set: %s (%d columns)", combo_name, len(feature_cols))
        result = run_feature_set(df, feature_cols, label_col, teacher_col, n_classes, seed=seed)
        rows.append({"feature_set": combo_name, **{k: v for k, v in result.items() if k != "pooled_confusion_matrix"}})

    result_df = pd.DataFrame(rows).sort_values("pooled_uar", ascending=False).reset_index(drop=True)
    logger.info("\n%s", result_df.to_string(index=False))
    return result_df


def probe_feature_fusion_cli(cfg: DictConfig) -> int:
    """`tea probe-feature-fusion` entry point.

    Requires `probes.feature_fusion.mtkd_json` and `.sentiment_fi_json`.
    Runs the ablation for both `base="softmax"` and `base="embedding"`
    (report Table 23's two conditions) unless `probes.feature_fusion.base`
    restricts to one.
    """
    from tea.features.master_table import build_master_table

    ff = cfg.probes.get("feature_fusion", {})
    if not ff.get("mtkd_json") or not ff.get("sentiment_fi_json"):
        logger.error("Set probes.feature_fusion.mtkd_json and .sentiment_fi_json")
        return 2

    df, mtkd_classes, sentiment_classes = build_master_table(
        csv_root=cfg.paths.annotation_root, audio_root=cfg.paths.audio_root,
        mtkd_json_path=ff.mtkd_json, sentiment_fi_json_path=ff.sentiment_fi_json,
        sentiment_en_json_path=ff.get("sentiment_en_json"),
        embedding_root=ff.get("embedding_root") or cfg.paths.embedding_root,
    )

    label_map = {c: i for i, c in enumerate(mtkd_classes)}
    df["_label_id"] = df["gt_label"].map(label_map)
    groups = get_feature_groups(df, mtkd_cols=[f"mtkd_{c}" for c in mtkd_classes])

    bases = [ff.base] if ff.get("base") else [b for b in ("softmax", "embedding") if b in groups]
    all_results = {}
    for base in bases:
        logger.info("=== Base representation: %s ===", base)
        all_results[base] = run_experiment(df, groups, label_col="_label_id", teacher_col="teacher_id", n_classes=len(mtkd_classes), base=base, seed=cfg.seed)

    if ff.get("output_dir"):
        from tea.utils.paths import ensure_dir, resolve

        out_dir = ensure_dir(resolve(ff.output_dir))
        for base, result_df in all_results.items():
            result_df.to_csv(out_dir / f"feature_fusion_{base}.csv", index=False)
        logger.info("Saved -> %s", out_dir)

    return 0
