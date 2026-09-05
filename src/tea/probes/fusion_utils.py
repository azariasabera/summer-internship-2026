# src/tea/probes/fusion_utils.py

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tea.utils.logging import get_logger

logger = get_logger(__name__)


def get_feature_groups(df: pd.DataFrame, mtkd_cols: list[str]) -> dict[str, list[str]]:
    """Group features into categories for ablation experiments.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing features and labels.
    mtkd_cols : list[str]
        List of columns corresponding to the MTKD softmax outputs.

    Returns
    -------
    dict[str, list[str]]
        Dictionary mapping group names to lists of feature column names.
    """
    groups: dict[str, list[str]] = {"softmax": list(mtkd_cols)}

    embedding_cols = [c for c in df.columns if c.startswith("embedding_")]
    if embedding_cols:
        groups["embedding"] = embedding_cols

    softmax_derived = [c for c in ("mtkd_entropy", "mtkd_margin", "mtkd_max_prob") if c in df.columns]
    if softmax_derived:
        groups["softmax_derived"] = softmax_derived

    acoustic = [
        c for c in ("rms_mean", "rms_std", "f0_mean", "f0_std", "voiced_ratio", "speech_rate")
        if c in df.columns
    ]
    if acoustic:
        groups["acoustic"] = acoustic

    text_cols = [c for c in df.columns if c.startswith("text_fi_") or c.startswith("text_en_")]
    if text_cols:
        groups["text_sentiment"] = text_cols

    return groups


def get_feature_combinations(groups: dict[str, list[str]], base: str = "softmax") -> list[tuple[str, ...]]:
    """Generate all combinations of feature groups, always including the base group.

    Parameters
    ----------
    groups : dict[str, list[str]]
        Dictionary mapping group names to lists of feature column names.
    base : str
        The base group that must be included in every combination. Default is "softmax". Could also be "embedding" or "softmax_derived".
    
    Returns
    -------
    list[tuple[str, ...]]
        List of tuples, each representing a combination of feature group names.
    """
    optional = [g for g in groups if g != base]
    combos = []
    for r in range(len(optional) + 1):
        for subset in itertools.combinations(optional, r):
            combos.append((base,) + subset)
    return combos


def make_model(seed: int = 42) -> Pipeline:
    """Create a scikit-learn pipeline with imputation, scaling, and logistic regression.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    Pipeline
        A scikit-learn Pipeline object that includes a SimpleImputer, StandardScaler, and LogisticRegression.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            C=1.0,
            max_iter=2000,
            solver="lbfgs",
            multi_class="multinomial",
            random_state=seed,
        )),
    ])


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate UAR and WAR metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.   

    Returns
    -------
    dict[str, float]
        Dictionary containing 'uar' (Unweighted Average Recall) and 'war' (Weighted Average Recall) metrics.    
    """
    return {
        "uar": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "war": float(accuracy_score(y_true, y_pred)),
    }


def run_feature_set(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    teacher_col: str,
    n_classes: int,
    seed: int = 42,
) -> dict[str, Any]:
    """Run a single feature set through Leave-One-Group-Out cross-validation and calculate metrics.
    Group in our case is the teacher ID, so we are doing Leave-One-Teacher-Out CV.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing features, labels, and teacher IDs.
    feature_cols : list[str]
        List of feature column names to use for this run.
    label_col : str
        Name of the column containing the true labels.
    teacher_col : str
        Name of the column containing the teacher IDs for grouping.
    n_classes : int
        Number of unique classes in the label column. Useful if some folds may not contain all classes.
    seed : int
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    dict[str, Any]
        Dictionary containing metrics such as number of features, number of folds, mean UAR, mean WAR, 
        standard deviations, and pooled metrics across all folds.
    """
    X = df[feature_cols]
    y = df[label_col].to_numpy()
    teacher_ids = df[teacher_col].to_numpy()

    fold_metrics: list[dict[str, float]] = []
    oof_true: list[int] = []
    oof_pred: list[int] = []

    logo = LeaveOneGroupOut()
    for train_idx, test_idx in logo.split(X, y, groups=teacher_ids):
        y_train = y[train_idx]

        if len(np.unique(y_train)) < n_classes:
            continue

        model = make_model(seed=seed)
        model.fit(X.iloc[train_idx], y_train)
        preds = model.predict(X.iloc[test_idx])

        fold_metrics.append(calculate_metrics(y[test_idx], preds))
        oof_true.extend(y[test_idx].tolist())
        oof_pred.extend(preds.tolist())

    if not fold_metrics:
        logger.warning("No valid folds for feature set with %d columns", len(feature_cols))
        return {
            "n_features": len(feature_cols),
            "n_folds": 0,
            "mean_uar": float("nan"),
            "mean_war": float("nan"),
            "pooled_uar": float("nan"),
            "pooled_war": float("nan"),
        }

    pooled = calculate_metrics(np.asarray(oof_true), np.asarray(oof_pred))

    return {
        "n_features": len(feature_cols),
        "n_folds": len(fold_metrics),
        "mean_uar": float(np.mean([m["uar"] for m in fold_metrics])),
        "mean_war": float(np.mean([m["war"] for m in fold_metrics])),
        "std_uar": float(np.std([m["uar"] for m in fold_metrics])),
        "std_war": float(np.std([m["war"] for m in fold_metrics])),
        "pooled_uar": pooled["uar"],
        "pooled_war": pooled["war"],
    }

def run_experiment(
    df: pd.DataFrame,
    groups: dict[str, list[str]],
    label_col: str,
    teacher_col: str,
    n_classes: int,
    base: str = "softmax",
    seed: int = 42,
) -> pd.DataFrame:
    """Run every combination and return a comparison table.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing features, labels, and teacher IDs.
    groups : dict[str, list[str]]
        Dictionary mapping group names to lists of feature column names.
    label_col : str
        Name of the column containing the true labels.
    teacher_col : str
        Name of the column containing the teacher IDs for grouping.
    n_classes : int
        Number of unique classes in the label column. Useful if some folds may not contain all classes.
    base : str
        The base group that must be included in every combination. Default is "softmax". My bases are "softmax" or "embedding".
    seed : int
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    pd.DataFrame
        DataFrame containing metrics for each feature set combination, including deltas vs the base feature set.
    """
    combos = get_feature_combinations(groups, base=base)
    rows = []

    for combo in combos:
        feature_cols = [c for g in combo for c in groups[g]]
        combo_name = "+".join(combo)
        logger.info("Running feature set: %s (%d columns)", combo_name, len(feature_cols))

        result = run_feature_set(
            df=df,
            feature_cols=feature_cols,
            label_col=label_col,
            teacher_col=teacher_col,
            n_classes=n_classes,
            seed=seed,
        )
        rows.append({"feature_set": combo_name, **result})

    result_df = pd.DataFrame(rows)

    # Delta vs pure base
    baseline_mask = result_df["feature_set"] == base
    if baseline_mask.any():
        base_uar = result_df.loc[baseline_mask, "mean_uar"].iloc[0]
        base_war = result_df.loc[baseline_mask, "mean_war"].iloc[0]
        result_df["delta_uar_vs_base"] = result_df["mean_uar"] - base_uar
        result_df["delta_war_vs_base"] = result_df["mean_war"] - base_war

    result_df = result_df.sort_values("mean_uar", ascending=False).reset_index(drop=True)
    logger.info("\n%s", result_df.to_string(index=False))
    return result_df