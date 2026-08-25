# src/tea/features/softmax.py

"""MTKD softmax-derived scalar features (entropy, margin, max-prob)."""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-8


def softmax_entropy(probs: np.ndarray) -> np.ndarray:
    """Normalized entropy in [0, 1]. `probs`: `(N, C)`."""
    p = np.clip(probs, EPS, 1.0)
    n_classes = probs.shape[1]
    raw_entropy = -np.sum(p * np.log(p), axis=1)
    max_entropy = np.log(n_classes)
    return raw_entropy / max_entropy


def softmax_margin(probs: np.ndarray) -> np.ndarray:
    """Top1 - Top2 probability. Large margin = confident, unambiguous call."""
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    return sorted_probs[:, 0] - sorted_probs[:, 1]


def softmax_max_prob(probs: np.ndarray) -> np.ndarray:
    """Top-1 probability."""
    return np.max(probs, axis=1)


def mtkd_derived_features(mtkd_probs: np.ndarray, prefix: str = "mtkd") -> pd.DataFrame:
    """Raw probs (as `<prefix>_p0..pN`) + entropy + margin + max_prob, for `tea.confidence`'s feature table.

    Parameters
    ----------
    mtkd_probs:
        `(N, C)` array of MTKD softmax outputs.
    prefix:
        Column-name prefix.
    """
    n_classes = mtkd_probs.shape[1]
    cols = {f"{prefix}_p{i}": mtkd_probs[:, i] for i in range(n_classes)}
    cols[f"{prefix}_entropy"] = softmax_entropy(mtkd_probs)
    cols[f"{prefix}_margin"] = softmax_margin(mtkd_probs)
    cols[f"{prefix}_max_prob"] = softmax_max_prob(mtkd_probs)
    return pd.DataFrame(cols)


def add_softmax_engineered_features(df: pd.DataFrame, class_cols: list[str] = ("neutral", "sadness", "happiness", "anger")) -> pd.DataFrame:
    """Keep original class-name columns and append `mtkd_entropy`/`mtkd_margin`/`mtkd_max_prob`, for `tea.probes.feature_fusion`.

    Parameters
    ----------
    df:
        Must contain `class_cols` (the raw MTKD softmax columns).
    class_cols:
        Column names holding the per-class probabilities, in class order.
    """
    class_cols = list(class_cols)
    probs = df[class_cols].to_numpy(dtype=float)
    probs = np.clip(probs, EPS, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)  # renormalize in case of tiny numerical drift

    df = df.copy()
    df["mtkd_entropy"] = softmax_entropy(probs)
    df["mtkd_margin"] = softmax_margin(probs)
    df["mtkd_max_prob"] = softmax_max_prob(probs)
    return df
