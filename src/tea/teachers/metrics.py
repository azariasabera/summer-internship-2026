# src/tea/teachers/metrics.py

"""Metrics used by teacher training and evaluation.

Provides helpers for computing UAR and WAR from model predictions.
"""

from __future__ import annotations

from sklearn.metrics import recall_score


def recalls(actual: list[int], predicted: list[int]) -> tuple[float, float]:
    return (
        recall_score(actual, predicted, average="macro"),
        recall_score(actual, predicted, average="weighted"),
    )