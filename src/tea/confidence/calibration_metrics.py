# src/tea/confidence/calibration_metrics.py

"""Shared metrics for evaluating confidence/reliability scores.

Used by `binary.py`, `tcp.py`, and `temperature.py` so all three report
numbers on the same footing and are directly comparable against each
other and against raw max-softmax.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def expected_calibration_error(confidences: np.ndarray, correctness: np.ndarray, n_bins: int = 10) -> float:
    """Standard binned ECE: within each confidence bin, `|mean(confidence) - mean(correctness)|`, weighted by bin size.

    Lower is better calibrated. Note: ECE alone can't tell you whether
    correct/incorrect are SEPARATED (that's AUROC's job) -- a model that
    outputs a constant 0.8 for everything can have low ECE if 80% of
    predictions happen to be correct, while providing zero separation.
    Always read ECE alongside AUROC.
    """
    confidences = np.asarray(confidences, dtype=float)
    correctness = np.asarray(correctness, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(confidences)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if not mask.any():
            continue
        bin_acc = correctness[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def brier_score_binary(confidences: np.ndarray, correctness: np.ndarray) -> float:
    """Mean squared error between a scalar confidence score and the binary correctness outcome. Lower is better."""
    confidences = np.asarray(confidences, dtype=float)
    correctness = np.asarray(correctness, dtype=float)
    return float(np.mean((confidences - correctness) ** 2))


def brier_score_multiclass(probs: np.ndarray, true_labels: np.ndarray, n_classes: int) -> float:
    """Standard multiclass Brier score: MSE between the full probability vector and the one-hot true label.

    Use this for instance-dependent temperature scaling, which outputs a
    full reshaped distribution rather than a single scalar.
    """
    onehot = np.eye(n_classes)[true_labels]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def negative_log_likelihood(probs: np.ndarray, true_labels: np.ndarray, eps: float = 1e-8) -> float:
    """Mean NLL of the true class under the given probability vectors."""
    p_true = probs[np.arange(len(true_labels)), true_labels]
    return float(-np.mean(np.log(np.clip(p_true, eps, 1.0))))


def auroc_correctness(scores: np.ndarray, correctness: np.ndarray) -> float:
    """AUROC treating "correct" as the positive class.

    This is the metric temperature scaling can NEVER move (it's
    rank-invariant), so it's the key number for judging whether a given
    approach is actually adding separation beyond max-softmax. Returns
    NaN if the fold has only one class of correctness (AUROC undefined).
    """
    correctness = np.asarray(correctness)
    if len(np.unique(correctness)) < 2:
        return float("nan")
    return float(roc_auc_score(correctness, scores))


def risk_coverage_curve(scores: np.ndarray, correctness: np.ndarray, n_points: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Error rate ("risk") at each coverage level if you abstain on the lowest-confidence tail.

    Sorts chunks by descending confidence and computes, at each coverage
    level (fraction of chunks kept), the error rate among the kept chunks.

    Returns
    -------
    tuple
        `(coverages, risks)`, both length `n_points`.
    """
    scores = np.asarray(scores)
    correctness = np.asarray(correctness, dtype=float)
    order = np.argsort(-scores)
    correctness_sorted = correctness[order]
    coverages = np.linspace(1.0 / len(scores), 1.0, n_points)
    risks = []
    for c in coverages:
        k = max(1, int(round(c * len(correctness_sorted))))
        risks.append(1.0 - correctness_sorted[:k].mean())
    return coverages, np.array(risks)


def conf_correct_incorrect_stats(conf: np.ndarray, is_correct: np.ndarray, name: str = "", plot: bool = True) -> dict:
    """Print (and optionally plot) confidence statistics for correct vs. incorrect predictions.

    Parameters
    ----------
    conf:
        Confidence scores.
    is_correct:
        Boolean/0-1 correctness array, same length as `conf`.
    name:
        Label for the printed/plotted title.
    plot:
        If True, shows a histogram and boxplot (requires a display/backend).
    """
    conf = np.asarray(conf, dtype=float)
    is_correct = np.asarray(is_correct).astype(bool)

    conf_correct, conf_incorrect = conf[is_correct], conf[~is_correct]
    n_cor, n_inc = len(conf_correct), len(conf_incorrect)
    mean_cor = conf_correct.mean() if n_cor else np.nan
    std_cor = conf_correct.std() if n_cor else np.nan
    mean_inc = conf_incorrect.mean() if n_inc else np.nan
    std_inc = conf_incorrect.std() if n_inc else np.nan

    print(f"\n{name}")
    print(f"  Correct   (n={n_cor:4d}): {mean_cor:.3f} +/- {std_cor:.3f}")
    print(f"  Incorrect (n={n_inc:4d}): {mean_inc:.3f} +/- {std_inc:.3f}")
    print(f"  Gap (correct - incorrect): {mean_cor - mean_inc:.3f}")

    if plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 4))
        bins = np.linspace(0, 1, 31)
        plt.hist(conf_incorrect, bins=bins, alpha=0.6, color="red", label=f"Incorrect (n={n_inc})", density=True)
        plt.hist(conf_correct, bins=bins, alpha=0.6, color="green", label=f"Correct (n={n_cor})", density=True)
        plt.xlabel("Confidence")
        plt.ylabel("Density")
        plt.title(f"{name}: Confidence Distribution")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(5, 5))
        bp = plt.boxplot([conf_incorrect, conf_correct], tick_labels=["Incorrect", "Correct"], patch_artist=True)
        bp["boxes"][0].set_facecolor("red")
        bp["boxes"][1].set_facecolor("green")
        plt.ylabel("Confidence")
        plt.title(f"{name}: Confidence Boxplot")
        plt.tight_layout()
        plt.show()

    return {
        "n_correct": n_cor, "n_incorrect": n_inc,
        "mean_correct": mean_cor, "std_correct": std_cor,
        "mean_incorrect": mean_inc, "std_incorrect": std_inc,
        "gap": mean_cor - mean_inc,
    }
