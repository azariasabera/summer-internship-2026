# src/tea/analysis/temporal.py

"""Temporal analysis: consistency scoring (report Section 2.2.2), extended
stability metrics (blip/run-length), a permutation-test baseline, and
smoothed emotion-probability arcs over time (report Section 4.4, Figures 12/13).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from tea.utils.constants import SAMPLE_RATE
from tea.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Report Section 2.2.2: Temporal Consistency
# ---------------------------------------------------------------------------


def transition_count(labels: list) -> int:
    """Number of adjacent-pair label changes. Ported from `per_video.ipynb`."""
    return sum(labels[i] != labels[i - 1] for i in range(1, len(labels)))


def temporal_consistency(labels: list) -> float:
    """Report Eq. 2.1: proportion of adjacent segments sharing the same label.

    `consistency = 1 - transitions / (N - 1)`. Returns NaN for `N < 2`
    (undefined -- not enough segments to have an adjacent pair).

    Parameters
    ----------
    labels:
        Sequence of per-segment labels, in temporal order.
    """
    n = len(labels)
    if n < 2:
        return np.nan
    return 1.0 - transition_count(labels) / (n - 1)


def merge_close_segments(df: pd.DataFrame, gap_threshold_sec: float, label_col: str, sr: int = SAMPLE_RATE) -> pd.DataFrame:
    """Treat speech segments separated by a small non-speech gap as "adjacent" for consistency purposes.

    Parameters
    ----------
    df:
        Chronologically-sorted per-video chunk dataframe with `start`/`end` (samples) and `label_col`.
    gap_threshold_sec:
        Max non-speech gap (seconds) to treat as "no real break."
    label_col:
        Column of labels to run consistency over (`gt_label` or `pred_label`).
    sr:
        Sample rate for converting `start`/`end` to seconds.
    """
    df = df.sort_values("start").reset_index(drop=True)
    gap_threshold_samples = gap_threshold_sec * sr

    run_id = 0
    run_ids = [run_id]
    for i in range(1, len(df)):
        gap = df.loc[i, "start"] - df.loc[i - 1, "end"]
        if gap > gap_threshold_samples:
            run_id += 1
        run_ids.append(run_id)

    df = df.copy()
    df["run_id"] = run_ids
    return df


def consistency_report(df: pd.DataFrame, gt_col: str = "gt_label", pred_col: str = "pred_label", gap_thresholds_sec: tuple[float, ...] = (0.0, 2.0, 3.0)) -> pd.DataFrame:
    """Reproduce report Section 2.2.2's three consistency numbers for one video, across gap thresholds.
    
    Parameters
    ----------
    df:
        One video's chunk dataframe with `start`, `end`, `gt_label`, `pred_label`.
    gt_col, pred_col:
        Column names.
    gap_thresholds_sec:
        Which gap thresholds to report (`0.0` = no merging).
    """
    rows = []

    # (1) full sequence including non-speech, non-speech as its own class
    full_gt = df[gt_col].fillna("non-speech").tolist()
    full_pred = df[pred_col].tolist()
    rows.append({"scope": "full_sequence_incl_non_speech", "gap_threshold_sec": None, "gt_consistency": temporal_consistency(full_gt), "pred_consistency": temporal_consistency(full_pred)})

    # (2)/(3) labeled-only, with each gap threshold
    labeled = df.loc[df[gt_col].notna()].reset_index(drop=True)
    for gap in gap_thresholds_sec:
        merged = merge_close_segments(labeled, gap, gt_col)
        gt_consistencies, pred_consistencies = [], []
        for _, run_df in merged.groupby("run_id"):
            if len(run_df) < 2:
                continue
            gt_consistencies.append(temporal_consistency(run_df[gt_col].tolist()))
            pred_consistencies.append(temporal_consistency(run_df[pred_col].tolist()))
        rows.append({
            "scope": "labeled_only", "gap_threshold_sec": gap,
            "gt_consistency": float(np.nanmean(gt_consistencies)) if gt_consistencies else np.nan,
            "pred_consistency": float(np.nanmean(pred_consistencies)) if pred_consistencies else np.nan,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Extended stability metrics (blips, run length) -- per_video.ipynb cell 6
# ---------------------------------------------------------------------------


def get_runs(labels: list, durations: list) -> list[tuple]:
    """Group a label sequence into contiguous runs, returning `(label, total_duration)` per run."""
    runs, idx = [], 0
    for label, group in itertools.groupby(labels):
        group = list(group)
        run_len = len(group)
        runs.append((label, sum(durations[idx : idx + run_len])))
        idx += run_len
    return runs


def compute_stability_metrics(labels: list, durations: list) -> dict:
    """Transition count, "blip" count (single-chunk deviation that immediately reverts), mean run length, run count.

    Parameters
    ----------
    labels, durations:
        Aligned sequences with no NaNs (filter before calling).
    """
    n = len(labels)
    if n == 0:
        return dict(transition_count=np.nan, blip_count=np.nan, mean_run_sec=np.nan, n_runs=np.nan)

    n_transitions = transition_count(labels)
    n_blips = sum(
        (labels[i] != labels[i - 1]) and (i + 1 < n and labels[i] != labels[i + 1]) and (i + 1 < n and labels[i - 1] == labels[i + 1])
        for i in range(1, n - 1)
    )
    runs = get_runs(labels, durations)
    mean_run_sec = float(np.mean([r[1] for r in runs])) if runs else np.nan
    return dict(transition_count=n_transitions, blip_count=n_blips, mean_run_sec=mean_run_sec, n_runs=len(runs))


def process_video_stability(df: pd.DataFrame, gt_col: str = "gt_label", pred_col: str = "pred_label", duration_col: str = "duration_sec") -> dict:
    """Per-video stability summary row: transitions/min, blips, mean run length, gt-vs-pred transition ratio.

    Parameters
    ----------
    df:
        One video's chunk dataframe.
    """
    total_duration_min = df[duration_col].sum() / 60

    gt_df = df.loc[df[gt_col].notna()]
    gt_metrics = compute_stability_metrics(gt_df[gt_col].tolist(), gt_df[duration_col].tolist())
    pred_metrics = compute_stability_metrics(df[pred_col].tolist(), df[duration_col].tolist())

    row = {
        "duration_min": round(total_duration_min, 2), "n_gt_chunks": len(gt_df), "n_pred_chunks": len(df),
        "gt_transitions": gt_metrics["transition_count"],
        "gt_transitions_per_min": round(gt_metrics["transition_count"] / total_duration_min, 2) if total_duration_min > 0 else np.nan,
        "gt_blips": gt_metrics["blip_count"],
        "gt_mean_run_sec": round(gt_metrics["mean_run_sec"], 2) if not np.isnan(gt_metrics["mean_run_sec"]) else np.nan,
        "pred_transitions": pred_metrics["transition_count"],
        "pred_transitions_per_min": round(pred_metrics["transition_count"] / total_duration_min, 2) if total_duration_min > 0 else np.nan,
        "pred_blips": pred_metrics["blip_count"],
        "pred_mean_run_sec": round(pred_metrics["mean_run_sec"], 2) if not np.isnan(pred_metrics["mean_run_sec"]) else np.nan,
    }
    row["pred_gt_transition_ratio"] = round(row["pred_transitions_per_min"] / row["gt_transitions_per_min"], 2) if row.get("gt_transitions_per_min") else np.nan
    return row


# ---------------------------------------------------------------------------
# Permutation-test baseline -- per_video.ipynb cell 7
# ---------------------------------------------------------------------------


def shuffled_baseline(labels: list, n_shuffles: int = 1000, seed: int = 0) -> np.ndarray:
    """Transition-count distribution under `n_shuffles` random reorderings of the same multiset of labels."""
    rng = np.random.default_rng(seed)
    labels = np.array(labels)
    counts = np.empty(n_shuffles)
    for i in range(n_shuffles):
        counts[i] = transition_count(rng.permutation(labels).tolist())
    return counts


def stability_test(labels: list, n_shuffles: int = 1000, seed: int = 0) -> dict:
    """Is this label sequence more temporally structured than a random shuffle of the same labels would be?

    `stability_score > 0` -> more stable than chance (real temporal
    structure); `~ 0` -> indistinguishable from chance; `< 0` -> less
    stable than chance (unusual, worth investigating if it happens).

    Parameters
    ----------
    labels:
        Label sequence.
    n_shuffles, seed:
        Permutation test settings.
    """
    actual = transition_count(labels)
    shuffled = shuffled_baseline(labels, n_shuffles, seed)
    mean_s, std_s = shuffled.mean(), shuffled.std()

    z = (actual - mean_s) / std_s if std_s > 0 else np.nan
    stability_score = (mean_s - actual) / mean_s if mean_s > 0 else np.nan
    p_value = float((shuffled <= actual).mean())

    return dict(actual_transitions=actual, shuffled_mean=round(mean_s, 2), shuffled_std=round(std_s, 2), z_score=round(z, 2) if not np.isnan(z) else np.nan, stability_score=round(stability_score, 3) if not np.isnan(stability_score) else np.nan, p_value=round(p_value, 4))


# ---------------------------------------------------------------------------
# Smoothed emotion arcs -- temporal.txt (report Section 4.4, Figures 12/13)
# ---------------------------------------------------------------------------

EMOTION_COLORS = {"neutral": "grey", "sadness": "#4C72B0", "happiness": "#E6C84A", "anger": "#C44E52"}


def load_video_probabilities(csv_path: str | Path, mtkd_json_path: str | Path, video_id: str, class_order: list[str]) -> pd.DataFrame:
    """Join one video's annotation CSV to its MTKD softmax probabilities, with time in seconds.

    Parameters
    ----------
    csv_path:
        Path to the video's annotation CSV.
    mtkd_json_path:
        MTKD prediction JSON (must contain per-class probabilities, not just argmax).
    video_id:
        Key into the prediction JSON.
    class_order:
        Emotion class columns to attach (usually `CLASS_ORDER`).
    """
    with open(mtkd_json_path) as f:
        preds = json.load(f)[video_id]

    df = pd.read_csv(csv_path)
    for emo in class_order:
        df[emo] = df["name"].map(lambda n: preds.get(n, {}).get(emo))
    df[class_order] = df[class_order].astype(float)

    df["start_sec"] = df["start"] / SAMPLE_RATE
    df["end_sec"] = df["end"] / SAMPLE_RATE
    df["mid_sec"] = (df["start_sec"] + df["end_sec"]) / 2
    return df


def moving_average(df: pd.DataFrame, emotions: list[str], window: int = 5) -> pd.DataFrame:
    """Add `<emotion>_smooth` columns via centered rolling mean."""
    df = df.copy()
    for emo in emotions:
        df[f"{emo}_smooth"] = df[emo].rolling(window, center=True, min_periods=1).mean()
    return df


def gaussian_smooth(df: pd.DataFrame, emotions: list[str], sigma: float = 2.0, radius: int | None = None, truncate: float = 4.0) -> pd.DataFrame:
    """Add `<emotion>_smooth` columns via a 1D Gaussian filter (kernel width = `2*radius+1`, `radius = round(sigma*truncate)` if not given)."""
    from scipy.ndimage import gaussian_filter1d

    df = df.copy()
    for emo in emotions:
        df[f"{emo}_smooth"] = gaussian_filter1d(df[emo], sigma=sigma, radius=radius, truncate=truncate)
    return df


def plot_emotion_arc(df: pd.DataFrame, emotions: list[str] = tuple(EMOTION_COLORS.keys()), show_non_speech: bool = True):
    """Plot raw (faint) + smoothed (bold) emotion-probability curves over time, with non-speech regions shaded.

    Requires `<emotion>_smooth` columns (see `moving_average`/`gaussian_smooth`).

    Parameters
    ----------
    df:
        Output of `moving_average`/`gaussian_smooth`, applied on top of `load_video_probabilities`.
    emotions:
        Which emotion columns to plot.
    show_non_speech:
        Shade non-speech regions (needs a `type` column).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(15, 6))

    if show_non_speech and "type" in df.columns:
        for _, row in df.loc[df["type"] == "non-speech"].iterrows():
            ax.axvspan(row.start_sec, row.end_sec, color="gray", alpha=0.5, zorder=0)

    for emo in emotions:
        ax.plot(df.mid_sec, df[emo], color=EMOTION_COLORS.get(emo, "black"), alpha=0.25, lw=1)
        ax.plot(df.mid_sec, df[f"{emo}_smooth"], color=EMOTION_COLORS.get(emo, "black"), lw=3, label=emo)

    ax.set_ylim(-0.08, 1.02)
    ax.set_ylabel("Probability")
    ax.set_xlabel("Time (seconds)")
    ax.legend()
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def temporal_cli(cfg: DictConfig) -> int:
    """`tea temporal` -- run the consistency + stability report across every video's annotation CSV, pooled and per-video.

    Requires `analysis.mtkd_json` (must contain per-class probabilities
    for the smoothed-arc plots to work; argmax-only JSONs still work for
    the consistency/stability metrics via the `pred_label` join).
    """
    from tea.analysis.classroom import join_predictions

    ac = cfg.analysis
    if not ac.get("mtkd_json"):
        logger.error("Set analysis.mtkd_json")
        return 2

    df = join_predictions(ac.mtkd_json, cfg.paths.annotation_root, excluded_videos=set(cfg.classroom.excluded_videos))

    consistency_rows, stability_rows = [], []
    for video, group in df.groupby("video"):
        cr = consistency_report(group)
        cr.insert(0, "video", video)
        consistency_rows.append(cr)
        stability_rows.append({"video": video, **process_video_stability(group)})

    consistency_df = pd.concat(consistency_rows, ignore_index=True)
    stability_df = pd.DataFrame(stability_rows)

    logger.info("=== Consistency (pooled across videos, gap_threshold=0) ===")
    pooled = consistency_df.loc[(consistency_df["scope"] == "labeled_only") & (consistency_df["gap_threshold_sec"] == 0.0)]
    logger.info("mean gt_consistency=%.4f mean pred_consistency=%.4f", pooled["gt_consistency"].mean(), pooled["pred_consistency"].mean())

    logger.info("=== Stability summary ===")
    logger.info("\n%s", stability_df)

    if ac.get("output_dir"):
        from tea.utils.paths import ensure_dir, resolve

        out_dir = ensure_dir(resolve(ac.output_dir))
        consistency_df.to_csv(out_dir / "temporal_consistency.csv", index=False)
        stability_df.to_csv(out_dir / "temporal_stability_summary.csv", index=False)
        logger.info("Saved -> %s", out_dir)

    return 0

def emotion_arc_per_video(cfg: DictConfig) -> int:
    """Plot MA and Gaussian smoothed temporal emotion arc of the specified video.

    Requires ``analysis.mtkd_json`` (with per-class probabilities) and
    ``analysis.emotion_arc.video``. Saves both moving-average and
    Gaussian-smoothed figures under ``analysis.emotion_arc.plot_save_dir``.
    """
    from tea.utils.constants import CLASS_ORDER
    from tea.utils.paths import ensure_dir, resolve

    ac = cfg.analysis
    if not ac.get("mtkd_json"):
        logger.error("Set analysis.mtkd_json")
        return 2

    ea = ac.get("emotion_arc")
    if ea is None or not ea.get("video"):
        logger.error("Set analysis.emotion_arc.video to a video ID")
        return 2

    video_id = ea.video

    # Annotation CSV for this video
    csv_path = resolve(cfg.paths.annotation_root) / f"{video_id}.csv"
    if not csv_path.is_file():
        logger.error("Annotation CSV not found: %s", csv_path)
        return 2

    df = load_video_probabilities(
        csv_path=csv_path,
        mtkd_json_path=ac.mtkd_json,
        video_id=video_id,
        class_order=list(CLASS_ORDER),
    )

    out_dir = ensure_dir(resolve(ea.plot_save_dir))

    # --- Moving-average smoothed arc ---
    ma_cfg = ea.get("ma", {})
    window = ma_cfg.get("window", 5)
    df_ma = moving_average(df, emotions=list(CLASS_ORDER), window=window)
    fig_ma = plot_emotion_arc(df_ma, emotions=list(CLASS_ORDER))
    fig_ma.suptitle(f"{video_id} – moving average (window={window})")
    ma_path = out_dir / f"{video_id}_ma.png"
    fig_ma.savefig(ma_path, dpi=150, bbox_inches="tight")
    logger.info("Saved MA arc -> %s", ma_path)

    # --- Gaussian-smoothed arc ---
    g_cfg = ea.get("gaussian", {})
    sigma = g_cfg.get("sigma", 2.0)
    radius = g_cfg.get("radius")          # None is fine → scipy computes it
    truncate = g_cfg.get("truncate", 4.0)
    df_gauss = gaussian_smooth(
        df, emotions=list(CLASS_ORDER), sigma=sigma, radius=radius, truncate=truncate
    )
    fig_gauss = plot_emotion_arc(df_gauss, emotions=list(CLASS_ORDER))
    fig_gauss.suptitle(f"{video_id} – Gaussian smooth (σ={sigma})")
    gauss_path = out_dir / f"{video_id}_gaussian.png"
    fig_gauss.savefig(gauss_path, dpi=150, bbox_inches="tight")
    logger.info("Saved Gaussian arc -> %s", gauss_path)

    return 0