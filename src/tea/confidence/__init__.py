# src/tea/confidence/__init__.py

"""Confidence/reliability recalibration on top of a frozen MTKD student
(report Section 4.3, Table "Preliminary result").

Three approaches, all MTKD-frozen, all evaluated on the same LOTO CV
footing via `tea.confidence.calibration_metrics`:

| Module           | Approach                                         |
|------------------|--------------------------------------------------|
| `binary.py`      | P(correct | features), BCE                       |
| `tcp.py`         | True Class Probability regression + ranking loss |
| `temperature.py` | Instance-dependent temperature scaling           |

`calibration_metrics.py` is shared scoring infrastructure.
`loto_scaled_folds` factors out the identical per-fold StandardScaler + degenerate-fold-skip 
logic for the three strategies.
"""

from tea.confidence.binary import BinaryReliabilityNet, run_loto_cv_binary
from tea.confidence.calibration_metrics import (
    auroc_correctness, brier_score_binary, brier_score_multiclass, conf_correct_incorrect_stats,
    expected_calibration_error, negative_log_likelihood, risk_coverage_curve,
)
from tea.confidence.tcp import TCPReliabilityNet, run_loto_cv_tcp
from tea.confidence.temperature import TemperatureNet, run_loto_cv_temperature

__all__ = [
    "BinaryReliabilityNet", "run_loto_cv_binary",
    "TCPReliabilityNet", "run_loto_cv_tcp",
    "TemperatureNet", "run_loto_cv_temperature",
    "auroc_correctness", "expected_calibration_error", "brier_score_binary", "brier_score_multiclass",
    "negative_log_likelihood", "risk_coverage_curve", "conf_correct_incorrect_stats",
]


def confidence_cli(cfg) -> int:
    """`tea confidence` -- build the feature table and run all three reliability approaches.

    Requires `confidence.mtkd_json`, `confidence.sentiment_fi_json`.
    Optional: `confidence.sentiment_en_json`, `confidence.use_three_class`
    (4-class vs 3-class MTKD), `confidence.methods` (subset of
    `binary,tcp,temperature`, default all three).

    Parameters
    ----------
    cfg:
        Resolved Hydra configuration.
    """
    from tea.features.master_table import build_feature_table, build_master_table
    from tea.utils.logging import get_logger

    logger = get_logger(__name__)
    cc = cfg.confidence

    if not cc.get("mtkd_json") or not cc.get("sentiment_fi_json"):
        logger.error("Set confidence.mtkd_json and confidence.sentiment_fi_json")
        return 2

    df, mtkd_classes, sentiment_classes = build_master_table(
        csv_root=cfg.paths.annotation_root, audio_root=cfg.paths.chunk_audio_dir,
        mtkd_json_path=cc.mtkd_json, sentiment_fi_json_path=cc.sentiment_fi_json,
        sentiment_en_json_path=cc.get("sentiment_en_json"),
        use_three_class=cc.get("use_three_class", False),
    )

    mtkd_prob_cols = [f"mtkd_{c}" for c in mtkd_classes]
    text_fi_cols = [f"text_fi_{c}" for c in sentiment_classes]
    text_en_cols = [f"text_en_{c}" for c in sentiment_classes] if cc.get("sentiment_en_json") else None

    feature_df = build_feature_table(df, mtkd_prob_cols=mtkd_prob_cols, text_fi_cols=text_fi_cols, text_en_cols=text_en_cols)
    label_map = {c: i for i, c in enumerate(mtkd_classes)}
    labels = df["gt_label"].map(label_map).to_numpy()
    teacher_ids = df["teacher_id"].to_numpy()
    video_ids = df["video_id"].to_numpy()
    raw_mtkd_probs = df[mtkd_prob_cols].to_numpy()

    methods = list(cc.get("methods", ["binary", "tcp", "temperature"]))
    all_results = {}

    if "binary" in methods:
        logger.info("=== Binary correctness classifier ===")
        all_results["binary"] = run_loto_cv_binary(feature_df, labels, teacher_ids, raw_mtkd_probs, video_ids=video_ids)
    if "tcp" in methods:
        logger.info("=== TCP regression + ranking ===")
        all_results["tcp"] = run_loto_cv_tcp(feature_df, labels, teacher_ids, raw_mtkd_probs, video_ids=video_ids)
    if "temperature" in methods:
        logger.info("=== Instance-dependent temperature scaling ===")
        all_results["temperature"] = run_loto_cv_temperature(feature_df, labels, teacher_ids, raw_mtkd_probs, n_classes=len(mtkd_classes), video_ids=video_ids)

    if cc.get("output_dir"):
        from tea.utils.paths import ensure_dir, resolve

        out_dir = ensure_dir(resolve(cc.output_dir))
        for name, (results_df, per_chunk_df) in all_results.items():
            results_df.to_csv(out_dir / f"{name}_results.csv", index=False)
            per_chunk_df.to_csv(out_dir / f"{name}_per_chunk.csv", index=False)
        logger.info("Saved results -> %s", out_dir)

    return 0
