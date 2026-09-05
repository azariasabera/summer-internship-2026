# src/tea/probes/feature_fusion.py

"""Handcrafted-feature-fusion ablation.

Asks, "Does adding acoustic / text / softmax-derived features on top of the raw
MTKD softmax (or the raw 768-d embedding) improve ground-truth emotion
prediction?"
"""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig

from tea.probes.fusion_utils import get_feature_groups, run_experiment
from tea.utils.logging import get_logger

logger = get_logger(__name__)


def probe_feature_fusion_cli(cfg: DictConfig) -> int:
    """`tea probe-feature-fusion` entry point (Hydra).
    
    Parameters
    ----------
    cfg : DictConfig
        Hydra config.

    Returns
    -------
    int
        Exit code (0 = success, 1 = failure).
    """
    from tea.features.master_table import build_master_table
    from tea.utils.paths import ensure_dir, resolve

    ff = cfg.probes.get("feature_fusion", {})
    if not ff.get("mtkd_json") or not ff.get("sentiment_fi_json"):
        logger.error("Set probes.feature_fusion.mtkd_json and .sentiment_fi_json")
        return 2

    # build master table (features + labels)
    df, mtkd_classes, _ = build_master_table(
        csv_root=cfg.paths.annotation_root,
        audio_root=ff.get("audio_root") or cfg.paths.chunk_audio_dir,
        mtkd_json_path=ff.mtkd_json,
        sentiment_fi_json_path=ff.sentiment_fi_json,
        sentiment_en_json_path=ff.get("sentiment_en_json"),
        embedding_root=ff.get("embedding_root") or cfg.paths.get("embedding_root"),
        compute_acoustic=True,
    )

    exclude = list(ff.get("exclude") or []) # which video to exclude

    if exclude:
        n_before = len(df)
        if "video_id" not in df.columns:
            logger.error("Cannot exclude videos: 'video_id' column not found in master table")
            return 2
        mask = ~df["video_id"].isin(exclude)
        df = df.loc[mask].reset_index(drop=True)
        n_removed = n_before - len(df)
        logger.info("Excluded %d rows belonging to video %s (remaining: %d)", n_removed, exclude, len(df))

        if len(df) == 0:
            logger.error("All rows were excluded – nothing left to evaluate.")
            return 2

    # integer labels
    label_map = {c: i for i, c in enumerate(mtkd_classes)}
    df = df.copy()
    df["_label_id"] = df["gt_label"].map(label_map)

    # drop any rows that failed the mapping
    n_before = len(df)
    df = df.dropna(subset=["_label_id"]).reset_index(drop=True)
    if len(df) < n_before:
        logger.warning("Dropped %d rows with unmapped gt_label", n_before - len(df))

    n_classes = len(mtkd_classes)
    groups = get_feature_groups(df, mtkd_cols=[f"mtkd_{c}" for c in mtkd_classes])

    # which base representations to run
    if ff.get("base"):
        bases = [ff.base]
    else:
        bases = [b for b in ("softmax", "embedding") if b in groups]

    all_results: dict[str, pd.DataFrame] = {}
    for base in bases:
        logger.info("=== Base representation: %s ===", base)
        all_results[base] = run_experiment(
            df=df,
            groups=groups,
            label_col="_label_id",
            teacher_col="teacher_id",
            n_classes=n_classes,
            base=base,
            seed=ff.get("seed", cfg.get("seed", 42)),
        )

    # save results to CSV (optionally save master table)
    if ff.get("output_dir"):
        out_dir = ensure_dir(resolve(ff.output_dir))
        for base, result_df in all_results.items():
            result_df.to_csv(out_dir / f"feature_fusion_{base}.csv", index=False)
        logger.info("Results saved → %s", out_dir)

        if ff.get("save_master_table", False):
            pickle_name = ff.get("master_table_name", "master_table.pkl")
            df.to_pickle(out_dir / pickle_name)
            logger.info("Master table saved → %s", out_dir / pickle_name)

    return 0