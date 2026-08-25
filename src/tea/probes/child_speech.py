# src/tea/probes/child_speech.py

"""Child-speech-presence probe: can a frozen MTKD emotion model's pooled
embeddings predict whether child speech was present, despite never being
trained for that task?
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from tea.utils.logging import get_logger

logger = get_logger(__name__)


def load_dataset(embedding_root: str | Path, annotation_root: str | Path) -> pd.DataFrame:
    """Join each video's pooled embeddings to its annotation CSV's child-speech flag.

    Expects the `tea.mtkd.embeddings.EmbeddingExtractor.process_video_folder`
    output layout (`pooled.npy`, `names.npy`/`metadata.csv` per video
    under `embedding_root`) and an `overlap` (0/1 child-speech) column in
    each annotation CSV.

    Parameters
    ----------
    embedding_root:
        Directory of one subfolder per video, each with `pooled.npy` + `metadata.csv`.
    annotation_root:
        Directory of per-video annotation CSVs (must have `name`, `overlap` columns).

    Returns
    -------
    pd.DataFrame
        One row per chunk: `video_id`, `chunk_id`, `child_speech` (0/1),
        `embedding_0..767`.
    """
    embedding_root, annotation_root = Path(embedding_root), Path(annotation_root)
    video_dirs = sorted(p for p in embedding_root.iterdir() if p.is_dir())

    frames = []
    for video_dir in video_dirs:
        video_id = video_dir.name
        pooled_path, meta_path = video_dir / "pooled.npy", video_dir / "metadata.csv"
        csv_path = annotation_root / f"{video_id}.csv"

        if not pooled_path.exists() or not meta_path.exists():
            logger.warning("%s: missing pooled.npy/metadata.csv, skipping", video_id)
            continue
        if not csv_path.exists():
            logger.warning("%s: no annotation csv found, skipping", video_id)
            continue

        pooled = np.load(pooled_path)
        meta = pd.read_csv(meta_path)
        ann = pd.read_csv(csv_path)

        if len(pooled) != len(meta):
            raise ValueError(f"{video_id}: pooled.npy ({len(pooled)}) / metadata.csv ({len(meta)}) length mismatch")

        ann_lookup = dict(zip(ann["name"].astype(str), ann["overlap"]))
        child_speech = meta["chunk"].astype(str).map(ann_lookup)

        valid = child_speech.notna()
        if not valid.all():
            logger.info("%s: %d/%d chunks have no matching annotation, dropping", video_id, (~valid).sum(), len(valid))

        df = pd.DataFrame(pooled[valid.to_numpy()], columns=[f"embedding_{i}" for i in range(pooled.shape[1])])
        df.insert(0, "chunk_id", meta.loc[valid, "chunk"].to_numpy())
        df.insert(0, "video_id", video_id)
        df["child_speech"] = child_speech[valid].astype(int).to_numpy()
        frames.append(df)

    if not frames:
        raise ValueError("No videos loaded -- check embedding_root/annotation_root and that CSVs have an 'overlap' column.")

    full = pd.concat(frames, ignore_index=True)
    logger.info(
        "Loaded %d chunks from %d videos. Child speech present: %d/%d (%.1f%%)",
        len(full), full["video_id"].nunique(), full["child_speech"].sum(), len(full), 100 * full["child_speech"].mean(),
    )
    return full


class ChildSpeechProbe:
    """GroupKFold logistic-regression probe of pooled embeddings for child-speech presence.

    Parameters
    ----------
    n_splits:
        Number of GroupKFold folds (grouped by `video_id`).
    C:
        Inverse regularization strength for `LogisticRegression`.
    max_iter:
        Max solver iterations.
    seed:
        Random seed (used by `LogisticRegression`'s solver where applicable).
    """

    def __init__(self, n_splits: int = 5, C: float = 1.0, max_iter: int = 1000, seed: int = 42) -> None:
        self.n_splits = n_splits
        self.C = C
        self.max_iter = max_iter
        self.seed = seed

    def run(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Run GroupKFold cross-validation.

        Parameters
        ----------
        df:
            Output of `load_dataset`.

        Returns
        -------
        tuple
            `(fold_results_df, overall_metrics)`. `overall_metrics` is
            computed by pooling every fold's held-out predictions
            (out-of-fold), not by averaging per-fold metrics.
        """
        embedding_cols = [c for c in df.columns if c.startswith("embedding_")]
        X, y, groups = df[embedding_cols].to_numpy(), df["child_speech"].to_numpy(), df["video_id"].to_numpy()

        gkf = GroupKFold(n_splits=self.n_splits)
        fold_rows = []
        oof_true, oof_pred = np.zeros(len(y), dtype=int), np.zeros(len(y), dtype=int)

        for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
            scaler = StandardScaler().fit(X[train_idx])
            X_train, X_test = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])

            clf = LogisticRegression(C=self.C, max_iter=self.max_iter, random_state=self.seed)
            clf.fit(X_train, y[train_idx])
            preds = clf.predict(X_test)

            oof_true[test_idx], oof_pred[test_idx] = y[test_idx], preds
            fold_rows.append({
                "fold": fold_i, "n_train": len(train_idx), "n_test": len(test_idx),
                "test_videos": sorted(set(groups[test_idx])),
                "accuracy": accuracy_score(y[test_idx], preds),
                "precision": precision_score(y[test_idx], preds, zero_division=0),
                "recall": recall_score(y[test_idx], preds, zero_division=0),
                "f1": f1_score(y[test_idx], preds, zero_division=0),
            })
            logger.info("Fold %d (%d test videos): acc=%.4f f1=%.4f", fold_i, len(fold_rows[-1]["test_videos"]), fold_rows[-1]["accuracy"], fold_rows[-1]["f1"])

        overall = {
            "war": accuracy_score(oof_true, oof_pred),
            "uar": recall_score(oof_true, oof_pred, average="macro"),
            "precision": precision_score(oof_true, oof_pred, zero_division=0),
            "recall": recall_score(oof_true, oof_pred, zero_division=0),
            "f1": f1_score(oof_true, oof_pred, zero_division=0),
        }
        logger.info(
            "Overall (pooled OOF): WAR=%.4f UAR=%.4f precision=%.4f recall=%.4f f1=%.4f",
            overall["war"], overall["uar"], overall["precision"], overall["recall"], overall["f1"],
        )
        return pd.DataFrame(fold_rows), overall


def probe_child_speech_cli(cfg: DictConfig) -> int:
    """`tea probe-child-speech` entry point.

    Requires `probes.embedding_root` (defaults to `cfg.paths.embedding_root`).
    """
    pc = cfg.probes.get("child_speech", {})
    embedding_root = pc.get("embedding_root") or cfg.paths.embedding_root

    df = load_dataset(embedding_root, cfg.paths.annotation_root)
    probe = ChildSpeechProbe(
        n_splits=pc.get("n_splits", 5), C=pc.get("C", 1.0), max_iter=pc.get("max_iter", 1000), seed=cfg.seed
    )
    fold_df, overall = probe.run(df)

    if pc.get("output_dir"):
        from tea.utils.paths import ensure_dir, resolve

        out_dir = ensure_dir(resolve(pc.output_dir))
        fold_df.to_csv(out_dir / "child_speech_probe_folds.csv", index=False)
        logger.info("Saved -> %s", out_dir / "child_speech_probe_folds.csv")

    return 0
