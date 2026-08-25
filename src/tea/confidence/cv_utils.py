# src/tea/confidence/cv_utils.py

"""Shared cross-validation helpers for the three reliability nets."""

from __future__ import annotations

import numpy as np

# UNTIL I CONFIRM I COMMENT THE CURRENT IMPLEMENTATION AND DIRECTLY COPY THE ORIGINAL ONE.
# def internal_val_split(
#     n_train: int, video_ids_train: np.ndarray | None = None, val_frac: float = 0.15, seed: int = 0
# ) -> tuple[np.ndarray, np.ndarray]:
#     """Split `n_train` training indices into a fit set and an early-stopping set.

#     Parameters
#     ----------
#     n_train:
#         Number of training samples.
#     video_ids_train:
#         If given, the split is grouped by video id (a video's chunks never
#         straddle both sides) -- consistent with every other early-stopping
#         split in this codebase. If `None`, falls back to a plain random split.
#     val_frac:
#         Fraction of data (or of unique videos, if grouped) held out for early stopping.
#     seed:
#         Random seed.

#     Returns
#     -------
#     tuple
#         `(train_idx, early_stop_idx)`, both `np.ndarray` of positional indices into the training arrays.
#     """
#     rng = np.random.RandomState(seed)
#     idx = np.arange(n_train)

#     if video_ids_train is None:
#         rng.shuffle(idx)
#         n_val = max(1, int(round(val_frac * n_train)))
#         return idx[n_val:], idx[:n_val]

#     unique_videos = np.unique(video_ids_train)
#     if len(unique_videos) < 2:
#         # Only one video in this training fold -- can't group-split meaningfully, fall back to random.
#         rng.shuffle(idx)
#         n_val = max(1, int(round(val_frac * n_train)))
#         return idx[n_val:], idx[:n_val]

#     shuffled_videos = unique_videos.copy()
#     rng.shuffle(shuffled_videos)
#     n_val_videos = max(1, int(round(val_frac * len(shuffled_videos))))
#     val_videos = set(shuffled_videos[:n_val_videos])

#     es_mask = np.isin(video_ids_train, list(val_videos))
#     es_idx, tr_idx = idx[es_mask], idx[~es_mask]

#     if len(tr_idx) == 0 or len(es_idx) == 0:
#         # Degenerate grouping (e.g. one video holds almost all the data) -- fall back to random.
#         rng.shuffle(idx)
#         n_val = max(1, int(round(val_frac * n_train)))
#         return idx[n_val:], idx[:n_val]

#     return tr_idx, es_idx


def internal_val_split(n_train: int, video_ids_train: np.ndarray = None,
                        val_frac: float = 0.15, seed: int = 0):
    """
    Returns (train_idx, val_idx) into the TRAIN fold only. The outer LOTO
    test fold is never touched by this.

    If video_ids_train is given, the split is grouped by video_id so a
    video's chunks never straddle both sides (avoids leakage from adjacent
    chunks of the same recording). If omitted, falls back to a random
    per-chunk split -- fine for a first pass, but chunks from the same
    video can end up on both sides in that case.
    """
    rng = np.random.RandomState(seed)
    if video_ids_train is not None:
        unique_videos = np.unique(video_ids_train)
        rng.shuffle(unique_videos)
        n_val_videos = max(1, int(round(val_frac * len(unique_videos))))
        val_videos = set(unique_videos[:n_val_videos])
        val_idx = np.where(np.isin(video_ids_train, list(val_videos)))[0]
        train_idx = np.where(~np.isin(video_ids_train, list(val_videos)))[0]
        if len(val_idx) == 0 or len(train_idx) == 0:
            # degenerate case (too few videos) -- fall back to random split
            perm = rng.permutation(n_train)
            n_val = max(1, int(round(val_frac * n_train)))
            val_idx, train_idx = perm[:n_val], perm[n_val:]
    else:
        perm = rng.permutation(n_train)
        n_val = max(1, int(round(val_frac * n_train)))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
    return train_idx, val_idx


def loto_scaled_folds(feature_df, teacher_ids: np.ndarray, labels: np.ndarray, pred_all: np.ndarray):
    """Shared LOTO-fold generator: per held-out teacher, standardize features (fit on train only) and check the val fold has both correctness classes.

    Every reliability net (`binary.py`/`tcp.py`/`temperature.py`) use it.

    Parameters
    ----------
    feature_df:
        Output of `tea.features.build_feature_table`.
    teacher_ids, labels, pred_all:
        Aligned arrays: teacher grouping, true labels, and frozen-MTKD
        argmax predictions (`pred_all` is used only to compute
        `is_correct` for the degenerate-fold check).

    Yields
    ------
    tuple
        `(held_out, train_mask, val_mask, X_train_scaled, X_val_scaled)`
        for every held-out teacher whose val fold has both correct and
        incorrect predictions.
    """
    from sklearn.preprocessing import StandardScaler

    X = feature_df.to_numpy()
    is_correct_all = (pred_all == labels).astype(np.float32)

    for held_out in np.unique(teacher_ids):
        train_mask = teacher_ids != held_out
        val_mask = ~train_mask

        y_correct_val = is_correct_all[val_mask]
        if len(np.unique(y_correct_val)) < 2:
            print(f"  {held_out}: skipped, val fold has only one class of correctness")
            continue

        scaler = StandardScaler().fit(X[train_mask])
        yield held_out, train_mask, val_mask, scaler.transform(X[train_mask]), scaler.transform(X[val_mask])
