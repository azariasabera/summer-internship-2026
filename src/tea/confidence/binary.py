# src/tea/confidence/binary.py

"""Approach 1 of 3: binary correctness classifier."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from tea.confidence.calibration_metrics import auroc_correctness, brier_score_binary, expected_calibration_error
from tea.confidence.cv_utils import internal_val_split, loto_scaled_folds


class BinaryReliabilityNet(nn.Module):
    """Tiny MLP -> single logit -> sigmoid = P(MTKD prediction is correct).

    Parameters
    ----------
    input_dim:
        Number of input features.
    hidden_dim:
        Hidden layer width.
    dropout:
        Dropout probability.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 20, dropout: float = 0.35) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # raw logit; use BCEWithLogitsLoss


def train_one_fold(
    X_train, is_correct_train, X_val, video_ids_train=None, val_frac: float = 0.15, hidden_dim: int = 20,
    epochs: int = 300, lr: float = 1e-3, weight_decay: float = 1e-3, patience: int = 30, seed: int = 0,
    device: torch.device | None = None,
):
    """Train one `BinaryReliabilityNet` with early stopping on an internal (video-grouped) validation slice.

    Parameters mirror the module-level defaults used across all three
    reliability nets; see `calibrate.py`'s equivalent for context.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    tr_idx, es_idx = internal_val_split(len(X_train), video_ids_train, val_frac, seed)

    model = BinaryReliabilityNet(X_train.shape[1], hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # class-balance the BCE loss -- correct/incorrect counts are rarely 50/50
    n_pos = max(1, int(is_correct_train[tr_idx].sum()))
    n_neg = max(1, len(tr_idx) - n_pos)
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32, device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Xt = torch.tensor(X_train[tr_idx], dtype=torch.float32, device=device)
    yt = torch.tensor(is_correct_train[tr_idx], dtype=torch.float32, device=device)
    Xes = torch.tensor(X_train[es_idx], dtype=torch.float32, device=device)
    yes = torch.tensor(is_correct_train[es_idx], dtype=torch.float32, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)

    best_state, best_es_loss, bad_epochs = None, float("inf"), 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = bce(model(Xt), yt)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            es_loss = nn.functional.binary_cross_entropy_with_logits(model(Xes), yes).item()
        if es_loss < best_es_loss - 1e-5:
            best_es_loss, best_state, bad_epochs = es_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
        if bad_epochs > patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_scores = torch.sigmoid(model(Xv)).cpu().numpy()
    return model, val_scores


def run_loto_cv_binary(
    feature_df: pd.DataFrame, labels: np.ndarray, teacher_ids: np.ndarray, raw_mtkd_probs: np.ndarray,
    n_classes: int | None = None, video_ids: np.ndarray | None = None, seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """LOTO CV for the binary correctness classifier.

    MTKD stays frozen throughout: `pred = argmax(raw_mtkd_probs)`. Reports
    per held-out teacher: AUROC/ECE/Brier for raw max-softmax (floor) vs.
    the learned binary reliability score.

    Parameters
    ----------
    feature_df:
        Output of `tea.features.build_feature_table`.
    labels, teacher_ids, raw_mtkd_probs:
        Aligned per-chunk arrays.
    n_classes:
        Accepted but unused -- kept so this matches the call signature of
        `run_loto_cv_tcp`/`run_loto_cv_temperature`.
    video_ids:
        Used for video-grouped internal-validation splitting within each fold.
    seed:
        Random seed.

    Returns
    -------
    tuple
        `(results_df, per_chunk_df)`.
    """
    pred_all = raw_mtkd_probs.argmax(axis=1)
    is_correct_all = (pred_all == labels).astype(np.float32)
    max_softmax_all = raw_mtkd_probs.max(axis=1)

    results, per_chunk_rows = [], []
    for held_out, train_mask, val_mask, X_train, X_val in loto_scaled_folds(feature_df, teacher_ids, labels, pred_all):
        y_correct_val = is_correct_all[val_mask]
        video_ids_train = video_ids[train_mask] if video_ids is not None else None

        _, val_scores = train_one_fold(X_train, is_correct_all[train_mask], X_val, video_ids_train=video_ids_train, seed=seed)

        results.append({
            "held_out_teacher": held_out, "n_val": int(val_mask.sum()), "n_correct": int(y_correct_val.sum()),
            "auroc_max_softmax": auroc_correctness(max_softmax_all[val_mask], y_correct_val),
            "auroc_learned": auroc_correctness(val_scores, y_correct_val),
            "ece_max_softmax": expected_calibration_error(max_softmax_all[val_mask], y_correct_val),
            "ece_learned": expected_calibration_error(val_scores, y_correct_val),
            "brier_max_softmax": brier_score_binary(max_softmax_all[val_mask], y_correct_val),
            "brier_learned": brier_score_binary(val_scores, y_correct_val),
        })
        for local_i, global_idx in enumerate(np.where(val_mask)[0]):
            per_chunk_rows.append({
                "held_out_teacher": held_out, "video_id": video_ids[global_idx] if video_ids is not None else None,
                "teacher_id": teacher_ids[global_idx], "gt_label": int(labels[global_idx]),
                "pred_label": int(pred_all[global_idx]), "is_correct": bool(is_correct_all[global_idx]),
                "max_softmax": float(max_softmax_all[global_idx]), "learned_conf": float(val_scores.squeeze()[local_i]),
            })

    results_df, per_chunk_df = pd.DataFrame(results), pd.DataFrame(per_chunk_rows)
    print("\nLOTO average results (binary correctness classifier):")
    for col in ["auroc_max_softmax", "auroc_learned", "ece_max_softmax", "ece_learned", "brier_max_softmax", "brier_learned"]:
        print(f"  {col}: {results_df[col].mean():.4f}")
    return results_df, per_chunk_df
