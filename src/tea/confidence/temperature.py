# src/tea/confidence/temperature.py

"""Approach 3 of 3: instance-dependent temperature scaling.

Why this can never hurt WAR/UAR: dividing every logit in a sample by the
SAME positive scalar T preserves rank order, so `argmax(scaled) ==
argmax(raw)` always, by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from tea.confidence.calibration_metrics import (
    auroc_correctness, brier_score_binary, brier_score_multiclass, expected_calibration_error,
    negative_log_likelihood,
)
from tea.confidence.cv_utils import internal_val_split, loto_scaled_folds


class TemperatureNet(nn.Module):
    """Tiny MLP -> single scalar T, bounded to `[t_min, t_max]` via a scaled sigmoid.

    Bounding prevents degenerate near-zero T (would blow up logits) or
    huge T (would flatten everything to uniform) early in training.

    Parameters
    ----------
    input_dim:
        Number of input features.
    hidden_dim:
        Hidden layer width.
    dropout:
        Dropout probability.
    t_min, t_max:
        Bounds on the predicted temperature.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 20, dropout: float = 0.35, t_min: float = 0.3, t_max: float = 5.0) -> None:
        super().__init__()
        self.t_min, self.t_max = t_min, t_max
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x).squeeze(-1)
        return self.t_min + (self.t_max - self.t_min) * torch.sigmoid(raw)


def reshape_logits(raw_probs: torch.Tensor, T: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert `raw_probs` (N, C) back to logits via `log(p)`, then divide by per-sample `T` (N,).

    Returns scaled logits ready for `nn.CrossEntropyLoss` (which applies `log_softmax` itself).
    """
    logits = torch.log(raw_probs.clamp(min=eps))
    return logits / T.unsqueeze(-1)


def train_one_fold(
    X_train, raw_probs_train, y_train, X_val, raw_probs_val, video_ids_train=None, val_frac: float = 0.15,
    hidden_dim: int = 20, epochs: int = 300, lr: float = 1e-3, weight_decay: float = 1e-3, patience: int = 30,
    seed: int = 0, device: torch.device | None = None,
):
    """Train one `TemperatureNet` (cross-entropy on the reshaped distribution) with early stopping."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    tr_idx, es_idx = internal_val_split(len(X_train), video_ids_train, val_frac, seed)

    model = TemperatureNet(X_train.shape[1], hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()

    Xt = torch.tensor(X_train[tr_idx], dtype=torch.float32, device=device)
    pt = torch.tensor(raw_probs_train[tr_idx], dtype=torch.float32, device=device)
    yt = torch.tensor(y_train[tr_idx], dtype=torch.long, device=device)
    Xes = torch.tensor(X_train[es_idx], dtype=torch.float32, device=device)
    pes = torch.tensor(raw_probs_train[es_idx], dtype=torch.float32, device=device)
    yes = torch.tensor(y_train[es_idx], dtype=torch.long, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    pv = torch.tensor(raw_probs_val, dtype=torch.float32, device=device)

    best_state, best_es_loss, bad_epochs = None, float("inf"), 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        T = model(Xt)
        loss = ce(reshape_logits(pt, T), yt)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            Tes = model(Xes)
            es_loss = ce(reshape_logits(pes, Tes), yes).item()
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
        Tv = model(Xv)
        scaled_probs = torch.softmax(reshape_logits(pv, Tv), dim=1).cpu().numpy()
        Tv = Tv.cpu().numpy()
    return model, scaled_probs, Tv


def run_loto_cv_temperature(
    feature_df: pd.DataFrame, labels: np.ndarray, teacher_ids: np.ndarray, raw_mtkd_probs: np.ndarray,
    n_classes: int, video_ids: np.ndarray | None = None, seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """LOTO CV for instance-dependent temperature scaling.

    Reports per held-out teacher: `argmax_unchanged` (sanity check --
    should always be True; False means a bug, report it immediately),
    AUROC/ECE/Brier(binary) for raw vs. reshaped, Brier(multiclass)/NLL
    (which only make sense here since this outputs a full distribution),
    and mean/std predicted T (sanity-check it's actually varying per
    sample rather than collapsing to a constant global-T-equivalent).

    Parameters
    ----------
    feature_df, labels, teacher_ids, raw_mtkd_probs:
        See `binary.run_loto_cv_binary`.
    n_classes:
        Required (unlike the other two) -- used for the multiclass Brier score.
    video_ids, seed:
        See `binary.run_loto_cv_binary`.
    """
    pred_all = raw_mtkd_probs.argmax(axis=1)
    is_correct_all = (pred_all == labels).astype(np.float32)
    max_softmax_all = raw_mtkd_probs.max(axis=1)

    results, per_chunk_rows = [], []
    for held_out, train_mask, val_mask, X_train, X_val in loto_scaled_folds(feature_df, teacher_ids, labels, pred_all):
        y_correct_val = is_correct_all[val_mask]
        video_ids_train = video_ids[train_mask] if video_ids is not None else None

        _, scaled_probs_val, T_val = train_one_fold(
            X_train, raw_mtkd_probs[train_mask], labels[train_mask], X_val, raw_mtkd_probs[val_mask],
            video_ids_train=video_ids_train, seed=seed,
        )

        argmax_unchanged = bool(np.array_equal(scaled_probs_val.argmax(axis=1), raw_mtkd_probs[val_mask].argmax(axis=1)))
        if not argmax_unchanged:
            print(f"  {held_out}: WARNING -- argmax changed after temperature scaling, this should be impossible by construction; check reshape_logits.")

        scaled_max_prob = scaled_probs_val.max(axis=1)

        results.append({
            "held_out_teacher": held_out, "n_val": int(val_mask.sum()), "n_correct": int(y_correct_val.sum()),
            "argmax_unchanged": argmax_unchanged, "mean_T": float(T_val.mean()), "std_T": float(T_val.std()),
            "auroc_max_softmax": auroc_correctness(max_softmax_all[val_mask], y_correct_val),
            "auroc_scaled": auroc_correctness(scaled_max_prob, y_correct_val),
            "ece_max_softmax": expected_calibration_error(max_softmax_all[val_mask], y_correct_val),
            "ece_scaled": expected_calibration_error(scaled_max_prob, y_correct_val),
            "brier_binary_max_softmax": brier_score_binary(max_softmax_all[val_mask], y_correct_val),
            "brier_binary_scaled": brier_score_binary(scaled_max_prob, y_correct_val),
            "brier_multiclass_raw": brier_score_multiclass(raw_mtkd_probs[val_mask], labels[val_mask], n_classes),
            "brier_multiclass_scaled": brier_score_multiclass(scaled_probs_val, labels[val_mask], n_classes),
            "nll_raw": negative_log_likelihood(raw_mtkd_probs[val_mask], labels[val_mask]),
            "nll_scaled": negative_log_likelihood(scaled_probs_val, labels[val_mask]),
        })
        for local_i, global_idx in enumerate(np.where(val_mask)[0]):
            per_chunk_rows.append({
                "held_out_teacher": held_out, "video_id": video_ids[global_idx] if video_ids is not None else None,
                "teacher_id": teacher_ids[global_idx], "gt_label": int(labels[global_idx]),
                "pred_label": int(pred_all[global_idx]), "is_correct": bool(is_correct_all[global_idx]),
                "max_softmax": float(max_softmax_all[global_idx]), "learned_conf": float(scaled_max_prob[local_i]),
            })

    results_df, per_chunk_df = pd.DataFrame(results), pd.DataFrame(per_chunk_rows)
    print("\nLOTO average results (instance-dependent temperature scaling):")
    print(f"  all folds argmax-unchanged: {results_df['argmax_unchanged'].all()}")
    for col in ["mean_T", "std_T", "auroc_max_softmax", "auroc_scaled", "ece_max_softmax", "ece_scaled",
                "brier_binary_max_softmax", "brier_binary_scaled", "brier_multiclass_raw", "brier_multiclass_scaled",
                "nll_raw", "nll_scaled"]:
        print(f"  {col}: {results_df[col].mean():.4f}")
    return results_df, per_chunk_df
