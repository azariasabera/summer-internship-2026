# src/tea/confidence/tcp.py

"""Approach 2 of 3: True Class Probability (TCP) regression + pairwise
ranking loss (ConfidNet, Corbiere et al. 2019). https://arxiv.org/abs/1910.04851"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from tea.confidence.calibration_metrics import auroc_correctness, brier_score_binary, expected_calibration_error
from tea.confidence.cv_utils import internal_val_split, loto_scaled_folds


class TCPReliabilityNet(nn.Module):
    """Tiny MLP -> single scalar in [0,1] via sigmoid. Regresses TCP.

    Parameters
    ----------
    input_dim:
        Number of input features.
    hidden_dim:
        Hidden layer width.
    dropout:
        Dropout probability.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 16, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


def pairwise_ranking_loss(scores: torch.Tensor, is_correct: torch.Tensor, margin: float = 0.1, max_pairs: int = 2000) -> torch.Tensor:
    """Penalize incorrect-chunk scores that aren't at least `margin` below correct-chunk scores, on random pairs.

    Pairs are sampled uniformly at random rather than stratified by class,
    so anger's influence on this loss is proportional to how often it
    appears -- worth checking per-class AUROC separately if anger
    reliability specifically matters to you.
    """
    correct_idx = torch.nonzero(is_correct == 1).squeeze(-1)
    incorrect_idx = torch.nonzero(is_correct == 0).squeeze(-1)
    if correct_idx.numel() == 0 or incorrect_idx.numel() == 0:
        return torch.tensor(0.0, device=scores.device)

    n_pairs = min(max_pairs, correct_idx.numel() * incorrect_idx.numel())
    ci = correct_idx[torch.randint(0, correct_idx.numel(), (n_pairs,), device=scores.device)]
    ii = incorrect_idx[torch.randint(0, incorrect_idx.numel(), (n_pairs,), device=scores.device)]

    diff = scores[ii] - scores[ci] + margin
    return torch.clamp(diff, min=0).mean()


def train_one_fold(
    X_train, tcp_train, is_correct_train, X_val, video_ids_train=None, val_frac: float = 0.15, hidden_dim: int = 16,
    epochs: int = 300, lr: float = 1e-3, weight_decay: float = 1e-3, lam_rank: float = 1.0, patience: int = 30,
    seed: int = 0, device: torch.device | None = None,
):
    """Train one `TCPReliabilityNet` (MSE + lambda * pairwise-ranking loss) with early stopping on MSE alone."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    tr_idx, es_idx = internal_val_split(len(X_train), video_ids_train, val_frac, seed)

    model = TCPReliabilityNet(X_train.shape[1], hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    Xt = torch.tensor(X_train[tr_idx], dtype=torch.float32, device=device)
    tcp_t = torch.tensor(tcp_train[tr_idx], dtype=torch.float32, device=device)
    correct_t = torch.tensor(is_correct_train[tr_idx], dtype=torch.float32, device=device)
    Xes = torch.tensor(X_train[es_idx], dtype=torch.float32, device=device)
    tcp_es = torch.tensor(tcp_train[es_idx], dtype=torch.float32, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)

    best_state, best_es_loss, bad_epochs = None, float("inf"), 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        scores = model(Xt)
        loss = nn.functional.mse_loss(scores, tcp_t) + lam_rank * pairwise_ranking_loss(scores, correct_t)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            es_loss = nn.functional.mse_loss(model(Xes), tcp_es).item()
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
        val_scores = model(Xv).cpu().numpy()
    return model, val_scores


def run_loto_cv_tcp(
    feature_df: pd.DataFrame, labels: np.ndarray, teacher_ids: np.ndarray, raw_mtkd_probs: np.ndarray,
    n_classes: int | None = None, video_ids: np.ndarray | None = None, seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """LOTO CV for TCP regression + ranking. MTKD stays frozen: `pred = argmax(raw_mtkd_probs)`.

    Parameters mirror `binary.run_loto_cv_binary`.
    """
    pred_all = raw_mtkd_probs.argmax(axis=1)
    is_correct_all = (pred_all == labels).astype(np.float32)
    max_softmax_all = raw_mtkd_probs.max(axis=1)
    tcp_all = raw_mtkd_probs[np.arange(len(labels)), labels]

    results, per_chunk_rows = [], []
    for held_out, train_mask, val_mask, X_train, X_val in loto_scaled_folds(feature_df, teacher_ids, labels, pred_all):
        y_correct_val = is_correct_all[val_mask]
        video_ids_train = video_ids[train_mask] if video_ids is not None else None

        _, val_scores = train_one_fold(X_train, tcp_all[train_mask], is_correct_all[train_mask], X_val, video_ids_train=video_ids_train, seed=seed)

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
    print("\nLOTO average results (TCP regression + ranking):")
    for col in ["auroc_max_softmax", "auroc_learned", "ece_max_softmax", "ece_learned", "brier_max_softmax", "brier_learned"]:
        print(f"  {col}: {results_df[col].mean():.4f}")
    return results_df, per_chunk_df
