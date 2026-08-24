# src/tea/mtkd/losses.py

"""Multi-teacher knowledge-distillation loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from omegaconf import DictConfig


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.view(a.size(0), -1), b.view(b.size(0), -1), dim=1).mean()


class MTKDLoss:
    """Multi-teacher KD loss with per-batch cosine-similarity teacher weighting.

    `loss = (1 - lambda_param) * CE(student, labels) + lambda_param * weighted_KL(student, teachers)`

    Each teacher's contribution to the KL term is weighted by
    `softmax(cosine_similarity(student_logits, teacher_logits) / cosine_temp)`
    across teachers.

    Parameters
    ----------
    temperature:
        Softmax temperature for both student and teacher logits in the KL term.
    lambda_param:
        Weight on the KL term vs. the CE term.
    cosine_temp:
        Temperature for the teacher-weighting softmax (lower = sharper, closer to winner-take-all).
    """

    def __init__(self, temperature: float = 5.0, lambda_param: float = 0.25, cosine_temp: float = 0.25) -> None:
        self.temperature = temperature
        self.lambda_param = lambda_param
        self.cosine_temp = cosine_temp
        self.ce = torch.nn.CrossEntropyLoss()
        self.kl = torch.nn.KLDivLoss(reduction="mean")  # could be "batchmean"?

    @classmethod
    def from_config(cls: type["MTKDLoss"], cfg: DictConfig) -> "MTKDLoss":
        """Build from `cfg.mtkd.hyperparams.{temperature,lambda_param,cosine_temp}`."""
        hp = cfg.mtkd.hyperparams
        return cls(temperature=hp.temperature, lambda_param=hp.lambda_param, cosine_temp=hp.cosine_temp)

    def __call__(self, student_logits: torch.Tensor, teacher_logits: dict[str, torch.Tensor], labels: torch.Tensor):
        """Compute the loss. __call__ makes the object behave like a function. loss=MTKDLoss(...) then loss(...) is a function call.

        Parameters
        ----------
        student_logits:
            `(B, 4)` student logits.
        teacher_logits:
            `{"EN": tensor, "FI": tensor, "FR": tensor}`, each `(B, 4)`,
            already permuted into canonical class order by the caller
            (see `frozen_teachers.logit_permutations`).
        labels:
            `(B,)` integer class labels.

        Returns
        -------
        tuple
            `(loss, aux)` where `aux` holds `loss_ce`, `loss_kl`, per-teacher
            cosine similarity, and per-teacher weight, for logging.
        """
        langs = list(teacher_logits.keys())
        student_log_soft = F.log_softmax(student_logits / self.temperature, dim=-1)

        cos_sims = torch.stack([_cosine_similarity(student_logits, teacher_logits[l]) for l in langs])
        weights = F.softmax(cos_sims / self.cosine_temp, dim=-1)

        kd_losses = {}
        loss_kl = 0.0
        for lang, w in zip(langs, weights):
            teacher_soft = F.softmax(teacher_logits[lang] / self.temperature, dim=-1)
            kd_losses[lang] = self.kl(student_log_soft, teacher_soft) * (self.temperature**2)
            loss_kl = loss_kl + w * kd_losses[lang]

        loss_ce = self.ce(student_logits, labels)
        loss = (1.0 - self.lambda_param) * loss_ce + self.lambda_param * loss_kl

        aux = {
            "loss_ce": loss_ce.item(),
            "loss_kl": loss_kl.item(),
            "cosine_sim": {l: c.item() for l, c in zip(langs, cos_sims)},
            "teacher_weight": {l: w.item() for l, w in zip(langs, weights)},
        }
        return loss, aux
