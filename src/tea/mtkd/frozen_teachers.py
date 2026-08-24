# src/tea/mtkd/frozen_teachers.py

"""Loading the three frozen, pre-trained teacher models used during MTKD training."""

from __future__ import annotations

import torch
from omegaconf import DictConfig
from transformers import AutoModelForAudioClassification

from tea.utils.constants import CLASS_ORDER, ID2LABEL, LABEL2ID
from tea.utils.paths import resolve


def _load_one(cfg: DictConfig, language: str, device: torch.device) -> torch.nn.Module:
    model = AutoModelForAudioClassification.from_pretrained(
        cfg.mtkd.model_ckpt, num_labels=len(CLASS_ORDER), label2id=LABEL2ID, id2label=ID2LABEL
    )
    ckpt_path = resolve(cfg.mtkd.teacher_checkpoints[language])
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Teacher checkpoint not found for {language}: {ckpt_path}\n"
            f"Train it first with `tea train-teacher teachers.language={language} "
            f"teachers.session={cfg.mtkd.teacher_sessions[language]}`."
        )
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.freeze_feature_encoder()
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    model.eval()
    return model


def load_frozen_teachers(cfg: DictConfig, device: torch.device) -> dict[str, torch.nn.Module]:
    """Load all three teachers, frozen and in eval mode.

    Parameters
    ----------
    cfg:
        Resolved Hydra config (reads `cfg.mtkd.teacher_checkpoints`).
    device:
        Device to load onto.

    Returns
    -------
    dict
        `{"EN": model, "FI": model, "FR": model}`.
    """
    return {lang: _load_one(cfg, lang, device) for lang in ("EN", "FI", "FR")}


def logit_permutations(cfg: DictConfig, device: torch.device) -> dict[str, torch.Tensor]:
    """Index tensors reordering each teacher's logit columns into canonical class order.

    Only matters for checkpoints trained with a non-canonical head order;
    for anything retrained with the current `tea.teachers`, this is the identity permutation.

    Usage: `aligned_logits = teacher_logits[:, perm[lang]]`.
    """
    perms = {}
    for lang in ("EN", "FI", "FR"):
        order = CLASS_ORDER
        perms[lang] = torch.tensor([order.index(emo) for emo in CLASS_ORDER], device=device)
    return perms
