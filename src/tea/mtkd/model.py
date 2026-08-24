# src/tea/mtkd/model.py

"""Student model construction and checkpoint loading."""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig
from transformers import AutoModelForAudioClassification

from tea.utils.constants import CLASS_ORDER, ID2LABEL, LABEL2ID


def build_student(cfg: DictConfig, device: torch.device) -> torch.nn.Module:
    """Build a fresh MTKD student (wav2vec2-base, frozen conv feature encoder).

    Parameters
    ----------
    cfg:
        Resolved Hydra config (reads `cfg.mtkd.model_ckpt`).
    device:
        Device to place the model on.
    """
    model = AutoModelForAudioClassification.from_pretrained(
        cfg.mtkd.model_ckpt, num_labels=len(CLASS_ORDER), label2id=LABEL2ID, id2label=ID2LABEL
    )
    model.freeze_feature_encoder()
    model.to(device)
    return model


def load_student(cfg: DictConfig, checkpoint_path: str | Path, device: torch.device, eval_mode: bool = True):
    """Build a student and load a trained checkpoint into it.

    Parameters
    ----------
    cfg:
        Resolved Hydra config.
    checkpoint_path:
        Path to a `.pth` file saved by `Trainer.train`/`LOTOFineTuner`.
    device:
        Device to load onto.
    eval_mode:
        If True, calls `model.eval()` before returning. Leave False if the
        caller intends to continue training (they should still call
        `model.train()` themselves).

    Returns
    -------
    tuple
        `(model, epoch)` -- `epoch` is `"?"` if the checkpoint didn't record one.
    """
    model = build_student(cfg, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if eval_mode:
        model.eval()
    return model, checkpoint.get("epoch", "?")


def load_model(cfg: DictConfig, checkpoint_path: str | Path, device: torch.device):
    """Same as `load_student` but never calls `.eval()` -- used by the classroom
    fine-tuning loop, which immediately calls `freeze_for_variant` + `.train()` itself.
    """
    model = build_student(cfg, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, checkpoint.get("epoch", "?")
