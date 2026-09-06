# src/tea/mtkd/evaluate.py

"""Teacher-free evaluation of a trained MTKD student on a labeled test split.
Only the student is loaded, no KD-loss overhead.
"""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig
from sklearn.metrics import recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor

from tea.mtkd import data as mtkd_data
from tea.mtkd.model import load_model
from tea.mtkd.utils import collate_fn, confusion_matrix, preprocess_function
from tea.utils.logging import get_logger
from tea.utils.paths import resolve

logger = get_logger(__name__)


@torch.no_grad()
def run_eval(model, loader, device: torch.device) -> dict:
    """Run the student over `loader`, returning accuracy/UAR/WAR/confidence."""
    model.eval()
    total, correct = 0, 0
    actual, predicted, confidence = [], [], []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        inputs, labels = batch["input_values"].to(device), batch["label"].to(device)
        logits = model(inputs).logits
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()
        actual.extend(labels.tolist())
        predicted.extend(preds.tolist())

        max_p, max_i = probs.max(dim=1)
        confidence.extend(zip(max_p.tolist(), max_i.tolist()))

    return {
        "accuracy": correct / total,
        "uar": recall_score(actual, predicted, average="macro"),
        "war": recall_score(actual, predicted, average="weighted"),
        "actual": actual,
        "predicted": predicted,
        "confidence": confidence,
    }

def evaluate_mtkd_cli(cfg: DictConfig) -> int:
    """Evaluate a trained student on its test split (report Tables 4/5).

    Parameters
    ----------
    cfg:
        Resolved Hydra config.
    """
    linguality = cfg.mtkd.get("linguality")
    language = cfg.mtkd.get("language")
    session = cfg.mtkd.get("session")
    eval_checkpoint = cfg.mtkd.get("eval_checkpoint")
    use_default_eval_ckpt = cfg.mtkd.get("use_default_eval_ckpt", False)

    # Checkpoint selection priority:
    #   1. linguality + language + session
    #   2. eval_checkpoint
    #   3. default_student_checkpoint
    ckpt_path = None

    if linguality is not None and language is not None and session is not None:
        # Highest priority: construct checkpoint path from MTKD config
        ckpt_path = resolve(cfg.mtkd.checkpoint_save_dir) / f"MTKD_{linguality}_{language}_S{session}.pth"

    elif eval_checkpoint is not None:
        # Second priority: explicitly provided checkpoint
        ckpt_path = resolve(eval_checkpoint)

    elif use_default_eval_ckpt:
        # Lowest priority: configured default checkpoint
        ckpt_path = resolve(cfg.mtkd.default_student_checkpoint)

    else:
        logger.error(
            "No evaluation checkpoint specified. Set either "
            "(mtkd.linguality, mtkd.language, mtkd.session), "
            "mtkd.eval_checkpoint, or mtkd.use_default_eval_ckpt=true."
        )
        return 2

    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        return 1

    logger.info("Using evaluation checkpoint: %s", ckpt_path)

    ds = mtkd_data.build_dataset(cfg, linguality, language, session)

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(cfg.mtkd.model_ckpt)
    encoded = ds["test"].map(
        lambda ex: preprocess_function(ex, feature_extractor, cfg.mtkd.hyperparams.max_duration_sec),
        remove_columns="audio",
        batched=True,
    )

    test_loader = DataLoader(encoded, batch_size=cfg.mtkd.hyperparams.batch_size, collate_fn=collate_fn)

    model, epoch = load_model(cfg, ckpt_path, device)
    model.eval()

    logger.info("Loaded checkpoint (epoch %s) from %s", epoch, ckpt_path)

    results = run_eval(model, test_loader, device)

    logger.info(
        "MTKD %s %s S%s | UAR=%.4f WAR=%.4f Acc=%.4f",
        linguality,
        language,
        session,
        results["uar"],
        results["war"],
        results["accuracy"],
    )

    logger.info("Confusion matrix:\n%s", confusion_matrix(results["actual"], results["predicted"]))

    return 0

def evaluate_classroom_cli(cfg: DictConfig) -> int:
    """`tea evaluate-classroom` entry point -- delegates to `tea.analysis.classroom` once that module lands.

    Kept here as a placeholder pointer so the CLI command already exists;
    `tea.mtkd.evaluate` itself only knows how to evaluate against
    benchmark-dataset test splits (Tables 4/5), not the annotated
    classroom chunks (Tables 6-13), which is `tea.analysis`'s job.
    """
    logger.info("Classroom evaluation lives in tea.analysis -- not yet ported (see docs/reproducibility.md).")
    return 0