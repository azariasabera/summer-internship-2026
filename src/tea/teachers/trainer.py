# src/tea/teachers/trainer.py

"""Standalone training for monolingual teacher models.

Contains the TeacherTrainer class, model construction, evaluation,
checkpoint management, and train/dev/test workflows.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForAudioClassification,
    Wav2Vec2FeatureExtractor,
)

from tea.mtkd.utils import (
    collate_fn,
    confusion_matrix,
    plot_training,
    preprocess_function,
)

from tea.teachers.data import LOADERS
from tea.teachers.metrics import recalls

from tea.utils.constants import CLASS_ORDER, ID2LABEL, LABEL2ID
from tea.utils.logging import get_logger
from tea.utils.paths import ensure_dir, resolve
from tea.utils.seed import set_seed

warnings.filterwarnings("ignore")

logger = get_logger(__name__)

class TeacherTrainer:
    """Trains one monolingual wav2vec2-base classifier ("teacher").

    Checkpoint naming matches what `tea.mtkd` expects:
    `<checkpoint_root>/teachers/FT_Monolingual_<LANGUAGE>_S<SESSION>.pth`.

    Parameters
    ----------
    cfg:
        Resolved Hydra config; reads defaults from `cfg.teachers`.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")

    def build_model(self, num_classes: int) -> torch.nn.Module:
        """Build a fresh wav2vec2-base classifier with a frozen feature encoder."""
        model = AutoModelForAudioClassification.from_pretrained(
            self.cfg.teachers.model_ckpt, num_labels=num_classes, label2id=LABEL2ID, id2label=ID2LABEL
        )
        model.freeze_feature_encoder()
        model.to(self.device)
        return model

    def checkpoint_path(self, language: str, session: int) -> Path:
        """Resolve the checkpoint path for a given language/session, per `cfg.teachers.checkpoint_name_template`."""
        name = self.cfg.teachers.checkpoint_name_template.format(language=language, session=session)
        return resolve(self.cfg.paths.checkpoint_root) / "teachers" / name

    def _train_epoch(self, model, loader, optimizer, loss_fn, desc="Training") -> dict:
        from tqdm import tqdm

        model.train()
        total, correct, running_loss = 0, 0, 0.0
        actual, predicted = [], []

        for batch in tqdm(loader, desc=desc, leave=False):
            inputs, labels = batch["input_values"].to(self.device), batch["label"].to(self.device)
            optimizer.zero_grad()
            outputs = model(inputs).logits
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            running_loss += loss.item()
            actual.extend(labels.tolist())
            predicted.extend(preds.tolist())

        uar, war = recalls(actual, predicted)
        return {"loss": running_loss / len(loader), "accuracy": correct / total, "uar": uar, "war": war}

    @torch.no_grad()
    def _evaluate(self, model, loader, loss_fn, desc="Eval") -> dict:
        from tqdm import tqdm

        model.eval()
        total, correct, running_loss = 0, 0, 0.0
        actual, predicted = [], []

        for batch in tqdm(loader, desc=desc, leave=False):
            inputs, labels = batch["input_values"].to(self.device), batch["label"].to(self.device)
            outputs = model(inputs).logits
            loss = loss_fn(outputs, labels)

            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            running_loss += loss.item()
            actual.extend(labels.tolist())
            predicted.extend(preds.tolist())

        uar, war = recalls(actual, predicted)
        return {
            "loss": running_loss / len(loader),
            "accuracy": correct / total,
            "uar": uar,
            "war": war,
            "actual": actual,
            "predicted": predicted,
        }

    def train(self, language: str, session: int, epochs: int | None = None, lr: float | None = None) -> Path:
        """Train (or resume) one teacher, saving only the best-dev-UAR checkpoint.

        Parameters
        ----------
        language:
            "EN", "FI", or "FR".
        session:
            Dataset split/session index (must not exceed `cfg.teachers.max_session[language]`).
        epochs, lr:
            Override `cfg.teachers.hyperparams` values.

        Returns
        -------
        Path
            The checkpoint path written to.
        """
        hp = self.cfg.teachers.hyperparams
        epochs = epochs or hp.n_epochs
        lr = lr or hp.learning_rate

        max_s = self.cfg.teachers.max_session[language]
        if session > max_s:
            raise ValueError(f"{language} only has {max_s} session(s)/split(s).")

        set_seed(self.cfg.seed)
        ds = LOADERS[language](session, self.cfg)

        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.cfg.teachers.model_ckpt)
        encoded = ds.map(
            lambda ex: preprocess_function(ex, feature_extractor, hp.max_duration_sec),
            remove_columns="audio",
            batched=True,
        )
        train_loader = DataLoader(encoded["train"], batch_size=hp.batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(encoded["test"], batch_size=hp.batch_size, collate_fn=collate_fn)
        dev_loader = DataLoader(encoded["dev"], batch_size=hp.batch_size, shuffle=True, collate_fn=collate_fn)

        model = self.build_model(num_classes=len(CLASS_ORDER))
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        ckpt_path = self.checkpoint_path(language, session)
        ensure_dir(ckpt_path.parent)
        history_path = ckpt_path.with_name(ckpt_path.stem + "_history.json")

        history = {"train_loss": [], "dev_loss": [], "train_uar": [], "dev_uar": [], "train_war": [], "dev_war": []}
        best_dev_uar, epochs_no_improve, start_epoch = 0.0, 0, 1

        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_dev_uar = checkpoint.get("best_dev_uar", 0.0)
            logger.info("Resumed from epoch %d (best dev UAR=%.4f)", checkpoint["epoch"], best_dev_uar)
            if history_path.exists():
                with open(history_path) as f:
                    history = json.load(f)

        for epoch in range(start_epoch, epochs + 1):
            train_metrics = self._train_epoch(model, train_loader, optimizer, loss_fn, desc=f"Epoch {epoch} train")
            dev_metrics = self._evaluate(model, dev_loader, loss_fn, desc=f"Epoch {epoch} dev")
            logger.info(
                "[%d] train loss=%.4f UAR=%.4f WAR=%.4f | dev loss=%.4f UAR=%.4f WAR=%.4f",
                epoch,
                train_metrics["loss"],
                train_metrics["uar"],
                train_metrics["war"],
                dev_metrics["loss"],
                dev_metrics["uar"],
                dev_metrics["war"],
            )

            for key in ("train_loss", "dev_loss", "train_uar", "dev_uar", "train_war", "dev_war"):
                split, metric = key.split("_")
                history[key].append((train_metrics if split == "train" else dev_metrics)[metric])
            with open(history_path, "w") as f:
                json.dump(history, f, indent=4)

            if dev_metrics["uar"] > best_dev_uar:
                epochs_no_improve = 0
                best_dev_uar = dev_metrics["uar"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "training_loss": train_metrics["loss"],
                        "validation_loss": dev_metrics["loss"],
                        "train_uar": train_metrics["uar"],
                        "dev_uar": dev_metrics["uar"],
                        "dev_war": dev_metrics["war"],
                        "best_dev_uar": best_dev_uar,
                    },
                    ckpt_path,
                )
                logger.info("Saved BEST checkpoint -> %s (epoch=%d, dev UAR=%.4f)", ckpt_path, epoch, best_dev_uar)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= hp.patience:
                    logger.info("No dev UAR improvement for %d epochs, stopping early at epoch %d.", hp.patience, epoch)
                    break

        plot_training(history, str(ckpt_path.parent / "plots"), hp.n_epochs, hp.learning_rate, hp.batch_size)

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = self._evaluate(model, test_loader, loss_fn, desc="Final test")
        logger.info("Final test UAR=%.4f WAR=%.4f Acc=%.4f", test_metrics["uar"], test_metrics["war"], test_metrics["accuracy"])
        logger.info("Confusion matrix:\n%s", confusion_matrix(test_metrics["actual"], test_metrics["predicted"]))

        return ckpt_path