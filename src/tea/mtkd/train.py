# src/tea/mtkd/train.py

"""MTKD student training: fresh training, resuming, fine-tuning an existing
checkpoint, and optional noise-augmented training (report Table 21
"Retrain" rows).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor

from tea.mtkd import data as mtkd_data
from tea.mtkd import engine
from tea.mtkd.frozen_teachers import load_frozen_teachers, logit_permutations
from tea.mtkd.losses import MTKDLoss
from tea.mtkd.model import build_student
from tea.mtkd.utils import collate_fn, confusion_matrix, plot_training, preprocess_function
from tea.noise import NoiseAugmentor
from tea.utils.logging import get_logger
from tea.utils.paths import ensure_dir, resolve
from tea.utils.seed import set_seed

logger = get_logger(__name__)


class Trainer:
    """Trains an MTKD student against the three frozen teachers.

    Parameters
    ----------
    cfg:
        Resolved Hydra config; reads defaults from `cfg.mtkd`.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")

    def checkpoint_path(self, linguality: str, language: str, session: int, suffix: str = "") -> Path:
        """Resolve the checkpoint path for a training run, optionally with a noise-condition suffix."""
        name = f"MTKD_{linguality}_{language}_S{session}{suffix}.pth"
        return resolve(self.cfg.paths.checkpoint_root) / "mtkd" / name

    def _build_loaders(self, ds, feature_extractor, batch_size: int, augmentor: NoiseAugmentor | None):
        hp = self.cfg.mtkd.hyperparams

        def noisy_preprocess(examples):
            if augmentor is None:
                return preprocess_function(examples, feature_extractor, hp.max_duration_sec)
            augmented = []
            for audio in examples["audio"]:
                waveform = torch.from_numpy(audio["array"]).float()
                waveform, _, _ = augmentor.augment(waveform.unsqueeze(0))
                augmented.append(waveform.squeeze(0).cpu().numpy())
            return feature_extractor(
                augmented,
                sampling_rate=feature_extractor.sampling_rate,
                max_length=int(feature_extractor.sampling_rate * hp.max_duration_sec),
                padding=True,
                truncation=True,
            )

        plain_preprocess = lambda ex: preprocess_function(ex, feature_extractor, hp.max_duration_sec)

        encoded_train = ds["train"].map(noisy_preprocess, remove_columns="audio", batched=True)
        encoded_dev = ds["dev"].map(plain_preprocess, remove_columns="audio", batched=True)
        encoded_test = ds["test"].map(plain_preprocess, remove_columns="audio", batched=True)

        if augmentor is not None:
            logger.info("Noise augmentation summary: %s", augmentor.summary())

        train_loader = DataLoader(encoded_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        dev_loader = DataLoader(encoded_dev, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(encoded_test, batch_size=batch_size, collate_fn=collate_fn)
        return train_loader, dev_loader, test_loader

    def train(
        self,
        linguality: str,
        language: str,
        session: int,
        epochs: int | None = None,
        lr: float | None = None,
        mode: str = "train",
        augmentor: NoiseAugmentor | None = None,
        checkpoint_suffix: str = "",
    ) -> Path:
        """Train (fresh, resumed, or fine-tuned) an MTKD student.

        Parameters
        ----------
        linguality, language, session:
            See `tea.mtkd.data.build_dataset`.
        epochs, lr:
            Override `cfg.mtkd.hyperparams`. `lr` defaults to `finetune_lr`
            if `mode="finetune"`, else `learning_rate`.
        mode:
            `"train"` (fresh or auto-resume) or `"finetune"` (requires an
            existing checkpoint; continues training with the optimizer
            state reset, at a lower default LR).
        augmentor:
            Optional `NoiseAugmentor` applied to training-split waveforms
            only (report Table 21 "Retrain" rows). `None` = plain training.
        checkpoint_suffix:
            Appended to the checkpoint filename -- use this to keep
            noise-augmented runs from overwriting the plain checkpoint,
            e.g. `f"_lr{lr}_{noise_type}_{snr_min}_{snr_max}dB"`.

        Returns
        -------
        Path
            The checkpoint path written to.
        """
        hp = self.cfg.mtkd.hyperparams
        set_seed(self.cfg.seed)

        lr = lr if lr is not None else (hp.finetune_lr if mode == "finetune" else hp.learning_rate)
        ckpt_path = self.checkpoint_path(linguality, language, session, checkpoint_suffix)

        if mode == "finetune" and not ckpt_path.exists():
            raise FileNotFoundError(
                f"mode='finetune' requires an existing checkpoint, none found at {ckpt_path}. Run mode='train' first."
            )

        ds = mtkd_data.build_dataset(self.cfg, linguality, language, session)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.cfg.mtkd.model_ckpt)
        train_loader, dev_loader, test_loader = self._build_loaders(ds, feature_extractor, hp.batch_size, augmentor)

        teachers = load_frozen_teachers(self.cfg, self.device)
        perms = logit_permutations(self.cfg, self.device)
        student = build_student(self.cfg, self.device)
        optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=hp.weight_decay)
        loss_fn = MTKDLoss.from_config(self.cfg)

        ensure_dir(ckpt_path.parent)
        plot_dir = ckpt_path.parent / "plots"
        history_path = ckpt_path.with_name(ckpt_path.stem + "_history.json")

        history = {"train_loss": [], "dev_loss": [], "train_uar": [], "dev_uar": [], "train_war": [], "dev_war": []}
        best_dev_uar, epochs_no_improve, start_epoch = 0.0, 0, 1

        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            student.load_state_dict(checkpoint["model_state_dict"])
            if mode == "train":
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch = checkpoint["epoch"] + 1
            best_dev_uar = checkpoint.get("best_dev_uar", 0.0)
            logger.info("Loaded checkpoint epoch %d (best dev UAR=%.4f) [mode=%s, lr=%s]", checkpoint["epoch"], best_dev_uar, mode, lr)
            if history_path.exists():
                with open(history_path) as f:
                    history = json.load(f)

        epochs = epochs or hp.n_epochs
        for epoch in range(start_epoch, start_epoch + epochs):
            train_metrics = engine.train_epoch(student, teachers, perms, train_loader, optimizer, loss_fn, self.device, desc=f"Epoch {epoch} train")
            logger.info("[%d] train loss=%.4f UAR=%.4f WAR=%.4f Acc=%.4f", epoch, train_metrics["loss"], train_metrics["uar"], train_metrics["war"], train_metrics["accuracy"])

            dev_metrics = engine.evaluate(student, teachers, perms, dev_loader, loss_fn, self.device, desc=f"Epoch {epoch} dev")
            logger.info("[%d] dev   loss=%.4f UAR=%.4f WAR=%.4f Acc=%.4f", epoch, dev_metrics["loss"], dev_metrics["uar"], dev_metrics["war"], dev_metrics["accuracy"])

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
                        "model_state_dict": student.state_dict(),
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

        plot_training(history, str(plot_dir), hp.n_epochs, hp.learning_rate, hp.batch_size)

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        student.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = engine.evaluate(student, teachers, perms, test_loader, loss_fn, self.device, desc="Final test")
        logger.info("Final test UAR=%.4f WAR=%.4f Acc=%.4f", test_metrics["uar"], test_metrics["war"], test_metrics["accuracy"])
        logger.info("Confusion matrix:\n%s", confusion_matrix(test_metrics["actual"], test_metrics["predicted"]))

        return ckpt_path


def train_mtkd_cli(cfg: DictConfig) -> int:
    """`tea train-mtkd` entry point.

    Requires `mtkd.linguality`/`mtkd.language`/`mtkd.session`. Optional
    `mtkd.mode` (train/finetune) and a noise block for the Table 21 "Retrain"
    rows: `mtkd.noise.type` (single/full), `mtkd.noise.snr_min`, `mtkd.noise.snr_max`.
    """
    if cfg.mtkd.linguality is None or cfg.mtkd.language is None or cfg.mtkd.session is None:
        logger.error("Set mtkd.linguality / mtkd.language / mtkd.session")
        return 2

    trainer = Trainer(cfg)
    augmentor, suffix = None, ""
    noise_cfg = cfg.mtkd.get("noise", None)
    if noise_cfg and noise_cfg.get("type", "none") != "none":
        aug_cfg = cfg.noise.augment
        augmentor = NoiseAugmentor(
            noise_path=aug_cfg.noise_path,  # point this at your single-clip or full-collection noise source
            contam_prob=noise_cfg.get("contam_prob", aug_cfg.contam_prob),
            snr_min=noise_cfg.get("snr_min", aug_cfg.snr_min),
            snr_max=noise_cfg.get("snr_max", aug_cfg.snr_max),
        )
        suffix = f"_lr{cfg.mtkd.get('lr', cfg.mtkd.hyperparams.learning_rate)}_{noise_cfg.type}_{noise_cfg.get('snr_min')}_{noise_cfg.get('snr_max')}dB"

    trainer.train(
        cfg.mtkd.linguality,
        cfg.mtkd.language,
        cfg.mtkd.session,
        epochs=cfg.mtkd.get("epochs"),
        lr=cfg.mtkd.get("lr"),
        mode=cfg.mtkd.get("mode", "train"),
        augmentor=augmentor,
        checkpoint_suffix=suffix,
    )
    return 0
