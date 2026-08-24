# src/tea/mtkd/calibrate.py

"""Temperature-scaling calibration (report Section 2.3, Table 14)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor

from tea.mtkd import data as mtkd_data
from tea.mtkd.model import load_student
from tea.mtkd.utils import collate_fn, preprocess_function
from tea.utils.constants import CLASS_ORDER, ID2LABEL
from tea.utils.logging import get_logger

logger = get_logger(__name__)


class Calibrator:
    """Fits and evaluates a single scalar temperature T for softmax(logits / T).

    Parameters
    ----------
    cfg:
        Resolved Hydra config.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")

    @torch.no_grad()
    def collect_logits(self, model, loader) -> tuple[torch.Tensor, torch.Tensor]:
        """Run `model` over `loader`, returning `(logits, labels)` concatenated across all batches."""
        all_logits, all_labels = [], []
        for batch in loader:
            inputs = batch["input_values"].to(self.device)
            all_logits.append(model(inputs).logits.cpu())
            all_labels.append(batch["label"])
        return torch.cat(all_logits), torch.cat(all_labels)

    @staticmethod
    def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50, lr: float = 0.01) -> float:
        """Find the scalar T > 0 minimizing NLL of `softmax(logits / T)` against `labels`.

        Optimized in log-space so T can't go negative; starts at T=1 and only moves if it actually helps.
        """
        log_T = torch.zeros(1, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            T = log_T.exp()
            loss = F.cross_entropy(logits / T, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return log_T.exp().item()

    @staticmethod
    def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
        """ECE: bucket predictions by confidence, compare each bucket's mean confidence to its accuracy.

        0 = perfectly honest confidence, higher = more over/under-confident.
        """
        confidences, predictions = probs.max(dim=1)
        accuracies = predictions.eq(labels)

        ece = 0.0
        bin_edges = torch.linspace(0, 1, n_bins + 1)
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            in_bin = (confidences > lo) & (confidences <= hi)
            prop_in_bin = in_bin.float().mean().item()
            if prop_in_bin > 0:
                acc_in_bin = accuracies[in_bin].float().mean().item()
                conf_in_bin = confidences[in_bin].mean().item()
                ece += abs(conf_in_bin - acc_in_bin) * prop_in_bin
        return ece

    @staticmethod
    def per_class_breakdown(probs: torch.Tensor, labels: torch.Tensor, title: str) -> None:
        """Print n/mean-confidence/accuracy per PREDICTED class.

        If some classes stay near-100% confident regardless of
        correctness while others don't, the miscalibration is
        class-conditional.
        """
        confidences, predictions = probs.max(dim=1)
        accuracies = predictions.eq(labels)

        print(f"\n{title}")
        print(f"{'predicted class':16s} {'n':>6s} {'mean conf':>10s} {'accuracy':>10s}")
        for class_id in range(probs.shape[1]):
            mask = predictions == class_id
            n = mask.sum().item()
            if n == 0:
                continue
            mean_conf = confidences[mask].mean().item()
            acc = accuracies[mask].float().mean().item()
            print(f"{ID2LABEL[str(class_id)]:16s} {n:6d} {mean_conf:10.4f} {acc:10.4f}")

    @staticmethod
    def print_confusion_matrix(probs: torch.Tensor, labels: torch.Tensor) -> None:
        """Print confusion matrix + per-class recall + UAR/WAR. Identical before/after calibration by construction."""
        predictions = probs.argmax(dim=1).tolist()
        class_ids = list(range(len(CLASS_ORDER)))
        cm = sk_confusion_matrix(labels.tolist(), predictions, labels=class_ids)

        print("\nConfusion matrix (rows = true label, columns = predicted label):")
        header = " " * 12 + "".join(f"{name:>10s}" for name in CLASS_ORDER)
        print(header)
        for i, row in enumerate(cm):
            print(f"{CLASS_ORDER[i]:12s}" + "".join(f"{v:10d}" for v in row))

        recalls = []
        print("\nPer-class recall:")
        for i, class_name in enumerate(CLASS_ORDER):
            tp = cm[i, i]
            total = cm[i].sum()
            recall = tp / total if total > 0 else 0.0
            recalls.append(recall)
            print(f"{class_name:12s}: {recall:.4f}")

        uar = sum(recalls) / len(recalls)
        war = cm.trace() / cm.sum()
        print(f"\nWAR (Weighted Average Recall / Accuracy): {war:.4f}")
        print(f"UAR (Unweighted Average Recall):         {uar:.4f}")

    def run(
        self, linguality: str, language: str, session: int, split: str = "dev", checkpoint: str | Path | None = None
    ) -> float:
        """Fit and report temperature scaling for one checkpoint.

        Parameters
        ----------
        linguality, language, session:
            Which checkpoint/dataset to calibrate.
        split:
            Which split to FIT T on. Use `"dev"` -- never `"test"`.
        checkpoint:
            Override checkpoint path.

        Returns
        -------
        float
            The fitted temperature T.
        """
        ckpt_path = checkpoint or (
            Path(self.cfg.paths.checkpoint_root) / "mtkd" / f"MTKD_{linguality}_{language}_S{session}.pth"
        )
        model, epoch = load_student(self.cfg, ckpt_path, self.device)
        logger.info("Loaded checkpoint (epoch %s) from %s", epoch, ckpt_path)

        ds = mtkd_data.build_dataset(self.cfg, linguality, language, session)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.cfg.mtkd.model_ckpt)
        encoded = ds[split].map(
            lambda ex: preprocess_function(ex, feature_extractor, self.cfg.mtkd.hyperparams.max_duration_sec),
            remove_columns="audio",
            batched=True,
        )
        loader = DataLoader(encoded, batch_size=self.cfg.mtkd.hyperparams.batch_size, collate_fn=collate_fn)

        logger.info("Collecting logits on '%s' split (%d samples)...", split, len(encoded))
        logits, labels = self.collect_logits(model, loader)

        before_probs = F.softmax(logits, dim=1)
        before_ece = self.expected_calibration_error(before_probs, labels)
        before_conf = before_probs.max(dim=1).values.mean().item()
        before_acc = (before_probs.argmax(dim=1) == labels).float().mean().item()

        T = self.fit_temperature(logits, labels)

        after_probs = F.softmax(logits / T, dim=1)
        after_ece = self.expected_calibration_error(after_probs, labels)
        after_conf = after_probs.max(dim=1).values.mean().item()
        after_acc = (after_probs.argmax(dim=1) == labels).float().mean().item()

        print(f"\nFitted temperature T = {T:.3f}  (T > 1 means the model was overconfident)")
        print(f"{'':10s} {'mean confidence':>16s} {'accuracy':>10s} {'ECE':>10s}")
        print(f"{'before':10s} {before_conf:16.4f} {before_acc:10.4f} {before_ece:10.4f}")
        print(f"{'after':10s} {after_conf:16.4f} {after_acc:10.4f} {after_ece:10.4f}")
        print("\n(accuracy should be IDENTICAL before/after -- if it isn't, something's wrong)")

        self.per_class_breakdown(before_probs, labels, "Per-predicted-class breakdown (BEFORE calibration):")
        self.per_class_breakdown(after_probs, labels, "Per-predicted-class breakdown (AFTER calibration):")
        self.print_confusion_matrix(before_probs, labels)

        print(f"\nTo apply this calibration anywhere you compute softmax at inference:")
        print(f"    calibrated_probs = torch.softmax(logits / {T:.4f}, dim=-1)")

        return T


def calibrate_cli(cfg: DictConfig) -> int:
    """`tea calibrate` entry point.

    Requires `mtkd.linguality`, `mtkd.language`, `mtkd.session` overrides, e.g.:
    `tea calibrate mtkd.linguality=Multilingual mtkd.language=FI mtkd.session=8`
    """
    if cfg.mtkd.linguality is None or cfg.mtkd.language is None or cfg.mtkd.session is None:
        logger.error("Set mtkd.linguality / mtkd.language / mtkd.session")
        return 2

    Calibrator(cfg).run(cfg.mtkd.linguality, cfg.mtkd.language, cfg.mtkd.session)
    return 0
