# src/tea/classroom/finetune.py

"""Orchestrates: LOTO folds over teacher-grouped classroom data -> optional
FESC contamination injected into each fold's TRAIN split only -> pluggable
weighting -> fine-tune (full or head_only) -> baseline vs. fine-tuned
comparison per fold, plus a pooled out-of-fold summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.metrics import confusion_matrix, recall_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import Wav2Vec2FeatureExtractor

from tea.classroom.data import build_full_df, internal_val_split, teacher_grouped_folds, to_hf_dataset
from tea.classroom.fesc import build_noise_pool_for_fold, contaminate_fesc, estimate_snr_stats, fesc_pool_df
from tea.mtkd.model import load_model
from tea.mtkd.utils import (
    attach_loss_weight_column,
    collate_fn_weighted,
    compute_class_weights,
    freeze_for_variant,
    preprocess_function,
    weighted_ce,
)
from tea.utils.constants import CLASS_ORDER
from tea.utils.logging import get_logger
from tea.utils.paths import ensure_dir, resolve
from tea.utils.seed import set_seed

logger = get_logger(__name__)

ALL_LABELS = list(range(len(CLASS_ORDER)))


def _make_loader(df: pd.DataFrame, feature_extractor, max_duration_sec: float, batch_size: int, shuffle: bool = False) -> DataLoader:
    ds = to_hf_dataset(df).map(
        lambda ex: preprocess_function(ex, feature_extractor, max_duration_sec), remove_columns="audio", batched=True
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_weighted)


@torch.no_grad()
def _evaluate(model, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    actual, predicted = [], []
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        inputs, labels = batch["input_values"].to(device), batch["label"].to(device)
        preds = model(inputs).logits.argmax(dim=1)
        actual.extend(labels.tolist())
        predicted.extend(preds.tolist())
    uar = recall_score(actual, predicted, average="macro", labels=ALL_LABELS, zero_division=0)
    war = recall_score(actual, predicted, average="weighted", labels=ALL_LABELS, zero_division=0)
    return {"uar": uar, "war": war, "actual": actual, "predicted": predicted}


class LOTOFineTuner:
    """Runs leave-one-teacher-out classroom fine-tuning for one experiment configuration.

    Parameters
    ----------
    cfg:
        Resolved Hydra config; reads defaults from `cfg.classroom`.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")

    def run_fold(
        self,
        fold_name: str,
        held_out,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        base_checkpoint: str | Path,
        variant: str,
        use_class_weight: bool,
        use_confidence_weight: bool,
        augment_fesc: bool,
        epochs: int,
        lr: float | None,
        batch_size: int,
        save_model_dir: str | Path | None = None,
    ) -> dict:
        """Fine-tune and evaluate one LOTO fold.

        Parameters
        ----------
        fold_name, held_out:
            From `tea.classroom.data.teacher_grouped_folds`.
        train_df, test_df:
            This fold's train/test dataframes.
        base_checkpoint:
            MTKD student checkpoint to fine-tune from.
        variant:
            `"full"` or `"head_only"` (see `tea.mtkd.utils.freeze_for_variant`).
        use_class_weight, use_confidence_weight, augment_fesc:
            The three independent switches of report Table 16.
        epochs, lr, batch_size:
            Fine-tuning hyperparameters. `lr=None` -> 2e-6 for `full`, 1e-4 for `head_only`.
        save_model_dir:
            If set, saves the fine-tuned checkpoint here.
        """
        cc = self.cfg.classroom
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.cfg.mtkd.model_ckpt)
        logger.info("Fold (held out): %s  train=%d  test=%d", fold_name, len(train_df), len(test_df))
        logger.info("Before: %s", train_df.groupby("gt_label").size().to_dict())

        audio_root = resolve(self.cfg.paths.chunk_audio_dir)
        annotation_root = resolve(self.cfg.paths.annotation_root)

        if not audio_root.exists():
            raise FileNotFoundError(f"Chunk audio directory not found: {audio_root}")

        if not annotation_root.exists():
            raise FileNotFoundError(f"Annotation directory not found: {annotation_root}")

        # ── optional FESC contamination, TRAIN split only ────────────────
        if augment_fesc:
            noise_pool = build_noise_pool_for_fold(
                annotation_root, audio_root, train_df,
                extra_video_ids=tuple(cc.fesc.noise_extra_videos),
            )
            snr_stats = estimate_snr_stats(train_df, noise_pool["dynamic"] + noise_pool["mic"], seed=self.cfg.seed)
            fesc_df = fesc_pool_df(self.cfg, classes=tuple(cc.fesc.augment_classes))
            real_counts = train_df["gt_label"].value_counts().to_dict()
            aug_df = contaminate_fesc(
                fesc_df, noise_pool, snr_stats,
                output_dir=Path(self.cfg.paths.fesc_contaminated_dir) / str(fold_name).replace("+", "_"),
                classes=tuple(cc.fesc.augment_classes), cap_multiplier=cc.fesc.augment_cap_multiplier,
                real_class_counts=real_counts, n_noise_sources=tuple(cc.fesc.noise_n_sources),
                snr_std_db=cc.fesc.snr_std_db, snr_clip_min_db=cc.fesc.snr_clip_min_db, snr_clip_max_db=cc.fesc.snr_clip_max_db,
                seed=self.cfg.seed,
            )
            train_df = pd.concat([train_df, aug_df], ignore_index=True)
            logger.info("After: %s", train_df.groupby("gt_label").size().to_dict())

        # ── optional internal val split (early stopping only, never test) ─
        fit_df, val_df = internal_val_split(train_df, val_frac=cc.internal_val_frac, seed=self.cfg.seed)

        # ── weighting: attach BEFORE dataset construction so it survives shuffling ─
        class_weights = compute_class_weights(fit_df, clip_max=cc.class_weight_clip_max) if use_class_weight else None
        if class_weights is not None:
            fit_df = attach_loss_weight_column(
                fit_df, use_class_weight, use_confidence_weight, class_weights,
                confidence_discount_map=dict(cc.confidence_discount), augmented_discount=cc.augmented_discount,
            )
            logger.info("Class weights: %s", {CLASS_ORDER[i]: round(float(w), 4) for i, w in enumerate(class_weights)})
        else:
            logger.info("No class weighting.")

        train_loader = _make_loader(fit_df, feature_extractor, self.cfg.mtkd.hyperparams.max_duration_sec, batch_size, shuffle=True)
        val_loader = _make_loader(val_df, feature_extractor, self.cfg.mtkd.hyperparams.max_duration_sec, batch_size) if val_df is not None else None
        test_loader = _make_loader(test_df, feature_extractor, self.cfg.mtkd.hyperparams.max_duration_sec, batch_size)

        # ── baseline (no fine-tuning) on this fold's held-out teacher ────
        model, epoch = load_model(self.cfg, base_checkpoint, self.device)
        baseline_metrics = _evaluate(model, test_loader, self.device)
        logger.info("[baseline ckpt epoch %s] UAR=%.4f WAR=%.4f", epoch, baseline_metrics["uar"], baseline_metrics["war"])
        logger.info("Baseline confusion matrix:\n%s", confusion_matrix(baseline_metrics["actual"], baseline_metrics["predicted"], labels=ALL_LABELS))

        # ── fine-tune ────────────────────────────────────────────────────
        freeze_for_variant(model, variant)
        default_lr = 2e-6 if variant == "full" else 1e-4
        lr = lr or default_lr
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=self.cfg.mtkd.hyperparams.weight_decay
        )

        best_val_uar, best_state = -1.0, None
        for epoch_i in range(1, epochs + 1):
            model.train()
            running = 0.0
            for batch in train_loader:
                inputs, labels, weights = batch["input_values"].to(self.device), batch["label"].to(self.device), batch["loss_weight"].to(self.device)
                optimizer.zero_grad()
                loss = weighted_ce(model(inputs).logits, labels, weights)
                loss.backward()
                optimizer.step()
                running += loss.item()
            train_loss = running / max(len(train_loader), 1)

            if val_loader is not None:
                val_metrics = _evaluate(model, val_loader, self.device)
                logger.info("epoch %d: train_loss=%.4f internal_val UAR=%.4f", epoch_i, train_loss, val_metrics["uar"])
                if val_metrics["uar"] > best_val_uar:
                    best_val_uar = val_metrics["uar"]
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                logger.info("epoch %d: train_loss=%.4f", epoch_i, train_loss)
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}  # fixed-epoch: last = final

        model.load_state_dict(best_state)

        if save_model_dir is not None:
            tag = f"{variant[:1]}_cw{int(use_class_weight)}_cf{int(use_confidence_weight)}_a{int(augment_fesc)}_e{epochs}_lr{lr:.0e}"
            ckpt_path = ensure_dir(Path(save_model_dir)) / f"{fold_name}_{tag}.pth"
            torch.save(
                {
                    "fold": fold_name, "held_out": held_out, "variant": variant, "epoch": epochs, "learning_rate": lr,
                    "use_class_weight": use_class_weight, "use_confidence_weight": use_confidence_weight, "augment_fesc": augment_fesc,
                    "model_state_dict": best_state,
                },
                ckpt_path,
            )
            logger.info("Saved fine-tuned model -> %s", ckpt_path)

        final_metrics = _evaluate(model, test_loader, self.device)
        logger.info("[fine-tuned] UAR=%.4f WAR=%.4f", final_metrics["uar"], final_metrics["war"])
        logger.info("Fine-tuned confusion matrix:\n%s", confusion_matrix(final_metrics["actual"], final_metrics["predicted"], labels=ALL_LABELS))

        return {
            "fold": fold_name, "held_out": held_out,
            "n_train": len(fit_df), "n_val": len(val_df) if val_df is not None else 0, "n_test": len(test_df),
            "baseline_uar": baseline_metrics["uar"], "baseline_war": baseline_metrics["war"],
            "baseline_actual": baseline_metrics["actual"], "baseline_predicted": baseline_metrics["predicted"],
            "finetuned_uar": final_metrics["uar"], "finetuned_war": final_metrics["war"],
            "finetuned_actual": final_metrics["actual"], "finetuned_predicted": final_metrics["predicted"],
        }

    def run_all_folds(
        self, base_checkpoint: str | Path, variant: str, use_class_weight: bool, use_confidence_weight: bool,
        augment_fesc: bool, epochs: int | None = None, lr: float | None = None, batch_size: int | None = None,
        save_model_dir: str | Path | None = None,
    ) -> dict:
        """Run every LOTO fold for one experiment configuration and report pooled OOF metrics (report Tables 17-20).

        Parameters mirror `run_fold`; see there for details. Returns the
        full per-fold results plus pooled-OOF baseline/fine-tuned UAR/WAR
        and confusion matrices, and writes `results_<tag>.json` under `cfg.paths.generated_root/classroom_finetune/`.
        """
        cc = self.cfg.classroom
        set_seed(self.cfg.get("seed", 42))
        epochs = epochs if epochs is not None else cc.get("epochs", 5)
        batch_size = batch_size if batch_size is not None else cc.get("batch_size", 8)
        lr = lr if lr is not None else cc.get("lr", 2e-5)

        audio_root = resolve(self.cfg.paths.chunk_audio_dir)
        annotation_root = resolve(self.cfg.paths.annotation_root)

        if not audio_root.exists():
            raise FileNotFoundError(f"Chunk audio directory not found: {audio_root}")

        if not annotation_root.exists():
            raise FileNotFoundError(f"Annotation directory not found: {annotation_root}")

        logger.info("Chunk audio directory: %s", audio_root)
        logger.info("Annotation directory: %s", annotation_root)

        df = build_full_df(audio_root, annotation_root, set(cc.excluded_videos))

        results = []
        for fold_name, held_out, train_df, test_df in teacher_grouped_folds(df, n_splits=cc.cv):
            results.append(
                self.run_fold(
                    fold_name, held_out, train_df, test_df, base_checkpoint, variant,
                    use_class_weight, use_confidence_weight, augment_fesc, epochs, lr, batch_size, save_model_dir,
                )
            )

        tag = f"{variant}_cw{int(use_class_weight)}_conf{int(use_confidence_weight)}_aug{int(augment_fesc)}"
        output_dir = ensure_dir(Path(self.cfg.paths.generated_root) / "classroom_finetune")
        slim_results = [{k: v for k, v in r.items() if not k.startswith("finetuned_a") and not k.startswith("finetuned_p")} for r in results]
        with open(output_dir / f"results_{tag}.json", "w") as f:
            json.dump(slim_results, f, indent=2)

        def _pool(key_actual, key_pred):
            actual = sum([r[key_actual] for r in results], [])
            predicted = sum([r[key_pred] for r in results], [])
            uar = recall_score(actual, predicted, average="macro", labels=ALL_LABELS, zero_division=0)
            war = recall_score(actual, predicted, average="weighted", labels=ALL_LABELS, zero_division=0)
            cm = confusion_matrix(actual, predicted, labels=ALL_LABELS)
            return {"uar": uar, "war": war, "confusion_matrix": cm.tolist()}

        baseline_pooled = _pool("baseline_actual", "baseline_predicted")
        finetuned_pooled = _pool("finetuned_actual", "finetuned_predicted")

        # ── per-fold stats (mean ± std) for both metrics, both stages ────
        stats = {}
        for stage in ("baseline", "finetuned"):
            for metric in ("uar", "war"):
                vals = [r[f"{stage}_{metric}"] for r in results]
                stats[f"{stage}_fold_{metric}s"] = [round(float(v), 4) for v in vals]
                stats[f"{stage}_mean_{metric}"] = float(np.mean(vals))
                stats[f"{stage}_std_{metric}"] = float(np.std(vals))

        logger.info("SUMMARY [%s]", tag)
        for stage in ("baseline", "finetuned"):
            logger.info("%s Fold UARs: %s", stage.capitalize(), stats[f"{stage}_fold_uars"])
            logger.info("%s Fold WARs: %s", stage.capitalize(), stats[f"{stage}_fold_wars"])
            logger.info("%s Mean UAR = %.4f ± %.4f", stage.capitalize(), stats[f"{stage}_mean_uar"], stats[f"{stage}_std_uar"])
            logger.info("%s Mean WAR = %.4f ± %.4f", stage.capitalize(), stats[f"{stage}_mean_war"], stats[f"{stage}_std_war"])

        logger.info(
            "Baseline  pooled OOF UAR=%.4f WAR=%.4f\nConfusion matrix:\n%s",
            baseline_pooled["uar"], baseline_pooled["war"], np.array(baseline_pooled["confusion_matrix"]),
        )
        logger.info(
            "Finetuned pooled OOF UAR=%.4f WAR=%.4f\nConfusion matrix:\n%s",
            finetuned_pooled["uar"], finetuned_pooled["war"], np.array(finetuned_pooled["confusion_matrix"]),
        )

        return {
            "tag": tag, "per_fold": results,
            "baseline_pooled": baseline_pooled, "finetuned_pooled": finetuned_pooled,
            **stats,
        }

    def run_config(self, config_name: str, base_checkpoint: str | Path, variant: str, **kwargs) -> dict:
        """Run one of the named experiment configs from `conf/classroom/classroom.yaml` (A-F, report Table 16)."""
        config = self.cfg.classroom.configs[config_name]
        return self.run_all_folds(
            base_checkpoint, variant, config.use_class_weight, config.use_confidence_weight, config.augment_fesc, **kwargs
        )


def finetune_classroom_cli(cfg: DictConfig) -> int:
    """`tea finetune-classroom-loto` entry point.

    Requires `classroom.run.base_checkpoint` and `classroom.run.variant`.
    Either set `classroom.run.config=A`..`F` to use a named report
    configuration, or set `classroom.run.use_class_weight` /
    `use_confidence_weight` / `augment_fesc` individually.
    """
    run_cfg = cfg.classroom.get("run", {})
    base_checkpoint = run_cfg.get("base_checkpoint")
    variant = run_cfg.get("variant")
    if not base_checkpoint or not variant:
        logger.error("Set classroom.run.base_checkpoint and classroom.run.variant (full|head_only)")
        return 2

    tuner = LOTOFineTuner(cfg)
    save_model_dir = run_cfg.get("save_model_dir") or resolve(cfg.paths.classroom_finetune_dir)

    if run_cfg.get("config"):
        tuner.run_config(run_cfg.config, base_checkpoint, variant, save_model_dir=save_model_dir)
    else:
        tuner.run_all_folds(
            base_checkpoint, variant,
            use_class_weight=run_cfg.get("use_class_weight", False),
            use_confidence_weight=run_cfg.get("use_confidence_weight", False),
            augment_fesc=run_cfg.get("augment_fesc", False),
            save_model_dir=save_model_dir,
        )
    return 0
