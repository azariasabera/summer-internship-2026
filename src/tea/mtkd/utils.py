# src/tea/mtkd/utils.py

"""Shared feature-extraction, collation, plotting, and weighting helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from torch.nn.utils.rnn import pad_sequence

CONFIDENCE_DISCOUNT_DEFAULT = {1: 0.7, 2: 0.85, 3: 1.0}
AUGMENTED_DISCOUNT_DEFAULT = 0.8


def preprocess_function(examples: dict, feature_extractor, max_duration_sec: float = 20.0) -> dict:
    """Feature-extract a batch of `{"audio": [{"array": ...}, ...]}` examples.

    Parameters
    ----------
    examples:
        A batched HF dataset slice with an `"audio"` column.
    feature_extractor:
        A `Wav2Vec2FeatureExtractor`.
    max_duration_sec:
        Clips/pads every sample to this duration.
    """
    audio_arrays = [x["array"] for x in examples["audio"]]
    return feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * max_duration_sec),
        padding=True,
        truncation=True,
    )


def collate_fn(batch: list[dict]) -> dict:
    """Labeled-batch collator for train/eval loops."""
    batch = sorted(batch, key=lambda x: len(x["input_values"]), reverse=True)
    inputs = [torch.tensor(x["input_values"]) for x in batch]
    labels = [int(x["label"]) for x in batch]
    return {"input_values": pad_sequence(inputs, batch_first=True), "label": torch.tensor(labels)}


def collate_fn_infer(batch: list[dict]) -> dict:
    """Unlabeled-batch collator for inference; keeps file ids since there's no label to key off of."""
    batch = sorted(batch, key=lambda x: len(x["input_values"]), reverse=True)
    inputs = [torch.tensor(x["input_values"]) for x in batch]
    file_ids = [x["file_id"] for x in batch]
    return {"input_values": pad_sequence(inputs, batch_first=True), "file_id": file_ids}


def collate_fn_weighted(batch: list[dict]) -> dict:
    """Labeled-batch collator that also carries a `loss_weight` column (defaults to 1.0)."""
    batch = sorted(batch, key=lambda x: len(x["input_values"]), reverse=True)
    inputs = [torch.tensor(x["input_values"]) for x in batch]
    labels = [int(x["label"]) for x in batch]
    weights = [float(x.get("loss_weight", 1.0)) for x in batch]
    return {
        "input_values": pad_sequence(inputs, batch_first=True),
        "label": torch.tensor(labels),
        "loss_weight": torch.tensor(weights, dtype=torch.float32),
    }


def confusion_matrix(actual: list[int], predicted: list[int]):
    """Thin wrapper over `sklearn.metrics.confusion_matrix`."""
    return sk_confusion_matrix(actual, predicted)


def plot_training(history: dict, save_dir: str, n_epochs: int, learning_rate: float, batch_size: int) -> Path:
    """Save loss/UAR training curves to `save_dir`.

    Parameters
    ----------
    history:
        Dict with keys `train_loss`, `dev_loss`, `train_uar`, `dev_uar`.
    save_dir:
        Directory to write the PNG into.
    n_epochs, learning_rate, batch_size:
        Only used to build a descriptive filename.
    """
    import matplotlib.pyplot as plt

    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["dev_loss"], label="Dev")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_uar"], label="Train")
    plt.plot(epochs, history["dev_uar"], label="Dev")
    plt.xlabel("Epoch")
    plt.ylabel("UAR")
    plt.title("UAR Curve")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = save_dir_path / f"training_curves_e{n_epochs}_lr{learning_rate}_bs{batch_size}_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return save_path


# ---------------------------------------------------------------------------
# Classroom fine-tuning-only: class/confidence weighting
# ---------------------------------------------------------------------------


def compute_class_weights(train_df, num_classes: int = 4, clip_max: float = 6.0) -> torch.Tensor:
    """Inverse-frequency class weights, normalized so the smallest weight is ~1 and clipped at `clip_max`.

    Parameters
    ----------
    train_df:
        Training dataframe with an integer `label` column.
    num_classes:
        Number of classes.
    clip_max:
        Maximum allowed weight (chosen empirically, see `conf/classroom/classroom.yaml`).
    """
    counts = train_df["label"].astype(int).value_counts().reindex(range(num_classes), fill_value=0)
    counts = counts.clip(lower=1)
    inv_freq = counts.sum() / (num_classes * counts)
    weights = inv_freq / inv_freq.min()
    weights = weights.clip(upper=clip_max)
    return torch.tensor(weights.values, dtype=torch.float32)


def confidence_discount(confidence_series, discount_map: dict | None = None):
    """Map a 1-3 annotation-confidence column to a loss-weight discount factor."""
    discount_map = discount_map or CONFIDENCE_DISCOUNT_DEFAULT
    return confidence_series.map(discount_map).fillna(1.0)


def attach_loss_weight_column(
    df,
    use_class_weight: bool,
    use_confidence_weight: bool,
    class_weights: torch.Tensor | None = None,
    confidence_discount_map: dict | None = None,
    augmented_discount: float = AUGMENTED_DISCOUNT_DEFAULT,
):
    """Write a `loss_weight` column onto `df`, combining independent weighting factors.

    Weights are attached to the dataframe (not reconstructed from batch
    position later) so they survive `Dataset` construction and
    `DataLoader` shuffling correctly.

    Parameters
    ----------
    df:
        Training dataframe.
    use_class_weight:
        Multiply by `class_weights[label]` (rebalances rare classes).
    use_confidence_weight:
        Multiply by a gentle per-annotation-confidence discount.
    class_weights:
        Required if `use_class_weight=True` (see `compute_class_weights`).
    confidence_discount_map:
        Overrides the default {1: 0.7, 2: 0.85, 3: 1.0} mapping.
    augmented_discount:
        Flat multiplier applied to rows where `is_augmented=True` (FESC-injected), regardless of the switches above.
    """
    w = np.ones(len(df), dtype=np.float32)

    if use_class_weight:
        assert class_weights is not None, "pass class_weights from compute_class_weights()"
        w = w * class_weights[torch.tensor(df["label"].values, dtype=torch.long)].numpy()

    if use_confidence_weight and "confidence" in df.columns:
        w = w * confidence_discount(df["confidence"], confidence_discount_map).values.astype(np.float32)

    if "is_augmented" in df.columns:
        aug_mask = df["is_augmented"].values.astype(bool)
        w = np.where(aug_mask, w * augmented_discount, w)

    df["loss_weight"] = w
    return df


def weighted_ce(logits: torch.Tensor, labels: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
    """Weighted cross-entropy: `sum(w_i * CE_i) / sum(w_i)`."""
    per_sample = F.cross_entropy(logits, labels, reduction="none")
    return (per_sample * sample_weights).sum() / sample_weights.sum()


def freeze_for_variant(model, variant: str) -> None:
    """Freeze parameters according to a fine-tuning variant.

    ```
    Variant     Conv feature encoder   Transformer encoder   Projector + Classifier
    full        frozen (always)        trainable              trainable
    head_only   frozen (always)        frozen                 trainable
    ```

    `variant="full"` is a no-op here because the conv feature encoder is
    already frozen by `build_student`'s `freeze_feature_encoder()` --
    everything else stays trainable. `variant="head_only"` additionally
    freezes everything except parameter names containing `"projector"` or
    `"classifier"` (standard HF `Wav2Vec2ForSequenceClassification` naming).

    [Raw Audio Input] 
        │
    [Frozen CNN Feature Encoder] ──> Extracted acoustic frames
        │
    [Transformer Blocks (12 Layers)] ──> 3D Tensor: (Batch, Time Frames, 768)
        │
    [Projector Layer] ──> Linear downsample to (Batch, Time Frames, 256)
        │
    [MEAN POOLING LAYER] ──> Averages all frames across the time axis
        │                  Resulting 2D Shape: (Batch, 256)
        │
    [Classification Head] ──> Linear projection to 4 target emotion labels


    The pooling layer is a stateless mathematical operation

    projector (Trainable): Changes the frame dimension from 768 to 256.
    Mean Pooling (Stateless): Collapses the time dimension by averaging the frames. No parameters are stored or modified here.
    classifier (Trainable): Changes the 256 features into your 4 final emotion logits.

    Parameters
    ----------
    model:
        The student model to freeze in-place.
    variant:
        `"full"` or `"head_only"`.
    """
    if variant == "full":
        return

    head_keywords = ("projector", "classifier")
    n_trainable, n_total = 0, 0
    for name, param in model.named_parameters():
        n_total += 1
        keep = any(k in name for k in head_keywords)
        param.requires_grad = keep
        n_trainable += int(keep)

    print(f"head_only: {n_trainable}/{n_total} parameter tensors trainable (should be projector/classifier only)")


def describe_trainable_params(model) -> dict:
    """Return trainable vs. frozen parameter names, for sanity-checking `freeze_for_variant`.

    Replaces the original standalone `check.py` diagnostic script.
    """
    trainable, frozen = [], []
    for name, param in model.named_parameters():
        (trainable if param.requires_grad else frozen).append(name)
    return {"trainable": trainable, "frozen": frozen, "n_trainable": len(trainable), "n_total": len(trainable) + len(frozen)}
