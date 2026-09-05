# src/tea/classroom/utils.py

import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForAudioClassification

from tea.utils.constants import CLASS_ORDER, LABEL2ID, ID2LABEL

CONFIDENCE_DISCOUNT = {1: 0.7, 2: 0.85, 3: 1.0}
AUGMENTED_DISCOUNT = 0.8


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess_function(examples, feature_extractor, max_duration_sec=20.0):
    audio_arrays = [x["array"] for x in examples["audio"]]
    return feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * max_duration_sec),
        padding=True,
        truncation=True,
    )


def collate_fn_weighted(batch):
    batch = sorted(batch, key=lambda x: len(x["input_values"]), reverse=True)
    inputs = [torch.tensor(x["input_values"]) for x in batch]
    labels = [int(x["label"]) for x in batch]
    weights = [float(x.get("loss_weight", 1.0)) for x in batch]
    return {
        "input_values": pad_sequence(inputs, batch_first=True),
        "label": torch.tensor(labels),
        "loss_weight": torch.tensor(weights, dtype=torch.float32),
    }


def compute_class_weights(train_df, num_classes=4, clip_max=6.0):
    counts = train_df["label"].astype(int).value_counts().reindex(range(num_classes), fill_value=0)
    counts = counts.clip(lower=1)
    inv_freq = counts.sum() / (num_classes * counts)
    weights = inv_freq / inv_freq.min()
    weights = weights.clip(upper=clip_max)
    return torch.tensor(weights.values, dtype=torch.float32)


def confidence_discount(confidence_series):
    return confidence_series.map(CONFIDENCE_DISCOUNT).fillna(1.0)


def attach_loss_weight_column(df, use_class_weight: bool, use_confidence_weight: bool, class_weights=None):
    w = np.ones(len(df), dtype=np.float32)
    if use_class_weight:
        assert class_weights is not None, "pass class_weights from compute_class_weights()"
        w = w * class_weights[torch.tensor(df["label"].values, dtype=torch.long)].numpy()
    if use_confidence_weight and "confidence" in df.columns:
        w = w * confidence_discount(df["confidence"]).values.astype(np.float32)
    if "is_augmented" in df.columns:
        aug_mask = df["is_augmented"].values.astype(bool)
        w = np.where(aug_mask, w * AUGMENTED_DISCOUNT, w)
    df["loss_weight"] = w
    return df


def weighted_ce(logits, labels, sample_weights):
    per_sample = F.cross_entropy(logits, labels, reduction="none")
    return (per_sample * sample_weights).sum() / sample_weights.sum()


def freeze_for_variant(model, variant: str):
    if variant == "full":
        return
    head_keywords = ("projector", "classifier")
    n_trainable, n_total = 0, 0
    for name, param in model.named_parameters():
        n_total += 1
        keep = any(k in name for k in head_keywords)
        param.requires_grad = keep
        n_trainable += int(keep)
    print(
        f"head_only: {n_trainable}/{n_total} parameter tensors trainable "
        f"(should only be projector/classifier — verify using check.py)"
    )

def build_student(device, cfg: DictConfig):
    model = AutoModelForAudioClassification.from_pretrained(
        cfg.classroom.model_ckpt,
        num_labels=len(CLASS_ORDER),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )
    model.freeze_feature_encoder()
    model.to(device)
    return model


def load_model(checkpoint_path, device, cfg: DictConfig):
    model = build_student(device, cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, checkpoint.get("epoch", "?")