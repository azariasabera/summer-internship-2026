# src/tea/mtkd/engine.py

"""Student-training loop with multi-teacher KD, and teacher-free evaluation."""

from __future__ import annotations

import torch
from sklearn.metrics import recall_score
from tqdm import tqdm


def _recalls(actual: list[int], predicted: list[int]) -> tuple[float, float]:
    return (
        recall_score(actual, predicted, average="macro"),
        recall_score(actual, predicted, average="weighted"),
    )


def _teacher_logits(teachers: dict, perms: dict, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
    """Run all teachers on `inputs`, permuting each into canonical class order. No grad."""
    with torch.no_grad():
        return {lang: model(inputs).logits[:, perms[lang]] for lang, model in teachers.items()}


def train_epoch(student, teachers, perms, loader, optimizer, loss_fn, device, scheduler=None, desc="Training") -> dict:
    """Run one KD training epoch.

    Parameters
    ----------
    student:
        The student model being trained.
    teachers:
        `{lang: model}` frozen teachers (see `frozen_teachers.load_frozen_teachers`).
    perms:
        `{lang: index_tensor}` from `frozen_teachers.logit_permutations`.
    loader:
        Training `DataLoader` (labeled batches, `collate_fn`).
    optimizer:
        Optimizer for `student`'s parameters.
    loss_fn:
        An `MTKDLoss` instance.
    device:
        Device to run on.
    scheduler:
        Optional LR scheduler, stepped once per batch.
    desc:
        tqdm progress bar label.
    """
    student.train()
    total, correct, running_loss = 0, 0, 0.0
    actual, predicted = [], []

    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        inputs, labels = batch["input_values"].to(device), batch["label"].to(device)

        optimizer.zero_grad()
        t_logits = _teacher_logits(teachers, perms, inputs)
        s_logits = student(inputs).logits

        loss, aux = loss_fn(s_logits, t_logits, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        pbar.set_postfix(
            loss=loss.item(), lr=optimizer.param_groups[0]["lr"], **{f"w_{k}": round(v, 2) for k, v in aux["teacher_weight"].items()}
        )
        preds = s_logits.argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        running_loss += loss.item()
        actual.extend(labels.tolist())
        predicted.extend(preds.tolist())

    uar, war = _recalls(actual, predicted)
    return {"loss": running_loss / len(loader), "accuracy": correct / total, "uar": uar, "war": war}


@torch.no_grad()
def evaluate(student, teachers, perms, loader, loss_fn, device, desc="Eval") -> dict:
    """Evaluate the student with the KD loss active (for dev-set model selection during training).

    For teacher-free test-set evaluation (no KD loss, faster), use `tea.mtkd.evaluate.evaluate_student` instead.
    """
    student.eval()
    total, correct, running_loss = 0, 0, 0.0
    actual, predicted = [], []

    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        inputs, labels = batch["input_values"].to(device), batch["label"].to(device)

        t_logits = _teacher_logits(teachers, perms, inputs)
        s_logits = student(inputs).logits
        loss, aux = loss_fn(s_logits, t_logits, labels)

        pbar.set_postfix(loss=loss.item())
        preds = s_logits.argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        running_loss += loss.item()
        actual.extend(labels.tolist())
        predicted.extend(preds.tolist())

    uar, war = _recalls(actual, predicted)
    return {
        "loss": running_loss / len(loader),
        "accuracy": correct / total,
        "uar": uar,
        "war": war,
        "actual": actual,
        "predicted": predicted,
    }
