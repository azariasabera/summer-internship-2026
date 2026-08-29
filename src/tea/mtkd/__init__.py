# src/tea/mtkd/__init__.py

"""Multilingual Teacher Knowledge Distillation (MTKD) student.

Public API re-exports the pieces most callers need directly. Submodules:

| Module | Ported from |
|---|---|
| `model.py` | `mtkd_code.txt`'s `model.py` |
| `losses.py` | `mtkd_code.txt`'s `losses.py` (KLDivLoss reduction UNCHANGED, see its docstring) |
| `frozen_teachers.py` | `mtkd_code.txt`'s `teachers.py` |
| `data.py` | `mtkd_code.txt`'s `data.py` (single-language loaders live in `tea.teachers`) |
| `engine.py` | `mtkd_code.txt`'s `engine.py` |
| `utils.py` | `mtkd_code.txt`'s `utils.py` |
| `train.py` | `mtkd_code.txt`'s `train.py` + `train_modified.py` merged |
| `infer.py` | `infer.py` + `infer_per_fold.py` + `infer_whole.py` merged |
| `evaluate.py` | `mtkd_code.txt`'s `evaluate.py` |
| `calibrate.py` | `mtkd_code.txt`'s `calibrate.py` |
| `embeddings.py` | `mtkd_code.txt`'s `extract_embeddings.py` |

`check.py` (a standalone diagnostic script) was folded into
`utils.describe_trainable_params` rather than kept as a separate module.
"""

from tea.mtkd.calibrate import Calibrator, calibrate_cli
from tea.mtkd.evaluate import evaluate_classroom_cli, evaluate_mtkd_cli
from tea.mtkd.embeddings import EmbeddingExtractor, extract_embeddings_cli
from tea.mtkd.infer import Inferencer, infer_mtkd_cli
from tea.mtkd.losses import MTKDLoss
from tea.mtkd.train import Trainer, train_mtkd_cli

__all__ = [
    "Trainer",
    "Inferencer",
    "Calibrator",
    "EmbeddingExtractor",
    "MTKDLoss",
    "evaluate_mtkd_cli",
    "train_mtkd_cli",
    "infer_mtkd_cli",
    "calibrate_cli",
    "extract_embeddings_cli",
    "evaluate_classroom_cli",
]


def infer_mtkd(cfg):
    """CLI entry point registered as `tea infer-mtkd` (see `src/tea/cli.py`)."""
    return infer_mtkd_cli(cfg)


def evaluate_classroom(cfg):
    """CLI entry point registered as `tea evaluate-classroom` (see `src/tea/cli.py`)."""
    return evaluate_classroom_cli(cfg)
