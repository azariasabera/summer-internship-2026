# src/tea/classroom/__init__.py

"""Leave-one-teacher-out classroom fine-tuning, with optional FESC contamination.

| Module | Ported from |
|---|---|
| `data.py` | `mtkd_code.txt`'s `classroom_data.py` |
| `fesc.py` | `mtkd_code.txt`'s `fesc_contamination.py` |
| `finetune.py` | `mtkd_code.txt`'s `finetune_classroom.py` |
"""

from __future__ import annotations

from omegaconf import DictConfig

from tea.classroom.data import build_full_df, internal_val_split, teacher_grouped_folds, to_hf_dataset
from tea.classroom.fesc import build_noise_pool_for_fold, contaminate_fesc, estimate_snr_stats, fesc_pool_df
from tea.classroom.finetune import LOTOFineTuner, finetune_classroom_cli

__all__ = [
    "LOTOFineTuner",
    "build_full_df",
    "teacher_grouped_folds",
    "internal_val_split",
    "to_hf_dataset",
    "build_noise_pool_for_fold",
    "estimate_snr_stats",
    "fesc_pool_df",
    "contaminate_fesc",
    "finetune_classroom_cli",
]


def finetune_classroom_loto(cfg: DictConfig) -> int:
    """CLI entry point registered as `tea finetune-classroom-loto` (see `src/tea/cli.py`)."""
    return finetune_classroom_cli(cfg)