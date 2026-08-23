# src/tea/utils/seed.py

"""Deterministic seeding shared by every module that trains or samples."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch (if installed) for reproducibility.

    Parameters
    ----------
    seed:
        Seed value. Callers normally pass `cfg.seed` (see `conf/config.yaml`).

    Notes
    -----
    Torch is imported lazily so this module has no hard torch dependency
    for CLI paths that never touch a model (e.g. `tea chunk`).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
