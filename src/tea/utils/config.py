# src/tea/config.py

"""Utilities for loading the project's Hydra configuration."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


def load_config(overrides: Sequence[str]) -> DictConfig:
    """Load the project Hydra configuration.

    Parameters
    ----------
    overrides:
        Hydra-style configuration overrides, for example `["vad.atten_lim_db=15"]`.

    Returns
    -------
    DictConfig
        Composed Hydra configuration.
    """
    project_root = Path(__file__).resolve().parents[2]
    config_dir = project_root / "conf"

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg: DictConfig = compose(
            config_name="config",
            overrides=list(overrides),
        )

    return cfg
