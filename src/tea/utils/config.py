# src/tea/utils/config.py

"""Utilities for loading the project's Hydra configuration."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

def find_project_root() -> Path:
    """Find the project root directory by locating the top-level `conf/` directory.

    The project is expected to have a structure similar to:

        project/
        ├── conf/
        └── src/
            └── tea/
                └── utils/
                    └── config.py

    Starting from this file, each parent directory is searched until a directory containing `conf/` is found.

    Returns:
        Path: The project root directory containing `conf/`.

    Raises:
        RuntimeError: If no parent directory contains a `conf/` directory.
    """
    current: Path = Path(__file__).resolve()

    for parent in current.parents:
        if (parent / "conf").is_dir():
            return parent

    raise RuntimeError("Could not find project root containing conf/")

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
    project_root = find_project_root() # or Path(__file__).resolve().parents[3]
    config_dir = project_root / "conf"

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg: DictConfig = compose(
            config_name="config",
            overrides=list(overrides),
        )

    return cfg
