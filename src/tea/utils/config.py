# src/tea/utils/config.py

"""Utilities for loading the project's Hydra configuration."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import subprocess

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

def get_git_commit() -> str:
    """Return the current Git commit hash.

    Returns
    -------
    str
        The current Git commit hash, or `"unknown"` if Git is
        unavailable or the current directory is not a Git repository.
    """
    try:
        commit_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"

    return commit_hash.stdout.strip()

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

def get_config_path() -> Path:
    """Return the path to the project's Hydra configuration directory."""
    return find_project_root() / "conf" # or Path(__file__).resolve().parents[3] / "conf"

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
    config_path = get_config_path()
    git_commit = get_git_commit() # `git show commit_hash`` to get the commit message

    print(f"[tea] Git commit: {git_commit}")
    print(f"[tea] Config path: {config_path}")

    with initialize_config_dir(version_base=None, config_dir=str(config_path)):
        cfg: DictConfig = compose(
            config_name="config",
            overrides=list(overrides),
        )

    return cfg
