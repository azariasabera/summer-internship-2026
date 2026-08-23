# src/tea/utils/logging.py

"""Standard logger setup + run-provenance recording.

Per the Aalto group guideline: every recipe run should log the git commit
it was executed from, so an old result can always be traced back to the
exact code version that produced it. Hydra already records the resolved
config for every run; this module adds the one piece Hydra doesn't: git
state.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with a consistent format.

    Parameters
    ----------
    name:
        Usually `__name__` of the calling module.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def git_commit_hash(cwd: str | Path | None = None) -> str:
    """Return the current Git commit hash.

    Parameters
    ----------
    cwd:
        Directory to run Git from. Defaults to the current working directory.

    Returns
    -------
    str
        The current Git commit hash, or `"unknown"` if Git is unavailable
        or the directory is not inside a Git repository.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"

    return result.stdout.strip()

def log_run_provenance(logger: logging.Logger, cfg: DictConfig, run_dir: str | Path) -> None:
    """Log and persist the git commit + resolved config for the current run.

    Writes `provenance.yaml` (commit hash + resolved config) into `run_dir`,
    in addition to logging the commit hash. This is the file to check when
    an old result under `generated/` needs to be traced back to the code
    and settings that produced it.

    Parameters
    ----------
    logger:
        Logger obtained from `get_logger`.
    cfg:
        Resolved Hydra config for the current run.
    run_dir:
        Directory the current command is writing outputs to.
    """
    commit = git_commit_hash()
    logger.info("git commit: %s", commit)

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    provenance = {"git_commit": commit, "config": OmegaConf.to_container(cfg, resolve=True)}
    OmegaConf.save(OmegaConf.create(provenance), run_dir / "provenance.yaml")
