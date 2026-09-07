# src/tea/utils/logging.py

"""Standard logger setup + run-provenance recording.

logging.py
│
├── 1. Create a logger
│      get_logger()
│
├── 2. Record "what code/config produced this run?"
│      git_commit_hash()
│      make_run_dir()
│      log_run_provenance()
│
└── 3. Save terminal output to a file
       _Tee
       tee_stdout_stderr()
"""

from __future__ import annotations

import logging
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, TextIO

from omegaconf import DictConfig, OmegaConf


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with a consistent format:

    [H:M:S] module_name: message

    Parameters
    ----------
    name:
        Usually `__name__` of the calling module.
    """
    logger = logging.getLogger(name)
    if not logger.handlers: # to prevent adding another handler every time get_logger() is called.
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def get_git_commit(cwd: str | Path | None = None) -> str:
    """Return the current Git commit hash.

    Runs `git rev-parse HEAD` in the given directory.
    Returns `"unknown"` if Git is unavailable or the directory
    is not inside a Git repository.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def make_run_dir(logs_root: str | Path, command: str) -> Path:
    """Create a timestamped run directory under logs/runs/."""
    time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(logs_root) / "runs" / f"{time_stamp}_{command}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_info(logger: logging.Logger, cfg: DictConfig, run_dir: Path) -> None:
    """Save the Git commit and resolved config to provenance.yaml.

    Parameters
    ----------
    logger:
        Logger used to record the Git commit.
    cfg:
        Hydra configuration for the current run.
    run_dir:
        Directory where provenance.yaml will be saved.
    """

    commit = get_git_commit()
    logger.info("git commit: %s", commit)

    run_info = {
        "git_commit": commit,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    OmegaConf.save(OmegaConf.create(run_info), run_dir / "provenance.yaml")


class _Tee(TextIO):
    """Write to both the original stream and a file."""

    def __init__(self, original: TextIO, file: TextIO):
        self._original = original
        self._file = file

    def write(self, data: str) -> int:
        self._original.write(data)
        self._file.write(data)
        return len(data)

    def flush(self) -> None:
        self._original.flush()
        self._file.flush()

    # minimal TextIO compatibility
    def fileno(self) -> int:
        return self._original.fileno()

    def isatty(self) -> bool:
        return self._original.isatty()


@contextmanager
def tee_stdout_stderr(log_file: Path) -> Generator[None, None, None]:
    """Context manager that tees stdout + stderr to a file while still printing to terminal."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as f:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(old_stdout, f)
        sys.stderr = _Tee(old_stderr, f)
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr