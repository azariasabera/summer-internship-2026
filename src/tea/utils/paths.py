# src/tea/utils/paths.py

"""Path resolution helpers.

All real filesystem locations live in `conf/paths/paths.yaml` (or are
overridden per-machine on the command line, e.g.
`tea chunk paths.audio_root=/scratch/me/audio`). Every function here takes `cfg.paths` and
returns a `Path`, creating parent directories on write paths so callers
never need a manual `os.makedirs`.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig


def resolve(path_value: str | Path) -> Path:
    """Turn a config path string into a `Path`, expanding `~`.

    Parameters
    ----------
    path_value:
        Raw string from `cfg.paths.*`.
    """
    return Path(path_value).expanduser()


def ensure_dir(path: Path) -> Path:
    """Create `path` (and parents) if missing, and return it unchanged.

    Parameters
    ----------
    path:
        Directory that should exist after this call.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def video_csv_path(cfg: DictConfig, video_id: str) -> Path:
    """Path to the annotation CSV for one video (`<annotation_root>/<video_id>.csv`)."""
    return resolve(cfg.paths.annotation_root) / f"{video_id}.csv"


def video_prediction_path(cfg: DictConfig, run_name: str) -> Path:
    """Path to the MTKD prediction JSON for a run (`<prediction_root>/<run_name>.json`)."""
    return resolve(cfg.paths.prediction_root) / f"{run_name}.json"


def video_embedding_dir(cfg: DictConfig, video_id: str) -> Path:
    """Directory holding pooled embeddings for one video."""
    return resolve(cfg.paths.embedding_root) / video_id


def generated_dir(cfg: DictConfig, *parts: str) -> Path:
    """Build and create a subdirectory under `generated/` for a given stage's outputs.

    Parameters
    ----------
    parts:
        Path components appended to the generated root, e.g.
        `generated_dir(cfg, "classroom_finetune", "config_A_full")`.
    """
    base = resolve(cfg.paths.get("generated_root", "generated"))
    return ensure_dir(base.joinpath(*parts))
