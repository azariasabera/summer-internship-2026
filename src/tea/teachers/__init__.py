# src/tea/teachers/__init__.py

"""Public API and CLI entrypoints for teacher training."""

from __future__ import annotations

import warnings

from omegaconf import DictConfig

from tea.teachers.trainer import TeacherTrainer
from tea.teachers.data import LOADERS # for public API re-export
from tea.utils.logging import get_logger


warnings.filterwarnings("ignore")

logger = get_logger(__name__)

def train_teacher(cfg: DictConfig) -> int:
    """`tea train-teacher` -- train one teacher model.

    Parameters
    ----------
    cfg:
        Resolved Hydra config. Expects `cfg.teachers.language` and `cfg.teachers.session`
        to be set via CLI override, e.g.
        `tea train-teacher teachers.language=FI teachers.session=6`.
    """
    if cfg.teachers.language is None or cfg.teachers.session is None:
        logger.error("Set teachers.language and teachers.session, e.g. teachers.language=FI teachers.session=6")
        return 2

    trainer = TeacherTrainer(cfg)
    trainer.train(cfg.teachers.language, cfg.teachers.session)
    return 0