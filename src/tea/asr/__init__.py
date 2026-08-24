# src/tea/asr/__init__.py

"""Whisper-based transcription and translation.

Ported from `vad_chunking.txt` (`asr.py`). Logic unchanged: still
Whisper-large-v3 via the HF `pipeline` API, forced Finnish
transcribe/translate. The commented-out `transcribe_with_confidence_hf`
experiment (manual HF generate() call to get no_speech_prob/avg_logprob)
was never used in the reported pipeline and is dropped rather than ported --
flag if you actually want that path revived.

"""

from __future__ import annotations

from omegaconf import DictConfig

from tea.utils.logging import get_logger
from tea.asr.transcriber import Transcriber

logger = get_logger(__name__)


def transcribe_annotation_root(cfg: DictConfig) -> int:
    """CLI entry point for `tea sentiment`-adjacent transcription runs.

    Reads chunk audio referenced by `generated/chunks/` metadata (from
    `tea chunk`) and writes `transcription`/`translation` columns into the
    annotation CSVs under `cfg.paths.annotation_root`. Full wiring depends
    on `tea.vad`'s segment metadata format landing first.

    Parameters
    ----------
    cfg:
        Resolved Hydra configuration.
    """
    logger.info("ASR transcription: not yet wired to tea.vad segment metadata.")
    return 0
