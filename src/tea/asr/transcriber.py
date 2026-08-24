# src/tea/asr/transcriber.py

"""Whisper-based transcription and translation."""

from __future__ import annotations

import gc

from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import pipeline

AudioInput = str | np.ndarray

DEFAULT_MODEL = "openai/whisper-large-v3"

class Transcriber:
    """Thin wrapper around a HF Whisper ASR pipeline.

    Parameters
    ----------
    model_path:
        HF model id or local path. Defaults to `openai/whisper-large-v3`.
    device:
        Torch device string/int. Defaults to CUDA if available.
    """

    def __init__(self, model_path: str | None = None, device: int | str | None = None) -> None:
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_path or DEFAULT_MODEL,
            device=device,
        )

    @staticmethod
    def _load_audio(x: AudioInput, sr: int) -> np.ndarray:
        if isinstance(x, str):
            return librosa.load(x, sr=sr)[0]
        return x

    def transcribe(self, audio: AudioInput, sr: int = 16_000, language: str = "fi") -> str:
        """Transcribe one chunk in its source language.

        Parameters
        ----------
        audio:
            File path or waveform.
        sr:
            Sample rate (only used if `audio` is a path).
        language:
            Forced source language for Whisper's generate kwargs.
        """
        audio = self._load_audio(audio, sr)
        return self.pipe(audio, generate_kwargs={"language": language, "task": "transcribe"})["text"]

    def translate(self, audio: AudioInput, sr: int = 16_000, language: str = "fi") -> str:
        """Translate one chunk into English.

        Parameters
        ----------
        audio:
            File path or waveform.
        sr:
            Sample rate (only used if `audio` is a path).
        language:
            Forced source language for Whisper's generate kwargs.
        """
        audio = self._load_audio(audio, sr)
        return self.pipe(audio, generate_kwargs={"language": language, "task": "translate"})["text"]

    def transcribe_batch(
        self, audios: list[AudioInput], sr: int = 16_000, language: str = "fi", batch_size: int = 8
    ) -> list[dict]:
        """Batch-transcribe multiple chunks. See `transcribe` for per-item semantics."""
        loaded = [self._load_audio(a, sr) for a in audios]
        return self.pipe(
            loaded, batch_size=batch_size, generate_kwargs={"language": language, "task": "transcribe"}
        )

    def translate_batch(
        self, audios: list[AudioInput], sr: int = 16_000, language: str = "fi", batch_size: int = 8
    ) -> list[dict]:
        """Batch-translate multiple chunks. See `translate` for per-item semantics."""
        loaded = [self._load_audio(a, sr) for a in audios]
        return self.pipe(
            loaded, batch_size=batch_size, generate_kwargs={"language": language, "task": "translate"}
        )

    def close(self) -> None:
        """Release the pipeline and free CUDA memory (call when done with this instance)."""
        del self.pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
