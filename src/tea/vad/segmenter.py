# src/tea/vad/segmenter.py

"""VAD-based and fixed-size audio segmentation."""

from __future__ import annotations

import json
from pathlib import Path

import librosa
import torch
from omegaconf import DictConfig

from tea.utils.paths import ensure_dir
from tea.vad.refinement import build_full_timeline, refine_segments

from typing import Optional, Union
from tea.utils.logging import get_logger

logger = get_logger(__name__)

class Segmenter:
    """VAD-based and fixed-size audio chunking.

    Parameters
    ----------
    cfg:
        Resolved Hydra config; reads defaults from `cfg.vad`.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self._vad_model = None
        self._vad_utils = None

    def _load_vad_model(self, device: str | None = None):
        """Lazily load (and cache) the Silero VAD model on this instance."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if self._vad_model is None:
            logger.info("Loading Silero VAD model ...")
            self._vad_model, self._vad_utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
            )
        else:
            current_device = next(self._vad_model.parameters()).device
            if str(current_device) != str(device):
                self._vad_model = self._vad_model.to(device)

        return self._vad_model, self._vad_utils

    def chunk_vad(
        self,
        input_path: Union[str, Path],
        save: bool = False,
        save_dir: Optional[Union[str, Path]] = None,
        enhance: bool = False,
    ) -> list[dict]:
        """Run Silero VAD + refinement on one file or every `.wav` in a directory.

        Parameters
        ----------
        input_path:
            A `.wav` file or a directory of `.wav` files.
        save:
            If True, write one `<stem>.json` segment-metadata file per input
            under `save_dir` (or alongside the input if `save_dir` is None).
        save_dir:
            Output directory for segment metadata (see `save`).
        enhance:
            If True, denoise (via `tea.noise.Denoiser`) before running VAD.
            Off by default -- VAD boundaries in the report were computed on
            raw audio.

        Returns
        -------
        list[dict]
            One `{"audio_path", "sr", "total_samples", "segments"}` dict per
            processed file. `segments` is the refined `{"start", "end",
            "type"}` list in samples.
        """
        cfg = self.cfg.vad
        sr = cfg.get("sample_rate", 16_000)

        model, utils = self._load_vad_model()
        get_speech_timestamps = utils[0]
        read_audio = utils[2]

        input_path = Path(input_path)
        wav_files = [input_path] if input_path.is_file() else sorted(input_path.glob("*.wav"))
        if not wav_files:
            raise FileNotFoundError(f"No WAV files found in '{input_path}'.")

        output_dir = None
        if save:
            output_dir = Path(save_dir) if save_dir else (input_path if input_path.is_dir() else input_path.parent)
            ensure_dir(output_dir)

        def process_file(wav_path: Path) -> dict:
            if enhance:
                from tea.noise import Denoiser

                audio = Denoiser(self.cfg).enhance(wav_path)
            else:
                audio = read_audio(wav_path, sampling_rate=sr)

            speech_ts = get_speech_timestamps(
                audio,
                model,
                sampling_rate=sr,
                threshold=cfg.threshold,
                min_speech_duration_ms=cfg.min_speech_duration_ms,
                min_silence_duration_ms=cfg.min_silence_duration_ms,
                speech_pad_ms=cfg.speech_pad_ms,
            )

            segments = build_full_timeline(speech_ts=speech_ts, total_samples=len(audio))
            segments = refine_segments(
                segments,
                sr=sr,
                max_merge_gap=cfg.max_merge_gap,
                min_speech_ratio=cfg.min_speech_ratio,
                min_speech_duration=cfg.min_speech_duration,
                min_non_speech_duration=cfg.min_non_speech_duration,
                max_segment_duration=cfg.max_segment_duration,
                overlap=cfg.overlap,
            )

            info = {"audio_path": str(wav_path), "sr": sr, "total_samples": len(audio), "segments": segments}

            if save:
                with open(output_dir / f"{wav_path.stem}.json", "w") as f:
                    json.dump(info, f, indent=2)

            return info

        return [process_file(w) for w in wav_files]

    def chunk_fixed(
        self,
        input_path: Union[str, Path],
        chunk_size_s: float = 10.0,
        overlap_s: float = 0.0,
        sr: int = 16_000,
        min_chunk_s: float = 1.5,
        save: bool = False,
        save_dir: Optional[Union[str, Path]] = None,
    ) -> list[dict]:
        """Split audio into fixed-size segments (no VAD).

        Used as an alternative segmentation baseline. Trailing audio shorter
        than `min_chunk_s` is merged into the preceding chunk rather than
        emitted as a short tail.

        Parameters
        ----------
        input_path:
            A `.wav` file or a directory of `.wav` files.
        chunk_size_s, overlap_s, sr, min_chunk_s:
            See docstring in the original `prepare_segments.py`; semantics unchanged.
        save, save_dir:
            See `chunk_vad`.

        Returns
        -------
        list[dict]
            Same shape as `chunk_vad`'s output, but every segment has
            `"type": None` (no speech/non-speech distinction is made here).
        """
        if chunk_size_s <= 0:
            raise ValueError("chunk_size_s must be positive.")
        if overlap_s < 0:
            raise ValueError("overlap_s cannot be negative.")
        if min_chunk_s <= 0:
            raise ValueError("min_chunk_s must be positive.")
        if overlap_s >= chunk_size_s:
            raise ValueError("overlap_s must be smaller than chunk_size_s.")

        input_path = Path(input_path)
        wav_files = [input_path] if input_path.is_file() else list(input_path.glob("*.wav"))
        if not wav_files:
            raise FileNotFoundError(f"No WAV files found in '{input_path}'.")

        output_dir = None
        if save:
            output_dir = Path(save_dir) if save_dir else (input_path if input_path.is_dir() else input_path.parent)
            ensure_dir(output_dir)

        chunk_size = int(chunk_size_s * sr)
        overlap = int(overlap_s * sr)
        step = chunk_size - overlap
        min_chunk_samples = int(min_chunk_s * sr)

        def process_file(wav_path: Path) -> dict:
            audio, sr_loaded = librosa.load(wav_path, sr=sr)

            segments = []
            start = 0
            while start < len(audio):
                end = min(start + chunk_size, len(audio))
                tail_remaining = len(audio) - end
                if 0 < tail_remaining < min_chunk_samples:
                    end = len(audio)
                segments.append({"start": start, "end": end, "type": None})
                if end == len(audio):
                    break
                start += step

            info = {"audio_path": str(wav_path), "sr": sr_loaded, "total_samples": len(audio), "segments": segments}

            if save:
                with open(output_dir / f"{wav_path.stem}.json", "w") as f:
                    json.dump(info, f, indent=2)

            return info

        return [process_file(w) for w in wav_files]