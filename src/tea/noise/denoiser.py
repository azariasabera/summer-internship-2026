# src/tea/noise/denoiser.py

"""Denoising audio using DeepFilterNet."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchaudio
from omegaconf import DictConfig

from tea.utils.logging import get_logger
from tea.utils.paths import ensure_dir

from typing import Optional, Union

logger = get_logger(__name__)

AudioArray = np.ndarray

class Denoiser:
    """DeepFilterNet2-based speech enhancement.

    Handles both short clips (single forward pass) and long recordings
    (overlap-add chunked processing, to avoid the cuDNN failures / OOM that
    a single very long sequence can trigger). Falls back to chunked
    processing automatically if the native path throws.

    Parameters
    ----------
    cfg:
        Resolved Hydra config; reads defaults from `cfg.noise`.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg.noise
        self._model = None
        self._df_state = None

    def close(self) -> None:
        """Release the DeepFilterNet model and associated resources."""
        if self._model is not None:
            self._model.cpu()

        self._model = None
        self._df_state = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_model(self, device: str | None = None):
        from df import init_df

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if self._model is None:
            self._model, self._df_state, _ = init_df(model_base_dir="DeepFilterNet2")
            self._model = self._model.to(device)
            self._model.eval()
        else:
            current_device = next(self._model.parameters()).device
            if str(current_device) != str(device):
                self._model = self._model.to(device)

        return self._model, self._df_state

    @staticmethod
    def _get_fade(overlap: int, device=None):
        window = torch.hann_window(overlap * 2, periodic=False)
        fade_in = window[:overlap]
        fade_out = 1 - fade_in
        if device is not None:
            fade_in, fade_out = fade_in.to(device), fade_out.to(device)
        return fade_in, fade_out

    def _enhance_chunked(self, audio, sr, chunk_sec, overlap_sec, atten_lim_db):
        """Overlap-add chunked enhancement for long audio. See module docstring."""
        from df import enhance

        model, df_state = self._load_model()
        device = audio.device
        total_len = audio.shape[-1]
        chunk_size = int(chunk_sec * sr)
        overlap = int(overlap_sec * sr)
        if chunk_size <= overlap:
            raise ValueError("chunk_sec must be greater than overlap_sec")

        fade_in, fade_out = self._get_fade(overlap, device=device)
        chunks_out = []
        start = 0

        while start < total_len:
            end = min(start + chunk_size, total_len)
            chunk = audio[:, start:end].contiguous()
            with torch.no_grad():
                out = enhance(model, df_state, chunk, atten_lim_db=atten_lim_db).contiguous()

            if not chunks_out:
                chunks_out.append(out)
            else:
                prev = chunks_out[-1]
                valid_overlap = min(overlap, prev.shape[-1], out.shape[-1])
                if valid_overlap > 0:
                    prev_tail = prev[:, -valid_overlap:]
                    new_head = out[:, :valid_overlap]
                    blended = prev_tail * fade_out[:valid_overlap] + new_head * fade_in[:valid_overlap]
                    chunks_out[-1] = torch.cat([prev[:, :-valid_overlap], blended], dim=-1)
                    chunks_out.append(out[:, valid_overlap:])
                else:
                    chunks_out.append(out)

            start += chunk_size - overlap

        return torch.cat(chunks_out, dim=-1)

    def _enhance_native(self, audio, sr, atten_lim_db):
        """Single-pass enhancement. Only safe for short clips."""
        from df import enhance

        model, df_state = self._load_model()
        assert sr == 48000, "DeepFilterNet requires exactly 48000 Hz input"
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        audio = audio.contiguous()
        with torch.no_grad():
            enhanced = enhance(model=model, df_state=df_state, audio=audio, atten_lim_db=atten_lim_db)
        return enhanced.contiguous()

    def enhance(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        save: bool = False,
        atten_lim_db: Optional[int] = None,
    ) -> AudioArray:
        """Denoise one audio file and return it resampled to 16kHz.

        Parameters
        ----------
        input_path:
            Path to the audio file to enhance.
        output_path:
            Where to save the enhanced 16kHz audio, if `save=True`.
        save:
            If True, write `output_path`.
        atten_lim_db:
            Attenuation limit override. Defaults to `cfg.noise.atten_lim_db`
            (15 = milder, 0 = stronger).
        """
        cfg = self.cfg
        atten_lim_db = atten_lim_db if atten_lim_db is not None else cfg.atten_lim_db

        audio, sr = torchaudio.load(str(input_path))
        if sr != 48000:
            audio = torchaudio.functional.resample(audio, sr, 48000)
            sr = 48000
        audio = audio.contiguous()

        use_chunked = cfg.force_chunked or audio.shape[-1] > cfg.long_audio_threshold_sec * sr

        if use_chunked:
            enhanced = self._enhance_chunked(audio, sr, cfg.chunk_sec, cfg.overlap_sec, atten_lim_db)
        else:
            try:
                enhanced = self._enhance_native(audio, sr, atten_lim_db)
            except RuntimeError as e:
                logger.warning("native DeepFilterNet path failed (%s); falling back to chunked", e)
                enhanced = self._enhance_chunked(audio, sr, cfg.chunk_sec, cfg.overlap_sec, atten_lim_db)

        enhanced = enhanced.cpu()
        enhanced_16k = torchaudio.functional.resample(enhanced, 48000, 16_000).contiguous()

        if save and output_path:
            ensure_dir(Path(output_path).parent)
            torchaudio.save(str(output_path), enhanced_16k, 16_000)

        result = enhanced_16k.numpy()
        return result.squeeze(0) if result.shape[0] == 1 else result