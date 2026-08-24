# src/tea/noise/augmentation.py

"""Training-time additive noise augmentation for the benchmark datasets."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torchaudio

from tea.utils.logging import get_logger

from typing import Tuple, Union

logger = get_logger(__name__)


class NoiseAugmentor:
    """Mixes a fixed noise track into training waveforms at a random SNR.

    This is the "retrain 5-15/10-25/15-30dB" mechanism from the noise
    analysis (report Table 21).
    
    Parameters
    ----------
    noise_path:
        Path to a single noise `.wav` (mixed in on a rolling/tiled basis).
    contam_prob:
        Probability that a given sample is contaminated at all.
    snr_min, snr_max:
        Range to sample the mixing SNR (dB) from, uniformly, per contaminated sample.
    sample_rate:
        Target sample rate; the noise track is resampled to this if needed.
    seed:
        RNG seed for this augmentor's own `random.Random` instance (independent
        of the global seed, matching the original design).
    """

    def __init__(
        self,
        noise_path: Union[str, Path],
        contam_prob: float = 0.5,
        snr_min: float = 5.0,
        snr_max: float = 20.0,
        sample_rate: int = 16_000,
        seed: int = 42,
    ) -> None:
        self.contam_prob = contam_prob
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.sample_rate = sample_rate

        self.rng = random.Random(seed)
        self.total_samples = 0
        self.contaminated_samples = 0
        self.snr_history: list[float] = []

        noise_waveform, noise_sr = torchaudio.load(str(noise_path))
        if noise_sr != sample_rate:
            noise_waveform = torchaudio.functional.resample(noise_waveform, noise_sr, sample_rate)
        if noise_waveform.shape[0] > 1:
            noise_waveform = noise_waveform.mean(dim=0, keepdim=True)

        self.noise_waveform = noise_waveform
        self.noise_len = noise_waveform.shape[1]
        logger.info("Loaded noise: %s | duration: %.2fs", Path(noise_path).name, self.noise_len / sample_rate)

    def _mix_noise(self, clean: torch.Tensor, noise_segment: torch.Tensor, snr_db: float) -> torch.Tensor:
        eps = 1e-8
        clean_rms = torch.sqrt(torch.mean(clean**2) + eps)
        noise_rms = torch.sqrt(torch.mean(noise_segment**2) + eps)
        if noise_rms < 1e-8:
            return clean

        snr_linear = 10 ** (snr_db / 20)
        scale = clean_rms / (noise_rms * snr_linear)
        noise_scaled = noise_segment * scale

        if noise_scaled.shape[1] < clean.shape[1]:
            repeats = (clean.shape[1] // noise_scaled.shape[1]) + 1
            noise_scaled = noise_scaled.repeat(1, repeats)[:, : clean.shape[1]]
        else:
            noise_scaled = noise_scaled[:, : clean.shape[1]]

        return clean + noise_scaled

    def augment(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, bool, float]:
        """Possibly mix noise into `waveform`.

        Returns
        -------
        tuple
            `(augmented_waveform, was_contaminated, snr_used_db)`.
        """
        self.total_samples += 1
        if self.rng.random() > self.contam_prob:
            return waveform, False, 0.0

        self.contaminated_samples += 1
        snr = self.rng.uniform(self.snr_min, self.snr_max)
        self.snr_history.append(snr)

        length = waveform.shape[1]
        if length <= self.noise_len:
            start = self.rng.randint(0, self.noise_len - length)
            noise_seg = self.noise_waveform[:, start : start + length]
        else:
            repeats = int(np.ceil(length / self.noise_len))
            noise_seg = self.noise_waveform.repeat(1, repeats)[:, :length]

        mixed = self._mix_noise(waveform, noise_seg, snr)
        peak = mixed.abs().max()
        if peak > 1.0:
            mixed = mixed / peak

        return mixed, True, snr

    def summary(self) -> dict:
        """Return contamination-rate / SNR summary stats for this run."""
        avg_snr = float(np.mean(self.snr_history)) if self.snr_history else None
        contam_pct = 100 * self.contaminated_samples / self.total_samples if self.total_samples else 0.0
        return {
            "total_samples": self.total_samples,
            "contaminated_samples": self.contaminated_samples,
            "contamination_pct": contam_pct,
            "avg_snr_db": avg_snr,
            "snr_min_observed": min(self.snr_history) if self.snr_history else None,
            "snr_max_observed": max(self.snr_history) if self.snr_history else None,
        }

    def reset_stats(self) -> None:
        """Reset running contamination/SNR counters (not the RNG)."""
        self.total_samples = 0
        self.contaminated_samples = 0
        self.snr_history = []