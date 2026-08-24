# src/tea/noise/rir.py

"""Room impulse response (RIR) augmentation.

Applies simulated RIRs (OpenSLR-28) to speech BEFORE noise contamination,
so augmented utterances get far-field reverb characteristics ahead of
classroom background noise.
"""

from __future__ import annotations

import glob
import os
import random
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import fftconvolve

from tea.utils.constants import SAMPLE_RATE

EPS = 1e-8


class RIRAugmentor:
    """Applies simulated room-impulse-response reverb to speech.

    Parameters
    ----------
    rirs_noises_root:
        Root directory containing an OpenSLR-28-style `simulated_rirs/` subtree.
    room_sizes:
        Which `simulated_rirs/<size>/` subfolders to pool from.
    """

    def __init__(self, rirs_noises_root: str | Path, room_sizes: tuple[str, ...] = ("smallroom", "mediumroom")) -> None:
        self.rirs_noises_root = Path(rirs_noises_root)
        self.pool_by_size = {size: self._build_pool(size) for size in room_sizes}

    def _build_pool(self, size: str) -> list[str]:
        pool = glob.glob(os.path.join(self.rirs_noises_root, "simulated_rirs", size, "**", "*.wav"), recursive=True)
        if not pool:
            raise ValueError(f"No RIR wavs found under {self.rirs_noises_root}/simulated_rirs/{size}")
        return pool

    @staticmethod
    def _load_rir(path: str) -> np.ndarray:
        rir, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        peak_idx = int(np.argmax(np.abs(rir)))
        rir = rir[peak_idx:]  # trim to main impulse onward
        rir = rir / (np.linalg.norm(rir) + EPS)  # unit-energy normalize
        return rir.astype(np.float64)

    def apply(self, speech: np.ndarray, rir_path: str, dry_wet: float = 0.7) -> np.ndarray:
        """Convolve `speech` with the RIR at `rir_path` and mix dry/wet.

        Parameters
        ----------
        speech:
            Input waveform.
        rir_path:
            Path to one RIR `.wav` (e.g. from `sample_rir_path`).
        dry_wet:
            0 = untouched, 1 = fully reverberant. 0.6-0.8 keeps a direct-path
            component, avoiding an unrealistically "swimmy" result.
        """
        rir = self._load_rir(rir_path)
        speech64 = speech.astype(np.float64)

        wet = fftconvolve(speech64, rir, mode="full")[: len(speech64)]

        # RMS renormalize wet signal to match dry. Convolution changes energy;
        # without this, reverbed samples end up systematically quieter than
        # clean ones, and the model can shortcut on loudness instead of
        # learning the acoustic content.
        orig_rms = np.sqrt(np.mean(speech64**2) + EPS)
        wet_rms = np.sqrt(np.mean(wet**2) + EPS)
        wet = wet * (orig_rms / wet_rms)

        out = dry_wet * wet + (1 - dry_wet) * speech64
        peak = np.max(np.abs(out))
        if peak > 1.0:
            out = out / peak
        return out.astype(np.float32)

    def sample_path(self, size_weights: tuple[float, ...] = (0.7, 0.3)) -> str:
        """Randomly pick one RIR path, weighting room sizes by `size_weights` (same order as `room_sizes` at construction)."""
        sizes = list(self.pool_by_size.keys())
        chosen_size = random.choices(sizes, weights=size_weights, k=1)[0]
        return random.choice(self.pool_by_size[chosen_size])

    def augment(self, speech: np.ndarray, dry_wet_range: tuple[float, float] = (0.35, 0.75), size_weights: tuple[float, ...] = (0.7, 0.3)) -> np.ndarray:
        """Convenience: sample a random RIR + dry/wet ratio and apply in one call."""
        rir_path = self.sample_path(size_weights)
        dry_wet = random.uniform(*dry_wet_range)
        return self.apply(speech, rir_path, dry_wet=dry_wet)
