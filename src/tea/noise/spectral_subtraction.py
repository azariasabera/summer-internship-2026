# src/tea/noise/spectral_subtraction.py

"""Spectral subtraction denoising."""

from __future__ import annotations

import numpy as np
import pandas as pd
import librosa

from tea.utils.paths import resolve

AudioArray = np.ndarray

class SpectralSubtractor:
    """Basic spectral-subtraction denoiser using a non-speech noise reference.

    Estimates a noise magnitude spectrum (median across frames, robust to
    accidental speech in the noise reference) and subtracts it from each
    frame's magnitude spectrum, with a spectral floor to avoid musical-noise
    artifacts, then reconstructs with the original phase and overlap-adds.

    Parameters
    ----------
    n_fft:
        FFT size.
    hop:
        Hop size between frames.
    alpha:
        Over-subtraction factor.
    beta:
        Spectral floor, as a fraction of the estimated noise magnitude.
    """

    def __init__(self, n_fft: int = 1024, hop: int = 256, alpha: float = 1.0, beta: float = 0.05) -> None:
        self.n_fft = n_fft
        self.hop = hop
        self.alpha = alpha
        self.beta = beta

    def _frame_signal(self, x: AudioArray) -> AudioArray:
        n_fft, hop = self.n_fft, self.hop
        window = np.hanning(n_fft)
        if len(x) < n_fft:
            x = np.pad(x, (0, n_fft - len(x)))
        n_frames = 1 + (len(x) - n_fft) // hop
        frames = np.zeros((n_frames, n_fft))
        for i in range(n_frames):
            start = i * hop
            frames[i] = x[start : start + n_fft] * window
        return frames

    def subtract(self, speech: AudioArray, noise: AudioArray) -> AudioArray:
        """Subtract `noise`'s estimated spectrum from `speech`.

        Parameters
        ----------
        speech:
            Noisy signal to clean.
        noise:
            Reference noise-only signal (e.g. a non-speech chunk from the same recording).
        """
        n_fft, hop = self.n_fft, self.hop
        window = np.hanning(n_fft)

        y_frames = self._frame_signal(speech)
        noise_frames = self._frame_signal(noise)

        Y = np.fft.rfft(y_frames, axis=1)
        N = np.fft.rfft(noise_frames, axis=1)

        Y_mag, Y_phase = np.abs(Y), np.angle(Y)
        noise_mag = np.median(np.abs(N), axis=0)

        clean_mag = Y_mag - self.alpha * noise_mag
        clean_mag = np.maximum(clean_mag, self.beta * noise_mag)

        clean_complex = clean_mag * np.exp(1j * Y_phase)
        clean_frames = np.fft.irfft(clean_complex, n=n_fft, axis=1)

        output_length = (len(clean_frames) - 1) * hop + n_fft
        output = np.zeros(output_length)
        window_sum = np.zeros(output_length)

        for i, frame in enumerate(clean_frames):
            start, end = i * hop, i * hop + n_fft
            output[start:end] += frame * window
            window_sum[start:end] += window**2

        valid = window_sum > 1e-8
        output[valid] /= window_sum[valid]

        return output[: len(speech)]

    
def _audio_power(x: AudioArray) -> float:
    """Return mean-square audio power.

    Parameters
    ----------
    x:
        Audio array.
    """
    return float(np.mean(x**2))


def _select_low_energy_chunks(noise_candidates: list[dict], n_chunks: int) -> list[dict]:
    """Select the lowest-power candidate noise chunks.
    
    Parameters
    ----------
    noise_candidates:
        List of dictionaries about noise candidates including: name, start, end and power.
    n_chunks:
        Number of lowest-energy noise chunks to use.
    """
    if n_chunks <= 0:
        raise ValueError(f"n_chunks must be > 0, got {n_chunks}")

    candidates = sorted(noise_candidates, key=lambda item: item["power"])

    return candidates[:n_chunks]


def build_noise_reference(df: pd.DataFrame, n_chunks: int = 5) -> dict | None:
    """Build a noise reference for one video. Should be used for each recording separately.

    Loads the video's source WAV once, extracts all chunks whose
    `gt_label` is NaN, ranks them by mean-square power, selects
    the lowest-energy chunks, and concatenates them.

    Parameters
    ----------
    df:
        Annotation rows for a single video. Must contain:
        `name`, `start`, `end`, `gt_label`, and `audio_path`.
        
        Needs running `src.tea.utils.io.load_annotation_csvs` with 
        `add_audio_path` set to True. Then grouping by for e.g. audio_path 
        to get separate df for each recordings.

    n_chunks:
        Number of lowest-energy noise chunks to use.

    Returns
    -------
    dict | None
        `None` if no usable noise chunks exist.

        Otherwise:

        {
            "noise_signal": np.ndarray,
            "sr": int,
            "selected": list[dict],
            "n_candidates": int,
        }
    """
    if df.empty:
        return None

    if n_chunks <= 0:
        raise ValueError(f"n_chunks must be > 0, got {n_chunks}")

    # Get source audio path
    audio_paths = df["audio_path"].dropna().unique()

    if len(audio_paths) != 1:
        raise ValueError(f"Expected exactly one audio_path for a video, got: {audio_paths}")

    audio_path = resolve(audio_paths[0])

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load the recording WAV ONCE. I preserve the sample rate because CSV start/end are sample indices.
    waveform, sr = librosa.load(audio_path, sr=None, mono=True)

    # Find NaN gt_label chunks and calculate power
    noise_candidates = []

    for _, row in df.iterrows():

        if not pd.isna(row["gt_label"]):
            continue

        start = int(row["start"])
        end = int(row["end"])

        if start < 0 or end <= start:
            raise ValueError(f"{row['name']}: invalid segment start={start}, end={end}")

        if end > len(waveform):
            raise ValueError(f"{row['name']}: end={end} exceeds audio length={len(waveform)}")

        # Extract chunk directly from the already-loaded waveform.
        chunk = waveform[start:end]

        power = _audio_power(x=chunk)

        noise_candidates.append(
            {
                "name": row["name"],
                "start": start,
                "end": end,
                "power": power,
            }
        )

    # No usable noise chunks
    if not noise_candidates:
        return None

    # Select lowest-energy chunks
    selected = _select_low_energy_chunks(noise_candidates=noise_candidates, n_chunks=n_chunks)
    selected_audio = [waveform[item["start"]:item["end"]] for item in selected]

    # Concatenate selected noise chunks
    noise_signal = np.concatenate(selected_audio)

    return {
        "noise_signal": noise_signal,
        "sr": sr,
        "selected": selected,
        "n_candidates": len(noise_candidates),
    }