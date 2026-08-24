# src/tea/noise/spectral_subtraction.py

"""Spectral subtraction denoising."""

from __future__ import annotations

import numpy as np

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