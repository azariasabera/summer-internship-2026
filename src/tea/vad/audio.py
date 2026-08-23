# src/tea/vad/audio.py

"""Utilities for writing audio segments to files."""

from __future__ import annotations

from pathlib import Path

import librosa
import soundfile as sf

from tea.utils.paths import ensure_dir

from typing import Union

def save_audio_chunks(segments: list[dict], save_pth: Union[str, Path], sr: int = 16_000) -> None:
    """Write each segment's audio to `<save_pth>/<stem>/chunk_<type>_<i>.wav`.

    Parameters
    ----------
    segments:
        Output of `Segmenter.chunk_vad` / `chunk_fixed`.
    save_pth:
        Root directory to write chunk folders into.
    sr:
        Sample rate used for loading/writing.
    """
    for segment in segments:
        audio_path = segment["audio_path"]
        stem = Path(audio_path).stem

        y, _ = librosa.load(audio_path, sr=sr)
        out_dir = ensure_dir(Path(save_pth) / stem)

        for i, seg in enumerate(segment["segments"]):
            audio_chunk = y[seg["start"] : seg["end"]]
            typ_short = "n" if seg["type"] == "non-speech" else "s"
            fname = f"chunk_{typ_short}_{i}.wav" if seg["type"] is not None else f"chunk_{i}.wav"
            sf.write(out_dir / fname, audio_chunk, sr)