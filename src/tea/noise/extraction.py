# src/tea/noise/extraction.py

"""Noise-segment extraction from classroom recordings"""

from __future__ import annotations

from omegaconf import DictConfig
from pathlib import Path
from typing import Union
import pandas as pd

from tea.utils.logging import get_logger
from tea.utils.paths import ensure_dir, resolve

logger = get_logger(__name__)


def extract_noise_pool(
    annotation_root: Union[str, Path], video_ids: list[str], label_col: str = "gt_label"
) -> list[str]:
    """Collect audio paths of non-speech (unlabeled) chunks across videos.

    A chunk counts as "noise" if `label_col` is NaN for it. This works for
    every video, including ones normally excluded from speech train/test
    (e.g. `1B3261`), since only the audio is needed here, not an emotion label.

    Parameters
    ----------
    annotation_root:
        Directory of per-video annotation CSVs.
    video_ids:
        Videos to pull non-speech chunks from.
    label_col:
        Column whose NaN rows mark non-speech/unannotated chunks.
    """
    paths: list[str] = []
    root = Path(annotation_root)

    for video_id in sorted(set(video_ids)):
        csv_path = root / f"{video_id}.csv"
        if not csv_path.exists():
            logger.warning("%s: no annotation csv found, skipping for noise pool", video_id)
            continue

        df = pd.read_csv(csv_path)
        noise_rows = df.loc[df[label_col].isna()]
        if len(noise_rows) == 0:
            continue
        paths.extend(noise_rows["audio_path"].tolist())

    if not paths:
        raise ValueError(
            "Extracted noise pool is empty. Check that the CSVs contain "
            "NaN-labeled (non-speech) rows for the given video_ids."
        )
    return paths

def extract_noise(cfg: DictConfig) -> int:
    """`tea extract-noise`: collect non-speech chunk metadata.

    Writes the resulting noise chunk metadata to `cfg.noise.extraction.save_dir/noise_pool.json`. 
    If `cfg.noise.extraction.save_audio` is enabled, also extracts and saves full noise extract WAV file.
    """
    import json

    import librosa
    import soundfile as sf
    import numpy as np

    from tea.utils.io import load_annotation_csvs

    df = load_annotation_csvs(
        annotation_root=cfg.noise.csv_annotations,
        exclude=None,
        add_audio_path=True,
        json_dir=cfg.noise.json_annotations,
    )

    noise_rows = df.loc[df["gt_label"].isna()]

    if len(noise_rows) == 0:
        raise ValueError("Extracted noise pool is empty. Check that the annotations contain NaN-labeled rows.")

    pool = noise_rows[["audio_path", "start", "end"]].to_dict(orient="records")

    out_dir = ensure_dir(resolve(cfg.noise.extraction.save_dir))
    out_path = out_dir / "noise_pool.json"

    save_audio = cfg.noise.extraction.get("save_audio", False)
    sample_rate = int(cfg.noise.extraction.get("sample_rate", 16_000))

    if save_audio:
        noise_path = out_dir / "full_noise.wav"

        rng = np.random.default_rng(int(cfg.get("seed", 42)))
        shuffled_pool = pool.copy()
        rng.shuffle(shuffled_pool)

        with sf.SoundFile(noise_path, mode="w", samplerate=sample_rate, channels=1, subtype="PCM_16") as f:
            for item in shuffled_pool:
                audio_path = item["audio_path"]
                start = int(item["start"])
                end = int(item["end"])

                waveform, _ = librosa.load(
                    audio_path,
                    sr=sample_rate,
                    mono=True,
                    offset=start / sample_rate,
                    duration=(end - start) / sample_rate,
                )

                f.write(waveform)

        """noise_chunks = []

        for item in pool:
            audio_path = item["audio_path"]
            start = int(item["start"])
            end = int(item["end"])

            waveform, _ = librosa.load(audio_path, sr=sample_rate)
            noise_chunks.append(waveform[start:end])

        rng = np.random.default_rng(int(cfg.get("seed", 42)))
        rng.shuffle(noise_chunks)   
        noise_audio = np.concatenate(noise_chunks)
        noise_path = out_dir / "full_noise.wav"
        sf.write(noise_path, noise_audio, sample_rate)"""

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pool, f, indent=2)

    logger.info("Extracted %d noise chunks -> %s", len(pool), out_path)

    if save_audio:
        logger.info("Saved extracted noise audio to %s", noise_path)

    return 0