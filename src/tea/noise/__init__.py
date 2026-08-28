"""Noise processing, denoising, and noise augmentation."""

from __future__ import annotations

from omegaconf import DictConfig

from tea.noise.denoiser import Denoiser
from tea.noise.extraction import extract_noise_pool
from tea.utils.logging import get_logger
from tea.utils.paths import ensure_dir, resolve

logger = get_logger(__name__)


def denoise(cfg: DictConfig) -> int:
    """`tea denoise` -- run the configured denoising method over `cfg.paths.audio_root`.

    Parameters
    ----------
    cfg:
        Resolved Hydra config. `cfg.noise.method` selects `"deepfilter"`
        (default) or `"spectral"`.
    """
    method = cfg.noise.get("method", "deepfilter")
    audio_root = resolve(cfg.paths.audio_root)
    out_dir = ensure_dir(resolve(cfg.paths.get("generated_root", "generated")) / "denoised" / method)

    wav_files = sorted(audio_root.glob("*.wav")) if audio_root.is_dir() else [audio_root]
    logger.info("Denoising %d file(s) with method=%s -> %s", len(wav_files), method, out_dir)

    if method == "deepfilter":
        denoiser = Denoiser(cfg)
        for wav_path in wav_files:
            denoiser.enhance(
                wav_path, output_path=out_dir / wav_path.name, save=True, atten_lim_db=cfg.noise.atten_lim_db
            )
    elif method == "spectral":
        logger.info(
            "Spectral subtraction needs a per-file noise reference (non-speech chunk); "
            "wire this up once tea.vad chunk metadata + tea.analysis.noise are both in place."
        )
    else:
        logger.error("Unknown noise.method=%s", method)
        return 2

    return 0


def extract_noise(cfg: DictConfig) -> int:
    """`tea extract-noise`: collect non-speech chunk metadata.

    Writes the resulting noise chunk metadata to `paths.noise.noise_extract_save_dir/noise_pool.json`.

    If `noise.extraction.save_audio` is enabled, also extracts and saves full noise extract WAV file.
    """
    import json

    import librosa
    import soundfile as sf
    import numpy as np

    from tea.utils.io import load_annotation_csvs

    df = load_annotation_csvs(
        annotation_root=cfg.paths.annotation_root,
        exclude=None,
        add_audio_path=True,
        json_dir=cfg.paths.chunk_meta_dir,
    )

    noise_rows = df.loc[df["gt_label"].isna()]

    if len(noise_rows) == 0:
        raise ValueError("Extracted noise pool is empty. Check that the annotations contain NaN-labeled rows.")

    pool = noise_rows[["audio_path", "start", "end"]].to_dict(orient="records")

    out_dir = ensure_dir(resolve(cfg.paths.noise.noise_extract_save_dir))
    out_path = out_dir / "noise_pool.json"

    save_audio = cfg.noise.extraction.get("save_audio", False)
    sample_rate = int(cfg.noise.extraction.get("sample_rate", 16_000))

    if save_audio:
        audio_dir = ensure_dir(resolve(cfg.paths.noise.extraction.save_audio_dir))
        noise_chunks = []

        for item in pool:
            audio_path = item["audio_path"]
            start = int(item["start"])
            end = int(item["end"])

            waveform, _ = librosa.load(audio_path, sr=sample_rate)
            noise_chunks.append(waveform[start:end])

        rng = np.random.default_rng(int(cfg.get("seed", 42)))
        rng.shuffle(noise_chunks)   
        noise_audio = np.concatenate(noise_chunks)
        noise_path = audio_dir / "full_noise.wav"
        sf.write(noise_path, noise_audio, sample_rate)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pool, f, indent=2)

    logger.info("Extracted %d noise chunks -> %s", len(pool), out_path)

    if save_audio:
        logger.info("Saved extracted noise audio to %s", noise_path)

    return 0

# def denoise(cfg: DictConfig) -> int:
#     """Run the configured audio denoising pipeline.

#     Parameters
#     ----------
#     cfg:
#         Hydra configuration for the current run.

#     Returns
#     -------
#     int
#         Process exit status.
#     """
#     print("[tea] Running audio denoising")
#     print()
#     print("Noise configuration:")
#     print(OmegaConf.to_yaml(cfg.noise, resolve=True))

#     return 0


# def extract_noise(cfg: DictConfig) -> int:
#     """Extract noise segments from the benchmark data.

#     Parameters
#     ----------
#     cfg:
#         Hydra configuration for the current run.

#     Returns
#     -------
#     int
#         Process exit status.
#     """
#     print("[tea] Running noise extraction")
#     print()
#     print("Noise configuration:")
#     print(OmegaConf.to_yaml(cfg.noise, resolve=True))

#     return 0
