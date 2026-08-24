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
    """`tea extract-noise` -- extract non-speech chunk paths from annotated videos.

    Writes the resulting path list to `generated/noise_pool.json`.

    Parameters
    ----------
    cfg:
        Resolved Hydra config.
    """
    import json

    from tea.utils.io import load_annotation_csvs

    df = load_annotation_csvs(cfg.paths.annotation_root)
    video_ids = sorted(df["video"].unique())
    pool = extract_noise_pool(cfg.paths.annotation_root, video_ids)

    out_path = ensure_dir(resolve(cfg.paths.get("generated_root", "generated"))) / "noise_pool.json"
    with open(out_path, "w") as f:
        json.dump(pool, f, indent=2)

    logger.info("Extracted %d noise chunks -> %s", len(pool), out_path)
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
