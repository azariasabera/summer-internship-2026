"""Noise processing, denoising, and noise augmentation."""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf


def denoise(cfg: DictConfig) -> int:
    """Run the configured audio denoising pipeline.

    Parameters
    ----------
    cfg:
        Hydra configuration for the current run.

    Returns
    -------
    int
        Process exit status.
    """
    print("[tea] Running audio denoising")
    print()
    print("Noise configuration:")
    print(OmegaConf.to_yaml(cfg.noise))

    return 0


def extract_noise(cfg: DictConfig) -> int:
    """Extract noise segments from the benchmark data.

    Parameters
    ----------
    cfg:
        Hydra configuration for the current run.

    Returns
    -------
    int
        Process exit status.
    """
    print("[tea] Running noise extraction")
    print()
    print("Noise configuration:")
    print(OmegaConf.to_yaml(cfg.noise))

    return 0
