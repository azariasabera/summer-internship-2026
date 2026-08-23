"""Voice Activity Detection and speech-segment refinement."""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf


def chunk(cfg: DictConfig) -> int:
    """Run the VAD chunking pipeline.

    Parameters
    ----------
    cfg:
        Hydra configuration for the current run.

    Returns
    -------
    int
        Process exit status.
    """
    print("[tea] Running VAD chunking")
    print()
    print("VAD configuration:")
    print(OmegaConf.to_yaml(cfg.vad, resolve=True))

    return 0
