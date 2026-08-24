# src/tea/vad/__init__.py

"""
Voice Activity Detection and speech-segment processing.

Only the `chunk` command is exposed in this file in order to keep the CLI area small and consistent (all cli commands take cfg).
The other functions in tea.vad, such as `enrich_fixed_segments_from_vad`, are intended to be used as library functions.
"""

from __future__ import annotations

from omegaconf import DictConfig

from tea.utils.io import vad_json_to_annotation_csv
from tea.utils.logging import get_logger
from tea.utils.paths import ensure_dir, resolve
from tea.vad.segmenter import Segmenter

logger = get_logger(__name__)


def chunk(cfg: DictConfig) -> int:
    """Run VAD-based segmentation over the configured audio root.

    Parameters
    ----------
    cfg:
        Resolved Hydra configuration.

    Returns
    -------
    int
        Process exit status.
    """
    audio_root = resolve(cfg.paths.audio_root)
    out_dir = ensure_dir(resolve(cfg.paths.chunk_meta_dir))

    logger.info("Running VAD chunking on %s -> %s", audio_root, out_dir)

    segmenter = Segmenter(cfg)
    results = segmenter.chunk_vad(
        audio_root,
        save=True,
        save_dir=out_dir,
    )

    logger.info("Chunked %d file(s).", len(results))

    # Mirror each JSON as an annotation CSV
    n_csv = 0
    for json_path in sorted(resolve(out_dir).glob("*.json")):
        csv_path = resolve(cfg.paths.annotation_root) / f"{json_path.stem}.csv"
        vad_json_to_annotation_csv(json_path, csv_path)
        n_csv += 1
    logger.info("Wrote %d annotation CSV(s) -> %s", n_csv, cfg.paths.annotation_root)
    return 0


# from __future__ import annotations

# from omegaconf import DictConfig, OmegaConf


# def chunk(cfg: DictConfig) -> int:
#     """Run the VAD chunking pipeline.

#     Parameters
#     ----------
#     cfg:
#         Hydra configuration for the current run.

#     Returns
#     -------
#     int
#         Process exit status.
#     """
#     print("[tea] Running VAD chunking")
#     print()
#     print("VAD configuration:")
#     print(OmegaConf.to_yaml(cfg.vad, resolve=True))

#     return 0
