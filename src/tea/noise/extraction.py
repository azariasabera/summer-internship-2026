# src/tea/noise/extraction.py

"""Noise-segment extraction from classroom recordings"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tea.utils.logging import get_logger

from typing import Union

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