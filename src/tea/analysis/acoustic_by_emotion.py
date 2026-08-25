# src/tea/analysis/acoustic_by_emotion.py

"""Duration / loudness (RMS dB) / speaking-rate distributions grouped by
predicted emotion (report Figure 8 -> the "anger predictions are
associated with shorter, louder, faster chunks" finding).

Note: Loudness computation uses `librosa` directly on the raw audio (not `tea.features.acoustic`'s
RMS, which is linear-scale RMS for the confidence-estimation feature
table) -- kept as its own dB-scale function since that's what the report's
loudness figures use, and the two aren't interchangeable without a
`20*log10` conversion the report's plots already bake in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from tea.features.acoustic import clean_transcription
from tea.utils.logging import get_logger

logger = get_logger(__name__)

SAMPLE_RATE = 16_000


def compute_loudness_db(audio_path: str | Path, start_sample: int, end_sample: int, sr: int = SAMPLE_RATE) -> float:
    """Mean RMS loudness in dB (relative to the chunk's own peak) for one time range of one audio file.

    Returns NaN on any read failure rather than raising, so a batch run
    over many chunks doesn't die on one bad file.

    Parameters
    ----------
    audio_path:
        Path to the source (unchunked) audio file.
    start_sample, end_sample:
        Sample-index range to read.
    sr:
        Sample rate.
    """
    import librosa

    try:
        y, _ = librosa.load(audio_path, sr=sr, offset=start_sample / sr, duration=(end_sample - start_sample) / sr)
        if len(y) == 0:
            return np.nan
        rms = librosa.feature.rms(y=y)[0]
        loudness_db = librosa.power_to_db(rms**2, ref=np.max)
        return float(np.mean(loudness_db))
    except Exception as e:
        logger.warning("Loudness computation failed for %s [%d:%d]: %s", audio_path, start_sample, end_sample, e)
        return np.nan


def add_loudness(df: pd.DataFrame, audio_root: str | Path, video_col: str = "video", start_col: str = "start", end_col: str = "end") -> pd.DataFrame:
    """Add a `loudness` column, resolving each row's source audio file as `<audio_root>/<video>.<ext>`.

    Parameters
    ----------
    df:
        Must contain `video_col`, `start_col`, `end_col`.
    audio_root:
        Directory of full (unchunked) per-video audio files.
    """
    df = df.copy()
    audio_root = Path(audio_root)

    resolved_paths = {}
    for video in df[video_col].unique():
        for ext in (".wav", ".WAV", ".mp3", ".flac"):
            candidate = audio_root / f"{video}{ext}"
            if candidate.exists():
                resolved_paths[video] = str(candidate)
                break
        else:
            logger.warning("No audio found for %s under %s", video, audio_root)
            resolved_paths[video] = None

    df["loudness"] = df.apply(
        lambda r: compute_loudness_db(resolved_paths[r[video_col]], r[start_col], r[end_col]) if resolved_paths[r[video_col]] else np.nan,
        axis=1,
    )
    return df


def add_speaking_rate(pred_df: pd.DataFrame, annotation_root: str | Path, sample_rate: int = SAMPLE_RATE) -> pd.DataFrame:
    """Match each predicted chunk to its overlapping annotation-CSV speech segment and attach speaking rate (words/min).

    Word count is computed on the (repeated-phrase-cleaned) transcription;
    speaking rate below 0 or above 400 wpm is dropped as an ASR artifact,
    matching the original's sanity filter.

    Parameters
    ----------
    pred_df:
        Must contain `video`, `start`, `end`, `pred_label` (already joined via `tea.analysis.classroom.join_predictions`).
    annotation_root:
        Directory of per-video annotation CSVs with `type`, `transcription`, `duration_sec`, `start`, `end` columns.
    """
    annotation_dfs = []
    for video in pred_df["video"].unique():
        csv_path = Path(annotation_root) / f"{video}.csv"
        if not csv_path.exists():
            continue
        ann = pd.read_csv(csv_path)
        ann = ann.loc[ann["type"] == "speech"].copy()
        ann["clean_text"] = ann["transcription"].apply(clean_transcription)
        ann["word_count"] = ann["clean_text"].str.split().str.len().fillna(0).astype(int)
        ann["speaking_rate_wpm"] = 60 * ann["word_count"] / ann["duration_sec"]
        ann["video"] = video
        annotation_dfs.append(ann)

    if not annotation_dfs:
        raise ValueError("No matching annotation CSVs found for speaking-rate matching.")
    annotations = pd.concat(annotation_dfs, ignore_index=True)

    merged_rows = []
    for _, r in pred_df.iterrows():
        candidates = annotations.loc[annotations["video"] == r["video"]]
        if len(candidates) == 0:
            continue
        overlap = np.minimum(candidates["end"], r["end"]) - np.maximum(candidates["start"], r["start"])
        best = overlap.idxmax()
        if overlap.loc[best] <= 0:
            continue
        row = r.to_dict()
        row["speaking_rate_wpm"] = annotations.loc[best, "speaking_rate_wpm"]
        row["word_count"] = annotations.loc[best, "word_count"]
        merged_rows.append(row)

    out = pd.DataFrame(merged_rows)
    out = out.loc[(out["speaking_rate_wpm"] > 0) & (out["speaking_rate_wpm"] < 400)].reset_index(drop=True)
    return out


def summarize_by_predicted_emotion(df: pd.DataFrame, value_col: str, pred_col: str = "pred_label") -> pd.DataFrame:
    """Mean/median/count of `value_col`, grouped by `pred_col`. Feed this (or the raw `df`) directly into a seaborn boxplot."""
    return df.groupby(pred_col)[value_col].agg(mean="mean", median="median", count="count").sort_values("mean", ascending=False)


def acoustic_by_emotion_cli(cfg: DictConfig) -> int:
    """`tea acoustic-by-emotion` => duration/loudness/speaking-rate distributions by predicted emotion (report Figure 8).

    Requires `analysis.mtkd_json`.
    """
    from tea.analysis.classroom import join_predictions

    ac = cfg.analysis
    if not ac.get("mtkd_json"):
        logger.error("Set analysis.mtkd_json")
        return 2

    df = join_predictions(ac.mtkd_json, cfg.paths.annotation_root, excluded_videos=set(cfg.classroom.excluded_videos))
    df["duration"] = (df["end"] - df["start"]) / SAMPLE_RATE

    logger.info("=== Duration by predicted emotion ===")
    logger.info("\n%s", summarize_by_predicted_emotion(df, "duration"))

    if ac.get("audio_root"):
        df = add_loudness(df, ac.audio_root)
        logger.info("=== Loudness (dB) by predicted emotion ===")
        logger.info("\n%s", summarize_by_predicted_emotion(df.dropna(subset=["loudness"]), "loudness"))
    else:
        logger.info("analysis.audio_root not set -- skipping loudness (needs full per-video audio, not chunks).")

    rate_df = add_speaking_rate(df, cfg.paths.annotation_root)
    logger.info("=== Speaking rate (wpm) by predicted emotion ===")
    logger.info("\n%s", summarize_by_predicted_emotion(rate_df, "speaking_rate_wpm"))

    if ac.get("output_dir"):
        from tea.utils.paths import ensure_dir, resolve

        out_dir = ensure_dir(resolve(ac.output_dir))
        df.to_csv(out_dir / "duration_loudness_by_chunk.csv", index=False)
        rate_df.to_csv(out_dir / "speaking_rate_by_chunk.csv", index=False)
        logger.info("Saved -> %s", out_dir)

    return 0
