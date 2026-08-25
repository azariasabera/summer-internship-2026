# src/tea/features/acoustic.py

"""Per-chunk acoustic features (RMS, F0, voiced ratio) and a text-based
speech-rate proxy."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

EPS = 1e-8


def extract_acoustic_features(audio_path: str, sr: int = 16_000) -> dict:
    """Returns `rms_mean`, `rms_std`, `f0_mean`, `f0_std`, `voiced_ratio` for one chunk.

    Uses `librosa.pyin` for F0 (voiced-frame masked -- unvoiced frames
    never enter the mean/std). Returns a safe all-NaN dict (flagged, not
    silently zeroed) if the file is missing/unreadable, so bad audio paths
    surface as NaNs to catch during the merge rather than fake "confident
    zero pitch" rows.

    Parameters
    ----------
    audio_path:
        Path to the chunk's audio file.
    sr:
        Target sample rate.
    """
    import librosa

    nan_result = dict(rms_mean=np.nan, rms_std=np.nan, f0_mean=np.nan, f0_std=np.nan, voiced_ratio=np.nan)

    try:
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
    except Exception as e:
        warnings.warn(f"Could not load {audio_path}: {e}")
        return nan_result

    if y.size == 0:
        return nan_result

    rms = librosa.feature.rms(y=y)[0]

    try:
        f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"), sr=sr)
    except Exception:
        f0, voiced_flag = None, None

    if f0 is None or voiced_flag is None:
        f0_mean, f0_std, voiced_ratio = np.nan, np.nan, np.nan
    else:
        voiced_mask = np.asarray(voiced_flag, dtype=bool)
        voiced_f0 = np.asarray(f0)[voiced_mask]
        voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]
        voiced_ratio = float(np.mean(voiced_mask)) if voiced_mask.size else 0.0
        if voiced_f0.size:
            f0_mean, f0_std = float(np.mean(voiced_f0)), float(np.std(voiced_f0))
        else:
            f0_mean, f0_std = np.nan, np.nan

    return dict(rms_mean=float(np.mean(rms)), rms_std=float(np.std(rms)), f0_mean=f0_mean, f0_std=f0_std, voiced_ratio=voiced_ratio)


def batch_extract_acoustic_features(df: pd.DataFrame, audio_path_col: str = "audio_path", sr: int = 16_000, log_every: int = 100) -> pd.DataFrame:
    """Apply `extract_acoustic_features` over every row of `df`.

    Real disk I/O + DSP per chunk -- fine single-threaded for a few
    thousand short clips; parallelize (e.g. joblib) if this is slow on
    the full annotated set.

    Parameters
    ----------
    df:
        Must contain `audio_path_col`.
    audio_path_col:
        Column holding each chunk's audio path.
    sr:
        Target sample rate.
    log_every:
        Print progress every N chunks (0 disables).
    """
    from tea.utils.logging import get_logger

    logger = get_logger(__name__)
    records = []
    for i, path in enumerate(df[audio_path_col]):
        records.append(extract_acoustic_features(path, sr=sr))
        if log_every and (i + 1) % log_every == 0:
            logger.info("Acoustic features: %d / %d", i + 1, len(df))

    feat_df = pd.DataFrame(records, index=df.index)
    n_bad = feat_df.isna().any(axis=1).sum()
    if n_bad:
        warnings.warn(f"{n_bad} chunks got NaN acoustic features (missing/unreadable audio). These will be dropped at merge time -- check audio_root.")
    return feat_df


def clean_transcription(text, max_repeat_ratio: float = 0.5) -> str:
    """Collapse cases where ASR repeats the same short phrase many times.

    Complements (but is distinct from) `tea.features.sentiment`'s
    `_remove_repeated_phrases`. This one works at the n-gram-block level
    (splits into fixed-size chunks and checks what fraction are identical)
    rather than greedy longest-repeat detection; kept as-is since it's
    tuned for the speech-rate use case specifically.

    Parameters
    ----------
    text:
        Raw transcription (NaN-safe).
    max_repeat_ratio:
        If more than this fraction of n-gram blocks (for some block size
        `n`) are identical, keep only one repetition.
    """
    if pd.isna(text):
        return ""

    text = str(text).strip()
    words = text.lower().split()
    if len(words) == 0:
        return ""

    for n in range(1, min(10, len(words) // 2 + 1)):
        chunks = [words[i : i + n] for i in range(0, len(words), n)]
        if len(chunks) < 3:
            continue
        identical = sum(chunks[i] == chunks[0] for i in range(len(chunks)))
        if identical / len(chunks) > max_repeat_ratio:
            return " ".join(chunks[0])

    return " ".join(words)


def speech_rate_from_text(transcription: pd.Series, duration_sec: pd.Series) -> pd.Series:
    """Cheap, language-appropriate speech-rate proxy: whitespace word count / chunk duration.

    Parameters
    ----------
    transcription:
        Text column (ideally pre-cleaned via `clean_transcription`).
    duration_sec:
        Chunk duration in seconds.
    """
    word_count = transcription.fillna("").str.split().str.len()
    return word_count / duration_sec.clip(lower=1e-3)
