# src/tea/classroom/fesc.py

"""FESC contamination: "infects" FESC sadness/anger utterances with real
classroom background noise before injecting them into a fold's training
set, so the added minority-class data sounds closer to the far-field
classroom domain instead of raw close-talk acted audio.

Ported from `mtkd_code.txt`'s `fesc_contamination.py`. Noise-pool
extraction from annotated CSVs (`extract_noise_pool`) now delegates to
`tea.noise.extract_noise_pool` rather than redefining it -- this module
keeps only what's genuinely LOTO-fold-specific: composing a fold's noise
pool (training teachers' non-speech + always-included noise-only videos),
estimating this fold's empirical SNR gap, and the composite-noise mixing
that injects it into FESC audio.

RIR (room impulse response) augmentation is available as an OPTIONAL.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import TYPE_CHECKING

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from omegaconf import DictConfig
from tqdm.auto import tqdm

from tea.classroom.data import read_video_csv_raw
from tea.utils.constants import LABEL2ID, SAMPLE_RATE
from tea.utils.logging import get_logger

if TYPE_CHECKING: # to avoid circular import issues with RIRAugmentor. here i am only importing the type for type hints, not actually using it at runtime.
    from tea.noise.rir import RIRAugmentor

logger = get_logger(__name__)

EPS = 1e-8


def _rms_db(wave: np.ndarray) -> float:
    rms = np.sqrt(np.mean(wave.astype(np.float64) ** 2) + EPS)
    return 20 * np.log10(rms + EPS)


def _extract_noise_pool(csv_root: str | Path, audio_root: str | Path, video_ids: list[str]) -> list[str]:
    """Non-speech (NaN `gt_label`) chunk paths for a set of videos.

    This is the fold-aware. It needs both `csv_root` and
    `audio_root` to build chunk paths via `read_video_csv_raw`. 
    
    `tea.noise.extract_noise_pool` is a separate, simpler
    version for the standalone `tea extract-noise` CLI command, which
    only needs an already-path-bearing annotation table.
    """
    paths: list[str] = []
    for video_id in sorted(set(video_ids)):
        df = read_video_csv_raw(csv_root, audio_root, video_id)
        if df is None:
            logger.info("%s: no csv found, skipping for noise pool", video_id)
            continue
        noise_rows = df.loc[df["gt_label"].isna()]
        if len(noise_rows) == 0:
            continue
        paths.extend(noise_rows["audio_path"].tolist())

    if not paths:
        raise ValueError("Extracted noise pool is empty -- check csv_root/audio_root and that CSVs contain NaN-labeled rows.")
    return paths


def build_noise_pool_for_fold(
    csv_root: str | Path, audio_root: str | Path, train_df: pd.DataFrame, extra_video_ids: tuple[str, ...] = ("1B3261",)
) -> dict[str, list[str]]:
    """Compose this fold's noise source: training videos' non-speech chunks, plus always-included noise-only videos.

    The held-out teacher's videos are never in `train_df` to begin with,
    so nothing further needs excluding there.

    Parameters
    ----------
    csv_root, audio_root:
        Directories of per-video annotation CSVs and chunk audio.
    train_df:
        This fold's training dataframe (see `tea.classroom.data.build_full_df`).
    extra_video_ids:
        Video ids to always include as noise source even if excluded from
        speech train/test (e.g. `1B3261` -- only 6 annotated chunks, but its
        background audio is fine as noise).

    Returns
    -------
    dict
        `{"dynamic": [...], "mic": [...]}` -- dynamic = training teachers'
        own non-speech chunks, mic = the fixed extra noise-only videos.
    """
    train_video_ids = set(train_df["video_id"].unique())
    logger.info(
        "Noise pool videos: %d training + %d noise-only extras", len(train_video_ids), len(set(extra_video_ids))
    )

    dynamic_noise_paths = _extract_noise_pool(csv_root, audio_root, list(train_video_ids))
    mic_noise_paths = _extract_noise_pool(csv_root, audio_root, list(extra_video_ids))
    logger.info("Dynamic noise: %d chunks | Mic noise: %d chunks", len(dynamic_noise_paths), len(mic_noise_paths))

    return {"dynamic": dynamic_noise_paths, "mic": mic_noise_paths}


def estimate_snr_stats(train_df: pd.DataFrame, noise_pool_paths: list[str], n_sample: int = 200, seed: int = 42) -> dict:
    """Estimate this fold's empirical speech-vs-background loudness gap (dB), to sample augmentation SNR from.

    Parameters
    ----------
    train_df:
        This fold's training dataframe.
    noise_pool_paths:
        Combined (dynamic + mic) noise chunk paths.
    n_sample:
        Max number of speech/noise clips to sample for the RMS estimate.
    seed:
        Random seed for sampling.
    """
    speech_paths = train_df["audio_path"].sample(n=min(n_sample, len(train_df)), random_state=seed)
    speech_db = [_rms_db(librosa.load(p, sr=SAMPLE_RATE)[0]) for p in tqdm(speech_paths, desc="Computing speech RMS", leave=False)]

    noise_sample = random.sample(noise_pool_paths, min(n_sample, len(noise_pool_paths)))
    noise_db = [_rms_db(librosa.load(p, sr=SAMPLE_RATE)[0]) for p in tqdm(noise_sample, desc="Computing noise RMS", leave=False)]

    speech_db, noise_db = np.array(speech_db), np.array(noise_db)
    gap = speech_db.mean() - noise_db.mean()
    gap_std = float(np.sqrt(speech_db.var() + noise_db.var()))
    stats = {
        "mean_snr_db": float(gap),
        "std_snr_db": gap_std,
        "snr_min_db": float(gap - gap_std),
        "snr_max_db": float(gap + gap_std),
    }
    logger.info(
        "Estimated speech/background gap this fold: %.1f dB (+/- %.1f) -> sampling from [%.1f, %.1f] dB",
        stats["mean_snr_db"], stats["std_snr_db"], stats["snr_min_db"], stats["snr_max_db"],
    )
    return stats


def _load_and_trim(path: str, target_len: int) -> np.ndarray:
    wave, _ = librosa.load(path, sr=SAMPLE_RATE)
    if len(wave) < target_len:
        wave = np.tile(wave, int(np.ceil(target_len / len(wave))))
    start = random.randint(0, len(wave) - target_len) if len(wave) > target_len else 0
    return wave[start : start + target_len]


def _composite_noise(noise_pool_paths: dict[str, list[str]], target_len: int, n_sources: tuple[int, int] = (2, 3)) -> np.ndarray:
    """Sum a random mic-noise clip plus 2-3 randomly chosen environmental clips, each RMS-normalized before mixing.

    Mic-noise: from 1B3261, where there is no other sound. Environmental-noise: the non-speech parts from the rest of the videos.
    """
    dynamic_pool, mic_pool = noise_pool_paths["dynamic"], noise_pool_paths["mic"]
    composite = np.zeros(target_len, dtype=np.float64)

    chosen = [random.choice(mic_pool)]
    k = random.randint(*n_sources)
    chosen += random.sample(dynamic_pool, min(k, len(dynamic_pool)))

    for p in chosen:
        seg = _load_and_trim(p, target_len).astype(np.float64)
        rms = np.sqrt(np.mean(seg**2) + EPS)
        seg = seg / (rms + EPS)
        gain = random.uniform(0.2, 0.5) if p in mic_pool else random.uniform(0.5, 1.0)
        composite += seg * gain

    return composite


def _mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale `noise` to hit `snr_db` relative to `speech`, sum, and peak-normalize only if clipping would occur."""
    if len(noise) != len(speech):
        if len(noise) < len(speech):
            noise = np.tile(noise, int(np.ceil(len(speech) / len(noise))))
        noise = noise[: len(speech)]

    rms_speech = np.sqrt(np.mean(speech.astype(np.float64) ** 2) + EPS)
    rms_noise = np.sqrt(np.mean(noise.astype(np.float64) ** 2) + EPS)
    target_rms_noise = rms_speech / (10 ** (snr_db / 20))
    noise_scaled = noise * (target_rms_noise / (rms_noise + EPS))

    mixed = speech + noise_scaled
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32)


def fesc_pool_df(cfg: DictConfig, classes: tuple[str, ...] = ("sadness", "anger", "happiness")) -> pd.DataFrame:
    """Pool every FESC speaker session's train+dev+test splits for augmentation material.

    We're mining FESC for raw audio here, not evaluating on it, so nothing
    is held back. Uses the same json-splits location, path fixup, and
    label mapping as `tea.teachers.fesc`, just without the HF
    Dataset/Audio wrapping, since plain file paths are wanted for librosa here.

    Parameters
    ----------
    cfg:
        Resolved Hydra config.
    classes:
        Which emotion classes to keep from the pool.
    """
    label_map = {"1": "neutral", "2": "sadness", "3": "happiness", "4": "anger"}
    splits_root = Path(cfg.paths.splits_root)
    fesc_new_prefix = str(Path(cfg.paths.datasets_root) / "FESC") + "/"

    frames = []
    for session, folder in cfg.teachers.fesc_session_map.items():
        for split in ("train", "test", "dev"):
            path = splits_root / "Finnish-emotion-spilits" / folder / f"{split}.json"
            if not path.exists():
                logger.warning("Can't find %s", path)
                continue
            with open(path) as f:
                raw = json.load(f)
            df = pd.DataFrame.from_dict(raw, orient="index").reset_index()
            df = df.rename(columns={"index": "file_id", "label": "emo"})
            df = df.loc[df["emo"].astype(str) != "5"].reset_index(drop=True)
            df["audio_path"] = df["file_path"].str.replace(cfg.teachers.fesc_old_prefix, fesc_new_prefix, regex=False)
            df["gt_label"] = df["emo"].astype(str).map(label_map)
            df["label"] = df["gt_label"].map(LABEL2ID).astype(int)
            df["session"] = session
            frames.append(df[["audio_path", "gt_label", "label", "session"]])

    pool = pd.concat(frames, ignore_index=True).drop_duplicates(subset="audio_path").reset_index(drop=True)
    pool = pool.loc[pool["gt_label"].isin(classes)].reset_index(drop=True)
    logger.info("FESC pool: %d utterances across classes %s", len(pool), dict(pool["gt_label"].value_counts()))
    return pool

def compute_augmentation_sizes(
    class_counts: dict,
    alpha: float = 5.0,
    max_augmentation: float | None = None,
) -> dict[str, int]:
    """Adaptive sizing for FESC augmentation. Largest class(es) get 0.
    
    Parameters
    ----------
    class_counts:
        `{"sadness": n, "anger": n, ...}` from this fold's number of samples per class in the training fold.
    alpha:
        Controls the exponential decay of augmentation size relative to class count.
    max_augmentation:
        If given, the largest class(es) get 0, and the smallest class gets `max_augmentation`. 
        If None, the largest class gets 0 and the smallest class gets half the largest class's count.

    Returns
    -------
    dict
        Mapping from class -> number of synthetic samples to generate.
    """
    if not class_counts:
        return {}
    max_count = max(class_counts.values())
    if max_count == 0:
        return {cls: 0 for cls in class_counts}
    if max_augmentation is None:
        max_augmentation = max_count / 2
    return {
        cls: (
            0
            if count == max_count
            else round(max_augmentation * math.exp(-alpha * count / max_count))
        )
        for cls, count in class_counts.items()
    }

def contaminate_fesc(
    fesc_df: pd.DataFrame,
    noise_pool_paths: dict[str, list[str]],
    snr_stats: dict,
    output_dir: str | Path,
    classes: tuple[str, ...] = ("sadness", "anger"),
    # --- sizing ---
    augment_strategy: str = "cap",          # "cap" | "adaptive"
    cap_multiplier: float = 2.5,
    adaptive_alpha: float = 5.0,
    real_class_counts: dict | None = None,
    # --- noise / SNR ---
    n_noise_sources: tuple[int, int] = (2, 3),
    snr_std_db: float = 5.0,
    snr_clip_min_db: float = 10,
    snr_clip_max_db: float = 30,
    seed: int = 42,
    # --- optional RIR ---
    rir: RIRAugmentor | None = None,
    rir_prob: float = 0.7,
    rir_dry_wet_range: tuple[float, float] = (0.35, 0.75),
) -> pd.DataFrame:
    """Mix classroom noise into FESC minority-class utterances before injecting
    them into a fold's training set.

    Parameters
    ----------
    fesc_df:
        Output of `fesc_pool_df`.
    noise_pool_paths:
        Output of `build_noise_pool_for_fold`
        (`{"dynamic": [...], "mic": [...]}`).
    snr_stats:
        Output of `estimate_snr_stats`.
    output_dir:
        Directory where the mixed `.wav` files are written.
    classes:
        Which emotion classes to augment. Report used `("sadness", "anger")`.
    augment_strategy:
        `"cap"` (default, reported path) or `"adaptive"`.
    cap_multiplier:
        Used only when `augment_strategy="cap"`. Injected count per class
        <= `cap_multiplier * real_class_count`.
    adaptive_alpha:
        Used only when `augment_strategy="adaptive"`. Controls how quickly
        the number of synthetic samples falls as real class size grows
        (see `compute_augmentation_sizes`).
    real_class_counts:
        `{"sadness": n, "anger": n, ...}` from this fold's real training
        data. Required for both strategies when class-aware sizing is desired;
        if `None`, the full FESC pool for each class is used.
    n_noise_sources:
        Min/max number of environmental (dynamic) noise clips composited
        together per augmented sample. A mic-noise clip is always included.
    snr_std_db, snr_clip_min_db, snr_clip_max_db:
        Per-sample SNR is drawn from
        `Normal(snr_stats["mean_snr_db"], snr_std_db)` and clipped to
        `[snr_clip_min_db, snr_clip_max_db]`.
    seed:
        Random seed for sampling and SNR draws.
    rir:
        Optional `tea.noise.RIRAugmentor`. Default `None` disables RIR
        and reproduces the original reported behaviour exactly. If given,
        reverb is applied before noise mixing.
    rir_prob:
        Fraction of augmented samples that receive reverb when `rir` is
        provided.
    rir_dry_wet_range:
        Per-sample randomized dry/wet ratio range when `rir` is provided.

    Returns
    -------
    pd.DataFrame
        Augmented samples ready to be concatenated onto the fold's training
        dataframe.
    """
    random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rir_count = 0

    # Pre-compute adaptive sizes once (only used if strategy == "adaptive")
    adaptive_sizes = None
    if augment_strategy == "adaptive" and real_class_counts:
        adaptive_sizes = compute_augmentation_sizes(
            real_class_counts, alpha=adaptive_alpha
        )

    for cls in classes:
        cls_id = LABEL2ID[cls]
        pool = fesc_df.loc[fesc_df["label"].astype(int) == int(cls_id)]

        # sizing the augmentation
        if augment_strategy == "adaptive":
            n_samples = adaptive_sizes.get(cls, 0) if adaptive_sizes is not None else len(pool)
            chosen = pool.sample(
                n=n_samples,
                replace=(n_samples > len(pool)),
                random_state=seed,
            )
            logger.info("Contaminating %d/%d FESC '%s' utterances (adaptive)", len(chosen), len(pool), cls)
        else: # augment_strategy == "cap"
            cap = len(pool)
            if real_class_counts:
                cap = min(
                    cap,
                    int(round(cap_multiplier * max(real_class_counts.get(cls, 1), 1))),
                )
            chosen = pool.sample(n=cap, random_state=seed) if cap < len(pool) else pool
            logger.info(
                "Contaminating %d/%d FESC '%s' utterances (cap=%d)",
                len(chosen), len(pool), cls, cap,
            )

        for i, row in enumerate(chosen.itertuples()):
            speech, _ = librosa.load(row.audio_path, sr=SAMPLE_RATE)

            if rir is not None and random.random() < rir_prob:
                speech = rir.augment(speech, dry_wet_range=rir_dry_wet_range)
                rir_count += 1

            noise = _composite_noise(noise_pool_paths, len(speech), n_sources=n_noise_sources)
            snr_db = float(
                np.clip(
                    # I intentionally used the configurable augmentation std (snr_std_db) rather than
                    # snr_stats["std_snr_db"] for the SNR sampling.
                    random.gauss(snr_stats["mean_snr_db"], snr_std_db),
                    a_min=snr_clip_min_db,
                    a_max=snr_clip_max_db,
                )
            )
            mixed = _mix_at_snr(speech, noise, snr_db)

            out_path = output_dir / f"fesc_aug_{cls}_{i}.wav"
            sf.write(out_path, mixed, SAMPLE_RATE)

            rows.append({
                "audio_path": str(out_path),
                "label": int(cls_id),
                "gt_label": cls,
                "video_id": "__AUG__",
                "teacher_id": "__AUG__",
                "confidence": 3,
                "child_speech": 0,
                "is_augmented": True,
                "chunk_uid": f"aug_{cls}_{i}",
                "snr_db": snr_db,
            })

    aug_df = pd.DataFrame(rows)
    logger.info("Total contaminated FESC samples added: %d", len(aug_df))
    if rir is not None:
        logger.info("RIR applied to %d/%d samples", rir_count, len(aug_df))
    return aug_df
