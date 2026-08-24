"""Whisper-based transcription and translation.

Ported from `vad_chunking.txt` (`asr.py`). Loads Whisper-large-v3 via the HF
`pipeline` API, forced Finnish transcribe/translate.
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from omegaconf import DictConfig

import torch
import gc

from tea.asr.transcriber import Transcriber
from tea.utils.logging import get_logger
from tea.utils.paths import resolve

logger = get_logger(__name__)


def _resolve_audio_path(video_id: str, cfg: DictConfig, json_dir: Path | None = None) -> Path | None:
    """Locate the source waveform for a video.

    json_dir here is paths.chunk_meta_dir where `tea chunk` saves the jsons for each file.

    Preference order:
    1. ``audio_path`` stored inside the matching VAD JSON (if present).
    2. ``cfg.paths.audio_root / f"{video_id}.wav"``.
    """
    if json_dir is not None:
        json_path = Path(json_dir) / f"{video_id}.json"
        if json_path.exists():
            import json
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            ap = meta.get("audio_path")
            if ap and Path(ap).exists():
                return Path(ap)

    candidate = Path(resolve(cfg.paths.audio_root)) / f"{video_id}.wav"
    return candidate if candidate.exists() else None


def transcribe_annotation_root(cfg: DictConfig) -> int:
    """Run Whisper over every annotation CSV under ``annotation_root``.

    For each CSV:

    * load the matching source audio
    * slice every **speech** row by ``start``/``end`` (sample indices)
    * write ``transcription`` and ``translation`` columns
    * non-speech rows (and any row whose audio cannot be loaded) get empty strings

    Final column order:
        name, duration_sec, start, end, type, gt_label, confidence, overlap,
        transcription, translation

    Parameters
    ----------
    cfg:
        Resolved Hydra configuration. Expected keys:
        ``paths.annotation_root``, ``paths.audio_root``,
        optionally ``paths.chunk_meta_dir``, ``asr.model``, ``asr.language``,
        ``asr.batch_size``.

    Returns
    -------
    int
        Process exit status (0 on success).
    """
    annotation_root = Path(resolve(cfg.paths.annotation_root))
    chunk_meta_dir = Path(resolve(cfg.paths.chunk_meta_dir)) if cfg.paths.get("chunk_meta_dir") else None
    language = cfg.asr.get("language", "fi") if cfg.get("asr") else "fi"
    batch_size = int(cfg.asr.get("batch_size", 8)) if cfg.get("asr") else 8
    model_path = cfg.asr.get("model", None) if cfg.get("asr") else None

    csv_paths = sorted(annotation_root.glob("*.csv"))
    if not csv_paths:
        logger.warning("No annotation CSVs found under %s", annotation_root)
        return 0

    logger.info("ASR: %d CSV(s) under %s", len(csv_paths), annotation_root)
    transcriber = Transcriber(model_path=model_path)

    try:
        for csv_path in csv_paths:
            video_id = csv_path.stem
            df = pd.read_csv(csv_path)

            # Already done? skip unless forced
            if "transcription" in df.columns and "translation" in df.columns:
                if not cfg.get("asr", {}).get("overwrite", False):
                    logger.info("  %s: transcription columns already present, skipping", video_id)
                    continue

            audio_path = _resolve_audio_path(video_id, cfg, chunk_meta_dir)
            if audio_path is None:
                logger.warning("  %s: audio not found, writing empty ASR columns", video_id)
                df["transcription"] = ""
                df["translation"] = ""
                df.to_csv(csv_path, index=False)
                continue

            waveform, sr = librosa.load(str(audio_path), sr=16_000)

            # Collect speech slices
            speech_mask = df["type"].astype(str).str.lower().eq("speech")
            speech_indices = df.index[speech_mask].tolist()
            slices: list[np.ndarray] = []
            for idx in speech_indices:
                start = int(df.at[idx, "start"])
                end = int(df.at[idx, "end"])
                # VAD JSON stores sample indices at the original sr; we loaded at 16 kHz.
                # If the JSON sr differs we would need a scale factor — for now assume 16 kHz.
                slices.append(waveform[start:end])

            transcriptions = [""] * len(df)
            translations = [""] * len(df)

            if slices:
                # Batch transcribe / translate
                tr_out = transcriber.transcribe_batch(slices, sr=sr, language=language, batch_size=1)
                tl_out = transcriber.translate_batch(slices, sr=sr, language=language, batch_size=1)

                # pipeline returns list[dict] with key "text"
                for i, idx in enumerate(speech_indices):
                    transcriptions[idx] = tr_out[i]["text"].strip() if isinstance(tr_out[i], dict) else str(tr_out[i]).strip()
                    translations[idx] = tl_out[i]["text"].strip() if isinstance(tl_out[i], dict) else str(tl_out[i]).strip()

            df["transcription"] = transcriptions
            df["translation"] = translations

            # Enforce column order
            preferred = [
                "name", "duration_sec", "start", "end", "type",
                "gt_label", "confidence", "overlap",
                "transcription", "translation",
            ]
            # keep any extra columns that might already exist, after the preferred ones
            extra = [c for c in df.columns if c not in preferred]
            df = df[[c for c in preferred if c in df.columns] + extra]

            df.to_csv(csv_path, index=False)
            n_speech = int(speech_mask.sum())
            logger.info("  %s: %d speech / %d total rows transcribed", video_id, n_speech, len(df))

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    finally:
        transcriber.close()

    logger.info("ASR complete.")
    return 0