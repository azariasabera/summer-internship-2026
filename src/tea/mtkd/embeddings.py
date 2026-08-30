# src/tea/mtkd/embeddings.py

"""Extract embeddings, logits, and predictions from a trained MTKD student."""

from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from transformers import Wav2Vec2FeatureExtractor

from tea.mtkd.model import load_student
from tea.utils.constants import CLASS_ORDER, ID2LABEL, SAMPLE_RATE
from tea.utils.logging import get_logger
from tea.utils.paths import ensure_dir, resolve

logger = get_logger(__name__)

SOURCE_LANGUAGE = {"fesc": "FI", "iemocap": "EN", "cafe": "FR"}
SOURCE_MAX_SESSION = {"fesc": 9, "iemocap": 5, "cafe": 1}


def _extract_idx(name: str) -> int:
    """Extract the last integer in a filename, for chunk ordering (`chunk_s_12` -> 12)."""
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else -1


def _load_audio(path: str) -> np.ndarray:
    wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return wav.astype(np.float32)


def _load_video_annotations(video_folder: str, annotations_dir: str | None) -> dict:
    """Load `{chunk_name: gt_label}` for one video, defaulting missing labels to `"non-speech"`."""
    if annotations_dir is None:
        return {}
    video_name = Path(video_folder).name
    csv_path = Path(annotations_dir) / f"{video_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No annotation file found for {video_name}: {csv_path}")

    df = pd.read_csv(csv_path)
    if "name" not in df.columns or "gt_label" not in df.columns:
        raise ValueError(f"{csv_path} needs 'name' and 'gt_label' columns. Found: {list(df.columns)}")
    df["gt_label"] = df["gt_label"].fillna("non-speech")
    return dict(zip(df["name"], df["gt_label"]))


class EmbeddingExtractor:
    """Extracts pooled/projected embeddings, logits, and predictions from a trained student.

    Parameters
    ----------
    cfg:
        Resolved Hydra config.
    checkpoint:
        Path to the student checkpoint to load.
    """

    def __init__(self, cfg: DictConfig, checkpoint: str | Path) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
        self.model, epoch = load_student(cfg, checkpoint, self.device)
        logger.info("Loaded checkpoint (epoch %s) from %s", epoch, checkpoint)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(cfg.mtkd.model_ckpt)

    @torch.no_grad()
    def _extract_batch(self, wavs: list[np.ndarray], layer_ids: list[int]) -> list[dict]:
        """Run one batch through the student, returning per-sample pooled/projected/logits/probs/pred/confidence."""
        inputs = self.feature_extractor(
            wavs,
            sampling_rate=SAMPLE_RATE,
            padding=True,
            truncation=True,
            max_length=int(self.cfg.mtkd.hyperparams.max_duration_sec * SAMPLE_RATE),
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(self.device)
        outputs = self.model(input_values, output_hidden_states=True, return_dict=True)

        hidden = outputs.hidden_states[-1]
        pooled = hidden.mean(dim=1)
        projected = self.model.projector(pooled)
        logits = self.model.classifier(projected)
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        confs = probs.max(dim=-1).values

        layer_reps_batch = {
            f"layer_{lid}": outputs.hidden_states[lid].mean(dim=1).cpu().numpy() for lid in layer_ids
        }

        results = []
        for i in range(len(wavs)):
            results.append(
                {
                    "pooled": pooled[i].cpu().numpy(),
                    "projected": projected[i].cpu().numpy(),
                    "logits": logits[i].cpu().numpy(),
                    "probabilities": probs[i].cpu().numpy(),
                    "prediction": int(preds[i]),
                    "confidence": float(confs[i]),
                    "layer_reps": {key: layer_reps_batch[key][i] for key in layer_reps_batch},
                }
            )
        return results

    @staticmethod
    def _save_group(save_dir: Path, pooleds, projs, logits_all, probs_all, preds, names, layer_accum, rows, true_labels=None) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / "pooled.npy", np.stack(pooleds))
        np.save(save_dir / "projected.npy", np.stack(projs))
        np.save(save_dir / "logits.npy", np.stack(logits_all))
        np.save(save_dir / "probabilities.npy", np.stack(probs_all))
        np.save(save_dir / "predictions.npy", np.array(preds))
        np.save(save_dir / "names.npy", np.array(names))
        if true_labels is not None:
            np.save(save_dir / "labels.npy", np.array(true_labels))
        for key, values in layer_accum.items():
            np.save(save_dir / f"{key}.npy", np.stack(values))

        df = pd.DataFrame(rows)
        df.to_csv(save_dir / "metadata.csv", index=False)
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(rows, f, indent=2)

    def process_video_folder(
        self, folder: str, output_dir: str | Path, batch_size: int = 16, layer_ids: list[int] | None = None,
        annotations_dir: str | None = None,
    ) -> None:
        """Extract embeddings for every `.wav` chunk in one video's folder.

        Parameters
        ----------
        folder:
            Directory of ordered `.wav` chunks for one video.
        output_dir:
            Root output directory; results go to `<output_dir>/<folder_name>/`.
        batch_size:
            Inference batch size.
        layer_ids:
            Additional transformer layer indices to save mean-pooled representations for.
        annotations_dir:
            Directory of `<video_name>.csv` annotation files, for attaching ground truth to metadata.
        """
        layer_ids = layer_ids or []
        annotations = _load_video_annotations(folder, annotations_dir)

        wavs = sorted(glob.glob(os.path.join(folder, "*.wav")), key=lambda x: _extract_idx(Path(x).stem))
        if not wavs:
            return

        save_dir = Path(output_dir) / Path(folder).name
        pooleds, projs, logits_all, probs_all, preds, names, rows = [], [], [], [], [], [], []
        layer_accum = {f"layer_{lid}": [] for lid in layer_ids}

        for start in range(0, len(wavs), batch_size):
            batch_files = wavs[start : start + batch_size]
            batch_audio = [_load_audio(f) for f in batch_files]
            batch_feats = self._extract_batch(batch_audio, layer_ids)

            for wav_file, wav, feat in zip(batch_files, batch_audio, batch_feats):
                stem = Path(wav_file).stem
                true_label = annotations.get(stem, "non-speech")
                pooleds.append(feat["pooled"])
                projs.append(feat["projected"])
                logits_all.append(feat["logits"])
                probs_all.append(feat["probabilities"])
                preds.append(feat["prediction"])
                names.append(stem)
                for lid in layer_ids:
                    layer_accum[f"layer_{lid}"].append(feat["layer_reps"][f"layer_{lid}"])

                prob = feat["probabilities"]
                rows.append(
                    {
                        "chunk": stem,
                        "true_label": true_label,
                        "prediction": ID2LABEL[str(feat["prediction"])],
                        "confidence": feat["confidence"],
                        **{label: float(prob[i]) for i, label in enumerate(CLASS_ORDER)},
                        "filepath": wav_file,
                        "duration": float(len(wav) / SAMPLE_RATE),
                    }
                )

        self._save_group(save_dir, pooleds, projs, logits_all, probs_all, preds, names, layer_accum, rows)
        logger.info("%s: %d chunks -> %s", Path(folder).name, len(wavs), save_dir)

    def process_dataset_split(
        self, source: str, session: int, split_name: str, ds_split, output_dir: str | Path,
        batch_size: int = 16, layer_ids: list[int] | None = None,
    ) -> None:
        """Extract embeddings for one benchmark dataset split (fesc/iemocap/cafe).

        Parameters
        ----------
        source:
            `"fesc"`, `"iemocap"`, or `"cafe"`.
        session:
            Session/split index for the source dataset.
        split_name:
            `"train"`, `"test"`, or `"dev"`.
        ds_split:
            The HF `Dataset` for this split.
        output_dir, batch_size, layer_ids:
            See `process_video_folder`.
        """
        layer_ids = layer_ids or []
        n = len(ds_split)
        if n == 0:
            return

        save_dir = Path(output_dir) / source / f"S{session}" / split_name
        pooleds, projs, logits_all, probs_all, preds, names, true_labels, rows = [], [], [], [], [], [], [], []
        layer_accum = {f"layer_{lid}": [] for lid in layer_ids}

        for start in range(0, n, batch_size):
            batch = ds_split[start : start + batch_size]
            batch_audio = [item["array"].astype(np.float32) for item in batch["audio"]]
            batch_true = [int(l) for l in batch["label"]]
            batch_feats = self._extract_batch(batch_audio, layer_ids)

            for offset, (wav, true_label, feat) in enumerate(zip(batch_audio, batch_true, batch_feats)):
                idx = start + offset
                stem = f"{source}_S{session}_{split_name}_{idx:05d}"
                pooleds.append(feat["pooled"])
                projs.append(feat["projected"])
                logits_all.append(feat["logits"])
                probs_all.append(feat["probabilities"])
                preds.append(feat["prediction"])
                names.append(stem)
                true_labels.append(true_label)
                for lid in layer_ids:
                    layer_accum[f"layer_{lid}"].append(feat["layer_reps"][f"layer_{lid}"])

                prob = feat["probabilities"]
                rows.append(
                    {
                        "chunk": stem,
                        "true_label": ID2LABEL[str(true_label)],
                        "prediction": ID2LABEL[str(feat["prediction"])],
                        "confidence": feat["confidence"],
                        **{label: float(prob[i]) for i, label in enumerate(CLASS_ORDER)},
                        "duration": float(len(wav) / SAMPLE_RATE),
                    }
                )

        self._save_group(save_dir, pooleds, projs, logits_all, probs_all, preds, names, layer_accum, rows, true_labels=true_labels)
        logger.info("%s S%d %s: %d samples -> %s", source, session, split_name, n, save_dir)


def extract_embeddings_cli(cfg: DictConfig) -> int:
    """`tea extract-embeddings` entry point.

    Set `mtkd.embeddings.source` to `"videos"` (default) or a benchmark
    dataset name, plus `mtkd.linguality`/`language`/`session` for the
    checkpoint to use. See `EmbeddingExtractor` for the underlying API.
    """
    emb_cfg = cfg.mtkd.get("embeddings", {})
    source = emb_cfg.get("source", "videos")
    checkpoint = emb_cfg.get("checkpoint") or cfg.mtkd.default_student_checkpoint
    output_dir = ensure_dir(resolve(emb_cfg.get("output_dir", cfg.paths.embedding_root)))
    layer_ids = [int(x) for x in str(emb_cfg.get("layers", "")).split(",") if x.strip()]
    batch_size = emb_cfg.get("batch_size", 16)

    extractor = EmbeddingExtractor(cfg, checkpoint)

    if source == "videos":
        input_dir = emb_cfg.get("input_dir")
        if not input_dir:
            logger.error("Set mtkd.embeddings.input_dir for --source videos")
            return 2
        folders = sorted(f for f in glob.glob(os.path.join(input_dir, "*")) if os.path.isdir(f))
        logger.info("videos: %d", len(folders))
        for folder in folders:
            extractor.process_video_folder(
                folder, output_dir, batch_size=batch_size, layer_ids=layer_ids, annotations_dir=emb_cfg.get("annotations_dir")
            )
        return 0

    sessions_raw = str(emb_cfg.get("sessions", "all"))
    sessions = list(range(1, SOURCE_MAX_SESSION[source] + 1)) if sessions_raw.lower() == "all" else [
        int(x) for x in sessions_raw.split(",") if x.strip()
    ]
    splits = [s.strip() for s in str(emb_cfg.get("splits", "train,test,dev")).split(",") if s.strip()]
    language = SOURCE_LANGUAGE[source]

    from tea.teachers import LOADERS # lazy import to avoid circular dependency

    for session in sessions:
        ds = LOADERS[language](session, cfg)
        for split_name in splits:
            extractor.process_dataset_split(source, session, split_name, ds[split_name], output_dir, batch_size=batch_size, layer_ids=layer_ids)

    return 0
