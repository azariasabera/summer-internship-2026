# src/tea/mtkd/infer.py

"""Run a trained MTKD student on raw, unlabeled audio."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from datasets import Audio, Dataset
from omegaconf import DictConfig
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from torch.utils.data import DataLoader

from tea.mtkd.model import load_model
from tea.mtkd.utils import collate_fn_infer, preprocess_function
from tea.utils.constants import CLASS_ORDER, ID2LABEL
from tea.utils.io import get_teacher_id
from tea.utils.logging import get_logger
from tea.utils.paths import ensure_dir, resolve

logger = get_logger(__name__)

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def gather_files(input_path: str | Path) -> list[str]:
    """Return `[input_path]` if it's a file, or every audio file under it (recursive) if a directory."""
    path = Path(input_path)
    if path.is_file():
        return [str(path)]

    if path.is_dir():
        files = [str(f) for f in sorted(path.rglob("*")) if f.suffix.lower() in AUDIO_EXTS]
        if not files:
            raise FileNotFoundError(f"No audio files ({sorted(AUDIO_EXTS)}) found under {input_path}")
        return files

    raise FileNotFoundError(f"{input_path} is neither a file nor a directory")


def _load_video_annotations(video_folder: str, annotations_dir: str | None) -> dict:
    """Load `{chunk_name: gt_label}` for a single video folder.
    Returns an empty dict if no annotations_dir is provided or if the expected CSV file doesn't exist.
    """
    if annotations_dir is None:
        return {}
    csv_path = Path(annotations_dir) / f"{Path(video_folder).name}.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    if "name" not in df.columns or "gt_label" not in df.columns:
        raise ValueError(f"{csv_path} needs 'name' and 'gt_label' columns. Found: {list(df.columns)}")
    df["gt_label"] = df["gt_label"].fillna("non-speech")
    return dict(zip(df["name"], df["gt_label"]))


def collect_annotations(files: list[str], annotations_dir: str | None) -> dict:
    """Load `{video_name: {chunk_name: gt_label}}` for every video folder appearing in `files`."""
    folders = {str(Path(f).parent) for f in files}
    return {Path(folder).name: _load_video_annotations(folder, annotations_dir) for folder in folders}


def print_confusion_matrix(results: dict, annotations: dict) -> None:
    """Print a confusion matrix over whichever inferred files have a matching ground-truth annotation."""
    y_true, y_pred = [], []
    for file_path, pred in results.items():
        p = Path(file_path)
        folder, stem = p.parent.name, p.stem
        if folder not in annotations or stem not in annotations[folder]:
            continue
        gt = annotations[folder][stem]
        if gt not in CLASS_ORDER:
            continue
        y_true.append(gt)
        y_pred.append(pred["prediction"])

    if not y_true:
        print("\nNo matching ground-truth annotations found.")
        return

    cm = sk_confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    print("\nConfusion Matrix\n")
    print("{:>12}".format("") + "".join(f"{l:>12}" for l in CLASS_ORDER))
    for label, row in zip(CLASS_ORDER, cm):
        print(f"{label:>12}" + "".join(f"{v:>12}" for v in row))


class Inferencer:
    """Runs a trained MTKD student on raw audio with no ground-truth labels.

    Parameters
    ----------
    cfg:
        Resolved Hydra config.
    checkpoint:
        Path to a trained student checkpoint. Defaults to `cfg.mtkd.default_student_checkpoint`.
    temperature:
        Optional fixed inference-time temperature (divides logits before
        softmax; does not change the predicted class). `None` (default) =
        plain softmax, no scaling. Pass a value only if you've deliberately
        decided to apply a previously-fitted `Calibrator` temperature.
    """

    def __init__(self, cfg: DictConfig, checkpoint: str | Path | None = None, temperature: float | None = None) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
        checkpoint = checkpoint or resolve(cfg.mtkd.default_student_checkpoint)
        self.model, epoch = load_model(cfg, checkpoint, self.device)
        self.model.eval()
        logger.info("Loaded checkpoint (epoch %s) from %s", epoch, checkpoint)

        self.temperature = temperature if temperature is not None else cfg.mtkd.get("inference_temperature", None)
        if self.temperature is not None:
            logger.info("Inference-time temperature scaling enabled: T=%.4f", self.temperature)

    def _build_loader(self, files: list[str], batch_size: int) -> DataLoader:
        from transformers import Wav2Vec2FeatureExtractor

        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.cfg.mtkd.model_ckpt)
        ds = Dataset.from_dict({"audio": files, "file_id": files}).cast_column("audio", Audio(sampling_rate=16_000))
        encoded = ds.map(
            lambda ex: preprocess_function(ex, feature_extractor, self.cfg.mtkd.hyperparams.max_duration_sec),
            remove_columns="audio",
            batched=True,
        )
        return DataLoader(encoded, batch_size=batch_size, collate_fn=collate_fn_infer)

    @torch.no_grad()
    def _infer_loader(self, loader: DataLoader) -> dict:
        results = {}
        for batch in loader:
            inputs = batch["input_values"].to(self.device)
            logits = self.model(inputs).logits
            if self.temperature is not None:
                logits = logits / self.temperature
            probs = torch.softmax(logits, dim=1).cpu()

            for file_id, p in zip(batch["file_id"], probs):
                pred = int(torch.argmax(p))
                results[file_id] = {
                    "probabilities": {ID2LABEL[str(i)]: round(p[i].item(), 6) for i in range(len(p))},
                    "prediction": ID2LABEL[str(pred)],
                }
        return results

    def run(self, input_path: str | Path, batch_size: int = 8) -> dict:
        """Run inference over a single file or every audio file under a directory.

        Parameters
        ----------
        input_path:
            File or directory to run inference on.
        batch_size:
            Inference batch size.

        Returns
        -------
        dict
            `{file_path: {"probabilities": {...}, "prediction": label}}`.
        """
        files = gather_files(input_path)
        logger.info("Found %d file(s) to run inference on from %s.", len(files), input_path)
        loader = self._build_loader(files, batch_size)
        return self._infer_loader(loader)

    def run_per_teacher(self, input_path: str | Path, batch_size: int = 8) -> dict:
        """Same as `run`, but batches files grouped by teacher (video-folder-name minus `_videoN` suffix) first.

        Only affects batch composition, not per-file results -- use this if
        you need to keep each teacher's chunks contiguous for downstream
        streaming/logging, otherwise `run` is equivalent and simpler.
        """
        files = gather_files(input_path)
        groups: dict[str, list[str]] = defaultdict(list)
        for f in files:
            teacher = get_teacher_id(Path(f).parent.name)
            groups[teacher].append(f)

        all_results = {}
        for teacher_id in sorted(groups):
            teacher_files = groups[teacher_id]
            logger.info("Teacher %s: %d files", teacher_id, len(teacher_files))
            loader = self._build_loader(teacher_files, batch_size)
            all_results.update(self._infer_loader(loader))
        return all_results

    @staticmethod
    def save_grouped(results: dict, output_path: str | Path) -> None:
        """Save `results` grouped as `{video_folder: {chunk_stem: probabilities}}` -- the format `tea.analysis` expects."""
        grouped = defaultdict(dict)
        for file_path, data in results.items():
            p = Path(file_path)
            grouped[p.parent.name][p.stem] = data["probabilities"]

        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        with open(output_path, "w") as f:
            json.dump(grouped, f, indent=2, ensure_ascii=False)
        logger.info("Saved results -> %s", output_path)


def infer_mtkd_cli(cfg: DictConfig) -> int:
    """`tea infer-mtkd` entry point.

    Requires `mtkd.infer.input` (file or directory). Optional:
    `mtkd.infer.checkpoint`, `mtkd.infer.output` (json path),
    `mtkd.infer.annotations_dir` (for a confusion matrix against ground
    truth), `mtkd.infer.per_teacher` (bool), `mtkd.inference_temperature`.
    """
    infer_cfg = cfg.mtkd.get("infer", {})
    input_path = infer_cfg.get("input")
    if not input_path:
        logger.error("Set mtkd.infer.input=<file_or_dir>")
        return 2

    inferencer = Inferencer(cfg, checkpoint=infer_cfg.get("checkpoint"))
    batch_size = infer_cfg.get("batch_size") or cfg.mtkd.hyperparams.batch_size

    results = (
        inferencer.run_per_teacher(input_path, batch_size=batch_size)
        if infer_cfg.get("per_teacher", False)
        else inferencer.run(input_path, batch_size=batch_size)
    )

    if infer_cfg.get("save_output", True):
        output_path = infer_cfg.get("output")
        if output_path:
            inferencer.save_grouped(results, output_path)
        else:
            logger.warning("Save output was true, but couldn't find ouput save path.")

    if infer_cfg.get("eval", True):
        annotations = collect_annotations(gather_files(input_path), infer_cfg.get("annotations_dir"))
        print_confusion_matrix(results, annotations)
    return 0
