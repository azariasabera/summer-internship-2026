# src/tea/features/sentiment.py

"""FI/EN text-sentiment scoring, used as a feature source for
`tea.confidence` and `tea.probes` (handcrafted feature fusion).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import torch
from omegaconf import DictConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tea.utils.logging import get_logger
from tea.utils.paths import ensure_dir, resolve

logger = get_logger(__name__)

DEFAULT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _remove_repeated_phrases(text: str, max_ngram: int = 20) -> str:
    """Collapse immediately-repeated word sequences (a known ASR/Whisper hallucination artifact).

    Examples
    --------
    >>> _remove_repeated_phrases("hello hello hello")
    'hello'
    >>> _remove_repeated_phrases("how are you how are you how are you")
    'how are you'
    """
    if not text:
        return text

    words = text.split()
    out = []
    i = 0
    while i < len(words):
        removed = False
        max_len = min(max_ngram, (len(words) - i) // 2)

        for n in range(max_len, 0, -1):
            phrase = words[i : i + n]
            repeats = 1
            while i + repeats * n + n <= len(words) and words[i + repeats * n : i + (repeats + 1) * n] == phrase:
                repeats += 1
            if repeats > 1:
                out.extend(phrase)
                i += repeats * n
                removed = True
                break

        if not removed:
            out.append(words[i])
            i += 1

    return " ".join(out)


class SentimentScorer:
    """3-class (positive/neutral/negative) sentiment scoring via a multilingual XLM-RoBERTa model.

    Parameters
    ----------
    model_name:
        HF model id. Defaults to `cardiffnlp/twitter-xlm-roberta-base-sentiment`.
    """

    def __init__(self, model_name: str | None = None) -> None:
        model_name = model_name or DEFAULT_MODEL
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.labels = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]

    @staticmethod
    def preprocess(text) -> str:
        """Normalize whitespace and strip repeated-phrase ASR artifacts. Returns `""` for NaN/empty input."""
        if pd.isna(text):
            return ""
        return _remove_repeated_phrases(_normalize_whitespace(str(text)))

    @torch.no_grad()
    def predict_probs(self, text: str) -> dict[str, float | None]:
        """Return `{label: probability}` for one text, or all-`None` if the text is empty after preprocessing."""
        text = self.preprocess(text)
        if text == "":
            return {label: None for label in self.labels}

        encoded = self.tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
        logits = self.model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
        return {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}

    def build_video_json(self, csv_path: str | Path) -> tuple[dict, dict]:
        """Score every chunk's `transcription` (FI) and `translation` (EN) column in one annotation CSV.

        Parameters
        ----------
        csv_path:
            Per-video annotation CSV with `name`, `transcription`, `translation` columns.

        Returns
        -------
        tuple
            `(fi_probs, en_probs)`, each `{chunk_name: {label: prob}}`.
        """
        df = pd.read_csv(csv_path)
        fi, en = {}, {}
        for _, row in df.iterrows():
            chunk = row["name"]
            fi[chunk] = self.predict_probs(row.get("transcription", ""))
            en[chunk] = self.predict_probs(row.get("translation", ""))
        return fi, en

    def build_corpus_json(self, csv_root: str | Path) -> tuple[dict, dict]:
        """Score every annotation CSV under `csv_root`, keyed by video (filename stem).

        Parameters
        ----------
        csv_root:
            Directory of per-video annotation CSVs.

        Returns
        -------
        tuple
            `(all_fi, all_en)`, each `{video_name: {chunk_name: {label: prob}}}`.
        """
        csv_files = sorted(Path(csv_root).glob("*.csv"))
        logger.info("Found %d CSV files under %s", len(csv_files), csv_root)

        all_fi, all_en = {}, {}
        for csv_file in csv_files:
            logger.info("Processing %s", csv_file.name)
            fi, en = self.build_video_json(csv_file)
            all_fi[csv_file.stem] = fi
            all_en[csv_file.stem] = en

        return all_fi, all_en


def sentiment_cli(cfg: DictConfig) -> int:
    """`tea sentiment` -- score every annotation CSV's transcription/translation and write FI/EN JSONs.

    Parameters
    ----------
    cfg:
        Resolved Hydra config. Reads `cfg.paths.annotation_root`,
        writes to `cfg.paths.sentiment_fi` / `cfg.paths.sentiment_en`.
    """
    scorer = SentimentScorer(cfg.features.sentiment_model)
    all_fi, all_en = scorer.build_corpus_json(cfg.paths.annotation_root)

    fi_path, en_path = resolve(cfg.paths.sentiment_fi), resolve(cfg.paths.sentiment_en)
    ensure_dir(fi_path.parent)
    ensure_dir(en_path.parent)

    with open(fi_path, "w", encoding="utf-8") as f:
        json.dump(all_fi, f, ensure_ascii=False, indent=2)
    with open(en_path, "w", encoding="utf-8") as f:
        json.dump(all_en, f, ensure_ascii=False, indent=2)

    logger.info("Processed %d videos -> %s, %s", len(all_fi), fi_path, en_path)
    return 0
