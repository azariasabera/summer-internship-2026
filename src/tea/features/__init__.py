# src/tea/features/__init__.py

"""Feature construction shared by `tea.confidence` and `tea.probes`."""

from tea.features.acoustic import batch_extract_acoustic_features, clean_transcription, extract_acoustic_features, speech_rate_from_text
from tea.features.master_table import build_feature_table, build_master_table, rescale_confidence
from tea.features.sentiment import SentimentScorer, sentiment_cli
from tea.features.softmax import add_softmax_engineered_features, mtkd_derived_features, softmax_entropy, softmax_margin, softmax_max_prob

__all__ = [
    "SentimentScorer",
    "sentiment_cli",
    "extract_acoustic_features",
    "batch_extract_acoustic_features",
    "clean_transcription",
    "speech_rate_from_text",
    "softmax_entropy",
    "softmax_margin",
    "softmax_max_prob",
    "mtkd_derived_features",
    "add_softmax_engineered_features",
    "build_master_table",
    "build_feature_table",
    "rescale_confidence",
]
