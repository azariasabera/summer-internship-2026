# src/tea/utils/constants.py

"""
Canonical constants shared across the whole package.

Do not change CLASS_ORDER or EXCLUDED_VIDEOS without updating every
reported metric that depends on them.
"""

from __future__ import annotations

# Emotion classes in the order used by MTKD / teachers / all report tables
CLASS_ORDER: list[str] = ["neutral", "sadness", "happiness", "anger"]

LABEL2ID: dict[str, int] = {c: i for i, c in enumerate(CLASS_ORDER)}
ID2LABEL: dict[int, str] = {i: c for i, c in enumerate(CLASS_ORDER)}

# 3-class sentiment mapping used by some probes / text models
SENTIMENT_ORDER: list[str] = ["neutral", "negative", "positive"]

EMOTION_TO_SENTIMENT: dict[str, str] = {
    "neutral": "neutral",
    "sadness": "negative",
    "anger": "negative",
    "happiness": "positive",
}

# Videos hard-excluded from classroom training / evaluation folds
# (1B3261 has only ~6 annotated chunks, not usable as a fold)
EXCLUDED_VIDEOS: set[str] = {"1B3261"}

# Sample rate used everywhere after resampling
SAMPLE_RATE: int = 16_000
