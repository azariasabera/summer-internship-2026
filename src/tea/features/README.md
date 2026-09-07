# `tea.features`

Hand-crafted acoustic features, text-sentiment scores and the master feature table used by probes.

## Modules

| File | Role |
|------|------|
| `acoustic.py` | Extracts duration, loudness, speaking-rate and related low-level descriptors from audio chunks. |
| `sentiment.py` | Runs a multilingual sentiment model (FI/EN) on transcripts/translations and writes probability JSON files. Exposes `sentiment_cli(cfg)`. |
| `softmax.py` | Softmax / temperature helpers for converting raw logits into calibrated probabilities. |
| `master_table.py` | Builds the wide feature table that joins acoustic, sentiment and embedding columns for the fusion probes. |
| `__init__.py` | Re-exports the public helpers. |

## Command

- `tea sentiment` – compute Finnish and/or English text-sentiment probabilities from the transcripts already present in the annotation CSVs.

## CLI

```bash
tea sentiment
tea sentiment features.annotation_root=generated/annotations
tea sentiment \ 
    features.annotation_root=generated/annotations \
    features.sentiment_fi=generated/sentiment/sentiment_fi.json \
    features.sentiment_en=generated/sentiment/sentiment_en.json
```

Important configuration lives in `conf/features/features.yaml`.
