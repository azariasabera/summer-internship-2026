# `tea.features`

Single source of truth for feature construction. Used by `tea.confidence`
and `tea.probes`.

## Public API

```python
from tea.features import SentimentScorer

scorer = SentimentScorer()  # cardiffnlp/twitter-xlm-roberta-base-sentiment
probs = scorer.predict_probs("some transcribed text")  # {"positive": .., "neutral": .., "negative": ..}

fi_json, en_json = scorer.build_corpus_json("generated/annotations")
```

## CLI

```bash
tea sentiment
```

Writes `cfg.paths.sentiment_fi` / `cfg.paths.sentiment_en` (FI/EN
sentiment-probability JSONs, one per video per chunk), scoring every
annotation CSV's `transcription`/`translation` columns.

## Feature groups

| Group | Module | Status |
|---|---|---|
| Text sentiment (FI / EN) | `sentiment.py` | Done |
| MTKD softmax + derived (entropy, margin, max-prob) | `softmax.py` | Pending |
| Acoustic (RMS, F0, voiced ratio, speech rate) | `acoustic.py` | Pending |
| Master feature table (joins everything) | `master_table.py` | Pending |

## Notes

- `SentimentScorer.preprocess` strips repeated-phrase Whisper hallucination
  artifacts (e.g. `"hello hello hello"` -> `"hello"`) before scoring.

## Status

`SentimentScorer` ported and CLI-wired. `acoustic.py`, `softmax.py`, and
`master_table.py` are next.
