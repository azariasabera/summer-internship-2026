# `tea.asr`

Whisper-based transcription / translation and text-sentiment probability extraction.

## Planned CLI

```bash
tea sentiment paths.annotation_root=...   # writes sentiment_fi.json / sentiment_en.json
```

## Notes

- Model: `cardiffnlp/twitter-xlm-roberta-base-sentiment` for sentiment.
- Whisper-large-v3 for ASR (transcribe + translate).
- Repeated-phrase cleaning applied before sentiment scoring.

## Status

Scaffold.
