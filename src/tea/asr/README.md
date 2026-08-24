# `tea.asr`

Whisper-based transcription / translation and text-sentiment probability extraction.

## Public API

```python
from tea.asr import Transcriber

t = Transcriber()  # defaults to openai/whisper-large-v3
text_fi = t.transcribe(audio_path_or_waveform)
text_en = t.translate(audio_path_or_waveform)
t.close()  # release CUDA memory when done with this instance
```

Text-sentiment scoring (`cardiffnlp/twitter-xlm-roberta-base-sentiment`,
used as a feature source, not part of transcription itself) moved to
`tea.features` -- see that module.

## Planned CLI

```bash
tea apply_asr   # wires Transcriber output into annotation CSVs
```

## Status

`Transcriber` class ported and usable directly. The `tea apply_asr` CLI
command is registered but not yet wired to `tea.vad`'s chunk metadata
format. Still needs that integration point once chunk-level audio access is
finalized.
