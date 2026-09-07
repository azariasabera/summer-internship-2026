# `tea.asr`

Whisper-based transcription and translation of speech chunks.

## Modules

| File | Role |
|------|------|
| `transcriber.py` | `Transcriber` class wrapping Whisper (default `openai/whisper-large-v3`). Expects audio path or waveform; returns Finnish transcript or English translation. Manages GPU memory via `close()`. |
| `__init__.py` | Exposes `transcribe_annotation_root(cfg)` which walks the annotation CSVs produced by `tea chunk` / `tea merge-annotations`, runs Whisper on the corresponding audio, and writes `transcription` / `translation` columns. |

## Command

- `tea apply-asr` – transcribe + translate every speech chunk listed in the annotation root and update the CSVs in place.

## CLI

```bash
tea apply-asr
tea apply-asr asr.model_name=openai/whisper-large-v3 asr.device=cuda
tea apply-asr asr.annotation_csv_dir=generated/annotations

# If we don't want to run the model again but attach existing asr outputs
tea apply-asr asr.use_precomputed=true
```

Important configuration lives in `conf/asr/asr.yaml`:

| Key | Meaning |
|-----|---------|
| `asr.model_name` | HuggingFace / Whisper model identifier |
| `asr.device` | `cuda` / `cpu` |
| `asr.batch_size` | Inference batch size |
| `asr.language` | Source language hint (Finnish) |
| `asr.annotation_csv_dir` | Directory of per-video annotation CSVs |
| `asr.audio_root` | Directory containing the cut WAV chunks (if used) |

Pre-computed transcripts already present in the CSVs can be left untouched; the command only fills missing columns when run.
