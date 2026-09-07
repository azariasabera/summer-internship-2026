# `tea.vad`

Voice Activity Detection (Silero) and speech-segment refinement for classroom audio.

## Modules

| File | Role |
|------|------|
| `segmenter.py` | Core `Segmenter` class. Runs Silero VAD, merges/splits/refines segments according to duration and speech-ratio rules, writes per-video JSON metadata. Expects a resolved Hydra `cfg` (paths + vad.*). Returns a dict of video → list of segment dicts. |
| `audio.py` | Optional helpers that cut and save the actual WAV chunks when `vad.save_audios=true`. |
| `refinement.py` | Pure functions that apply the merge-gap, min-duration, max-duration and overlap rules. |
| `enrichment.py` | Helpers that enrich fixed-length windows with VAD-derived speech ratios (library use). |
| `video.py` | Thin wrappers for video-to-audio extraction when the source is still a video file. |
| `__init__.py` | Exposes the CLI entry point `chunk(cfg)`. |

## Command

- `tea chunk` – run VAD + refinement over `vad.audio_root` and write segment metadata + empty annotation CSVs.

## CLI

```bash
tea chunk
tea chunk vad.max_segment_duration=8.0 vad.overlap=1.0
tea chunk \
    vad.audio_root=data/classroom_audio \
    vad.save_audios=true \
    vad.audio_save_dir=generated/chunked_audios
```

Important configuration lives in `conf/vad/vad.yaml`:

| Key | Meaning |
|-----|---------|
| `vad.threshold` | Silero speech probability threshold |
| `vad.max_merge_gap` | Maximum gap (s) to merge adjacent speech regions |
| `vad.min_speech_ratio` | Minimum fraction of speech inside a kept segment |
| `vad.min_speech_duration` / `vad.min_non_speech_duration` | Duration filters (s) |
| `vad.max_segment_duration` | Split segments longer than this |
| `vad.overlap` | Overlap (s) applied when splitting long segments |
| `vad.save_audios` | Whether to also write the cut WAVs |
|Paths:|
| `vad.audio_root`       | Directory containing the full per-video WAV recordings |
| `vad.json_save_dir`    | Directory where VAD segment metadata JSON files are written |
| `vad.annotation_root`  | Directory where the generated annotation CSVs are written |
| `vad.audio_save_dir`   | Directory where chunked WAV files are written when `vad.save_audios=true` |
