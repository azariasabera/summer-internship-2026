# `tea.analysis`

Post-hoc evaluation, temporal analysis, noise-condition tables and acoustic-by-emotion summaries of classroom predictions.

## Modules

| File | Role |
|------|------|
| `classroom.py` | Joins MTKD prediction JSONs onto annotation CSVs, computes WAR / UAR / F1 / confusion matrices (overall and per-video), plus accuracy broken down by child-speech presence and annotation confidence. Exposes `evaluate_classroom_cli`. |
| `temporal.py` | Temporal-consistency scores, moving-average / Gaussian smoothing, and emotion-arc plotting. Exposes `temporal_cli` and `emotion_arc_per_video`. |
| `noise.py` | Builds class-distribution tables under different noise / denoising conditions. Exposes `noise_analysis_cli`. |
| `acoustic_by_emotion.py` | Adds loudness / speaking-rate columns and produces box-plots / summary tables stratified by predicted emotion. Exposes `acoustic_by_emotion_cli`. |
| `__init__.py` | Re-exports the public functions and CLI entry points. |

## Commands

- `tea evaluate-classroom` – overall & per-video metrics + child-speech / confidence breakdowns.
- `tea temporal` – temporal consistency scores and smoothed emotion arcs.
- `tea emotion-arc` – per-video emotion-arc plots.
- `tea noise-analysis` – class-distribution tables for noise / denoising conditions.
- `tea acoustic-by-emotion` – acoustic feature summaries stratified by predicted emotion.

## CLI

```bash
tea evaluate-classroom
tea evaluate-classroom analysis.mtkd_json=generated/predictions/MTKD_run.json \
                       paths.annotation_root=generated/annotations

tea temporal analysis.mtkd_json=generated/predictions/MTKD_run.json
tea emotion-arc analysis.mtkd_json=generated/predictions/MTKD_run.json

tea noise-analysis
tea acoustic-by-emotion analysis.mtkd_json=generated/predictions/MTKD_run.json \
                        analysis.audio_root=data/classroom_audio
```

Important configuration lives in `conf/analysis/analysis.yaml` (prediction JSON path, excluded videos, smoothing windows, noise-condition map, etc.).
