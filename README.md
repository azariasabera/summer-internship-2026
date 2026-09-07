# summer-internship-2026

## Teacher Emotion Analysis

Voice analysis of classroom teacher speech using Speech Emotion Recognition (SER) model outputs.

This repository implements the full internship pipeline:

- VAD-based audio chunking and annotation preparation
- optional speech denoising and noise-pool extraction
- Whisper ASR (transcription + translation)
- multilingual MTKD emotion recognition (train / evaluate / calibrate / infer)
- classroom-level LOTO fine-tuning with FESC
- classroom evaluation, temporal analysis, noise-condition tables
- confidence / reliability estimation
- child-speech and feature-fusion probes

Reusable library code lives under `src/tea/`. Experiment-specific commands and scripts live under `recipes/`. Configuration is managed by Hydra under `conf/`.

## Quick Start

### 1. Clone

```bash
git clone https://github.com/azariasabera/summer-internship-2026.git
cd summer-internship-2026
```

### 2. Environment

```bash
conda config --set channel_priority flexible   # if it was strict
conda env create -f environment.yml
conda activate tea
```

### 3. Install the package (editable)

```bash
pip install -e .
```

### 4. Verify the CLI

```bash
tea --help
```

## Running the pipeline

All pipeline stages are exposed as sub-commands of the single `tea` entry point:

```bash
tea <command> [Hydra overrides]
```

Examples:

```bash
tea chunk
tea merge-annotations
tea apply-asr
tea infer-mtkd
tea evaluate-classroom
tea temporal
tea confidence
```

Any configuration value can be overridden on the command line:

```bash
tea chunk vad.max_segment_duration=8.0 vad.overlap=1.0
tea infer-mtkd mtkd.infer.checkpoint=final_models/mtkd_student.pt
```

See `doc/pipeline.md` for the recommended stage order, input/output tables, and full parameter lists.

## Repository layout

| Directory | Description |
|-----------|-------------|
| `src/tea/` | Reusable Python package (VAD, ASR, noise, MTKD, analysis, probes, \ldots) |
| `conf/` | Hydra configuration groups |
| `recipes/` | Shell scripts that reproduce concrete experiments |
| `doc/` | Pipeline description (`pipeline.md`) and reproducibility notes |
| `final_models/` | Final / reusable checkpoints (not stored in Git; place them here) |
| `generated/` | Intermediate artefacts (chunks, annotations, predictions, embeddings) |
| `data/` | Links or mounts to external audio / video / prepared annotations |
| `logs/` | Application and SLURM logs |
| `paper/` | LaTeX source and figures |

Each major package under `src/tea/` contains its own `README.md` that lists the modules, the public CLI commands, and the most important configuration keys.

## Report → code mapping

| Report section | Primary command(s) |
|----------------|--------------------|
| 1 Annotation (VAD chunking, ASR) | `tea chunk`, `tea merge-annotations`, `tea apply-asr` |
| 2.1 Predicting teacher videos (baseline) | `tea infer-mtkd` + `tea evaluate-classroom` |
| 2.2 Temporal analysis | `tea temporal`, `tea emotion-arc` |
| 2.3 Confidence calibration (temperature) | `tea calibrate` |
| 3 Preliminary fine-tuning (LOTO, FESC) | `tea finetune-classroom` |
| 4.1 Noise analysis | `tea noise-analysis`, `tea denoise`, `tea extract-noise` |
| 4.2 Representation / feature analysis | `tea extract-embeddings`, `tea probe-feature-fusion` |
| 4.3 Confidence estimation | `tea confidence` |
| 4.4 Temporal visualisation | `tea emotion-arc`, `tea temporal` |
| Child-speech investigation | `tea probe-child-speech` |

Detailed reproducibility instructions (pre-computed artefacts, existing checkpoints, no re-training) are collected in `doc/reproducibility.md`.

## Recipes

The `recipes/` directory contains ready-to-run shell scripts for the most common workflows. Each sub-folder corresponds to a logical stage or analysis:

```
recipes/
├── vad/
├── asr/
├── noise/
├── mtkd/
├── classroom/
├── analysis/
├── confidence/
└── probes/
```

Run any recipe with:

```bash
bash recipes/analysis/evaluate_classroom.sh
```

## Configuration

All configuration is composed by Hydra from `conf/`. The most frequently overridden groups are:

- `paths` – filesystem locations
- `vad` – chunking parameters
- `asr` – Whisper settings
- `mtkd` – student model / checkpoint / inference
- `analysis` – prediction JSON, smoothing, noise conditions
- `confidence` / `probes` – reliability and probe settings

See `conf/README.md` for the full group list.

## Development

After `pip install -e .` any change under `src/tea/` is immediately visible to the `tea` CLI. Generated artefacts, large models and logs are intentionally kept out of Git.
