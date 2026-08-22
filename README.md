# summer-internship-2026

## Teacher Emotion Analysis

Voice analysis of classroom teacher speech using Speech Emotion Recognition (SER) model outputs.

This repository contains the internship pipeline for:

- VAD-based audio chunking
- speech denoising
- ASR
- multilingual MTKD emotion recognition
- classroom-level evaluation
- confidence estimation
- feature probes
- noise analysis
- temporal analysis

The project separates reusable software from experiment-specific recipes. Reusable functionality lives under `src/tea/`, while `recipes/` contains the commands and scripts used to run and reproduce specific experiments.

## Quick Start

### 1. Clone the repository

```bash
git clone <your-repo-url> summer-internship-2026
cd summer-internship-2026
```

### 2. Create the Conda environment

The repository provides a Conda environment containing the dependencies required by the project.

```bash
conda env create -f environment.yml
conda activate tea
```

### 3. Install the project

Install the Python package in editable mode:

```bash
pip install -e .
```

Editable installation means that changes made under src/tea/ are immediately available without reinstalling the package.

### 4. Check the CLI

```bash
tea --help
```

The tea command provides the main interface to the project.

For example:

```bash
tea chunk
tea denoise
tea sentiment
tea infer-mtkd
tea evaluate-classroom
```

## Running the Pipeline

The main interface for running individual pipeline stages is:

```bash
tea <command> [Hydra overrides]
```

For example:

```bash
tea chunk
```

Configuration can be overridden directly from the command line:

```bash
tea chunk vad.max_segment_duration=8.0
```

Multiple overrides can be supplied:

```bash
tea chunk \
    vad.max_segment_duration=8.0 \
    vad.overlap=1.0 \
    paths.audio_root=/my/audio
```

For example:

```bash
tea chunk vad.overlap=1.0
```

overrides the configuration:

```yaml
vad:
  overlap: 1.0
```

for that run only.

The tea command selects the requested pipeline operation, while Hydra loads the project configuration and applies any command-line overrides.

## Recipes

The `recipes/` directory contains experiment-specific recipes.

A recipe records the commands and settings used for a particular experiment or result. Recipes may be shell scripts, SLURM submission scripts, or other scripts used to run experiments.

For example:

```note
recipes/
├── vad/
│   ├── baseline.sh
│   └── attenuation_sweep.sh
├── mtkd/
│   └── train.sh
├── classroom/
│   └── evaluate.sh
└── noise/
    └── analysis.sh
```

A recipe might contain a parameter sweep such as:

```bash
tea chunk noise.atten_lim_db=0
tea chunk noise.atten_lim_db=5
tea chunk noise.atten_lim_db=10
tea chunk noise.atten_lim_db=15
```

This makes the exact commands used for an experiment explicit and reproducible.

Recipes can also contain SLURM configuration for running experiments on Triton:

```sh
#!/bin/bash
#SBATCH --job-name=vad
#SBATCH --output=logs/slurm_%j.out

tea chunk noise.atten_lim_db=0
tea chunk noise.atten_lim_db=15
```

They can then be submitted with:

```bash
sbatch recipes/noise/attenuation_sweep.sh
```

Recipes are kept under version control so that the commands used to obtain reported results remain available even if the reusable software changes later.

## Configuration

Configuration is managed with Hydra.

The configuration hierarchy lives under:

`conf/`

A typical configuration is structured into sections such as:

```yaml
paths:
  ...

vad:
  ...

asr:
  ...

mtkd:
  ...

confidence:
  ...
```

Configuration values can be overridden directly from the command line:

```bash
tea chunk vad.threshold=0.7
```

or:

```bash
tea evaluate-classroom paths.prediction_root=generated/predictions
```

Hydra creates a separate run directory for each execution. Generated Hydra logs and run metadata are stored under:

`hydra-logs/`

These generated files are not committed to Git.

See `conf/README.md` and [`docs/reproducibility.md`](docs/reproducibility.md) for more information about the configuration structure.

## Repository Layout

| Directory | Description |
|---|---|
| `src/tea/` | Reusable Python package and library code |
| `recipes/` | Experiment-specific recipes and SLURM scripts |
| `conf/` | Hydra configuration files |
| `notebooks/` | Exploratory notebooks |
| `docs/` | Experiment documentation and reproducibility notes |
| `paper/` | LaTeX source, figures, and paper material |
| `generated/` | Generated/intermediate artefacts |
| `final_models/` | Final/reusable model checkpoints |
| `data/` | Links to external datasets, audio, and video |
| `logs/` | Application and SLURM logs |
| `hydra-logs/` | Hydra run metadata and configuration |
| `tmp/` | Temporary files |

### `src/tea/`

Contains reusable software that is not tied to one particular experiment or dataset.

Code here should be written so that it can be reused across experiments and, where appropriate, future projects.

### `recipes/`

Contains experiment-specific commands and scripts.

Recipes are allowed to be specific to this project and may contain fixed parameters, parameter sweeps, or SLURM submission settings.

### `conf/`

Contains the Hydra configuration used by the software and recipes.

### `generated/`

Contains intermediate results that can be regenerated from the original data using the appropriate recipes.

These files are not committed to Git.

### `final_models/`

Contains final model checkpoints that have reuse value.

These files are generally not stored directly in Git. Each model should be accompanied by documentation describing how it was produced and how it should be used.

### `data/`

Contains links to external corpora and datasets rather than copies of large datasets.

Machine-specific paths should preferably be handled through symbolic links or configuration rather than hard-coded into the source code.

### `logs/`

Contains application and SLURM logs.

Logs are not committed to Git.

### `tmp/`

Contains temporary files that can be safely deleted during development.

## Reproducing Results

The recommended workflow for reproducing a result is:

- Clone the repository.
- Create the Conda environment.
- Install the project with ```bash pip install -e .```.
- Obtain or link the required datasets and model checkpoints.
- Configure the relevant paths.
- Locate the corresponding recipe under `recipes/`.
- Run the recipe or submit it to SLURM/Triton.
- Inspect the generated results and Hydra run configuration.

For example:

```bash
bash recipes/vad/attenuation_sweep.sh
```

or, for a SLURM recipe:

```bash
sbatch recipes/vad/attenuation_sweep.sh
```

See `docs/reproducibility.md` for the complete reproduction procedure.

## Large Artefacts

Large files such as model checkpoints, embeddings, generated JSON files, and datasets are not stored directly in the repository.

The following directories may contain placeholder files describing expected artefacts:

```list
final_models/
generated/
data/
```

Each placeholder should document:

- the expected filename
- how to use an existing artefact
- how to regenerate the artefact when possible

## Report Mapping

| Report item | Command / recipe |
|---|---|
| Classroom WAR / UAR / confusion | `tea evaluate-classroom` |
| Confidence estimation | `tea confidence` |
| Child-speech probe | `tea probe-child-speech` |
| Feature fusion | `tea probe-feature-fusion` |
| Noise distribution analysis | `tea noise-analysis` |
| Temporal consistency | `tea temporal` |
| VAD segment statistics | `tea chunk` |


The exact commands and parameter settings used for reported experiments should be recorded in the corresponding recipes under `recipes/`.

## Development

After installing the project with:

```bash
pip install -e .
```

changes to the source code under src/tea/ are immediately reflected when running:

```bash
tea <command>
```

The project does not need to be reinstalled after every source-code change.

For quick debugging, individual Python modules can also be executed directly when appropriate, but the normal project interface is the tea CLI and the experiment recipes.

## Git and Reproducibility

Software and recipes are kept under version control.

Generated data, logs, temporary files, and large model artefacts are not normally committed.

Recipes should record the commands and configuration needed to reproduce important experiments.

When possible, experiment runs should also record the Git commit from which the software was executed. This makes it possible to identify the exact version of the code used to produce an old result.

## Status

The repository is currently a scaffold. Modules are being implemented incrementally.

See the README.md inside each major module for its current implementation status.


```text
                  ┌─────────────────────┐
                  │     src/tea/        │
                  │                     │
                  │ reusable software   │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │       tea CLI       │
                  │                     │
                  │ tea chunk ...       │
                  │ tea evaluate ...    │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │      recipes/       │
                  │                     │
                  │ exact experiments   │
                  │ parameter sweeps    │
                  │ SLURM jobs          │
                  └─────────────────────┘
