# `tea.teachers`

Monolingual teacher fine-tuning (used as the teacher models inside MTKD).

## Modules

| File | Role |
|------|------|
| `trainer.py` | Training loop for a single-language Wav2Vec2 / WavLM teacher. Expects a Hydra config; writes checkpoints under the configured run directory. |
| `data.py` | Dataset and DataLoader construction for the monolingual teacher corpora. |
| `metrics.py` | WAR / UAR / confusion helpers used during validation. |
| `__init__.py` | Exposes the CLI entry point `train_teacher(cfg)`. |

## Command

- `tea train-teacher` – fine-tune a monolingual teacher on the configured language/session (normally run on a GPU cluster).

## CLI

```bash
tea train-teacher
tea train-teacher teachers.language=FI teachers.session=6
tea train-teacher teachers.hyperparams.batch_size=16
```

Important configuration lives in `conf/teachers/teachers.yaml`. Checkpoints are written under `generated/checkpoints/teachers` if not explicitly changed using `teachers.checkpoint_save_dir=new-path`.
