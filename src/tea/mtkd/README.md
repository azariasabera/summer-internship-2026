# `tea.mtkd`

Multilingual Teacher Knowledge Distillation (MTKD) student model: training, evaluation, calibration, inference and embedding extraction.

## Modules

| File | Role |
|------|------|
| `model.py` | Student architecture (Wav2Vec2 encoder + classification head). |
| `losses.py` | Distillation loss (KL + CE) with temperature. |
| `frozen_teachers.py` | Loads and freezes the monolingual teacher checkpoints used as distillation targets. |
| `data.py` | Multi-language DataLoaders. |
| `engine.py` | Train / validate step helpers. |
| `train.py` | Full training entry point (`train_mtkd_cli`). |
| `evaluate.py` | Checkpoint evaluation on held-out benchmark splits (`evaluate_mtkd_cli`). |
| `calibrate.py` | Temperature calibration of a trained student (`calibrate_cli`). |
| `infer.py` | Batch inference that writes `pred_label` / probability columns (or a nested JSON) for classroom chunks (`infer_mtkd_cli`). |
| `embeddings.py` | Extracts pooled Wav2Vec2 embeddings for later probes (`extract_embeddings_cli`). |
| `utils.py` | Shared helpers (checkpoint loading, weighting, etc.). |

## Commands

- `tea train-mtkd` – train a multilingual student (GPU / Triton).
- `tea evaluate-mtkd` – evaluate a student checkpoint on benchmark splits.
- `tea calibrate` – fit temperature (and optional bias) on a validation set.
- `tea infer-mtkd` – run a frozen student on classroom annotation CSVs / audio and write predictions.
- `tea extract-embeddings` – dump pooled embeddings for the probe modules.

## CLI

```bash
# Training
tea train-mtkd # this will not work since `linguality`, `language`, and `session` have to be specified.
tea train-mtkd \
    mtkd.linguality=Multilingual \
    mtkd.language=FI \
    mtkd.session=6 \
    mtkd.batch_size=16 \
    mtkd.lr=2e-5 \
    mtkd.epochs=20

# Training with additive noise
tea train-mtkd \
    mtkd.linguality=Multilingual \
    mtkd.language=FI \
    mtkd.session=6 \
    mtkd.batch_size=16 \
    mtkd.lr=2e-5 \
    mtkd.epochs=20 \
    mtkd.noise.use=true \
    mtkd.noise.contam_prob=0.5 \
    mtkd.noise.snr_min=10 \
    mtkd.noise.snr_max=30

# Evaluation & calibration
tea evaluate-mtkd mtkd.linguality=Multilingual mtkd.language=FI mtkd.session=6 # if there is a model trained with that configuration in generated/checkpoints
tea evaluate-mtkd mtkd.checkpoint=final_models/mtkd/MTKD_Multilingual_FI_S6.pth # or
tea evaluate-mtkd mtkd.use_defualt_eval_ckpt=true # or

tea calibrate mtkd.checkpoint=final_models/mtkd/MTKD_Multilingual_FI_S6.pth

# Inference on classroom data (uses existing checkpoint)
tea infer-mtkd
tea infer-mtkd mtkd.infer.checkpoint=final_models/mtkd/MTKD_Multilingual_FI_S6.pth \
               mtkd.infer.annotations_dir=generated/annotations \
               mtkd.infer.output=generated/predictions/MTKD_run.json \
               mtkd.infer.batch_size=16

# Embeddings for probes
tea extract-embeddings \
    mtkd.embeddings.source=video \
    mtkd.embeddings.checkpoint=final_models/mtkd/MTKD_Multilingual_FI_S6.pth
```

Important configuration lives in `conf/mtkd/mtkd.yaml`. Checkpoint paths, temperature, class order and noise-augmentation ranges are all controlled from there (or via Hydra overrides).
