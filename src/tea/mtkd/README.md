# `tea.mtkd`

Multilingual Teacher Knowledge Distillation (MTKD) student: train,
evaluate, infer, calibrate, extract embeddings.

## Public API

```python
from tea.mtkd import Trainer, Inferencer, Calibrator, EmbeddingExtractor, evaluate_student

trainer = Trainer(cfg)
ckpt_path = trainer.train(linguality="Multilingual", language="FI", session=8)

inferencer = Inferencer(cfg, checkpoint=ckpt_path)
results = inferencer.run("path/to/video_folder")  # or a single file
Inferencer.save_grouped(results, "generated/predictions/MTKD_run.json")

results = evaluate_student(cfg, linguality="Multilingual", language="FI", session=8)  # Tables 4/5

Calibrator(cfg).run(linguality="Multilingual", language="FI", session=8)  # Table 14

extractor = EmbeddingExtractor(cfg, checkpoint=ckpt_path)
extractor.process_video_folder("path/to/video_folder", output_dir="generated/embeddings")
```

## CLI

```bash
tea train-mtkd mtkd.linguality=Multilingual mtkd.language=FI mtkd.session=8
tea train-mtkd mtkd.linguality=Multilingual mtkd.language=FI mtkd.session=8 mtkd.mode=finetune
tea train-mtkd mtkd.linguality=Multilingual mtkd.language=FI mtkd.session=8 mtkd.noise.type=full mtkd.noise.snr_min=15 mtkd.noise.snr_max=30
tea infer-mtkd mtkd.infer.input=path/to/videos mtkd.infer.output=generated/predictions/run.json
tea evaluate-mtkd mtkd.linguality=Multilingual mtkd.language=FI mtkd.session=8
tea calibrate mtkd.linguality=Multilingual mtkd.language=FI mtkd.session=8
tea extract-embeddings mtkd.embeddings.source=videos mtkd.embeddings.input_dir=path/to/videos
```

## Submodules

| Module | Ported from | Notes |
|---|---|---|
| `model.py` | `model.py` | |
| `losses.py` | `losses.py` | `KLDivLoss(reduction="mean")`, temperature, lambda_param **kept exactly unchanged** |
| `frozen_teachers.py` | `teachers.py` | named to avoid clashing with top-level `tea.teachers` |
| `data.py` | `data.py` | single-language loaders live in `tea.teachers`, not here |
| `engine.py` | `engine.py` | |
| `utils.py` | `utils.py` | shared with `tea.teachers`; also holds classroom-only weighting helpers used by `tea.classroom` |
| `train.py` | `train.py` + `train_modified.py` | merged: noise augmentation is now `augmentor=None` by default, not a separate script |
| `infer.py` | `infer.py` + `infer_per_fold.py` + `infer_whole.py` | merged into one `Inferencer`; inference-time temperature is optional (`None` by default), was previously hard-coded to `1.2972` |
| `evaluate.py` | `evaluate.py` | teacher-free test-split evaluation (Tables 4/5) |
| `calibrate.py` | `calibrate.py` | Table 14 |
| `embeddings.py` | `extract_embeddings.py` | |

## Notes

- All hyperparameters (temperature, lambda_param, cosine_temp, LR, batch
  size, teacher checkpoint paths) live in `conf/mtkd/mtkd.yaml`.
- `tea.classroom` depends on this package for `model.py` (checkpoint
  loading) and `utils.py` (weighting helpers). It is fine-tuning an
  MTKD-trained checkpoint, not a separate model architecture.
