# Pipeline command order

Everyday **classroom reproduction** (no training) follows the stages below.
Training commands are listed last; use them on Triton when you need to rebuild
checkpoints under `final_models/`.

Override any path or hyper-parameter with Hydra, e.g.:

```bash
tea chunk paths.audio_root=/scratch/.../classroom_audio
```

---

## Stage 1 — Data preparation

| # | Command | Inputs | Outputs |
|---|---------|--------|---------|
| 1 | `tea chunk` | `paths.audio_root` (WAVs) | `generated/chunks/*.json`, `generated/annotations/*.csv` (structure only) |
| 2 | `tea merge-annotations` | `paths.prepared_annotation_root` (`data/annotations`) + generated CSVs | same CSVs with `gt_label`, `confidence`, `overlap` filled |
| 3 | `tea denoise` *(optional)* | raw or chunked audio | denoised audio (config-dependent) |
| 4 | `tea extract-noise` *(optional)* | annotated videos | noise pool for augmentation |
| 5 | `tea apply-asr` | annotation CSVs + audio | CSVs gain `transcription`, `translation` |
| 6 | `tea sentiment` *(optional)* | transcripts | `generated/sentiment/sentiment_{fi,en}.json` |

**Typical local sequence after you already have hand-annotations:**

```bash
tea chunk
tea merge-annotations
tea apply-asr
```

If VAD boundaries already match your internship CSVs, you can skip `chunk`
and only point `paths.annotation_root` at a working copy of the prepared files
(or run `merge-annotations` which will copy them when no generated CSV exists).

---

## Stage 2 — Inference (frozen student)

Place checkpoints under `final_models/` (see placeholders there).

| # | Command | Inputs | Outputs |
|---|---------|--------|---------|
| 7 | `tea infer-mtkd` | annotation CSVs + `paths.mtkd_student_ckpt` | prediction JSON + `pred_label` / `pred_confidence` on CSVs |
| 8 | `tea extract-embeddings` *(optional)* | audio + student | `generated/embeddings/<video>/` |

```bash
tea infer-mtkd
# optional, needed for child-speech probe:
tea extract-embeddings
```

---

## Stage 3 — Analysis / probes / confidence

All of these assume Stage 1–2 outputs exist.

| # | Command | What it produces |
|---|---------|------------------|
| 9 | `tea evaluate-classroom` | WAR, UAR, confusion, child-speech & annotation-confidence breakdowns |
| 10 | `tea temporal` | temporal consistency scores + smoothed emotion arcs |
| 11 | `tea noise-analysis` | noise filtering / contamination distribution tables |
| 12 | `tea acoustic-by-emotion` | acoustic box-plots by predicted emotion |
| 13 | `tea confidence` | binary / TCP / temperature LOTO tables (AUROC, ECE, …) |
| 14 | `tea probe-child-speech` | child-speech logistic probe metrics |
| 15 | `tea probe-feature-fusion` | feature-fusion experiment tables |

```bash
tea evaluate-classroom
tea temporal
tea confidence
tea probe-child-speech
tea probe-feature-fusion
```

---

## Stage 4 — Training (Triton / multi-GPU)

Not required if `final_models/` already contains the report checkpoints.

| Command | Purpose |
|---------|---------|
| `tea train-teacher` | Monolingual EN/FI/FR teachers |
| `tea train-mtkd` | Multilingual student (incl. noise-augmented runs via config) |
| `tea finetune-classroom` | LOTO classroom fine-tune configs A–F |

Use the shell/SLURM scripts under `recipes/` for the exact flags that matched
the report.

---

## Stage 5 — Checkpoint evaluation helpers

| Command | Purpose |
|---------|---------|
| `tea evaluate-mtkd` | Benchmark-split evaluation of a student `.pth` |
| `tea calibrate` | Temperature / bias calibration |

---

## Notebooks

Exploratory notebooks that call the same APIs live under `notebooks/`.
They mirror Stages 1–3 (no training). See `notebooks/README.md`.

---

## Logs

| Directory | Contents |
|-----------|----------|
| `logs/` | SLURM / application text logs |
| `hydra-logs/` | Hydra resolved configs per run (gitignored) |
