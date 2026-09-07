# Reproducibility

This document maps every major section of the internship progress report
(`internship_report.pdf`) onto the corresponding `tea` commands and
configuration.  The goal is to re-obtain the reported tables and figures
**without re-training models and without re-running ASR**.

Assumptions:

- Pre-computed annotation CSVs (with `gt_label`, `transcription`, …) already
  exist under `data/annotations/` (or have been produced by Stage 1).
- Final student checkpoints live under `final_models/`.
- Prediction JSONs and embeddings can be regenerated with the commands below
  if they are missing.

---

## 1. Annotation

### 1.2 VAD-based chunking

```bash
tea chunk vad.save_audios=true
# or with explicit overrides
tea chunk \
    vad.audio_root=data/classroom_audio \
    vad.save_audios=true \
    vad.save_json_data=true
```

Produces the segment JSONs and the empty annotation CSV skeleton.
The refinement parameters in `conf/vad/vad.yaml` must stay identical to the
values used for the original annotations (see `src/tea/vad/README.md`).

### 1.3–1.5 ASR & annotation workflow

After chunking you should have:

- `generated/chunks/*.json` (segment metadata)
- `generated/annotations/*.csv` (columns: `name`, `duration_sec`, `start`, `end`, `type`)

First merge the hand-labelled columns:

```bash
tea merge-annotations
```

This adds `gt_label`, `confidence` and `overlap` to the CSVs under
`paths.annotation_root` (default `generated/annotations`).

For ASR outputs prefer the pre-computed versions:

```bash
tea apply-asr asr.use_precomputed=true
```

If you really need to re-run Whisper, create the dedicated environment from
`conda-envs/environment_for_asr.yml` and then:

```bash
tea apply-asr asr.use_precomputed=false
```

---

## 2. Result analysis

### 2.1 Baseline student on the benchmark (Tables 4 / 5)

Before touching the classroom videos, evaluate the frozen MTKD student on the
held-out benchmark test split (FESC / IEMOCAP / CaFE).  This reproduces the
numbers reported in Tables 4–5.

```bash
# Evaluate the Multilingual FI session-6 student (most common checkpoint)
tea evaluate-mtkd \
    mtkd.linguality=Multilingual \
    mtkd.language=FI \
    mtkd.session=6

# or with an explicit checkpoint
tea evaluate-mtkd \
    mtkd.eval_checkpoint=final_models/mtkd/MTKD_Multilingual_FI_S6.pth
```

### 2.1 Predicting the teacher-videos with the baseline model

Run the same frozen student on the classroom chunks and write the prediction
JSON that all later analysis commands consume:

```bash
tea infer-mtkd \
    mtkd.infer.checkpoint=final_models/mtkd/MTKD_Multilingual_FI_S6.pth \
    mtkd.infer.input=generated/chunked_audios \
    mtkd.infer.output=generated/mtkd_inference/infer_result.json \
    mtkd.infer.eval=true
```

(`mtkd.infer.eval=true` also prints a quick UAR/WAR on the classroom data.)

Classroom-level metrics (WAR / UAR / confusion, child-speech & confidence
breakdowns – report Tables 6–13):

```bash
tea evaluate-classroom \
    analysis.mtkd_json=generated/mtkd_inference/infer_result.json \
    analysis.annotation_dir=generated/annotations
```

Add the 3-class polarity view if needed:

```bash
tea evaluate-classroom \
    analysis.mtkd_json=generated/mtkd_inference/infer_result.json \
    analysis.three_class=true
```

### 2.2 Temporal analysis

```bash
tea temporal \
    analysis.mtkd_json=generated/mtkd_inference/infer_result.json
```

### 2.3 Acoustic-by-emotion summaries (duration / loudness / speaking-rate)

```bash
tea acoustic-by-emotion \
    analysis.mtkd_json=generated/mtkd_inference/infer_result.json \
    analysis.audio_root=data/classroom_audio
```

### 2.4 Confidence calibration – Temperature scaling (Table 14)

Fit a single scalar temperature on the **benchmark** development set:

```bash
tea calibrate mtkd.use_default_calibrate_ckpt=true
```

The command prints the optimal `T`.  You can later apply it at inference time:

```bash
tea infer-mtkd \
    mtkd.infer.checkpoint=final_models/mtkd/MTKD_Multilingual_FI_S6.pth \
    mtkd.inference_temperature=<T> \
    mtkd.infer.output=generated/mtkd_inference/infer_result_T.json
```

---

## 3. Preliminary fine-tuning

To run head-only fine-tuning (make sure to change `use_class_weight`, `use_confidence_weight`, `augment_fesc`
according to Table 16 in the report):

```bash
tea finetune-classroom \
    classroom.base_checkpoint=final_models/mtkd/MTKD_Multilingual_FI_S6.pth \
    classroom.variant=head_only \
    classroom.use_class_weight=true \
    classroom.use_confidence_weight=true \
    classroom.augment_fesc=false \
    classroom.epochs=5 \
    classroom.lr=1e-4 \
    classroom.batch_size=8
```

To run full fine-tuning (make sure to change `use_class_weight`, `use_confidence_weight`, `augment_fesc`
according to Table 16 in the report):

```bash
tea finetune-classroom \
    classroom.base_checkpoint=final_models/mtkd/MTKD_Multilingual_FI_S6.pth \
    classroom.variant=full \
    classroom.use_class_weight=true \
    classroom.use_confidence_weight=true \
    classroom.augment_fesc=false \
    classroom.epochs=5 \
    classroom.lr=2e-5 \
    classroom.batch_size=8
```

---

## 4. Further analysis

### 4.1 Noise analysis (Table 21)

First produce one prediction JSON per noise / filtering condition:

```bash
# 1) for DeepFilterNet 15 dB
tea denoise \
    noise.method=deepfilter \
    noise.deepfilter.atten_lim_db=15 

tea infer-mtkd \
    mtkd.infer.input=generated/noise/deepfilter/atten_lim_db_15 \
    mtkd.infer.output=generated/mtkd_inference/infer_result_15dB.json
```

```bash
# 2) for DeepFilterNet 0 dB
tea denoise \
    noise.method=deepfilter \
    noise.deepfilter.atten_lim_db=0

tea infer-mtkd \
    mtkd.infer.input=generated/noise/deepfilter/atten_lim_db_0 \
    mtkd.infer.output=generated/mtkd_inference/infer_result_0dB.json
```

```bash
# 3) for spectral subtraction
tea denoise noise.method=spectral 

tea infer-mtkd \
    mtkd.infer.input=generated/noise/spectral \
    mtkd.infer.output=generated/mtkd_inference/infer_result_spectral.json
```

For the noise-contaminated retrain, the checkpoints are already available in final_models/mtkd

```bash
# 4) model retrained with noise in 5-15 dB range
tea infer-mtkd \
    mtkd.infer.checkpoint=final_models/mtkd/MTKD_Multilingual_FI_S6_5.0_15.0dB.pth \
    mtkd.infer.output=generated/mtkd_inference/infer_result_5_15.json
```

```bash
# 5) model retrained with noise in 10-25 dB range
tea infer-mtkd \
    mtkd.infer.checkpoint=final_models/mtkd/MTKD_Multilingual_FI_S6_10.0_25.0dB.pth \
    mtkd.infer.output=generated/mtkd_inference/infer_result_10_25.json
```

```bash
# 6) model retrained with noise in 15-30 dB range
tea infer-mtkd \
    mtkd.infer.checkpoint=final_models/mtkd/MTKD_Multilingual_FI_S6_15.0_30.0dB.pth \
    mtkd.infer.output=generated/mtkd_inference/infer_result_15_30.json
```

Then build the distribution table:

```bash
tea noise-analysis
```

If any of the six above (also the base inference) are missing they will be missing from the table, so debug from that.

### 4.2 Representation and feature analysis

Extract embeddings (required by the probes):

```bash
tea extract-embeddings \
    mtkd.embeddings.checkpoint=final_models/mtkd/MTKD_Multilingual_FI_S6.pth \
    mtkd.embeddings.source=videos \
    mtkd.embeddings.input_dir=generated/chunked_audios \
    mtkd.embeddings.output_dir=generated/embeddings
```

Text-sentiment features:

```bash
tea sentiment
```

Feature-fusion (report Table 23):

```bash
tea probe-feature-fusion \
    probes.feature_fusion.mtkd_json=generated/mtkd_inference/infer_result.json \
    probes.feature_fusion.sentiment_fi_json=generated/sentiment/sentiment_fi.json \
    probes.feature_fusion.embedding_root=generated/embeddings
```

### 4.3 Confidence estimation (reliability nets)

```bash
tea confidence \
    confidence.mtkd_json=generated/mtkd_inference/infer_result.json \
    confidence.sentiment_fi_json=generated/sentiment/sentiment_fi.json \
    confidence.methods=[binary,tcp,temperature]
```

### 4.4 Temporal visualisation

Already covered by the commands in §2.2:

```bash
tea temporal   analysis.mtkd_json=generated/mtkd_inference/infer_result.json
tea emotion-arc analysis.mtkd_json=generated/mtkd_inference/infer_result.json
```

---

## Child-speech probe (used in §2.1.3 and §4)

```bash
tea probe-child-speech \
    probes.child_speech.embedding_root=generated/embeddings \
    probes.annotation_dir=generated/annotations
```
