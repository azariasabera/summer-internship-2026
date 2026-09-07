# Checks / Practical command reference

Cascading checklist to verify that each stage works and produces the expected artefacts.
Focus is on commands + the most useful overrides.

---

## 1. Preparation

### 1.1 Chunking
```bash
tea chunk
tea chunk paths.audio_root=somewhere vad.threshold=0.7
tea chunk vad.save_audios=true vad.save_json_data=false   # only audio
```

### 1.2 Merge hand-labelled annotations
```bash
tea merge-annotations
tea merge-annotations paths.prepared_annotation_root=somewhere
```

### 1.3 ASR
Prefer pre-computed results for exact reproducibility:

```bash
tea apply-asr asr.use_precomputed=true
```

If you must re-run Whisper, use the dedicated environment:

```bash
conda env create -f conda-envs/environment_for_asr.yml
conda activate tea-asr
pip install -e .
tea apply-asr asr.use_precomputed=false
```

(The main `tea` environment can give slightly different transcriptions because of torch version differences.)

### 1.4 Optional: noise pool & denoising
```bash
tea extract-noise
tea extract-noise noise.extraction.save_audio=true

tea denoise noise.method=deepfilter noise.deepfilter.atten_lim_db=15
tea denoise noise.method=spectral
```

### 1.5 Optional: text sentiment (needed by probes / confidence)
```bash
tea sentiment
```

---

## 2. Benchmark evaluation & calibration (Stage 3)

```bash
tea evaluate-mtkd mtkd.linguality=Multilingual mtkd.language=FI mtkd.session=6
# or
tea evaluate-mtkd mtkd.eval_checkpoint=final_models/mtkd/MTKD_Multilingual_FI_S6.pth

tea calibrate mtkd.linguality=Multilingual mtkd.language=FI mtkd.session=6
```

---

## 3. Classroom inference (Stage 4)

```bash
tea infer-mtkd

tea infer-mtkd mtkd.infer.per_teacher=true

tea infer-mtkd \
    mtkd.infer.input=generated/chunked_audios \
    mtkd.infer.checkpoint=final_models/mtkd/MTKD_Multilingual_FI_S6.pth \
    mtkd.infer.output=generated/mtkd_inference/MTKD_run.json \
    mtkd.infer.eval=true \
    mtkd.infer.annotations_dir=generated/annotations \
    mtkd.inference_temperature=1.30   # if you calibrated
```

### Embeddings (needed by probes)
```bash
tea extract-embeddings \
    mtkd.embeddings.source=videos \
    mtkd.embeddings.input_dir=generated/chunked_audios \
    mtkd.embeddings.annotations_dir=generated/annotations \
    mtkd.embeddings.output_dir=generated/embeddings \
    mtkd.embeddings.checkpoint=final_models/mtkd/MTKD_Multilingual_FI_S6.pth

# or on a benchmark
tea extract-embeddings \
    mtkd.embeddings.source=fesc \
    mtkd.embeddings.sessions=all \
    mtkd.embeddings.splits=[train,dev,test] \
    mtkd.embeddings.layers=[3,6,9,12]
```

---

## 4. Analysis (Stage 5) – produces most report tables

```bash
tea evaluate-classroom analysis.mtkd_json=generated/mtkd_inference/MTKD_run.json
tea evaluate-classroom analysis.three_class=true

tea temporal     analysis.mtkd_json=generated/mtkd_inference/MTKD_run.json
tea emotion-arc  analysis.mtkd_json=generated/mtkd_inference/MTKD_run.json

tea noise-analysis
tea acoustic-by-emotion analysis.mtkd_json=generated/mtkd_inference/MTKD_run.json
```

---

## 5. Probes & confidence (Stage 6)

```bash
tea confidence
tea probe-child-speech
tea probe-feature-fusion
```

---

## 6. Training (only if you need new checkpoints)

### Teacher
```bash
tea train-teacher teachers.language=FI teachers.session=6
```

### MTKD student
```bash
tea train-mtkd \
    mtkd.linguality=Multilingual \
    mtkd.language=FI \
    mtkd.session=6 \
    mtkd.epochs=20 \
    mtkd.lr=2e-5 \
    mtkd.batch_size=16

# with noise augmentation
tea train-mtkd \
    mtkd.linguality=Multilingual mtkd.language=FI mtkd.session=6 \
    mtkd.noise.use=true \
    mtkd.noise.contam_prob=0.5 \
    mtkd.noise.snr_min=15 mtkd.noise.snr_max=30
```

### Classroom LOTO fine-tune
```bash
tea finetune-classroom \
    classroom.base_checkpoint=final_models/mtkd/MTKD_Multilingual_FI_S6.pth \
    classroom.variant=head_only \
    classroom.use_class_weight=true \
    classroom.use_confidence_weight=true \
    classroom.epochs=5 \
    classroom.lr=1e-4
```

---

## Notes
- Prefer `asr.use_precomputed=true` whenever possible.
- After any new training/fine-tuning, re-run `infer-mtkd` (and optionally `extract-embeddings`) before the analysis commands.
- All paths default to values in `conf/paths/paths.yaml`; override only what you need.
