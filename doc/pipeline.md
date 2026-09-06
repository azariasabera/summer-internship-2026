# Pipeline command order

Below is a summary of the commands in the order they were run for the experiments in the report.
There are a total of 6 stages and in each stage, the commands are listed in the order they were run.

## Stage 1 : Data preparation

| # | Command | Config location & how to change it | Outputs |
|---|---------|------------------------------------|---------|
| 1 | `tea chunk` | **Main config:** `conf/vad/vad.yaml` | `generated/chunks/*.json` (segment metadata)<br>`generated/annotations/*.csv` (structure only – empty `gt_label` / `confidence` / `overlap`)<br>Optionally chunked audio under `generated/chunked_audios/` if `vad.save_audios=true` |
| 2 | `tea merge-annotations` | **Paths:** `conf/paths/paths.yaml`<br>• `prepared_annotation_root` (the hand-labelled CSVs)<br>• `annotation_root` (target CSVs produced by `tea chunk`) | Same CSVs under `generated/annotations/*.csv` now filled with `gt_label`, `confidence`, `overlap` (and any other columns present in the prepared files) |
| 3 | `tea apply-asr` | **Main config:** `conf/asr/asr.yaml` | Annotation CSVs gain `transcription` and `translation` columns |
| 4 | `tea denoise` *(optional)* | **Main config:** `conf/noise/noise.yaml`<br>• `method`: `deepfilter` \| `spectral` | Denoised audio written to the configured save dir (`generated/noise/deepfilter/` or `…/spectral/`) |
| 5 | `tea extract-noise` *(optional)* | **Main config:** `conf/noise/noise.yaml` → `extraction` section | Noise pool metadata (+ optional audio) under `generated/noise/` (used later for FESC / augmentation) |
| 6 | `tea sentiment` *(optional)* | **Main config:** `conf/features/features.yaml` | `generated/sentiment/sentiment_fi.json` and/or `generated/sentiment/sentiment_en.json` |

**This is the order that the following modules should be executed:**

```bash
tea chunk # then
tea merge-annotations # then
tea apply-asr
```

## How to call

All commands accept Hydra overrides of the form `group.key=value`.  
Paths are resolved from `conf/paths/paths.yaml`; change them there or on the CLI.

---

### 1. `tea chunk`

Runs Silero VAD + segment refinement and writes per-video JSON + empty annotation CSVs.

| Config key | What it does | Requiredness |
|------------|--------------|--------------|
| `paths.audio_root` | Directory containing the raw classroom WAVs | **Required** (default: `data/classroom_audio`) |
| `vad.audio_root` | Same as above (aliased) | Optional (inherits `paths.audio_root`) |
| `vad.out_dir` / `vad.save_dir` | Where chunked audio is written when `save_audios=true` | Optional (default: `generated/chunked_audios`) |
| `vad.annotation_root` | Where the empty CSVs are written | Optional (default: `generated/annotations`) |
| `vad.save_audios` | Also export the audio chunks as WAV files | Optional (default: `false`) |
| `vad.save_json_data` | Write the segment-boundary JSONs | Optional (default: `true`) |
| `vad.threshold` | Silero speech probability threshold | Optional (default: `0.5`) |
| `vad.min_speech_duration_ms` | Minimum speech duration kept by Silero | Optional (default: `250`) |
| `vad.min_silence_duration_ms` | Minimum silence duration kept by Silero | Optional (default: `300`) |
| `vad.speech_pad_ms` | Padding added around detected speech | Optional (default: `100`) |
| `vad.sample_rate` | Expected sample rate of the input audio | Optional (default: `16000`) |
| `vad.max_merge_gap` | Max gap (s) between segments that will be merged | Optional (default: `2.0`) |
| `vad.min_speech_ratio` | Minimum fraction of a segment that must be speech | Optional (default: `0.5`) |
| `vad.min_speech_duration` | Minimum final speech segment length (s) | Optional (default: `2.0`) |
| `vad.min_non_speech_duration` | Minimum final non-speech segment length (s) | Optional (default: `2.0`) |
| `vad.max_segment_duration` | Hard upper bound on segment length (s) | Optional (default: `10.0`) |
| `vad.overlap` | Overlap (s) used when splitting long segments | Optional (default: `1.0`) |

```bash
tea chunk
# or with overrides
tea chunk vad.threshold=0.6 vad.max_segment_duration=8.0 paths.audio_root=/my/wavs
```

### 2. `tea merge-annotations`

Copies MY labels (`gt_label`, `confidence`, `overlap`, …) from the prepared CSVs into the CSVs produced by `tea chunk`.

| Config key | What it does | Requiredness |
|------------|--------------|--------------|
| `paths.prepared_annotation_root` | Directory of the hand-labelled CSVs | **Required** (default: `data/annotations`) |
| `paths.annotation_root` | Target CSVs written by `tea chunk` | **Required** (default: `generated/annotations`) |

```bash
tea merge-annotations
# or
tea merge-annotations paths.prepared_annotation_root=/path/to/labels
```

---

### 3. `tea apply-asr`

Runs Whisper (or copies pre-computed text) and adds `transcription` + `translation` columns to the annotation CSVs.

| Config key | What it does | Requiredness |
|------------|--------------|--------------|
| `asr.asr_model` | HuggingFace model id for Whisper | Optional (default: `openai/whisper-large-v3`) |
| `asr.batch_size` | Batch size for Whisper inference | Optional (default: `8`) |
| `asr.language` | Language code passed to Whisper | Optional (default: `fi`) |
| `asr.use_precomputed` | If `true`, skip Whisper and only copy existing text columns | Optional (default: `false`) |
| `paths.annotation_root` | CSVs that will be updated in-place | **Required** (default: `generated/annotations`) |

```bash
tea apply-asr
# or
tea apply-asr asr.language=fi asr.batch_size=16 asr.use_precomputed=false
```

**Note:** For `tea apply-asr`, I have discovered discrepancies in the transcriptions and translations because of environment differences. In order to reproduce the original
ASR outputs, I have included a separate environment file `environment_for_asr.yml` that contains the library versions used to obtain the original ASR results. Use this environment when reproducing the ASR outputs from the experiments. Or for simplicity allow `use_precomputed=true` to copy the original outputs from the prepared annotation files.

---

### 4. `tea denoise` *(optional)*

Denoises the classroom audio with either DeepFilterNet or spectral subtraction.

| Config key | What it does | Requiredness |
|------------|--------------|--------------|
| `noise.json_annotations` | Directory containing the per-video JSON metadata and audio_path | **Required** (default: `generated/chunks`) |
| `noise.csv_annotations` | Directory containing the per-video CSV metadata | **Required** (default: `generated/annotations`) |
| `noise.method` | Denoising backend (`deepfilter` or `spectral`) | Optional (default: `deepfilter`) |
| `noise.deepfilter.atten_lim_db` | Attenuation limit | Optional (default: `15`) |
| `noise.deepfilter.long_audio_threshold_sec` | Threshold above which audio is processed in chunks | Optional (default: `600`) |
| `noise.deepfilter.chunk_sec` | Chunk length when force-chunking | Optional (default: `60`) |
| `noise.deepfilter.overlap_sec` | Overlap between chunks | Optional (default: `0.5`) |
| `noise.deepfilter.force_chunked` | Always process in chunks | Optional (default: `false`) |
| `noise.deepfilter.save_dir` | Output directory for DeepFilterNet | Optional (default: `generated/noise/deepfilter`) |
| `noise.spectral.n_fft` | FFT size for spectral subtraction | Optional (default: `1024`) |
| `noise.spectral.hop` | Hop size | Optional (default: `256`) |
| `noise.spectral.alpha` | Over-subtraction factor | Optional (default: `1.0`) |
| `noise.spectral.beta` | Spectral floor | Optional (default: `0.05`) |
| `noise.spectral.n_noise_chunks` | Number of noise-only chunks used for estimate | Optional (default: `5`) |
| `noise.spectral.save_dir` | Output directory for spectral method | Optional (default: `generated/noise/spectral`) |

```bash
tea denoise
# stronger attenuation
tea denoise noise.method=deepfilter noise.deepfilter.atten_lim_db=0
```

---

### 5. `tea extract-noise` *(optional)*

Builds a noise pool from non-speech regions (used later for retraining with noise augmentation).

| Config key | What it does | Requiredness |
|------------|--------------|--------------|
| `noise.extraction.save_audio` | Also write the extracted noise waveforms | Optional (default: `false`) |
| `noise.extraction.sample_rate` | Sample rate of the extracted noise | Optional (default: `16000`) |
| `noise.extraction.save_dir` | Directory for the noise pool | Optional (default: `generated/noise`) |
| `noise.csv_annotations` | CSVs used to locate non-speech regions | **Required** (default: `generated/annotations`) |
| `noise.json_annotations` | Directory containing the per-video JSON metadata and audio path | **Required** (default: `generated/chunks`) |

```bash
tea extract-noise
# also save the audio
tea extract-noise noise.extraction.save_audio=true
```

---

### 6. `tea sentiment` *(optional)*

Runs a multilingual sentiment model on the transcripts and writes per-language JSON files.
Used in parts where I created modules that use hand-crafted features for prediction, e.g., the child-speech probe and the feature-fusion probe.

| Config key | What it does | Requiredness |
|------------|--------------|--------------|
| `features.sentiment_model` | HuggingFace model id | Optional (default: `cardiffnlp/twitter-xlm-roberta-base-sentiment`) |
| `features.annotation_root` | CSVs that contain the transcripts | Optional (default: `generated/annotations`) |
| `features.sentiment_fi` | Output path for Finnish sentiment | Optional (default: `generated/sentiment/sentiment_fi.json`) |
| `features.sentiment_en` | Output path for English sentiment | Optional (default: `generated/sentiment/sentiment_en.json`) |

```bash
tea sentiment
# or
tea sentiment features.sentiment_model=cardiffnlp/twitter-xlm-roberta-base-sentiment
```

---

## Stage 2: Training

| # | Command | Config location & how to change it | Outputs |
|---|---------|------------------------------------|---------|
| 1 | `tea train-teacher` | **Main config:** `conf/teachers/teachers.yaml` | Monolingual teacher checkpoints under `final_models/teachers/` (e.g. `FT_Monolingual_FI_S6.pth`) |
| 2 | `tea train-mtkd` | **Main config:** `conf/mtkd/mtkd.yaml` | Multilingual (or monolingual) MTKD student checkpoint under `final_models/mtkd/` |
| 3 | `tea finetune-classroom` | **Main config:** `conf/classroom/classroom.yaml` | LOTO fine-tuned checkpoints (optional) under `generated/checkpoints/classroom_finetuned/` + results JSON / summary under `generated/classroom_finetune/` |

## How to call?


### 1. `tea train-teacher`

Fine-tunes a monolingual Wav2Vec2 base model on one language/session of FESC, IEMOCAP or CaFE.

| Config key | What it does | Requiredness |
|------------|--------------|--------------|
| `teachers.language` | Language to train (`EN` / `FI` / `FR`) | **Required** (no default – must be set on CLI) |
| `teachers.session` | Session / fold index for that language | **Required** (no default – must be set on CLI) |
| `teachers.model_ckpt` | HuggingFace base model | Optional (default: `facebook/wav2vec2-base`) |
| `teachers.hyperparams.n_epochs` | Max training epochs | Optional (default: `20`) |
| `teachers.hyperparams.learning_rate` | Learning rate | Optional (default: `1e-5`) |
| `teachers.hyperparams.batch_size` | Batch size | Optional (default: `8`) |
| `teachers.hyperparams.max_duration_sec` | Max audio duration (s) | Optional (default: `20.0`) |
| `teachers.hyperparams.patience` | Early-stopping patience | Optional (default: `6`) |
| `teachers.checkpoint_save_dir` | Directory where the `.pth` is written | Optional (default: `generated/checkpoints/teachers`) |
| `teachers.checkpoint_name_template` | Filename pattern | Optional (default: `FT_Monolingual_{language}_S{session}.pth`) |
| `teachers.splits_root` | JSON split files root | Optional (default: `paths.splits_root`) |
| `teachers.datasets_root` | Resampled audio root | Optional (default: `paths.datasets_root`) |
| `teachers.cafe_root` | CaFE audio root (FR only) | Optional (default: `paths.cafe_root`) |
| `teachers.fesc_old_prefix` | Prefix rewritten in FESC JSON paths | Optional |
| `teachers.iemocap_old_root` | Prefix rewritten in IEMOCAP JSON paths | Optional |
| `teachers.cafe_old_root` | Prefix rewritten in CaFE JSON paths | Optional |

```bash
tea train-teacher teachers.language=FI teachers.session=6
# or with hyper-parameter overrides
tea train-teacher teachers.language=EN teachers.session=2 \
    teachers.hyperparams.learning_rate=5e-6 teachers.hyperparams.n_epochs=15
```

---

### 2. `tea train-mtkd`

Trains the multilingual (or monolingual) MTKD student that distils from the three teachers.

| Config key | What it does | Requiredness |
|------------|--------------|--------------|
| `mtkd.linguality` | `Monolingual` or `Multilingual` | **Required** (no default; must be set on CLI) |
| `mtkd.language` | Target language (`EN` / `FI` / `FR`) | **Required** (no defaul; must be set on CLI) |
| `mtkd.session` | Target session / fold | **Required** (no default; must be set on CLI) |
| `mtkd.mode` | `train` (from scratch) or `finetune` (continue) | Optional (default: `train`) |
| `mtkd.model_ckpt` | HuggingFace base model | Optional (default: `facebook/wav2vec2-base`) |
| `mtkd.epochs` | Override for `hyperparams.n_epochs` | Optional (default: `null` → uses hyperparams) |
| `mtkd.lr` | Override for learning rate | Optional (default: `null` → uses hyperparams) |
| `mtkd.batch_size` | Override for batch size | Optional (default: `null` → uses hyperparams) |
| `mtkd.teacher_sessions` | Which session each teacher was trained on | Optional (defaults shown in yaml) |
| `mtkd.teacher_checkpoints` | Explicit paths to the three teacher `.pth` files | Optional (defaults point at `paths.teacher_ckpt_dir`) |
| `mtkd.checkpoint_save_dir` | Directory where the student is saved | Optional (default: `generated/checkpoints/mtkd`) |
| `mtkd.noise.use` | Enable additive noise augmentation | Optional (default: `false`) |
| `mtkd.noise.contam_prob` | Probability of contaminating a sample | Optional (default: `0.5`) |
| `mtkd.noise.snr_min` / `mtkd.noise.snr_max` | SNR range for contamination | Optional (falls back to `noise.augment` if null) |
| `mtkd.hyperparams.n_epochs` | Max epochs | Optional (default: `10`) |
| `mtkd.hyperparams.learning_rate` | LR used when `mode=train` | Optional (default: `1e-5`) |
| `mtkd.hyperparams.finetune_lr` | LR used when `mode=finetune` | Optional (default: `1e-6`) |
| `mtkd.hyperparams.batch_size` | Batch size | Optional (default: `8`) |
| `mtkd.hyperparams.temperature` | KD softmax temperature | Optional (default: `5.0`) |
| `mtkd.hyperparams.lambda_param` | Weight of the KL term | Optional (default: `0.25`) |
| `mtkd.hyperparams.cosine_temp` | Temperature for teacher weighting | Optional (default: `0.25`) |
| `mtkd.hyperparams.max_duration_sec` | Max audio duration (s) | Optional (default: `20.0`) |
| `mtkd.hyperparams.seed` | Random seed | Optional (default: `42`) |
| `mtkd.hyperparams.weight_decay` | AdamW weight decay | Optional (default: `0.01`) |
| `mtkd.hyperparams.warmup_ratio` | LR warmup ratio | Optional (default: `0.1`) |
| `mtkd.hyperparams.patience` | Early-stopping patience | Optional (default: `5`) |

```bash
tea train-mtkd mtkd.linguality=Multilingual mtkd.language=FI mtkd.session=6
# continue training an existing student
tea train-mtkd mtkd.linguality=Multilingual mtkd.language=FI mtkd.session=6 \
    mtkd.mode=finetune mtkd.lr=1e-6
# with noise augmentation
tea train-mtkd mtkd.linguality=Multilingual mtkd.language=FI mtkd.session=6 \
    mtkd.noise.use=true
```

---

### 3. `tea finetune-classroom`

Leave-one-teacher-out (LOTO) fine-tuning of the MTKD student on the classroom data.  
Supports class weighting, confidence weighting and optional augmentation with noise-contaminated FESC.

| Config key | What it does | Requiredness |
|------------|--------------|--------------|
| `classroom.base_checkpoint` | MTKD student checkpoint to start from | **Required** (default: `paths.mtkd_student_ckpt`) |
| `classroom.variant` | `full` or `head_only` | **Required** (default: `head_only`) |
| `classroom.use_class_weight` | Apply inverse-frequency class weights | Optional (default: `true`) |
| `classroom.use_confidence_weight` | Down-weight low-confidence labels | Optional (default: `true`) |
| `classroom.augment_fesc` | Inject FESC-contaminated samples into the train fold | Optional (default: `false`) |
| `classroom.audio_root` | Directory of chunked classroom audio | Optional (default: `paths.chunk_audio_dir`) |
| `classroom.csv_root` | Directory of annotation CSVs | Optional (default: `paths.annotation_root`) |
| `classroom.model_ckpt` | HuggingFace base model (architecture only) | Optional (default: `facebook/wav2vec2-base`) |
| `classroom.epochs` | Number of fine-tuning epochs | Optional (default: `5`) |
| `classroom.lr` | Learning rate (`null` → 1e-4 for head_only, 2e-6 for full) | Optional (default: `null`) |
| `classroom.batch_size` | Batch size | Optional (default: `8`) |
| `classroom.weight_decay` | AdamW weight decay | Optional (default: `0.01`) |
| `classroom.max_duration_sec` | Max audio duration (s) | Optional (default: `20.0`) |
| `classroom.seed` | Random seed | Optional (default: global `seed`) |
| `classroom.cv` | Cross-validation scheme (`loto` or an integer) | Optional (default: `loto`) |
| `classroom.internal_val_frac` | Fraction of train fold held out for early stopping | Optional (default: `0.0`) |
| `classroom.excluded_videos` | Video IDs always skipped | Optional (default: `[1B3261]`) |
| `classroom.output_dir` | Where results JSON / summary are written | Optional (default: `generated/classroom_finetune`) |
| `classroom.save_model_dir` | Where fine-tuned fold checkpoints are saved (`null` = do not save) | Optional (default: `paths.classroom_finetune_dir`) |
| `classroom.augment_classes` | Emotion classes to augment with FESC | Optional (default: `[sadness, anger]`) |
| `classroom.augment_cap_multiplier` | Cap on number of augmented samples per class | Optional (default: `2.5`) |
| `classroom.noise_extra_videos` | Extra videos used only as noise sources | Optional (default: `[1B3261]`) |
| `classroom.noise_n_sources` | Number of noise sources mixed per sample | Optional (default: `[2, 3]`) |
| `classroom.fesc_output_dir` | Cache directory for contaminated FESC audio | Optional (default: `generated/fesc_contaminated`) |
| `classroom.use_rir` | Also convolve FESC samples with a random RIR | Optional (default: `false`) |
| `classroom.augment_strategy` | `cap` or `adaptive` | Optional (default: `cap`) |
| `classroom.augment_adaptive_alpha` | Alpha used when `augment_strategy=adaptive` | Optional (default: `5.0`) |

```bash
# default report configuration (class + confidence weighting, head_only)
tea finetune-classroom

# explicit overrides matching the report tables
tea finetune-classroom \
    classroom.variant=head_only \
    classroom.use_class_weight=true \
    classroom.use_confidence_weight=true \
    classroom.augment_fesc=false \
    classroom.epochs=5 \
    classroom.lr=1e-4 \
    classroom.batch_size=8 \
    classroom.base_checkpoint=final_models/mtkd/MTKD_Multilingual_FI_S6.pth
```

---

## Stage 3 : Checkpoint evaluation & calibration

| # | Command | Config location & how to change it | Outputs |
|---|---------|------------------------------------|---------|
| 1 | `tea evaluate-mtkd` | **Main config:** `conf/mtkd/mtkd.yaml` (runtime keys: `linguality`, `language`, `session`, `eval_checkpoint`, `use_default_eval_ckpt`) | Printed UAR / WAR / Accuracy + confusion matrix on the held-out benchmark test split (report Tables 4/5). No files written. |
| 2 | `tea calibrate` | **Main config:** `conf/mtkd/mtkd.yaml` (runtime keys: `linguality`, `language`, `session`, `calibrate_checkpoint`, `use_default_calibrate_ckpt`) | Fitted temperature `T`, before/after ECE & confidence, per-class breakdown and confusion matrix (report Table 14). No files written; the scalar `T` is printed for later use at inference. |

## How to call?

### 1. `tea evaluate-mtkd`

Teacher-free evaluation of a trained MTKD student on its labelled benchmark test split (FESC / IEMOCAP / CaFE).  

Checkpoint selection priority (first match wins):

1. `mtkd.linguality` + `mtkd.language` + `mtkd.session` → constructs  
   `{checkpoint_save_dir}/MTKD_{linguality}_{language}_S{session}.pth`
2. `mtkd.eval_checkpoint` (explicit path)
3. `mtkd.use_default_eval_ckpt=true` → uses `mtkd.default_student_checkpoint`

| Config key | What it does | Requiredness |
|------------|--------------|--------------|
| `mtkd.linguality` | `Monolingual` or `Multilingual` (used to locate the checkpoint and build the test set) | **Required** unless `eval_checkpoint` or `use_default_eval_ckpt` is set |
| `mtkd.language` | `EN` / `FI` / `FR` | **Required** unless `eval_checkpoint` or `use_default_eval_ckpt` is set |
| `mtkd.session` | Session / fold index | **Required** unless `eval_checkpoint` or `use_default_eval_ckpt` is set |
| `mtkd.eval_checkpoint` | Explicit path to a `.pth` file | Optional (overrides the constructed path) |
| `mtkd.use_default_eval_ckpt` | Fall back to `mtkd.default_student_checkpoint` | Optional (default: `false`) |
| `mtkd.default_student_checkpoint` | Default student used when `use_default_eval_ckpt=true` | Optional (default: `paths.mtkd_student_ckpt`) |
| `mtkd.checkpoint_save_dir` | Directory that contains the trained students | Optional (default from yaml / paths) |
| `mtkd.model_ckpt` | HuggingFace base model (architecture only) | Optional (default: `facebook/wav2vec2-base`) |
| `mtkd.hyperparams.batch_size` | Evaluation batch size | Optional (default: `8`) |
| `mtkd.hyperparams.max_duration_sec` | Max audio duration (s) | Optional (default: `20.0`) |

```bash
# Evaluate the student trained for Multilingual FI session 6
tea evaluate-mtkd mtkd.linguality=Multilingual mtkd.language=FI mtkd.session=6

# Explicit checkpoint path
tea evaluate-mtkd mtkd.eval_checkpoint=final_models/mtkd/MTKD_Multilingual_FI_S6.pth

# Use the configured default student
tea evaluate-mtkd mtkd.use_default_eval_ckpt=true
```

---

### 2. `tea calibrate`

Fits a single scalar temperature `T` that minimises NLL of `softmax(logits / T)` on the **dev** split of the same benchmark. 
Accuracy is unchanged by construction; only confidence / ECE change.

Checkpoint selection priority (identical logic to `evaluate-mtkd`):

1. `mtkd.linguality` + `mtkd.language` + `mtkd.session`
2. `mtkd.calibrate_checkpoint` (explicit path – linguality/language/session are parsed from the filename)
3. `mtkd.use_default_calibrate_ckpt=true` → `mtkd.default_student_checkpoint`

| Config key | What it does | Requiredness |
|------------|--------------|--------------|
| `mtkd.linguality` | `Monolingual` or `Multilingual` | **Required** unless `calibrate_checkpoint` or `use_default_calibrate_ckpt` is set |
| `mtkd.language` | `EN` / `FI` / `FR` | **Required** unless `calibrate_checkpoint` or `use_default_calibrate_ckpt` is set |
| `mtkd.session` | Session / fold index | **Required** unless `calibrate_checkpoint` or `use_default_calibrate_ckpt` is set |
| `mtkd.calibrate_checkpoint` | Explicit path to a `.pth` file | Optional (overrides the constructed path) |
| `mtkd.use_default_calibrate_ckpt` | Fall back to `mtkd.default_student_checkpoint` | Optional (default: `false`) |
| `mtkd.default_student_checkpoint` | Default student used when `use_default_calibrate_ckpt=true` | Optional (default: `paths.mtkd_student_ckpt`) |
| `mtkd.checkpoint_save_dir` | Directory that contains the trained students | Optional |
| `mtkd.model_ckpt` | HuggingFace base model (architecture only) | Optional (default: `facebook/wav2vec2-base`) |
| `mtkd.hyperparams.batch_size` | Batch size while collecting logits | Optional (default: `8`) |
| `mtkd.hyperparams.max_duration_sec` | Max audio duration (s) | Optional (default: `20.0`) |
| `mtkd.inference_temperature` | (Not used by this command – only relevant at inference time) | Optional (default: `null`) |

```bash
# Calibrate the student trained for Multilingual FI session 6
tea calibrate mtkd.linguality=Multilingual mtkd.language=FI mtkd.session=6

# Explicit checkpoint (linguality/language/session parsed from the filename)
tea calibrate mtkd.calibrate_checkpoint=final_models/mtkd/MTKD_Multilingual_FI_S6.pth

# Use the configured default student
tea calibrate mtkd.use_default_calibrate_ckpt=true
```

**Note:** The fitted temperature is only printed. To apply it later (e.g. in `tea infer-mtkd`) set

```bash
mtkd.inference_temperature=<T>
```

or hard-code `torch.softmax(logits / T, dim=-1)` in any downstream script.

## Stage 4 : Inference (frozen checkpoints)

| # | Command | Config location & how to change it | Outputs |
|---|---------|------------------------------------|---------|
| 1 | `tea infer-mtkd` | **Main config:** `conf/mtkd/mtkd.yaml` → `infer` block + `inference_temperature` | Prediction JSON (grouped by video) under `generated/mtkd_inference/` (or the path you set). Optionally prints UAR/WAR + confusion matrix when annotations are available. |
| 2 | `tea extract-embeddings` *(optional)* | **Main config:** `conf/mtkd/mtkd.yaml` → `embeddings` block | Per-video (or per-benchmark-split) folders under `generated/embeddings/` containing `pooled.npy`, `projected.npy`, `logits.npy`, `probabilities.npy`, `predictions.npy`, `names.npy`, optional layer reps, and `metadata.csv` / `metadata.json`. |

## How to call

### 1. `tea infer-mtkd`

Runs a frozen MTKD student on raw audio (a single file or a whole directory tree of chunks).  
Writes a JSON of per-chunk probabilities grouped by video folder (the format expected by Stage 5 analysis).  
If annotation CSVs are supplied it also prints a confusion matrix, UAR and WAR.

| Config key | What it does | Requiredness |
|------------|--------------|--------------|
| `mtkd.infer.input` | File or directory of audio to run on | **Required** (default: `paths.chunk_audio_dir`) |
| `mtkd.infer.checkpoint` | Student checkpoint to load | Optional (default: `paths.mtkd_student_ckpt` / `mtkd.default_student_checkpoint`) |
| `mtkd.infer.save_output` | Whether to write the prediction JSON | Optional (default: `true`) |
| `mtkd.infer.output` | Path of the prediction JSON | Optional (default: `paths.infer.infer_output_path`) |
| `mtkd.infer.eval` | If `true` and annotations exist, print UAR/WAR + confusion matrix | Optional (default: `true`) |
| `mtkd.infer.annotations_dir` | Directory of `<video_id>.csv` files used for the optional evaluation | Optional (default: `paths.annotation_root`) |
| `mtkd.infer.per_teacher` | Group files by teacher before batching / report mean-per-teacher metrics | Optional (default: `false`) |
| `mtkd.infer.batch_size` | Inference batch size | Optional (default: `8`, falls back to `mtkd.hyperparams.batch_size`) |
| `mtkd.inference_temperature` | Scalar `T` applied as `softmax(logits / T)`. `null` = plain softmax | Optional (default: `null`) |
| `mtkd.model_ckpt` | HuggingFace base model (architecture only) | Optional (default: `facebook/wav2vec2-base`) |
| `mtkd.hyperparams.max_duration_sec` | Max audio duration (s) | Optional (default: `20.0`) |

```bash
# Default: run on the chunked classroom audio, save predictions, print metrics
tea infer-mtkd

# Explicit paths + temperature scaling from Stage 3
tea infer-mtkd \
    mtkd.infer.input=generated/chunked_audios \
    mtkd.infer.checkpoint=final_models/mtkd/MTKD_Multilingual_FI_S6.pth \
    mtkd.infer.output=generated/mtkd_inference/infer_result.json \
    mtkd.inference_temperature=1.30

# Per-teacher metrics
tea infer-mtkd mtkd.infer.per_teacher=true
```

---

### 2. `tea extract-embeddings` *(optional)*

Extracts pooled / projected embeddings, logits, probabilities and (optionally) intermediate transformer-layer representations.  
Used by the child-speech and feature-fusion probes in Stage 6.

Two modes, selected by `mtkd.embeddings.source`:

- `videos` (default) – walk a directory of per-video chunk folders (classroom data).
- `fesc` / `iemocap` / `cafe` – walk the corresponding benchmark sessions & splits.

| Config key | What it does | Requiredness |
|------------|--------------|--------------|
| `mtkd.embeddings.source` | `videos` \| `fesc` \| `iemocap` \| `cafe` | Optional (default: `videos`) |
| `mtkd.embeddings.checkpoint` | Student checkpoint to load | Optional (default: `mtkd.default_student_checkpoint`) |
| `mtkd.embeddings.output_dir` | Root directory for the extracted arrays | Optional (default: `paths.embedding_root`) |
| `mtkd.embeddings.batch_size` | Extraction batch size | Optional (default: `16`) |
| `mtkd.embeddings.layers` | Extra transformer layer indices to mean-pool and save (0–12) | Optional (default: `[]`) |
| **When `source=videos`** | | |
| `mtkd.embeddings.input_dir` | Directory whose sub-folders are per-video chunk sets | **Required** for this mode (default: `paths.chunk_audio_dir`) |
| `mtkd.embeddings.annotations_dir` | Directory of `<video_id>.csv` files (attaches `true_label` to metadata) | Optional (default: `paths.annotation_root`) |
| **When `source=fesc\|iemocap\|cafe`** | | |
| `mtkd.embeddings.sessions` | Session list or `"all"` | Optional (default: `"all"`) |
| `mtkd.embeddings.splits` | Which splits to process | Optional (default: `[train, dev, test]`) |
| `mtkd.model_ckpt` | HuggingFace base model (architecture only) | Optional (default: `facebook/wav2vec2-base`) |
| `mtkd.hyperparams.max_duration_sec` | Max audio duration (s) | Optional (default: `20.0`) |

```bash
# Classroom videos (default)
tea extract-embeddings

# Explicit classroom paths + extra layers
tea extract-embeddings \
    mtkd.embeddings.source=videos \
    mtkd.embeddings.input_dir=generated/chunked_audios \
    mtkd.embeddings.output_dir=generated/embeddings \
    mtkd.embeddings.layers=[9,12]

# Benchmark dataset (e.g. FESC session 6, test split only)
tea extract-embeddings \
    mtkd.embeddings.source=fesc \
    mtkd.embeddings.sessions=[6] \
    mtkd.embeddings.splits=[test]
```

---

## Stage 5 : Analysis / plots / tables

| # | Command | Config location & how to change it | Outputs |
|---|---------|------------------------------------|---------|
| 1 | `tea evaluate-classroom` | **Main config:** `conf/analysis/analysis.yaml` | Printed per-video + overall WAR/UAR/F1 + confusion matrices, accuracy-by-child-speech, accuracy-by-annotation-confidence, child-speech-by-predicted-emotion. Optionally writes `joined_predictions.csv` + `overall_confusion_matrix.csv` under `analysis.output_dir`. |
| 2 | `tea temporal` | **Main config:** `conf/analysis/analysis.yaml` | Printed temporal-consistency score and extended stability metrics (transitions/min, blips, mean run length) per video and pooled. |
| 3 | `tea noise-analysis` | **Main config:** `conf/analysis/analysis.yaml` → `noise_conditions` | Predicted-class distribution across noise conditions (printed + optional CSV). Optionally also reports non-speech prediction distributions. |
| 4 | `tea acoustic-by-emotion` | **Main config:** `conf/analysis/analysis.yaml` | Duration / loudness (dB) / speaking-rate summaries grouped by predicted emotion. Optional CSVs under `analysis.output_dir`. |
| 5 | `tea emotion-arc` | **Main config:** `conf/analysis/analysis.yaml` → `emotion_arc` | Smoothed emotion-probability arc plot for one video, saved under `emotion_arc.plot_save_dir`. |

**Dependencies (what each command actually needs)**

Stage 5 is post-hoc analysis. Every command consumes the prediction JSON
produced by `tea infer-mtkd` (Stage 4) and the annotation CSVs produced by
Stage 1 (`tea chunk` → `tea merge-annotations`, optionally `tea apply-asr`).

| Command | Upstream requirements |
|---------|------------------------|
| `tea evaluate-classroom` | Stage 4 prediction JSON + Stage 1 annotation CSVs (`gt_label`, `confidence`, `overlap`) |
| `tea temporal` | Same as above; annotation CSVs must also contain `start` / `end` |
| `tea noise-analysis` | **One Stage-4 prediction JSON per noise condition** listed under `analysis.noise_conditions`, plus Stage 1 annotation CSVs |
| `tea acoustic-by-emotion` | Stage 4 prediction JSON + Stage 1 annotation CSVs; **loudness** also needs the original full-length audio (`analysis.audio_root`); **speaking-rate** needs `transcription` from `tea apply-asr` |
| `tea emotion-arc` | Stage 4 prediction JSON with per-class probabilities + the chosen video’s annotation CSV |

Nothing in this stage needs the Stage 2/3 training or calibration checkpoints
unless you intentionally re-ran Stage 4 with a different checkpoint or
`mtkd.inference_temperature`.

## How to call?

Shared keys used by almost every Stage-5 command:

| Config key | What it does | Requiredness |
|------------|--------------|--------------|
| `analysis.mtkd_json` | Path to the MTKD prediction JSON produced by `tea infer-mtkd` | **Required** for evaluate-classroom / temporal / acoustic-by-emotion / emotion-arc (default: `paths.infer.infer_output_path`) |
| `analysis.annotation_dir` | Directory of per-video annotation CSVs | Optional (default: `paths.annotation_root`) |
| `analysis.output_dir` | Where optional CSVs / plots are written | Optional (default: `paths.analysis.analysis_output_dir`) |

---

### 1. `tea evaluate-classroom`

Joins the prediction JSON onto the annotation CSVs and reports:

- 4-class WAR / UAR / Macro-F1 / Weighted-F1 + confusion matrix (per-video and pooled)
- optional 3-class polarity comparison (`happiness→positive`, `anger/sadness→negative`, `neutral→neutral`)
- accuracy split by child-speech presence (`overlap > 0`)
- accuracy split by annotation confidence (1 / 2 / 3)
- proportion of child speech by GT vs. predicted emotion

| Config key | What it does | Requiredness |
|------------|--------------|--------------|
| `analysis.mtkd_json` | Prediction JSON | **Required** (default: `paths.infer.infer_output_path`) |
| `analysis.annotation_dir` | Annotation CSVs | Optional (default: `paths.annotation_root`) |
| `analysis.three_class` | Also run the 3-class sentiment evaluation | Optional (default: `false`) |
| `analysis.output_dir` | Write `joined_predictions.csv` + `overall_confusion_matrix.csv` | Optional |

```bash
tea evaluate-classroom

# with 3-class polarity comparison and explicit paths
tea evaluate-classroom \
    analysis.mtkd_json=generated/mtkd_inference/infer_result.json \
    analysis.three_class=true
```

---

### 2. `tea temporal`

Computes temporal-consistency scores and extended stability metrics (transitions per minute, blips, mean run length, GT-vs-pred transition ratio) for every video, both pooled and per-video.  
Also supports gap-threshold merging of near-adjacent speech segments.

| Config key | What it does | Requiredness |
|------------|--------------|--------------|
| `analysis.mtkd_json` | Prediction JSON (probabilities preferred) | **Required** |
| `analysis.annotation_dir` | Annotation CSVs (must contain `start` / `end`) | Optional (default: `paths.annotation_root`) |
| `analysis.output_dir` | Optional directory for any written tables | Optional |

```bash
tea temporal
tea temporal analysis.mtkd_json=generated/mtkd_inference/infer_result.json
```

---

### 3. `tea noise-analysis`

Builds the predicted-class distribution table across multiple noise / filtering / retrain conditions.  
Only speech chunks (those that have a `gt_label`) are counted.  
Optionally also reports what the model predicts on non-speech chunks (sanity check).

| Config key | What it does | Requiredness |
|------------|--------------|--------------|
| `analysis.noise_conditions` | Mapping `{condition_name: prediction_json_path}` | **Required** (defaults already listed in `conf/analysis/analysis.yaml`) |
| `analysis.annotation_dir` | Annotation CSVs used to identify speech chunks | Optional (default: `paths.annotation_root`) |
| `analysis.check_non_speech` | Also print non-speech prediction distributions | Optional (default: `false`) |
| `analysis.output_dir` | Write `noise_condition_distribution.csv` | Optional |

Default conditions (from `conf/analysis/analysis.yaml`):

```yaml
noise_conditions:
  original: ${paths.infer.infer_output_path}
  deepfilternet_15dB: generated/mtkd_inference/infer_result_15dB.json
  deepfilternet_0dB: generated/mtkd_inference/infer_result_0dB.json
  spectral_subtraction: generated/mtkd_inference/infer_result_spectral.json
  retrain_5_15dB: generated/mtkd_inference/infer_result_5_15.json
  retrain_10_25dB: generated/mtkd_inference/infer_result_10_25.json
  retrain_15_30dB: generated/mtkd_inference/infer_result_15_30.json
```

```bash
tea noise-analysis

# override / add a condition on the CLI
tea noise-analysis \
    +analysis.noise_conditions.my_run=generated/mtkd_inference/my_run.json \
    analysis.check_non_speech=true
```

---

### 4. `tea acoustic-by-emotion`

Computes duration, loudness (RMS dB) and speaking-rate (words/min) distributions grouped by **predicted** emotion.  
Loudness requires the original full-length per-video audio (not the chunked files).

| Config key | What it does | Requiredness |
|------------|--------------|--------------|
| `analysis.mtkd_json` | Prediction JSON | **Required** |
| `analysis.annotation_dir` | Annotation CSVs (for speaking-rate matching) | Optional (default: `paths.annotation_root`) |
| `analysis.audio_root` | Directory of full (unchunked) per-video audio files | Optional – loudness is skipped if unset (default: `paths.audio_root`) |
| `analysis.output_dir` | Write per-chunk CSVs | Optional |
| `classroom.excluded_videos` | Video IDs skipped during the join | Optional (default: `[1B3261]`) |

```bash
tea acoustic-by-emotion

# explicit paths
tea acoustic-by-emotion \
    analysis.mtkd_json=generated/mtkd_inference/infer_result.json \
    analysis.audio_root=data/classroom_audio
```

---

### 5. `tea emotion-arc`

Plots the raw + smoothed emotion-probability curves over time for a single video.  
Smoothing can be a moving average or a Gaussian filter.

| Config key | What it does | Requiredness |
|------------|--------------|--------------|
| `analysis.mtkd_json` | Prediction JSON (must contain per-class probabilities) | **Required** |
| `analysis.annotation_dir` | Annotation CSVs (for time stamps + non-speech shading) | Optional (default: `paths.annotation_root`) |
| `analysis.emotion_arc.video` | Video ID to plot | **Required** (default: `1B2251`) |
| `analysis.emotion_arc.plot_save_dir` | Directory where the figure is saved | Optional (default: `paths.analysis.plot_save_dir`) |
| `analysis.emotion_arc.ma.window` | Moving-average window size | Optional (default: `5`) |
| `analysis.emotion_arc.gaussian.sigma` | Gaussian sigma | Optional (default: `2.0`) |
| `analysis.emotion_arc.gaussian.radius` | Gaussian radius (`null` → derived from sigma × truncate) | Optional (default: `null`) |
| `analysis.emotion_arc.gaussian.truncate` | Gaussian truncate factor | Optional (default: `4.0`) |

```bash
tea emotion-arc

# different video + Gaussian smoothing parameters
tea emotion-arc \
    analysis.emotion_arc.video=1B2251 \
    analysis.emotion_arc.gaussian.sigma=1.5
```
