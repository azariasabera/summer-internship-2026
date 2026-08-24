# Reproducibility notes

This file is the reproduction map: for each report item, which command
(and recipe) produces it, and any known caveats. It grows as each module
is ported. See module's `README.md`'s "Status" section for what's done.

## Conventions

- Every table below names a **recipe** (`recipes/<module>/<name>.sh`), not
  raw code. Run the recipe, or read it to see the exact `tea` command.
- "Caveats" flags anything ambiguous in the original dump that was resolved
  with a judgment call, so it can be revisited.

## VAD chunking (Section 1.2 of the progress report)

| Item | Recipe | Notes |
|---|---|---|
| Segment boundaries (Table 3, Fig. 1) | `recipes/vad/baseline.sh` | Parameters must stay exactly as in `conf/vad/vad.yaml`; see `src/tea/vad/README.md`. |

**Caveats:** none -- the VAD + refinement algorithm (`Segmenter._refine_segments`) was ported unchanged from `vad_chunking.txt`.

**External/missing pieces still needed for full reproduction of Section 1:**
- `my_annotation.ipynb` (VAD-vs-manual disagreement stats, Table 1) was referenced but not included in the original upload -- only its output tables were. Not yet reconstructed.
- The joystick-comparison analysis (Section 1.7) reads supervisor-provided ground truth from `../Projects/data_teacher_emotions_gt/csv/joystick_*.csv`, which lives outside this repo and outside the classroom corpus. Point `paths.joystick_gt_root` (not yet added to `conf/paths/paths.yaml`) at wherever that data is mounted before running it.

## Noise processing (Section 4.1, Table 21)

| Item | Recipe | Notes |
|---|---|---|
| DeepFilterNet 15dB / 0dB rows | `recipes/noise/attenuation_sweep.sh` | |
| Custom Spectral Subtraction row | *(not yet wired to a recipe)* | `SpectralSubtractor` class exists in `tea.noise`; needs a per-chunk noise-reference selection step from `tea.analysis` before it has a CLI entry point. |
| Retrain 5-15 / 10-25 / 15-30 dB rows | *(not yet wired to a recipe)* | `NoiseAugmentor` class exists in `tea.noise`; needs `tea.mtkd`'s training entry point to actually consume it during a training run. |

**Caveats:** `NoiseAugmentor`'s original convenience constructors hard-coded two Triton paths (`/scratch/work/galamaa1/my_mtkd/noise/{single,full}_noise.wav`). Replaced with `cfg.noise.augment.noise_path` — confirmed the Table 21 "Retrain" rows used `full_noise.wav` (the ~51-minute pooled collection); `conf/noise/noise.yaml`'s default already points at `data/noise/full_noise.wav` to match.

## Not yet ported

Confidence estimation (binary/TCP/temperature), features, probes, and
analysis (classroom evaluation, temporal consistency, noise-distribution
tables) are still scaffold-only. See each module's `README.md` for its
specific status.

## Teachers, MTKD, classroom fine-tuning (Sections 2-3, Tables 4-20)

| Item | Recipe | Notes |
|---|---|---|
| Teacher checkpoints | `recipes/teachers/train_all.sh` | |
| MTKD baseline (Tables 4-13) | `recipes/mtkd/train_baseline.sh` | |
| LOTO fine-tuning configs A-F (Tables 16-20) | `recipes/classroom/finetune_configs.sh` | |
| Confidence calibration (Table 14) | *(no recipe yet)* | `tea calibrate` command exists and works; add a recipe once a specific checkpoint/session is settled on |
| Noise-augmented retraining (Table 21 "Retrain" rows) | *(no recipe yet)* | `tea train-mtkd mtkd.noise.type=full mtkd.noise.snr_min=... mtkd.noise.snr_max=...` works; needs `data/noise/full_noise.wav` in place first |
| Embedding extraction (Section 4.2 probes) | *(no recipe yet)* | `tea extract-embeddings` works; recipe depends on `tea.probes` (next installment) to actually consume the embeddings |

**Consolidations made in this installment** (confirmed near-identical
logic before merging, not silently changed):
- `train.py` + `train_modified.py` -> one `Trainer.train(augmentor=None)`.
- `infer.py` + `infer_per_fold.py` + `infer_whole.py` -> one `Inferencer`
  with `.run()` / `.run_per_teacher()`. `infer.py`'s hard-coded
  `TEMPERATURE = 1.2972` is now `cfg.mtkd.inference_temperature: null`
  (opt-in only), per the note already left in that file.
- `teacher_code.txt`'s dataset loaders/utils were a strict subset of
  `mtkd_code.txt`'s -- `tea.teachers` is now the one canonical copy of both,
  `tea.mtkd` imports from it.

**Known gap, resolved:** RIR (room impulse response) augmentation is now
implemented in `tea.noise.RIRAugmentor`, wired opt-in through
`tea.classroom.fesc.contaminate_fesc`'s `rir` parameter
(`classroom.use_rir=true`). Off by default; the default path reproduces
the original, reported contamination pipeline exactly. Ported from
`mtkd_code.txt` section 1.21, which also included an alternative,
exponential-decay-based augmentation-count strategy
(`compute_augmentation_sizes` in `fesc_contamination2.py`) and a matching
alternate orchestrator (`finetune_classroom2.py`) -- **neither of those
two was ported**, since they were presented as an unreported, exploratory
variant ("other finetuning") and the canonical `cap_multiplier`-based
sizing in `tea.classroom.fesc.contaminate_fesc` is what matches Tables
16-20. Flag if you want that alternate sizing strategy added as a
second, explicitly-optional function alongside the canonical one.

## Text sentiment (feeds Section 4.2/4.3 handcrafted features)

| Item | Recipe | Notes |
|---|---|---|
| FI/EN sentiment JSONs | `tea sentiment` (no dedicated recipe script yet, single command) | Moved from `tea.asr` to `tea.features` -- see that module's README |
