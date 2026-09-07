# Checks

In this file, I will try to create a cascading check over the commands and make sure the codes' outputs align with what I got during my report, and also that the codes do what I intended them to do, plus they don't have any structural limitation nor errors.

1. Preparation

1.1. Chunking

This one is the very first thing I need to make sure that it works because everything else depends on it, such as my annotation, and duration of chunks, amt of chunks ... etc.

command:

```bash
tea chunk
```

with all parameters (those configurations that affect the chunking):

```bash
tea chunk paths.audio_root=somewhere vad.threshold=0.7
```

if the goal is only to get the chunk audios:

```bash
tea chunk vad.save_audios=true vad.save_json_data=false
```

1.2. Adding my already-created annotation labels into each chunk in the csv

```bash
tea merge-annotations
```

or if the reference annotation directory that contains the csvs per video file is located somewhere than in `paths.prepared_annotation_root`:

```bash
tea merge-annotations paths.prepared_annotation_root=somewhere
```

1.3. Applying ASR for the speech chunks

Here, a speech chunk is defined as a chunk for which I provided a reference annotation. If the `gt_label` column is empty, I have manually verified that the chunk does not contain speech content.

An alternative approach would be to use the `type` column and select rows where type is `speech` and excluding rows where it is `non-speech`. However, this column is based on the automatic chunking system and contains both false positives and false negatives. Therefore, for applying ASR, I use the manually verified `gt_label` annotations instead.

When applying ASR, I noticed that the Whisper transcription and translation outputs were not reproducible when using the main `tea` environment. After investigating the issue, I found that the PyTorch version differed between the environment used to generate the original ASR results and the `tea` environment (torch 2.6.0 versus torch 2.2.2). Since these ASR outputs were also used in downstream analyses, such as calculating speaking rate and generating predictions from a text-sentiment model, and these outputs were subsequently used as handcrafted features in the post-hoc analyses, maintaining consistency is important.

I therefore provide three options for applying ASR:

* Use the main `tea` environment. This is the simplest option, but the resulting transcription and translation outputs may differ from those used in the original experiments.

```bash
tea apply-asr
```

* Use the dedicated `tea-asr` environment. I added a separate environment in `environment_for_asr.yml` containing the library versions used to obtain the original ASR results. This environment should be used when reproducing the ASR outputs from the experiments.

If the environment has not been created yet:

```bash
conda env create -f environment_for_asr.yml
```

Then activate it and run ASR:

```bash
conda activate tea-asr
pip install -e .
tea apply-asr
```

* Use the precomputed ASR results. The configuration option `use_precomputed` allows the transcription and translation columns to be added directly from the already prepared annotation files, without running Whisper again.

```bash
tea apply-asr asr.use_precomputed=true
```

This is my preferred option when the original transcription and translation results are already available and the goal is to reproduce the downstream analyses exactly without rerunning ASR.

2. Extract noise

To get the metadata for the noise pool:

```bash
tea extract-noise
```

To also save randomlly concatenated single WAV of all noises in the pool:

```bash
tea extract-noise noise.extraction.save_audio=true
```

3. Denoise

```bash
tea denoise noise.method=deepfilter noise.deepfilter.atten_lim_db=0
```

```bash
tea denoise noise.method=spectral
```

4. Eval

```bash
tea evaluate-mtkd mtkd.linguality=Multilingual mtkd.language=FI mtkd.session=6
```

5. Infer

```bash
tea infer-mtkd 
```

```bash
tea infer-mtkd mtkd.infer.per_teacher=true 
```

```bash
tea infer-mtkd \
    mtkd.infer.input=path_to_wavs \
    mtkd.infer.checkpoint=path_to_ckpt \
    mtkd.infer.save_output=true \
    mtkd.infer.output=where_to_save \
    mtkd.infer.eval=true \
    mtkd.infer.annotations_dir=paths_to_csvs_that have_gt_label \
    mtkd.infer.per_teacher=false \
    mtkd.infer.batch_size=8
```

6. Train

```bash
tea train-mtkd \
    mtkd.linguality=Multilingual \
    mtkd.language=FI \
    mtkd.session=6 \
    mtkd.epochs=20 \
    mtkd.lr=2e-5 \
    mtkd.batch_size=16
```

```bash
tea train-mtkd \
    mtkd.linguality=Multilingual \
    mtkd.language=FI \
    mtkd.session=6 \
    mtkd.epochs=20 \
    mtkd.lr=2e-5 \
    mtkd.batch_size=16 \
    mtkd.noise.use=true \
    noise.augment.noise_path=use_default_after_running_extract_command \
    mtkd.noise.contam_prob=0.5 \
    mtkd.noise.snr_min=15 \
    mtkd.noise.snr_max=30 
```

7. Calibrate

```bash
tea calibrate \
    mtkd.linguality=Multilingual \
    mtkd.language=FI \
    mtkd.session=6
```

8. Embeddings

```bash
tea extract-embeddings \
    mtkd.embeddings.source=videos \
    mtkd.embeddings.input_dir=generated/chunked_audios \
    mtkd.embeddings.annotations_dir=generated/annotations \
    mtkd.embeddings.output_dir=generated/embeddings \
    mtkd.embeddings.checkpoint=final_models/mtkd/MTKD_Multilingual_FI_S6_.pth
```

```bash
tea extract-embeddings \
    mtkd.embeddings.source=fesc \
    mtkd.embeddings.sessions=all \
    mtkd.embeddings.splits=[train,dev,test] \
    mtkd.embeddings.layers=[3,6,9,12]
```

9. Finetune

```bash
tea finetune-classroom \
    classroom.run.config=B \
    classroom.run.variant=full \
    classroom.run.base_checkpoint=final_models/mtkd/MTKD_Multilingual_FI_S6.pth \
    classroom.epochs=5 \
    classroom.lr=2e-5 \
    classroom.batch_size=8
```

```bash
tea finetune-classroom \
    classroom.run.config=B \
    classroom.run.variant=head_only \
    classroom.run.base_checkpoint=final_models/mtkd/MTKD_Multilingual_FI_S6.pth \
    classroom.epochs=5 \
    classroom.lr=1e-4 \
    classroom.batch_size=8
```