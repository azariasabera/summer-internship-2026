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

