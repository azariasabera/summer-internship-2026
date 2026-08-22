# `tea.vad`

Voice Activity Detection (Silero) and custom speech-segment refinement.

This module is responsible for determining **where speech occurs** in classroom
audio and producing the final speech segments used by downstream processing. Following VAD,
the detected segments are refined to achieve appropriate durations. Non-speech regions are
retained and flagged rather than removed. The resulting segments are then used to divide
the audio into smaller chunks for downstream processing.

## Responsibilities

- Voice Activity Detection using Silero
- Speech/non-speech segmentation
- Merging nearby speech regions
- Filtering segments based on speech ratio and duration
- Splitting long segments
- Applying segment overlap
- Producing segment metadata

Denoising and noise augmentation are handled separately by `tea.noise`.

## Critical reproducibility note

The refinement parameters in `conf/vad/default.yaml` **must stay identical** to
the values that generated the original annotations:

- `max_merge_gap = 2.0`
- `min_speech_ratio = 0.5`
- `min_speech_duration = 2.0`
- `min_non_speech_duration = 2.0`
- `max_segment_duration = 10.0`
- `overlap = 1.0`

Changing these parameters changes chunk durations and therefore can affect
downstream results.

## Planned public API

```python
from tea.vad import Segmenter

segmenter = Segmenter(cfg)
segments = segmenter.chunk_vad(audio_path)
```

## CLI

```bash
tea chunk
```

Hydra configuration can be overridden from the command line:

```bash
tea chunk vad.max_segment_duration=8.0
```

## Outputs

Segment metadata is written to:

`generated/chunks/`

## Status

Scaffold. Implementation is under development.
