# `tea.noise`

Noise processing, denoising, noise extraction, and noise augmentation.

This module contains functionality related to the acoustic noise present in the
classroom recordings and methods for modifying or analysing it.

## Responsibilities

The module may contain:

- DeepFilterNet denoising
- Custom spectral subtraction
- Noise-segment extraction from classroom recordings
- Noise contamination of benchmark audio using NoiseAugmentor during model RETRAINING.
- Room impulse response (RIR) augmentation for benchmark audio used to augment minority-class samples when FINE-TUNING.
- Other noise-related experiments

**Note**: Remember augmentation is used in two different contexts:

- First is to retrain the mtkd model by contaminating portions of the benchmark audio with noise.
- Second is to augment minority-class samples (such as anger and sadness) in the classroom dataset for finetuning.

The exact functionality will be added incrementally as the experiments are
implemented.

## Public API

```python
from tea.noise import Denoiser, SpectralSubtractor, NoiseAugmentor, RIRAugmentor, extract_noise_pool

denoiser = Denoiser(cfg)
clean = denoiser.enhance("audio.wav", output_path="clean.wav", save=True)

subtractor = SpectralSubtractor()
clean = subtractor.subtract(speech=noisy_chunk, noise=non_speech_reference)

augmentor = NoiseAugmentor(noise_path="data/noise/full_noise.wav", snr_min=15, snr_max=30)
augmented_waveform, was_contaminated, snr_db = augmentor.augment(waveform)

noise_paths = extract_noise_pool(annotation_root="generated/annotations", video_ids=[...])

# Optional
rir = RIRAugmentor("data/rir/openslr28", room_sizes=("smallroom", "mediumroom"))
reverbed = rir.augment(speech_waveform)
```

`RIRAugmentor` is used (opt-in only, `classroom.use_rir=true`) by
`tea.classroom.fesc.contaminate_fesc`, applied to FESC audio before noise
mixing. It lives here, not in `tea.classroom`, because it's a general
acoustic-condition transform like the other three classes above.

## CLI

Denoising:

```bash
tea denoise
```

For example:

```bash
tea denoise noise.method=deepfilter
```

or:

```bash
tea denoise noise.method=spectral
```

## Noise extraction

```bash
tea extract-noise
```

The exact commands may change as the pipeline is implemented.

## Reproducibility

Noise processing should be configured through Hydra rather than hard-coded
parameters so that the exact processing conditions can be reproduced.

For example:

```bash
tea denoise noise.method=deepfilter noise.atten_lim_db=15
```

## Outputs

Depending on the operation, generated files may include:

- enhanced audio
- extracted noise segments
- noise statistics
- augmented audio
- metadata describing the applied transformation

Generated artefacts belong under generated/ and will not normally be
committed to Git.

## Status

Spectral subtraction and the noise-augmented retraining path exist as
classes but aren't wired to a CLI command yet since they need `tea.analysis`
(to select per-chunk noise references) and `tea.mtkd` (to actually run a
training loop) respectively. `RIRAugmentor` is fully wired, opt-in, via
`tea.classroom.fesc.contaminate_fesc`.
