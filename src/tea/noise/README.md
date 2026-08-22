# `tea.noise`

Noise processing, denoising, noise extraction, and noise augmentation.

This module contains functionality related to the acoustic noise present in the
classroom recordings and methods for modifying or analysing it.

## Responsibilities

The module may contain:

- DeepFilterNet denoising
- Custom spectral subtraction
- Noise-segment extraction
- Noise contamination of benchmark audio
- Room impulse response (RIR) augmentation, if used
- Other noise-related experiments

The exact functionality will be added incrementally as the experiments are
implemented.

## Planned components

```text
deepfilter.py
    DeepFilterNet-based enhancement

spectral.py
    Custom spectral subtraction

extraction.py
    Extraction of noise segments from benchmark recordings

rir.py
    Room impulse response / reverberation augmentation
```

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

Scaffold. Implementation is under development.
