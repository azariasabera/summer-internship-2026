# `tea.noise`

Denoising, noise-pool extraction and noise augmentation for classroom and benchmark audio.

## Modules

| File | Role |
|------|------|
| `denoiser.py` | DeepFilterNet wrapper. Expects a directory of WAVs; writes denoised copies. |
| `spectral_subtraction.py` | Classical spectral-subtraction denoiser (optional alternative). |
| `extraction.py` | Builds a non-speech noise pool from annotated classroom videos (used later by FESC / augmentation). |
| `augmentation.py` | `NoiseAugmentor` – mixes background noise at controlled SNR ranges into training utterances. |
| `rir.py` | Room-impulse-response convolution helpers used by classroom fine-tuning (FESC). |
| `__init__.py` | Exposes the CLI entry points `denoise(cfg)` and `extract_noise(cfg)`. |

## Commands

- `tea denoise` – run DeepFilterNet (or spectral subtraction) over a configured audio root.
- `tea extract-noise` – harvest non-speech regions and build the noise pool used by later augmentation stages.

## CLI

```bash
tea denoise # default method=deepfilter
tea denoise noise.method=deepfilter noise.atten_lim_db=15
tea denoise noise.method=spectral

tea extract-noise # here save_audio=false so only saves noise_pool.json metadata
tea extract-noise noise.extraction.save_audio=true noise.extraction.save_dir=generated/noise
```

Important configuration lives in `conf/noise/noise.yaml`:

| Key | Meaning |
|-----|---------|
| `noise.method` | `deepfilter` or `spectral` |
| `noise.atten_lim_db` | Attenuation limit for DeepFilterNet |
| `noise.extraction.*` | Filters applied when harvesting non-speech regions |
| `noise.augment.noise_path` | Path to the pooled noise file used by `NoiseAugmentor` |
| `noise.augment.snr_min` / `snr_max` | SNR range for training-time contamination |
