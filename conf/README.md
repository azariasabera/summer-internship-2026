# Configuration (`conf/`)

Hierarchical Hydra configs. The composition root is `config.yaml`.

## Groups

| Group | File | Responsibility |
|---|---|---|
| `analysis` | `analysis/default.yaml` | Metrics, temporal, and noise analysis settings |
| `asr` | `asr/default.yaml` | Whisper / transcription settings |
| `classroom` | `classroom/default.yaml` | LOTO fine-tuning, FESC contamination, and RIR |
| `confidence` | `confidence/default.yaml` | Binary / TCP / temperature settings |
| `features` | `features/default.yaml` | Feature groups to compute |
| `mtkd` | `mtkd/default.yaml` | Student model hyperparameters and class order |
| `noise` | `noise/default.yaml` | Noise distribution and augmentation settings |
| `paths` | `paths/default.yaml` | All filesystem paths |
| `probes` | `probes/default.yaml` | Child-speech and feature-fusion settings |
| `teachers` | `teachers/default.yaml` | Monolingual fine-tuned teacher settings |
| `vad` | `vad/default.yaml` | Silero VAD, refinement, and denoising defaults |

## Override examples

```bash
# Change only the audio root
tea chunk paths.audio_root=/m/triton/work/.../vad-based

# Stronger denoising
tea denoise vad.atten_lim_db=0

# Different MTKD checkpoint
tea infer-mtkd paths.mtkd_student_ckpt=final_models/mtkd_student/other.pth

# Change noise configuration
tea noise-analysis noise.atten_lim_db=10
 

Resolved config is logged on every run (see `hydra-logs/`).
