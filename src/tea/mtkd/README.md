# `tea.mtkd`

Multilingual MTKD student: train, evaluate, infer.

## Planned CLI

```bash
tea train-mtkd linguality=Multilingual language=FI session=6
tea infer-mtkd paths.mtkd_student_ckpt=...
tea train-with-classroom-noise snr_min=15 snr_max=30
```

## Notes

- Knowledge-distillation temperature and λ stay as in the original training code.
- Class order fixed to `neutral, sadness, happiness, anger`.

## Status

Scaffold. Original code lives in the internship dump; will be ported carefully.
