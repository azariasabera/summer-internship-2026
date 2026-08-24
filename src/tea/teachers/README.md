# `tea.teachers`

Monolingual fine-tuned teachers (EN / FI / FR) used by MTKD.

## Public API

```python
from tea.teachers import TeacherTrainer, LOADERS

trainer = TeacherTrainer(cfg)
ckpt_path = trainer.train("FI", session=6)
```

`LOADERS = {"EN": iemocap, "FI": fesc, "FR": cafe}` -- the canonical
single-language dataset loaders. `tea.mtkd.data` imports these directly
rather than redefining them (the original dump had them duplicated
between `teacher_code.txt` and `mtkd_code.txt`).

## CLI

```bash
tea train-teacher teachers.language=FI teachers.session=6
```

## Status

Dataset loaders (`iemocap`/`fesc`/`cafe`) and `TeacherTrainer` (checkpoint-resumable,
early-stopping-on-dev-UAR training) are implemented and CLI-wired.

## Notes

- Checkpoint naming: `<checkpoint_root>/teachers/FT_Monolingual_<LANG>_S<SESSION>.pth`.
- `conf/teachers/teachers.yaml` holds hyperparameters, dataset session maps,
  and the old-path-prefix strings used to rewrite raw json split files onto
  `paths.datasets_root`. Update `*_old_prefix`/`*_old_root` only if you have
  new raw split files pointing elsewhere.
