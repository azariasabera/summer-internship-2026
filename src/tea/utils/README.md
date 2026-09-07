# `tea.utils`

Shared helpers used across the whole package.

## Modules

| File | Role |
|------|------|
| `config.py` | `load_config(overrides)` – compose and resolve the Hydra configuration tree. |
| `paths.py` | `resolve()`, `ensure_dir()` – turn config path strings into absolute paths and create directories. |
| `io.py` | JSON / CSV / nested-prediction loaders and writers; also `merge_annotations` (CLI entry) that copies ground-truth columns from prepared CSVs into the generated annotation files. |
| `logging.py` | Standard logger factory plus helpers that record git commit / seed. |
| `constants.py` | Canonical class order (`neutral`, `sadness`, `happiness`, `anger`), label maps, excluded-video list. |
| `seed.py` | `set_seed(seed)` for numpy / torch / random. |

## Commands

- `tea merge-annotations` – copy `gt_label`, `confidence`, `overlap` (and any other prepared columns) from the hand-labelled CSVs into the empty annotation CSVs produced by `tea chunk`.

## CLI

```bash
tea merge-annotations
tea merge-annotations paths.prepared_annotation_root=data/prepared_annotations paths.annotation_root=generated/annotations
```

Paths are taken from `conf/paths/paths.yaml`.
