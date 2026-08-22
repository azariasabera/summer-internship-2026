# `tea.utils`

Shared helpers used by every other module.

## Planned contents

| Module | Responsibility |
|--------|----------------|
| `constants.py` | Canonical class order, label↔id maps, excluded videos |
| `config.py` | Thin Hydra helpers (load resolved config, log snapshot) |
| `logging.py` | Standard logger + git-commit / seed recording |
| `paths.py` | Resolve paths from config, ensure directories exist |
| `io.py` | Common JSON / CSV / checkpoint loaders |
| `seed.py` | `set_seed(seed)` for numpy / torch / random |

## Conventions

- Class order is always `["neutral", "sadness", "happiness", "anger"]` unless a recipe explicitly maps to 3-class sentiment.
- Video `1B3261` is excluded from training/evaluation folds by default (too few annotated chunks).
- Teacher id is obtained by stripping a trailing `_videoN` suffix.

## Status

Scaffold only. Implementation follows immediately after VAD.
