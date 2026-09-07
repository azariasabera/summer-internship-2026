# `tea.confidence`

Reliability estimation for the student predictions: binary correctness probes, TCP, and instance-level temperature.

## Modules

| File | Role |
|------|------|
| `binary.py` | Trains a simple binary classifier that predicts whether the emotion prediction is correct. |
| `tcp.py` | True-Class-Probability style reliability head. |
| `temperature.py` | Instance-level temperature scaling. |
| `calibration_metrics.py` | ECE, MCE, reliability diagrams and related metrics. |
| `cv_utils.py` | Cross-validation helpers used by the reliability nets. |
| `__init__.py` | Exposes the unified CLI entry point `confidence_cli(cfg)`. |

## Command

- `tea confidence` – run the configured reliability estimators (binary / TCP / temperature) on a set of predictions + embeddings and write the resulting scores / tables.

## CLI

```bash
tea confidence
tea confidence confidence.method=binary
tea confidence confidence.method=tcp confidence.embeddings=generated/embeddings
tea confidence confidence.method=temperature
```

Important configuration lives in `conf/confidence/confidence.yaml` (method selection, embedding path, CV folds, etc.).
