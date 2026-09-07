# `tea.probes`

Lightweight probes that investigate the model representations: child-speech detection and feature-fusion tables.

## Modules

| File | Role |
|------|------|
| `child_speech.py` | Logistic (or linear) probe that predicts the presence of child speech from pooled WavLM embeddings. Exposes `probe_child_speech_cli`. |
| `feature_fusion.py` | Builds and evaluates tables that fuse hand-crafted acoustic / sentiment features with the embeddings. Exposes `probe_feature_fusion_cli`. |
| `fusion_utils.py` | Shared helpers for feature concatenation, scaling and simple classifiers. |
| `__init__.py` | Re-exports the CLI entry points. |

## Commands

- `tea probe-child-speech` – train / evaluate a child-speech probe on extracted embeddings.
- `tea probe-feature-fusion` – produce feature-fusion performance tables.

## CLI

```bash
tea probe-child-speech
tea probe-child-speech probes.embeddings=generated/embeddings \
                       probes.annotation_dir=generated/annotations

tea probe-feature-fusion # runs both `softmax` and `embedding` as bases
tea probe-feature-fusion probes.base=softmax # or probes.base=embedding
```

Important configuration lives in `conf/probes/probes.yaml`.
