# `tea.probes`

Representation probes on top of a frozen MTKD student (report Sections
4.2-4.3).

## Public API

```python
from tea.probes import ChildSpeechProbe, load_dataset, get_feature_groups, run_experiment

df = load_dataset(embedding_root="generated/embeddings", annotation_root="generated/annotations")
probe = ChildSpeechProbe(n_splits=5)
fold_df, overall = probe.run(df)  # report Section 4.2, WAR 70.65% / UAR 69.23%

from tea.features import build_master_table
mt_df, mtkd_classes, _ = build_master_table(..., embedding_root="generated/embeddings")
groups = get_feature_groups(mt_df, mtkd_cols=[f"mtkd_{c}" for c in mtkd_classes])
result_df = run_experiment(mt_df, groups, label_col="_label_id", teacher_col="teacher_id", n_classes=4, base="softmax")
```

## CLI

```bash
tea probe-child-speech
tea probe-feature-fusion probes.feature_fusion.mtkd_json=... probes.feature_fusion.sentiment_fi_json=...
```

## Submodules

| Module | Ported from |
|---|---|
| `child_speech.py` | `probe_related.txt` |
| `feature_fusion.py` | `probe_related2.txt` -- only the unique experiment-running logic (`get_feature_groups`/`get_feature_combinations`/`run_experiment`); the data-loading it also contained (confirmed pasted twice, reformatted, within that one file) is `tea.features.build_master_table` |

## Status

Fully ported.

## Notes

- `feature_fusion.py`'s `base="embedding"` condition reproduces the
  report's finding that the raw 768-d embedding performs WORSE as a base
  representation than the 4-value softmax alone (Table 23).
