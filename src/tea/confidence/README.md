# `tea.confidence`

Confidence/reliability recalibration on top of a frozen MTKD student
(report Section 4.3).

## Public API

```python
from tea.confidence import run_loto_cv_binary, run_loto_cv_tcp, run_loto_cv_temperature
from tea.features import build_feature_table, build_master_table

df, mtkd_classes, sentiment_classes = build_master_table(
    csv_root="generated/annotations", audio_root="data/classroom_audio",
    mtkd_json_path="generated/predictions/MTKD_run.json",
    sentiment_fi_json_path="generated/sentiment/sentiment_fi.json",
)
feature_df = build_feature_table(df, mtkd_prob_cols=[f"mtkd_{c}" for c in mtkd_classes], text_fi_cols=[f"text_fi_{c}" for c in ["neutral","negative","positive"]])
labels = df["gt_label"].map({c: i for i, c in enumerate(mtkd_classes)}).to_numpy()

results_df, per_chunk_df = run_loto_cv_binary(feature_df, labels, df["teacher_id"].to_numpy(), df[[f"mtkd_{c}" for c in mtkd_classes]].to_numpy())
```

## CLI

```bash
tea confidence confidence.mtkd_json=generated/predictions/MTKD_run.json confidence.sentiment_fi_json=generated/sentiment/sentiment_fi.json
```

## Submodules

| Module | Ported from |
|---|---|
| `calibration_metrics.py` | `conf_estim.txt` |
| `cv_utils.py` | **Reconstructed** -- `internal_val_split` was imported but never defined in the original dump; `loto_scaled_folds` is a new factoring of logic that was repeated identically 3x |
| `binary.py` | `confidence_binary.py` |
| `tcp.py` | `confidence_tcp.py` |
| `temperature.py` | `confidence_temperature.py` |

## Status

Fully ported. The core math (`expected_calibration_error`, `auroc_correctness`,
`cv_utils.internal_val_split`'s no-overlap guarantee) passed a standalone
functional sanity check with hand-computed expected values.
