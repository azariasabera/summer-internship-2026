# `tea.analysis`

Post-hoc analysis of MTKD predictions on the classroom recordings
(report Sections 2.1-2.2, 4.1, 4.4).

## Public API

```python
from tea.analysis import join_predictions, evaluate_per_video_and_overall, temporal_consistency, compare_conditions

df = join_predictions("generated/predictions/MTKD_run.json", "generated/annotations")
per_video, overall = evaluate_per_video_and_overall(df)  # report Tables 6-8
print(overall["war"], overall["uar"])

consistency = temporal_consistency(df.loc[df.video == "1B2251", "gt_label"].tolist())  # report Eq. 2.1

table21 = compare_conditions({"Original": "...", "DeepFilterNet 15 dB": "...", "Retrain 15-30 dB": "..."})
```

## CLI

```bash
tea evaluate-classroom analysis.mtkd_json=generated/predictions/MTKD_run.json
tea acoustic-by-emotion analysis.mtkd_json=... analysis.audio_root=data/classroom_audio
tea temporal analysis.mtkd_json=...
tea noise-analysis +analysis.noise_conditions.Original=... +analysis.noise_conditions."DeepFilterNet_15dB"=...
```

## Submodules

| Module | Ported from |
|---|---|
| `classroom.py` | `checking_pred.ipynb` -- the native 4-class and 3-class-sentiment evaluation blocks were duplicated in the notebook, now one `evaluate_predictions` function called twice |
| `acoustic_by_emotion.py` | `per_video.ipynb` (duration/loudness/speaking-rate cells, report Figure 8) |
| `temporal.py` | `per_video.ipynb` (stability cells) + `temporal.txt` + `temporal_analysis.txt` -- see below |
| `noise.py` | `noise_analysis.txt` + `noise.ipynb` (7 near-duplicate evaluate-and-tabulate blocks, one per Table 21 row, consolidated into one `class_distribution` function) |

## Status

Fully ported.
