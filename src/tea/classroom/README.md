# `tea.classroom`

Leave-one-teacher-out (LOTO) fine-tuning of an MTKD student on annotated
classroom recordings, with optional FESC contamination for the minority
classes (sadness/anger). Report Sections 3 and Tables 16-20.

## Public API

```python
from tea.classroom import LOTOFineTuner

tuner = LOTOFineTuner(cfg)

# Named report configuration (A-F, see conf/classroom/classroom.yaml)
results = tuner.run_config("E", base_checkpoint="final_models/mtkd/MTKD_Multilingual_FI_S8.pth", variant="full")

# Or set the three switches individually
results = tuner.run_all_folds(
    base_checkpoint="...", variant="full",
    use_class_weight=True, use_confidence_weight=False, augment_fesc=True,
)
print(results["finetuned_pooled"])  # pooled out-of-fold UAR/WAR/confusion (Tables 17-20)
```

## CLI

```bash
# Named config
tea finetune-classroom-loto classroom.run.base_checkpoint=final_models/mtkd/MTKD_Multilingual_FI_S8.pth classroom.run.variant=full classroom.run.config=E

# Or explicit switches
tea finetune-classroom-loto classroom.run.base_checkpoint=... classroom.run.variant=head_only classroom.run.use_class_weight=true
```

## Submodules

| Module | Ported from |
|---|---|
| `data.py` | `classroom_data.py` -- LOTO fold building, CSV loading |
| `fesc.py` | `fesc_contamination.py` -- noise-pool composition, SNR estimation, composite noise mixing |
| `finetune.py` | `finetune_classroom.py` -- the orchestrator: fold loop -> optional contamination -> weighting -> fine-tune -> pooled OOF report |

## Status

Fully ported. Verified: imports cleanly, `tea finetune-classroom-loto`
reaches real code end-to-end (fails only on missing classroom audio/CSVs
in this sandbox, not a wiring bug).

## Notes

- Internal validation split (`classroom.internal_val_frac`) is OFF by
  default (fixed-epoch training, evaluate on the fold's real test set only
  at the end). Set it > 0 only if you want early stopping and are comfortable with the
  video-grouped train/val split this carves out of TRAIN only.