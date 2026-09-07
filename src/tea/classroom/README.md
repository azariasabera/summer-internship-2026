# `tea.classroom`

Leave-one-teacher-out (LOTO) fine-tuning of an MTKD student on the classroom recordings, including FESC augmentation.

## Modules

| File | Role |
|------|------|
| `finetune.py` | LOTO fine-tuning loop (`finetune_classroom_cli`). Loads a pre-trained MTKD checkpoint, freezes or unfreezes layers according to the chosen config, applies weighted CE + optional FESC, and writes per-fold checkpoints. |
| `data.py` | Classroom dataset construction, fold splitting, and FESC contamination logic. |
| `fesc.py` | Feature-space / acoustic contamination helpers used by the fine-tuning configs. |
| `utils.py` | Weighting and fold utilities shared with the rest of the package. |
| `__init__.py` | Exposes the CLI entry point. |

## Command

- `tea finetune-classroom` – run one or more LOTO fine-tuning configurations (normally on a GPU cluster).

## CLI

```bash
tea finetune-classroom \
    classroom.use_class_weight=true \
    classroom.use_confidence_weight=true \
    classroom.augment_fesc=false \
    classroom.variant=head_only \
    classroom.base_checkpoint=final_models/mtkd/MTKD_Multilingual_FI_S6.pth \
    classroom.epochs=5 \
    classroom.lr=1e-4 \
    classroom.batch_size=8

tea finetune-classroom \
    classroom.use_class_weight=false \
    classroom.use_confidence_weight=false \
    classroom.augment_fesc=true \
    classroom.variant=full \
    classroom.base_checkpoint=final_models/mtkd/MTKD_Multilingual_FI_S6.pth \
    classroom.epochs=5 \
    classroom.lr=2e-5 \
    classroom.batch_size=8 \
```

**Note**: RIR augmentation can be applied during FESC noise contamination, but it is not currently supported in the pipeline.

Important configuration lives in `conf/classroom/classroom.yaml`. The six experimental configurations reported are selected by setting `classroom.use_classweight`, `classroom.use_confidence_weight` and `classroom.augment_fesc` true/false.
