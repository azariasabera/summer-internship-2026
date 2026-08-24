#!/bin/bash
# recipes/classroom/finetune_configs.sh
#
# Reproduces Tables 17/18: all six imbalance-handling configurations (A-F),
# both head_only and full fine-tuning strategies. Requires the MTKD
# baseline checkpoint from recipes/mtkd/train_baseline.sh.

set -euo pipefail

CHECKPOINT="final_models/mtkd/MTKD_Multilingual_FI_S8.pth"

for VARIANT in head_only full; do
    for CONFIG in A B C D E F; do
        tea finetune-classroom-loto \
            classroom.run.base_checkpoint="$CHECKPOINT" \
            classroom.run.variant="$VARIANT" \
            classroom.run.config="$CONFIG"
    done
done
