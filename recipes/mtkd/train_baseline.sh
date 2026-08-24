#!/bin/bash
# recipes/mtkd/train_baseline.sh
#
# Trains the multilingual MTKD student reported as the report's baseline
# (Tables 4-13): UAR 72.17%/WAR 70.80% Finnish-only, UAR 70.88%/WAR 72.50%
# combined multilingual test set. Requires the three teachers from
# recipes/teachers/train_all.sh to exist first.

set -euo pipefail

tea train-mtkd mtkd.linguality=Multilingual mtkd.language=FI mtkd.session=8
tea evaluate-mtkd mtkd.linguality=Monolingual mtkd.language=FI mtkd.session=8    # Table 4 (Finnish-only)
tea evaluate-mtkd mtkd.linguality=Multilingual mtkd.language=FI mtkd.session=8   # Table 5 (combined)
