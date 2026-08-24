#!/bin/bash
# recipes/teachers/train_all.sh
#
# Trains the three per-language teachers used by MTKD (report Section 2,
# "the retrained multilingual MTKD model" baseline). Sessions match
# conf/mtkd/mtkd.yaml's teacher_sessions.

set -euo pipefail

tea train-teacher teachers.language=EN teachers.session=2
tea train-teacher teachers.language=FI teachers.session=6
tea train-teacher teachers.language=FR teachers.session=1
