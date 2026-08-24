#!/bin/bash
# recipes/vad/baseline.sh
#
# Reproduces the segment boundaries used for the progress-report annotations.
# The values in conf/vad/vad.yaml ARE these settings -- this recipe is just
# the default config made explicit, so it stays correct even if the yaml
# defaults ever change for a different experiment.

set -euo pipefail

tea chunk \
    vad.threshold=0.5 \
    vad.min_speech_duration_ms=250 \
    vad.min_silence_duration_ms=300 \
    vad.speech_pad_ms=100 \
    vad.max_merge_gap=2.0 \
    vad.min_speech_ratio=0.5 \
    vad.min_speech_duration=2.0 \
    vad.min_non_speech_duration=2.0 \
    vad.max_segment_duration=10.0 \
    vad.overlap=1.0
