#!/bin/bash
# recipes/noise/attenuation_sweep.sh
#
# Reproduces the DeepFilterNet rows of Table 21 (noise_analysis.txt):
# "DeepFilterNet 15 dB" and "DeepFilterNet 0 dB".
#
# The "Custom Spectral Subtraction" and "Retrain X-Y dB" rows of the same
# table need tea.analysis.noise (distribution comparison) and
# tea.mtkd's training-with-noise entry point respectively -- both land in
# a later installment; this recipe covers only the denoising half for now.

set -euo pipefail

tea denoise noise.method=deepfilter noise.atten_lim_db=15
tea denoise noise.method=deepfilter noise.atten_lim_db=0
