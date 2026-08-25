#!/bin/bash -l
set -euo pipefail

# Run this shell script from the summer-internship-2026 directory.
# Make sure the tea Conda environment is activated before running.

mkdir -p data/classroom_audio

find video_dir \
  -type f \( -name '*.MP4' -o -name '*.MTS' \) -print0 |
while IFS= read -r -d '' f; do
  out="data/classroom_audio/$(basename "${f%.*}").wav"

  echo "Converting: $f → $out"

  # -nostdin: don't read input from the terminal
  # -y: overwrite existing output files
  # -i: input video file
  # -vn: extract audio only; don't produce video
  # -ac 1: convert audio to mono
  # -ar 16000: resample audio to 16 kHz
  # -c:a pcm_s16le: uncompressed 16-bit PCM audio
  ffmpeg -nostdin -y \
    -i "$f" \
    -vn \
    -ac 1 \
    -ar 16000 \
    -c:a pcm_s16le \
    "$out"
done