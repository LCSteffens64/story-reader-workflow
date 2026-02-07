#!/usr/bin/env bash
set -euo pipefail

# Normalize narration clips (per-file) using EBU R128 loudness.
# Default target: -16 LUFS, -1.5 dBTP, LRA 11
# Usage:
#   ./normalize_narration.sh /path/to/narration_dir
# Example:
#   ./normalize_narration.sh output/scramble/narration

TARGET_I="${TARGET_I:- -16}"
TARGET_TP="${TARGET_TP:- -1.5}"
TARGET_LRA="${TARGET_LRA:- 11}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/narration_dir" >&2
  exit 1
fi

NARR_DIR="$1"
if [[ ! -d "$NARR_DIR" ]]; then
  echo "ERROR: Not a directory: $NARR_DIR" >&2
  exit 1
fi

shopt -s nullglob
files=("$NARR_DIR"/*.wav)
if [[ ${#files[@]} -eq 0 ]]; then
  echo "No .wav files found in $NARR_DIR" >&2
  exit 1
fi

for f in "${files[@]}"; do
  tmp="${f%.wav}.normalized.wav"
  echo "Normalizing $(basename "$f")..."
  ffmpeg -y -i "$f" \
    -af "loudnorm=I=${TARGET_I}:TP=${TARGET_TP}:LRA=${TARGET_LRA}" \
    -ar 48000 -ac 1 "$tmp" \
    </dev/null >/dev/null 2>&1
  mv "$tmp" "$f"
  echo "  -> done"
 done

echo "All files normalized in $NARR_DIR"
