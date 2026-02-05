#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MUSIC_DIR="${ROOT_DIR}/music"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg is not installed or not on PATH." >&2
  exit 1
fi

shopt -s nullglob
ogx_files=("${MUSIC_DIR}"/*.ogx)
shopt -u nullglob

if [ ${#ogx_files[@]} -eq 0 ]; then
  echo "No .ogx files found in ${MUSIC_DIR}"
  exit 0
fi

for ogx in "${ogx_files[@]}"; do
  base="$(basename "${ogx}" .ogx)"
  out="${MUSIC_DIR}/${base}.mp3"
  echo "Converting ${ogx} -> ${out}"
  ffmpeg -y -i "${ogx}" -codec:a libmp3lame -qscale:a 2 "${out}" >/dev/null 2>&1
  rm -f "${ogx}"
done

echo "Done."
