#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MUSIC_DIR="${ROOT_DIR}/music"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg is not installed or not on PATH." >&2
  exit 1
fi

extensions=(ogx ogg oga opus flac wav webm)
audio_files=()

shopt -s nullglob nocaseglob
for ext in "${extensions[@]}"; do
  audio_files+=("${MUSIC_DIR}"/*.${ext})
done
shopt -u nullglob nocaseglob

if [ ${#audio_files[@]} -eq 0 ]; then
  echo "No supported audio files found in ${MUSIC_DIR} (extensions: ${extensions[*]})"
  exit 0
fi

for input_audio in "${audio_files[@]}"; do
  base="$(basename "${input_audio}")"
  base="${base%.*}"
  out="${MUSIC_DIR}/${base}.mp3"
  echo "Converting ${input_audio} -> ${out}"
  ffmpeg -y -i "${input_audio}" -codec:a libmp3lame -qscale:a 2 "${out}" >/dev/null 2>&1
  rm -f "${input_audio}"
done

echo "Done."
