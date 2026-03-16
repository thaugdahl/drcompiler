#!/usr/bin/env bash
# Invoke the drcc Docker image as a drop-in compiler.
#
# Usage (same flags as cc):
#   ./scripts/drcc-docker.sh -c foo.c -o foo.o
#   ./scripts/drcc-docker.sh foo.c bar.c -o program
#
# All file paths are resolved to absolute paths and bind-mounted read/write.

set -euo pipefail

IMAGE="${DRCC_IMAGE:-drcc}"

# Collect every unique parent directory referenced in the arguments so we
# can bind-mount them.  Also rewrite paths to their absolute form.
dirs=()
args=()
for arg in "$@"; do
  if [[ -e "$arg" ]]; then
    abs="$(realpath "$arg")"
    dir="$(dirname "$abs")"
    dirs+=("$dir")
    args+=("$abs")
  elif [[ "$arg" == -o ]]; then
    args+=("$arg")
  else
    # Peek ahead: if the previous arg was -o, the current arg is an
    # output path that may not exist yet — resolve its parent dir.
    if [[ ${#args[@]} -gt 0 && "${args[-1]}" == "-o" ]]; then
      abs="$(realpath -m "$arg")"
      dir="$(dirname "$abs")"
      mkdir -p "$dir"
      dirs+=("$dir")
      args+=("$abs")
    else
      args+=("$arg")
    fi
  fi
done

# Deduplicate mount directories.
mounts=()
seen=""
for d in "${dirs[@]}"; do
  if [[ ":$seen:" != *":$d:"* ]]; then
    mounts+=(-v "$d:$d")
    seen="$seen:$d"
  fi
done

exec docker run --rm "${mounts[@]}" "$IMAGE" "${args[@]}"
