#!/usr/bin/env bash
# Probe the host's CPU cost model via the drcc Docker image.
#
# Usage:
#   ./scripts/probe-cost-model-docker.sh costs.json
#   ./scripts/probe-cost-model-docker.sh           # prints to stdout
#
# Then compile with the result:
#   ./scripts/drcc-docker.sh --cpu-cost-model-file costs.json -c foo.c -o foo.o

set -euo pipefail

IMAGE="${DRCC_IMAGE:-drcc}"
OUTPUT="${1:-}"

args=()
if [[ -n "$OUTPUT" ]]; then
  abs="$(realpath -m "$OUTPUT")"
  dir="$(dirname "$abs")"
  mkdir -p "$dir"
  args+=(-v "$dir:$dir" --entrypoint
         /usr/local/share/drcompiler/scripts/probe-cost-model.sh
         "$IMAGE" "$abs")
else
  args+=(--entrypoint
         /usr/local/share/drcompiler/scripts/probe-cost-model.sh
         "$IMAGE")
fi

exec docker run --rm "${args[@]}"
