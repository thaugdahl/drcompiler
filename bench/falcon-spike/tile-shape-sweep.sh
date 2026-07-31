#!/usr/bin/env bash
# tile-shape-sweep.sh — is a kernel's tiling inert because the SHAPE is wrong,
# or because tiling cannot help it at all?
#
# For each kernel, force a spread of uniform tile sizes (bypassing the cost
# model via -tile-size) and measure predicted misses for each.  Then:
#
#   no size beats the untiled baseline  -> tiling genuinely cannot help; the
#                                          correct verdict is REJECT and the
#                                          gate is too permissive
#   some size beats it                  -> the traffic objective is picking a
#                                          bad shape; the gate is right
#
# Runs kernels concurrently (each lazystack gets a few threads).
set -u

REPO=/home/tor/Dev/PhD/DRComp/drcompiler.git/paper-eval
FALCON=${FALCON_DIR:-/home/tor/Dev/PhD/DRComp/falcon-artifact/falcon-artifact}
LAZY=$FALCON/cmake-build-release/bin/lazystack
SIZE=${SIZE:-L}
SRC=$FALCON/benchmark/polybench/mlir/$SIZE
OPT=$REPO/build/tools/dr-opt/dr-opt
OUT=${1:?usage: tile-shape-sweep.sh <outdir> <kernel>...}
mkdir -p "$OUT"
shift

LAZY_ARGS="--cs=512 -a 512 --cs=8192 -a 8192 --line-size=64 -n 4"
TIMEOUT=${TIMEOUT:-1800}
SIZES=${SIZES:-"16 32 64 128"}
JOBS=${JOBS:-6}

to22() { sed 's/vector<\([0-9]*\)xi32>/vector<\1xi64>/g' "$1"; }
to18() { sed 's/vector<\([0-9]*\)xi64>/vector<\1xi32>/g' "$1"; }

misses() {
  timeout "$TIMEOUT" "$LAZY" $LAZY_ARGS "$1" 2>/dev/null | python3 -c 'import json,sys
try: print(json.load(sys.stdin)["misses"])
except Exception: print("")'
}

# One (kernel, tile size) point -> a CSV line on stdout.
one() {
  local k=$1 ts=$2 src=$SRC/$1.mlir
  local base="$OUT/$k.ts$ts"
  to22 "$src" > "$base.in.mlir"
  if [[ $ts == untiled ]]; then
    to18 "$base.in.mlir" > "$base.mlir"
  else
    "$OPT" "$base.in.mlir" \
      --pass-pipeline="builtin.module(func.func(dr-affine-loop-distribute,dr-affine-loop-tile{tile-size=$ts}))" \
      -o "$base.out22.mlir" 2>/dev/null
    [[ -s "$base.out22.mlir" ]] || { echo "$k,$ts,DR-OPT-FAIL"; return; }
    to18 "$base.out22.mlir" > "$base.mlir"
  fi
  echo "$k,$ts,$(misses "$base.mlir")"
}
export -f one misses to22 to18
export SRC OUT OPT LAZY LAZY_ARGS TIMEOUT

echo "kernel,tile_size,misses"
for k in "$@"; do
  for ts in untiled $SIZES; do
    printf '%s\0%s\0' "$k" "$ts"
  done
done | xargs -0 -n2 -P "$JOBS" bash -c 'one "$0" "$1"'
