#!/usr/bin/env bash
# forced-tile.sh — was a REJECT correct?
#
# oracle-vs-model.sh can only show that a REJECTed kernel was left alone; it
# cannot say whether tiling would have helped.  This script force-tiles with
# an explicit tile size (bypassing the cost model via -tile-size) and measures
# the predicted miss count, so a REJECT can be scored:
#
#   misses drop a lot   -> FALSE NEGATIVE, the model refused a real win
#   misses flat or rise -> correct rejection
set -u

REPO=/home/tor/Dev/PhD/DRComp/drcompiler.git/paper-eval
FALCON=/home/tor/Dev/PhD/DRComp/falcon-artifact/falcon-artifact
LAZY=$FALCON/cmake-build-release/bin/lazystack
SRC=$FALCON/benchmark/polybench/mlir/L
OPT=$REPO/build/tools/dr-opt/dr-opt
OUT=${1:?usage: forced-tile.sh <outdir> <kernel>...}
mkdir -p "$OUT"
shift

LAZY_ARGS="--cs=512 -a 512 --cs=8192 -a 8192 --line-size=64 -n 8"
TIMEOUT=${TIMEOUT:-1200}
TS=${TS:-32}

to22() { sed 's/vector<\([0-9]*\)xi32>/vector<\1xi64>/g' "$1"; }
to18() { sed 's/vector<\([0-9]*\)xi64>/vector<\1xi32>/g' "$1"; }

echo "kernel,tile_size,misses_forced,missesL1_forced,missesL2_forced"

for k in "$@"; do
  src=$SRC/$k.mlir
  [[ -f $src ]] || continue
  to22 "$src" > "$OUT/$k.22.mlir"

  "$OPT" "$OUT/$k.22.mlir" \
    --pass-pipeline="builtin.module(func.func(dr-affine-loop-distribute,dr-affine-loop-tile{tile-size=$TS}))" \
    -o "$OUT/$k.forced.22.mlir" 2>/dev/null
  [[ -s "$OUT/$k.forced.22.mlir" ]] || { echo "$k,$TS,DR-OPT-FAIL,,"; continue; }
  to18 "$OUT/$k.forced.22.mlir" > "$OUT/$k.forced.mlir"

  timeout "$TIMEOUT" "$LAZY" $LAZY_ARGS "$OUT/$k.forced.mlir" \
    > "$OUT/$k.forced.json" 2>"$OUT/$k.forced.err"

  python3 - "$OUT/$k.forced.json" "$k" "$TS" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(f'{sys.argv[2]},{sys.argv[3]},{d.get("misses","")},'
          f'{d.get("misses_L1","")},{d.get("misses_L2","")}')
except Exception:
    print(f"{sys.argv[2]},{sys.argv[3]},TIMEOUT,,")
PY
done
