#!/usr/bin/env bash
# oracle-vs-model.sh — does drcompiler's tiling verdict agree with the change
# in predicted cache misses?
#
# For each PolyBench kernel (Falcon's own LARGE-size affine MLIR corpus):
#   1. lazystack on the untiled nest            -> baseline misses
#   2. dr-affine-loop-tile, capture the verdict -> TILE / REJECT + rationale
#   3. lazystack on the transformed nest        -> tiled misses
#
# A model that is doing its job REJECTs exactly the kernels whose miss count
# would not have improved.
#
# Cache: two fully-associative levels, 64 B lines --
#   L1 = 512 lines  = 32 KiB
#   L2 = 8192 lines = 512 KiB
# matching drcompiler's MachineModel defaults (dr-affine-loop-tile targets
# l2Size/2 = 256 KiB).
set -u

REPO=/home/tor/Dev/PhD/DRComp/drcompiler.git/paper-eval
FALCON=/home/tor/Dev/PhD/DRComp/falcon-artifact/falcon-artifact
LAZY=$FALCON/cmake-build-release/bin/lazystack
SRC=$FALCON/benchmark/polybench/mlir/L
OPT=$REPO/build/tools/dr-opt/dr-opt
OUT=${1:?usage: oracle-vs-model.sh <outdir> [kernel ...]}
mkdir -p "$OUT"

LAZY_ARGS="--cs=512 -a 512 --cs=8192 -a 8192 --line-size=64 -n 8"
TIMEOUT=${TIMEOUT:-900}

shift || true
KERNELS=("$@")
if [[ ${#KERNELS[@]} -eq 0 ]]; then
  mapfile -t KERNELS < <(find "$SRC" -maxdepth 1 -name '*.mlir' -printf '%f\n' | sed 's/\.mlir$//' | sort)
fi

# dr-opt is LLVM 22, the corpus was emitted by an LLVM 18-era Polygeist: the
# DLTI dense attr element type changed.  Same fixup drcc applies.
to22() { sed 's/vector<\([0-9]*\)xi32>/vector<\1xi64>/g' "$1"; }
to18() { sed 's/vector<\([0-9]*\)xi64>/vector<\1xi32>/g' "$1"; }

echo "kernel,verdict,misses_base,missesL1_base,missesL2_base,misses_tiled,missesL1_tiled,missesL2_tiled,rationale"

for k in "${KERNELS[@]}"; do
  src=$SRC/$k.mlir
  [[ -f $src ]] || continue

  to22 "$src" > "$OUT/$k.22.mlir"

  # 2. drcompiler's verdict + transformed IR.
  # Polygeist emits imperfect nests (init loop beside the compute loop), so
  # the tiler needs the distributor ahead of it to expose perfect bands --
  # this is the "enabler" pairing the pass was written for.
  rat=$("$OPT" "$OUT/$k.22.mlir" \
        --pass-pipeline="builtin.module(func.func(dr-affine-loop-distribute,dr-affine-loop-tile{emit-rationale=true}))" \
        -o "$OUT/$k.tiled.22.mlir" 2>&1 | sed -n 's/.*tile-rationale: //p' | paste -sd'|' -)
  [[ -s "$OUT/$k.tiled.22.mlir" ]] || { echo "$k,DR-OPT-FAIL,,,,,,,"; continue; }

  verdict=REJECT
  [[ "$rat" == *TILE* ]] && verdict=TILE

  to18 "$OUT/$k.tiled.22.mlir" > "$OUT/$k.tiled.mlir"

  # 1. + 3. miss counts before and after.
  timeout "$TIMEOUT" "$LAZY" $LAZY_ARGS "$src" > "$OUT/$k.base.json" 2>"$OUT/$k.base.err"
  timeout "$TIMEOUT" "$LAZY" $LAZY_ARGS "$OUT/$k.tiled.mlir" > "$OUT/$k.tiled.json" 2>"$OUT/$k.tiled.err"

  read -r mb m1b m2b < <(python3 - "$OUT/$k.base.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(d.get("misses", ""), d.get("misses_L1", ""), d.get("misses_L2", ""))
except Exception:
    print("TIMEOUT TIMEOUT TIMEOUT")
PY
)
  read -r mt m1t m2t < <(python3 - "$OUT/$k.tiled.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(d.get("misses", ""), d.get("misses_L1", ""), d.get("misses_L2", ""))
except Exception:
    print("TIMEOUT TIMEOUT TIMEOUT")
PY
)

  echo "$k,$verdict,$mb,$m1b,$m2b,$mt,$m1t,$m2t,\"$rat\""
done
