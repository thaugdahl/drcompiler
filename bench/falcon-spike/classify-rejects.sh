#!/usr/bin/env bash
# classify-rejects.sh — split the `out-of-model` rejections by actual cause.
#
# `analyzeBandReuse` fails for three unrelated reasons and the rationale string
# lumps them.  For each kernel, run the distributor (as the tiler pipeline
# does), then for every top-level band report:
#   depth      — 1 means the band is a bare outer loop with the interesting
#                nest below it (imperfect-nest failure)
#   bounds     — whether any loop in the band has a non-constant bound
#                (triangular: affine_map / min / max operand)
# so the coverage work can be aimed at whichever dominates.
set -u

REPO=/home/tor/Dev/PhD/DRComp/drcompiler.git/paper-eval
FALCON=/home/tor/Dev/PhD/DRComp/falcon-artifact/falcon-artifact
SRC=$FALCON/benchmark/polybench/mlir/L
OPT=$REPO/build/tools/dr-opt/dr-opt
OUT=${1:?usage: classify-rejects.sh <outdir> [kernel ...]}
mkdir -p "$OUT"
shift || true

KERNELS=("$@")
if [[ ${#KERNELS[@]} -eq 0 ]]; then
  mapfile -t KERNELS < <(find "$SRC" -maxdepth 1 -name '*.mlir' -printf '%f\n' | sed 's/\.mlir$//' | sort)
fi

echo "kernel,unanalyzable_bands,analyzable_bands,nonconst_bound_loops,max_perfect_depth"

for k in "${KERNELS[@]}"; do
  src=$SRC/$k.mlir
  [[ -f $src ]] || continue
  sed 's/vector<\([0-9]*\)xi32>/vector<\1xi64>/g' "$src" > "$OUT/$k.22.mlir"

  # Distribute first, exactly as the tiler pipeline does.
  "$OPT" "$OUT/$k.22.mlir" \
    --pass-pipeline='builtin.module(func.func(dr-affine-loop-distribute))' \
    -o "$OUT/$k.dist.mlir" 2>/dev/null
  [[ -s "$OUT/$k.dist.mlir" ]] || { echo "$k,DR-OPT-FAIL,,,"; continue; }

  # What does the reuse analysis make of the resulting bands?
  rem=$("$OPT" "$OUT/$k.dist.mlir" \
        --pass-pipeline='builtin.module(func.func(dr-test-reuse-analysis{accept-trip-upper-bounds=true}))' \
        -o /dev/null 2>&1)
  un=$(grep -c "band UNANALYZABLE" <<<"$rem")
  an=$(grep -c "band depth=" <<<"$rem")

  # Triangular / symbolic bound: `to #map(%iv)`, `to affine_map<...>(%iv)`,
  # or a min/max bound.  Constant-bound loops are `affine.for %x = 0 to 1234`.
  nc=$(grep -cE 'affine\.for .*(to|=) *(#[a-zA-Z_0-9]+\(|affine_map<|min |max )' \
       "$OUT/$k.dist.mlir")

  # Depth of the band the tiler actually sees: the perfect chain starting at
  # each TOP-LEVEL affine.for (getTopLevelTileableBands).  A depth of 1 means
  # the interesting nest sits below an imperfect outer loop.
  depth=$(python3 - "$OUT/$k.dist.mlir" <<'PY'
import re, sys
lines = [l.rstrip('\n') for l in open(sys.argv[1])]
fors = [(len(m.group(1)), i) for i, l in enumerate(lines)
        if (m := re.match(r'^(\s*)affine\.for', l))]
if not fors:
    print(0); raise SystemExit
top = min(ind for ind, _ in fors)
depths = []
for ind, i in fors:
    if ind != top:
        continue
    # Walk down: the chain is perfect while the next statement inside the body
    # is itself the only affine.for at the next indent level.
    d, cur, want = 1, i, ind + 2
    while True:
        body = [j for (ii, j) in fors if ii == want and j > cur]
        if not body:
            break
        j = body[0]
        # Perfect only if nothing else shares that indent inside this loop.
        sibs = [x for (ii, x) in fors if ii == want and cur < x]
        nxt_top = [x for (ii, x) in fors if ii <= ind and x > cur]
        limit = nxt_top[0] if nxt_top else len(lines)
        if len([x for x in sibs if x < limit]) != 1:
            break
        d, cur, want = d + 1, j, want + 2
    depths.append(d)
print(max(depths) if depths else 0)
PY
)
  echo "$k,$un,$an,$nc,$depth"
done
