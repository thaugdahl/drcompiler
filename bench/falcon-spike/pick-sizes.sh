#!/usr/bin/env bash
# pick-sizes.sh — choose, per kernel, the smallest PolyBench dataset size at
# which the tiling decision is actually load-bearing.
#
# Why this exists: cache-regress.sh originally used one global size (M, for
# speed).  M turned out to be a regime where the capacity constraint is slack
# for some kernels -- seidel-2d's full working set is only ~5x the tiler's
# target there, so a shape that exploits no reuse still "fits" and the model
# looks broken.  At L the same kernel is 122x over target and the model is
# fine.  A gate run only at M therefore produces verdicts that do not survive.
#
# Two conditions must hold, and BOTH are needed -- L2 pressure alone picked M
# for seidel-2d, the exact case that motivated this script:
#
#   1. L2 capacity misses exist in the untiled kernel (measured with the
#      oracle).  Without them the working set fits and tiling cannot help, so
#      the size says nothing about the cost model.
#   2. The band's working set exceeds the TILER'S TARGET by a healthy margin.
#      This is the condition that actually forces the tile search: at M
#      seidel-2d is only 4.9x over target, so a shape exploiting no reuse still
#      "fits" and the model looks broken; at L it is 122x over and the search
#      is forced into the right answer.
set -u

FALCON=${FALCON_DIR:-/home/tor/Dev/PhD/DRComp/falcon-artifact/falcon-artifact}
LAZY=$FALCON/cmake-build-release/bin/lazystack
BENCH=$FALCON/benchmark/polybench/mlir
OUT=${1:?usage: pick-sizes.sh <manifest.csv>}

LAZY_ARGS="--cs=512 -a 512 --cs=8192 -a 8192 --line-size=64 -n 4"
TIMEOUT=${TIMEOUT:-1800}
# Minimum share of total misses that must come from L2 for the size to count
# as exercising capacity pressure.
MIN_L2_FRAC=${MIN_L2_FRAC:-0.05}
JOBS=${JOBS:-6}
SIZES=${SIZES:-"S M L"}

REPO=${REPO:-/home/tor/Dev/PhD/DRComp/drcompiler.git/paper-eval}
OPT=${DR_OPT:-$REPO/build/tools/dr-opt/dr-opt}
# dr-affine-loop-tile's target: half the MachineModel L2.
TARGET=${TARGET:-262144}
MIN_RATIO=${MIN_RATIO:-20}

probe() { # kernel size -> "kernel,size,misses,missesL2,footprint"
  local k=$1 sz=$2 f=$BENCH/$2/$1.mlir
  [[ -f $f ]] || { echo "$k,$sz,,,"; return; }
  local m
  m=$(timeout "$TIMEOUT" "$LAZY" $LAZY_ARGS "$f" 2>/dev/null \
      | python3 -c "import json,sys
try:
    d=json.load(sys.stdin); print('%s,%s'%(d['misses'],d.get('misses_L2',0)))
except Exception: print(',')")
  # Largest analyzable band footprint after the distributor, i.e. the working
  # set the tile search must fit.  Empty when every band is out-of-model, in
  # which case the tiling decision is moot and any size will do.
  local fp tiles
  fp=$(sed 's/vector<\([0-9]*\)xi32>/vector<\1xi64>/g' "$f" \
       | "$OPT" --pass-pipeline='builtin.module(func.func(dr-affine-loop-distribute,dr-test-reuse-analysis{accept-trip-upper-bounds=true}))' \
         -o /dev/null 2>&1 \
       | sed -n 's/.*footprint=\([0-9]*\).*/\1/p' | sort -rn | head -1)
  # Does the pipeline actually TILE here?  A size where the tiler rejects
  # guards nothing about tiling -- 2mm and 3mm satisfy both capacity
  # conditions at S yet REJECT there, so an S row silently stopped protecting
  # their 9.3x / 9.1x wins measured at L.
  tiles=$(sed 's/vector<\([0-9]*\)xi32>/vector<\1xi64>/g' "$f" \
          | "$OPT" --pass-pipeline='builtin.module(func.func(dr-affine-loop-distribute,dr-affine-loop-tile{emit-rationale=true}))' \
            -o /dev/null 2>&1 \
          | grep -c 'tile-rationale: TILE')
  echo "$k,$sz,$m,$fp,$tiles"
}
export -f probe
export LAZY BENCH LAZY_ARGS TIMEOUT OPT

mapfile -t KERNELS < <(find "$BENCH/L" -maxdepth 1 -name '*.mlir' -printf '%f\n' \
                       | sed 's/\.mlir$//' | sort)

RAW=$(for k in "${KERNELS[@]}"; do for sz in $SIZES; do printf '%s\0%s\0' "$k" "$sz"; done; done \
      | xargs -0 -n2 -P "$JOBS" bash -c 'probe "$0" "$1"')

python3 - "$OUT" "$MIN_L2_FRAC" "$SIZES" "$TARGET" "$MIN_RATIO" <<PY
import sys
out, frac, order = sys.argv[1], float(sys.argv[2]), sys.argv[3].split()
target, min_ratio = int(sys.argv[4]), float(sys.argv[5])
rows = {}
for line in """$RAW""".strip().splitlines():
    p = line.split(",")
    if len(p) != 6 or not p[2]:
        continue
    fp = int(p[4]) if p[4] else 0
    tiles = int(p[5]) if p[5] else 0
    rows.setdefault(p[0], {})[p[1]] = (int(p[2]), int(p[3]), fp, tiles)

picked, notes = {}, {}
for k, per in sorted(rows.items()):
    choice = None
    for sz in order:
        if sz not in per:
            continue
        tot, l2, fp, tiles = per[sz]
        l2_ok = bool(tot) and l2 / tot >= frac
        # fp == 0 means no analyzable band, so the tile search never runs and
        # the ratio condition is vacuous.
        ratio_ok = fp == 0 or fp >= min_ratio * target
        if l2_ok and ratio_ok and tiles > 0:
            choice = sz
            break
    if choice is None:
        # No size exercises a tiling decision for this kernel, so the row can
        # only guard against regressions.  Take the SMALLEST size that still
        # has L2 pressure: the expensive L runs are concentrated in exactly
        # these never-tiling kernels (nussinov, lu, ludcmp, ...) and paying for
        # them buys no coverage of the tile gate.  Trade-off: a size-dependent
        # regression in one of these is missed -- see the FULL=1 note below.
        avail = [s for s in order if s in per]
        with_l2 = [s for s in avail if per[s][0] and per[s][1] / per[s][0] >= frac]
        choice = with_l2[0] if with_l2 else (avail[-1] if avail else order[-1])
        tot, l2, fp, tiles = per.get(choice, (0, 0, 0, 0))
        if not (tot and l2 / tot >= frac):
            notes[k] = "no-l2-pressure"
        elif not (fp == 0 or fp >= min_ratio * target):
            notes[k] = f"slack-capacity(fp/target={fp/target:.1f})"
        else:
            notes[k] = "never-tiles"  # REJECT everywhere: guards regressions only
    picked[k] = choice

with open(out, "w") as f:
    f.write("kernel,size,note\n")
    for k, sz in picked.items():
        f.write(f"{k},{sz},{notes.get(k,'')}\n")

from collections import Counter
print("picked sizes:", dict(Counter(picked.values())))
print(f"{len(notes)} kernel(s) with no size satisfying both conditions:")
for k in sorted(notes):
    print(f"    {k:<16} -> {picked[k]}  ({notes[k]})")
print("manifest ->", out)
PY
