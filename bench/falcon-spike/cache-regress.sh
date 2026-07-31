#!/usr/bin/env bash
# cache-regress.sh — cache-miss regression check for the affine transform suite.
#
# Uses Falcon (`lazystack`, PLDI 2024) as an exact miss-count oracle: for every
# PolyBench kernel it measures predicted cache misses BEFORE and AFTER the
# transform pipeline, and compares against a checked-in baseline.
#
# It catches two classes of defect that no existing test in this tree notices:
#
#   REGRESSION — a pass made the miss count worse.  Found in the wild:
#                gramschmidt, where dr-affine-loop-distribute acting alone
#                under a tiling REJECT doubled L2 misses (90.6M -> 181.0M).
#
#   INERT      — the cost model said TILE and the miss count did not move.
#                Found in the wild: covariance, correlation (1.0002x) and
#                trmm (0.9998x, marginally worse).
#
# Usage:
#   cache-regress.sh                 # check against the baseline, nonzero on regression
#   cache-regress.sh --update        # regenerate the baseline
#   SIZE=L cache-regress.sh          # force one global size, ignoring the manifest
#   PIPELINE='...' cache-regress.sh  # guard a different pass pipeline
#   JOBS=12 cache-regress.sh         # more concurrency
#
# By default each kernel is measured at the size pick-sizes.sh chose for it
# (bench/falcon-spike/sizes.csv), so the tiling decision is exercised rather
# than trivially satisfied.  Kernels run concurrently; regenerate the manifest
# with pick-sizes.sh if the corpus or the tiler's target changes.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FALCON=${FALCON_DIR:-/home/tor/Dev/PhD/DRComp/falcon-artifact/falcon-artifact}
LAZY=$FALCON/cmake-build-release/bin/lazystack
BENCH=$FALCON/benchmark/polybench/mlir
OPT=${DR_OPT:-$REPO/build/tools/dr-opt/dr-opt}
# Per-kernel dataset size, chosen by pick-sizes.sh so each kernel is measured
# where its tiling decision is actually load-bearing.  A single global size is
# wrong: at M, seidel-2d's working set is only 4.9x the tiler's target, so a
# shape exploiting no reuse still fits and the model looks broken -- at L it is
# 122x over and behaves correctly.  SIZE=<S|M|L> overrides the manifest.
MANIFEST=${MANIFEST:-$REPO/bench/falcon-spike/sizes.csv}
SIZE=${SIZE:-}
BASELINE=${BASELINE:-$REPO/bench/falcon-spike/baseline${SIZE:+-$SIZE}.csv}
JOBS=${JOBS:-6}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Cache geometry: two fully-associative levels, 64 B lines, L1 32 KiB /
# L2 512 KiB -- drcompiler's MachineModel defaults.
LAZY_ARGS=${LAZY_ARGS:-"--cs=512 -a 512 --cs=8192 -a 8192 --line-size=64 -n 8"}
TIMEOUT=${TIMEOUT:-900}
PIPELINE=${PIPELINE:-'builtin.module(func.func(dr-affine-loop-distribute,dr-affine-loop-tile{emit-rationale=true}))'}

# A transform must move misses by more than this to count as doing anything.
INERT_PCT=${INERT_PCT:-2}
# Tolerance on a baseline comparison, to absorb harmless codegen churn.
DRIFT_PCT=${DRIFT_PCT:-1}

UPDATE=0
[[ ${1:-} == --update ]] && UPDATE=1

if [[ ! -x $LAZY ]]; then
  echo "cache-regress: SKIP (no lazystack at $LAZY)" >&2
  echo "  set FALCON_DIR to the unpacked Falcon artifact to enable this check." >&2
  exit 0
fi
if [[ ! -d $BENCH ]]; then
  echo "cache-regress: SKIP (no corpus at $BENCH)" >&2
  exit 0
fi

# Work list: "<kernel> <size>" per line.
if [[ -n $SIZE ]]; then
  mapfile -t WORK < <(find "$BENCH/$SIZE" -maxdepth 1 -name '*.mlir' -printf '%f\n' \
                      | sed "s/\.mlir\$/ $SIZE/" | sort)
elif [[ -f $MANIFEST ]]; then
  mapfile -t WORK < <(tail -n +2 "$MANIFEST" | awk -F, 'NF>=2 {print $1, $2}' | sort)
else
  echo "cache-regress: no manifest at $MANIFEST -- run pick-sizes.sh, or set SIZE" >&2
  exit 1
fi
if [[ ${#WORK[@]} -eq 0 ]]; then
  echo "cache-regress: SKIP (empty work list)" >&2
  exit 0
fi

# dr-opt is LLVM 22; the corpus came from an LLVM 18-era Polygeist and the DLTI
# dense-attr element type changed.  Same fixup drcc applies.
to22() { sed 's/vector<\([0-9]*\)xi32>/vector<\1xi64>/g' "$1"; }
to18() { sed 's/vector<\([0-9]*\)xi64>/vector<\1xi32>/g' "$1"; }

misses() { # <mlir file> -> total misses, or the empty string
  timeout "$TIMEOUT" "$LAZY" $LAZY_ARGS "$1" 2>/dev/null \
    | python3 -c 'import json,sys
try: print(json.load(sys.stdin)["misses"])
except Exception: print("")'
}

one() { # kernel size -> "kernel,verdict,base,after"
  local k=$1 sz=$2 src=$BENCH/$2/$1.mlir rat verdict
  [[ -f $src ]] || { echo "$k,NO-SOURCE,,"; return; }
  to22 "$src" > "$TMP/$k.in.mlir"
  rat=$("$OPT" "$TMP/$k.in.mlir" --pass-pipeline="$PIPELINE" \
        -o "$TMP/$k.out22.mlir" 2>&1 | sed -n 's/.*tile-rationale: //p' | paste -sd'|' -)
  if [[ ! -s "$TMP/$k.out22.mlir" ]]; then
    echo "$k,DR-OPT-FAIL,,"
    return
  fi
  to18 "$TMP/$k.out22.mlir" > "$TMP/$k.out.mlir"
  verdict=REJECT
  [[ $rat == *TILE* ]] && verdict=TILE
  echo "$k,$verdict,$(misses "$src"),$(misses "$TMP/$k.out.mlir")"
}
export -f one misses to22 to18
export BENCH TMP OPT LAZY LAZY_ARGS TIMEOUT PIPELINE

measure() { # emits "kernel,verdict,base,after", kernels run concurrently
  printf '%s\n' "${WORK[@]}" \
    | xargs -n2 -P "$JOBS" bash -c 'one "$0" "$1"' \
    | sort
}

if (( UPDATE )); then
  { echo "kernel,verdict,misses_base,misses_after"; measure; } > "$BASELINE"
  echo "cache-regress: baseline written to $BASELINE ($(( $(wc -l < "$BASELINE") - 1 )) kernels, ${SIZE:-per-kernel sizes})"
  exit 0
fi

if [[ ! -f $BASELINE ]]; then
  echo "cache-regress: no baseline at $BASELINE -- run with --update first" >&2
  exit 1
fi

measure > "$TMP/now.csv"

python3 - "$BASELINE" "$TMP/now.csv" "$INERT_PCT" "$DRIFT_PCT" <<'PY'
import csv, sys

base_path, now_path, inert_pct, drift_pct = sys.argv[1:5]
inert_pct, drift_pct = float(inert_pct), float(drift_pct)


def load(path, header):
    rows = {}
    with open(path) as f:
        rdr = csv.reader(f)
        if header:
            next(rdr, None)
        for r in rdr:
            if len(r) >= 4:
                rows[r[0]] = r[1:4]
    return rows


old, new = load(base_path, True), load(now_path, False)

regressions, inert, newly_inert = [], [], []
improved, changed_verdict, missing = [], [], []

for k, (v, b, a) in sorted(new.items()):
    if k not in old:
        continue
    ov, ob, oa = old[k]
    if v != ov:
        changed_verdict.append((k, ov, v))
    if not a or not oa:
        missing.append(k)
        continue
    a_i, oa_i = int(a), int(oa)
    if a_i > oa_i * (1 + drift_pct / 100):
        regressions.append((k, oa_i, a_i))
    elif a_i < oa_i * (1 - drift_pct / 100):
        improved.append((k, oa_i, a_i))
    # Inert: claimed a tiling but the miss count barely moved.  Pre-existing
    # inert kernels are a warning; a kernel that WAS effective and has just
    # gone inert is a failure.
    if v == "TILE" and b:
        b_i = int(b)
        if b_i and a_i > b_i * (1 - inert_pct / 100):
            was_inert = (
                ov == "TILE" and ob and oa
                and int(oa) > int(ob) * (1 - inert_pct / 100)
            )
            (inert if was_inert else newly_inert).append((k, b_i, a_i))

for k in sorted(set(old) - set(new)):
    missing.append(k)


def show(title, rows, fmt):
    if not rows:
        return
    print(f"\n{title}")
    for r in rows:
        print("  " + fmt(r))


show("REGRESSION (misses worse than baseline)", regressions,
     lambda r: f"{r[0]:<16} {r[1]:>14,} -> {r[2]:>14,}  (+{100*(r[2]/r[1]-1):.2f}%)")
show(f"NEWLY INERT (TILE, was effective, now <{inert_pct:g}% fewer misses)",
     newly_inert,
     lambda r: f"{r[0]:<16} {r[1]:>14,} -> {r[2]:>14,}  ({r[1]/r[2]:.4f}x)")
show(f"inert, pre-existing (TILE but <{inert_pct:g}% fewer misses)", inert,
     lambda r: f"{r[0]:<16} {r[1]:>14,} -> {r[2]:>14,}  ({r[1]/r[2]:.4f}x)")
show("verdict changed", changed_verdict,
     lambda r: f"{r[0]:<16} {r[1]} -> {r[2]}")
show("improved (baseline is stale, rerun --update to accept)", improved,
     lambda r: f"{r[0]:<16} {r[1]:>14,} -> {r[2]:>14,}  ({r[1]/r[2]:.2f}x fewer)")
show("no data (timeout / dr-opt failure)", [(k,) for k in sorted(set(missing))],
     lambda r: r[0])

ok = len(new) - len(regressions) - len(newly_inert) - len(inert)
print(f"\n{ok}/{len(new)} kernels clean; "
      f"{len(regressions)} regression(s), {len(newly_inert)} newly inert, "
      f"{len(inert)} pre-existing inert, {len(improved)} improved")

sys.exit(1 if regressions or newly_inert else 0)
PY
