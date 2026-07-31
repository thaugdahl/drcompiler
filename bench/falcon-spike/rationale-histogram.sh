#!/usr/bin/env bash
# rationale-histogram.sh — what does the TILE GATE decide, per band, corpus-wide?
#
# Companion to reject-histogram.sh, one level up: that script asks whether the
# reuse analysis can see a band, this one asks what the tiler then does with
# it.  Run with ALLBANDS=0 and ALLBANDS=1 to separate "the model could not see
# the band" (out-of-model) from "the model saw it and said no".
set -u

FALCON=${FALCON_DIR:-/home/tor/Dev/PhD/DRComp/falcon-artifact/falcon-artifact}
BENCH=${BENCH:-$FALCON/benchmark/polybench/mlir}
SIZE=${SIZE:-L}
REPO=${REPO:-/home/tor/Dev/PhD/DRComp/drcompiler.git/paper-eval}
OPT=${DR_OPT:-$REPO/build/tools/dr-opt/dr-opt}
ALLBANDS=${ALLBANDS:-0}

PIPE="builtin.module(func.func(dr-affine-loop-distribute,dr-affine-loop-tile{emit-rationale=true all-bands=$ALLBANDS}))"

for f in "$BENCH/$SIZE"/*.mlir; do
  sed 's/vector<\([0-9]*\)xi32>/vector<\1xi64>/g' "$f" \
    | "$OPT" --pass-pipeline="$PIPE" -o /dev/null 2>&1 \
    | grep -o 'tile-rationale: TILE\|tile-rationale: REJECT reason=[a-z-]*\|tile-rationale: SKIP reason=[a-z-]*'
done | sed 's/tile-rationale: //' | sort | uniq -c | sort -rn

echo "--- kernels with at least one TILE (all-bands=$ALLBANDS) ---"
n=0
for f in "$BENCH/$SIZE"/*.mlir; do
  c=$(sed 's/vector<\([0-9]*\)xi32>/vector<\1xi64>/g' "$f" \
      | "$OPT" --pass-pipeline="$PIPE" -o /dev/null 2>&1 \
      | grep -c 'tile-rationale: TILE')
  [[ $c -gt 0 ]] && { n=$((n+1)); printf '    %-16s %s\n' "$(basename "$f" .mlir)" "$c"; }
done
echo "    total: $n"
