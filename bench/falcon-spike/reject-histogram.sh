#!/usr/bin/env bash
# reject-histogram.sh — why does analyzeBandReuse reject each PolyBench band?
#
# Tier 0 needs to be driven by measured weight, not by guesses about which
# modelling restriction hurts.  Prints one row per (kernel, band) that the
# analysis refuses, tagged with the ReuseReject reason, plus a histogram.
#
# Runs the same pipeline prefix the tiler sees (dr-affine-loop-distribute),
# since the distributor is what produces the bands the tile gate evaluates.
set -u

FALCON=${FALCON_DIR:-/home/tor/Dev/PhD/DRComp/falcon-artifact/falcon-artifact}
BENCH=${BENCH:-$FALCON/benchmark/polybench/mlir}
SIZE=${SIZE:-L}
REPO=${REPO:-/home/tor/Dev/PhD/DRComp/drcompiler.git/paper-eval}
OPT=${DR_OPT:-$REPO/build/tools/dr-opt/dr-opt}
UB=${UB:-true}   # accept-trip-upper-bounds
DISTRIBUTE=${DISTRIBUTE:-1}
ALLBANDS=${ALLBANDS:-1}

if [[ $DISTRIBUTE == 1 ]]; then
  PIPE="builtin.module(func.func(dr-affine-loop-distribute,dr-test-reuse-analysis{accept-trip-upper-bounds=$UB all-bands=$ALLBANDS}))"
else
  PIPE="builtin.module(func.func(dr-test-reuse-analysis{accept-trip-upper-bounds=$UB all-bands=$ALLBANDS}))"
fi

printf '%-16s %-6s %s\n' kernel bands detail
for f in "$BENCH/$SIZE"/*.mlir; do
  k=$(basename "$f" .mlir)
  out=$(sed 's/vector<\([0-9]*\)xi32>/vector<\1xi64>/g' "$f" \
        | "$OPT" --pass-pipeline="$PIPE" -o /dev/null 2>&1)
  ok=$(grep -c 'band depth=' <<<"$out")
  # reason=<tag> for every rejected band
  mapfile -t why < <(grep -o 'UNANALYZABLE reason=[a-z-]*' <<<"$out" \
                     | sed 's/.*reason=//' | sort | uniq -c \
                     | awk '{printf "%s:%s ", $2, $1}')
  printf '%-16s %-6s %s\n' "$k" "ok=$ok" "${why[*]:-—}"
done | tee /tmp/reject-rows.txt

echo
echo "=== histogram over all rejected bands ($SIZE, distribute=$DISTRIBUTE, ub=$UB) ==="
sed -n '2,$p' /tmp/reject-rows.txt | tr ' ' '\n' | grep ':' \
  | awk -F: '{h[$1]+=$2} END {for (k in h) printf "%8d  %s\n", h[k], k}' | sort -rn
echo "=== kernels with >=1 analyzable band ==="
sed -n '2,$p' /tmp/reject-rows.txt | awk '$2!="ok=0"' | wc -l
echo "=== kernels with ZERO analyzable bands ==="
sed -n '2,$p' /tmp/reject-rows.txt | awk '$2=="ok=0" {print "    " $1 "  " $3}'
