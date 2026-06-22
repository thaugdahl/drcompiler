#!/usr/bin/env bash
# polybench-par-survey.sh — parallel-axis opportunity map over all cached
# PolyBench kernels (bench/polybench-mlir/), via the dr-par-bubbles oracle.
# Columns: loops, PAR/SEQ axis counts, seq reasons, whole-function shard axis.
set -uo pipefail
export LC_ALL=C
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DROPT="${DROPT:-$REPO/build/tools/dr-opt/dr-opt}"
MLIR="$REPO/bench/polybench-mlir"
run(){ "$DROPT" --allow-unregistered-dialect --pass-pipeline="builtin.module(dr-par-bubbles{$1})" "$2" -o /dev/null 2>&1; }

printf "%-24s %5s %5s %5s  %-26s %s\n" "kernel" "loops" "PAR" "SEQ" "seq-reasons" "spmd-shard"
printf "%-24s %5s %5s %5s  %-26s %s\n" "------------------------" "-----" "---" "---" "--------------------------" "----------"
for f in "$MLIR"/*.mlir; do
  bn="$(basename "$f" .mlir)"
  d="$(run 'par-test-diagnostics' "$f")"
  npar=$(grep -c 'axis: PARALLEL' <<<"$d"); nseq=$(grep -c 'axis: SEQUENTIAL' <<<"$d")
  reasons=$(grep -oE 'SEQUENTIAL \([^)]*\)' <<<"$d" | sed 's/SEQUENTIAL (//;s/)//' | sort | uniq -c \
            | awk '{printf "%dx%s ",$1,$2$3$4}' | cut -c1-26)
  s="$(run 'par-test-spmd' "$f")"
  shard=$(grep -iE 'par-shard|par-spmd' <<<"$s" | head -1 | sed -E 's/^[^ ]+ remark: //' | cut -c1-40)
  [[ -z "$shard" ]] && shard="(none)"
  printf "%-24s %5d %5d %5d  %-26s %s\n" "$bn" "$((npar+nseq))" "$npar" "$nseq" "${reasons:-—}" "$shard"
done
