#!/usr/bin/env bash
set -euo pipefail

# Runtime benchmark harness for drcompiler synthetic suite.
# Compiles each .mlir program two ways (baseline / DR-optimized),
# runs both, and reports wall-clock time + speedup.

DR_OPT="${DR_OPT:-/home/tor/Dev/PhD/DRComp/drcompiler.git/onnx-mlir/build/tools/dr-opt/dr-opt}"
MLIR_OPT="${MLIR_OPT:-/home/tor/Dev/marco/install/llvm-project/bin/mlir-opt}"
MLIR_TRANSLATE="${MLIR_TRANSLATE:-/home/tor/Dev/marco/install/llvm-project/bin/mlir-translate}"
CLANG="${CLANG:-/home/tor/Dev/marco/install/llvm-project/bin/clang}"

ITERS="${1:-5}"
WORKDIR=$(mktemp -d)
trap "rm -rf $WORKDIR" EXIT

LOWER_PIPELINE="--lower-affine --convert-scf-to-cf --convert-cf-to-llvm \
  --finalize-memref-to-llvm --convert-arith-to-llvm --convert-func-to-llvm \
  --convert-math-to-llvm --convert-math-to-libm --reconcile-unrealized-casts"

compile() {
  local src="$1" dr_pipeline="$2" tag="$3"
  local base="$WORKDIR/${tag}"

  if [ -n "$dr_pipeline" ]; then
    "$DR_OPT" "$src" --pass-pipeline="$dr_pipeline" -o "${base}.dr.mlir" 2>/dev/null
    eval "$MLIR_OPT" "${base}.dr.mlir" $LOWER_PIPELINE -o "${base}.low.mlir" 2>/dev/null
  else
    eval "$MLIR_OPT" "$src" $LOWER_PIPELINE -o "${base}.low.mlir" 2>/dev/null
  fi

  "$MLIR_TRANSLATE" --mlir-to-llvmir "${base}.low.mlir" -o "${base}.ll" 2>/dev/null
  "$CLANG" -O2 "${base}.ll" -lm -o "${base}.exe" 2>/dev/null
}

time_median() {
  local exe="$1" n="$2"
  local times=()
  for ((i=0; i<n; i++)); do
    t=$( { /usr/bin/time -f '%e' "$exe" ; } 2>&1 >/dev/null | tail -1 )
    times+=("$t")
  done
  printf '%s\n' "${times[@]}" | sort -n | sed -n "$((( n + 1 ) / 2))p"
}

printf "%-45s %10s %10s %8s\n" "Benchmark" "Baseline" "DR-opt" "Speedup"
printf "%-45s %10s %10s %8s\n" "---------" "--------" "------" "-------"

run_bench() {
  local name="$1" src="$2" dr_pipe="$3"

  if ! compile "$src" "" "base_${name}" 2>/dev/null; then
    printf "%-45s %10s\n" "$name" "COMPILE_FAIL(base)"
    return
  fi
  if ! compile "$src" "$dr_pipe" "dr_${name}" 2>/dev/null; then
    printf "%-45s %10s %10s\n" "$name" "ok" "COMPILE_FAIL(dr)"
    return
  fi

  base_t=$(time_median "$WORKDIR/base_${name}.exe" "$ITERS")
  dr_t=$(time_median "$WORKDIR/dr_${name}.exe" "$ITERS")

  if [ "$(echo "$dr_t > 0" | bc -l)" = "1" ]; then
    speedup=$(echo "scale=2; $base_t / $dr_t" | bc -l)
  else
    speedup="inf"
  fi

  printf "%-45s %10ss %10ss %7sx\n" "$name" "$base_t" "$dr_t" "$speedup"
}

BENCHDIR="$(cd "$(dirname "$0")" && pwd)"

for f in "$BENCHDIR"/*.mlir; do
  name=$(basename "$f" .mlir)
  # Each file defines DR_PIPELINE in a comment: // DR_PIPELINE: ...
  dr_pipe=$(grep '^// DR_PIPELINE:' "$f" | head -1 | sed 's|^// DR_PIPELINE: *||')
  if [ -z "$dr_pipe" ]; then
    dr_pipe="builtin.module(data-recomputation{dr-recompute=true dr-cost-model=true})"
  fi
  run_bench "$name" "$f" "$dr_pipe"
done
