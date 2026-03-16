#!/usr/bin/env bash
#===----------------------------------------------------------------------===//
# benchmark.sh — Compare drcc pipeline configurations on a C source file.
#
# Usage: ./scripts/benchmark.sh <source.c> [iterations]
#
# Configurations tested:
#   1. baseline    — cgeist -O2 → lower → clang (no dr-opt)
#   2. greedy      — cgeist -O2 → dr-opt (recompute all) → lower → clang
#   3. cost-model  — cgeist -O0 → dr-opt (cost-model gated) → lower → clang
#   4. fission     — cgeist -O2 → dr-opt (memory-fission) → lower → clang
#   5. clang-only  — clang -O2 (native, no MLIR, reference)
#
# Each configuration is compiled and run N times; median wall-clock time
# is reported.
#===----------------------------------------------------------------------===//
set -euo pipefail

SRC="${1:?Usage: benchmark.sh <source.c> [iterations]}"
ITERS="${2:-5}"

# --- Resolve tool paths ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/build"

# Source tool paths from the generated drcc wrapper.
eval "$(grep -E '^(CGEIST|DR_OPT|MLIR_OPT|MLIR_TRANSLATE|CLANG)=' "${BUILD_DIR}/drcc")"

BASENAME="$(basename "$SRC" | sed 's/\.[^.]*$//')"
TMPDIR="$(mktemp -d "/tmp/drbench.${BASENAME}.XXXXXX")"
trap "rm -rf '$TMPDIR'" EXIT

LOWER_FLAGS="--lower-affine --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm --convert-math-to-llvm --convert-func-to-llvm --convert-index-to-llvm --finalize-memref-to-llvm --reconcile-unrealized-casts"

log() { echo "[bench] $*" >&2; }
die() { echo "bench: error: $*" >&2; exit 1; }

# --- Helper: cgeist → MLIR with DLTI fixup ---
emit_mlir() {
  local opt_level="$1" out="$2"
  "$CGEIST" "$SRC" -S --function='*' --raise-scf-to-affine --memref-fullrank \
    "-O${opt_level}" -o "$out" 2>/dev/null \
    || die "cgeist -O${opt_level} failed"
  sed -i 's/vector<\([0-9]*\)xi32>/vector<\1xi64>/g' "$out"
}

# --- Helper: MLIR → executable ---
mlir_to_exe() {
  local mlir="$1" exe="$2"
  local llvm_mlir="${mlir%.mlir}.llvm.mlir"
  local ll="${mlir%.mlir}.ll"
  "$MLIR_OPT" $LOWER_FLAGS "$mlir" -o "$llvm_mlir" 2>/dev/null \
    || die "mlir-opt lowering failed"
  "$MLIR_TRANSLATE" --mlir-to-llvmir "$llvm_mlir" -o "$ll" 2>/dev/null \
    || die "mlir-translate failed"
  "$CLANG" -O2 "$ll" -o "$exe" -lm 2>/dev/null \
    || die "clang backend failed"
}

# --- Helper: time N runs, report median ---
time_median() {
  local exe="$1" n="$2"
  local times=()
  for ((i = 0; i < n; i++)); do
    local t
    t=$( { time "$exe" >/dev/null 2>&1; } 2>&1 | grep real | sed 's/real\t//')
    # Convert to seconds
    local min sec
    min=$(echo "$t" | sed 's/m.*//')
    sec=$(echo "$t" | sed 's/.*m//' | sed 's/s//')
    local total
    total=$(echo "$min * 60 + $sec" | bc)
    times+=("$total")
  done
  # Sort and pick median
  IFS=$'\n' sorted=($(sort -n <<<"${times[*]}")); unset IFS
  local mid=$(( n / 2 ))
  echo "${sorted[$mid]}"
}

echo "=== Benchmark: $SRC ($ITERS iterations) ==="
echo ""

# --- Config 1: baseline (cgeist -O2, no dr-opt) ---
log "Building: baseline (cgeist -O2, no dr-opt)"
emit_mlir 2 "$TMPDIR/baseline.mlir"
mlir_to_exe "$TMPDIR/baseline.mlir" "$TMPDIR/baseline"
T_BASE=$(time_median "$TMPDIR/baseline" "$ITERS")
echo "  baseline (cgeist -O2):           ${T_BASE}s"

# --- Config 2: greedy recompute (cgeist -O2 + dr-opt recompute) ---
log "Building: greedy (cgeist -O2 + recompute-all)"
emit_mlir 2 "$TMPDIR/greedy.mlir"
"$DR_OPT" --pass-pipeline='builtin.module(raise-malloc-to-memref,data-recomputation{dr-recompute=true})' \
  "$TMPDIR/greedy.mlir" -o "$TMPDIR/greedy.opt.mlir" 2>/dev/null
mlir_to_exe "$TMPDIR/greedy.opt.mlir" "$TMPDIR/greedy"
T_GREEDY=$(time_median "$TMPDIR/greedy" "$ITERS")
echo "  greedy recompute:                ${T_GREEDY}s"

# --- Config 3: cost-model (cgeist -O0 + dr-opt cost-model) ---
log "Building: cost-model (cgeist -O0 + cost-model)"
emit_mlir 0 "$TMPDIR/costmodel.mlir"
"$DR_OPT" --pass-pipeline='builtin.module(raise-malloc-to-memref,data-recomputation{dr-recompute=true dr-cost-model=true})' \
  "$TMPDIR/costmodel.mlir" -o "$TMPDIR/costmodel.opt.mlir" 2>/dev/null
mlir_to_exe "$TMPDIR/costmodel.opt.mlir" "$TMPDIR/costmodel"
T_COST=$(time_median "$TMPDIR/costmodel" "$ITERS")
echo "  cost-model (cgeist -O0):         ${T_COST}s"

# --- Config 4: fission (cgeist -O2 + memory-fission) ---
log "Building: fission (cgeist -O2 + memory-fission)"
emit_mlir 2 "$TMPDIR/fission.mlir"
"$DR_OPT" --pass-pipeline='builtin.module(memory-fission)' \
  "$TMPDIR/fission.mlir" -o "$TMPDIR/fission.opt.mlir" 2>/dev/null
mlir_to_exe "$TMPDIR/fission.opt.mlir" "$TMPDIR/fission"
T_FISSION=$(time_median "$TMPDIR/fission" "$ITERS")
echo "  memory-fission (cgeist -O2):     ${T_FISSION}s"

# --- Config 5: clang -O2 reference ---
log "Building: clang -O2 reference"
"$CLANG" -O2 "$SRC" -o "$TMPDIR/clang_ref" -lm 2>/dev/null \
  || die "clang -O2 failed"
T_CLANG=$(time_median "$TMPDIR/clang_ref" "$ITERS")
echo "  clang -O2 (reference):           ${T_CLANG}s"

echo ""
echo "=== Summary ==="
printf "  %-35s %s\n" "Configuration" "Median time (s)"
printf "  %-35s %s\n" "---" "---"
printf "  %-35s %s\n" "baseline (cgeist -O2, no dr-opt)" "$T_BASE"
printf "  %-35s %s\n" "greedy recompute" "$T_GREEDY"
printf "  %-35s %s\n" "cost-model (cgeist -O0)" "$T_COST"
printf "  %-35s %s\n" "memory-fission (cgeist -O2)" "$T_FISSION"
printf "  %-35s %s\n" "clang -O2 (reference)" "$T_CLANG"
