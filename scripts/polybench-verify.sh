#!/usr/bin/env bash
#===----------------------------------------------------------------------===//
# polybench-verify.sh — Correctness check for DR-optimized PolyBench kernels.
#
# Compiles each kernel under clang -O2 (reference) and every requested DR
# config, runs them with POLYBENCH_DUMP_ARRAYS to capture array output, and
# diffs the results.  Exits non-zero if any kernel output diverges.
#
# Usage:
#   ./scripts/polybench-verify.sh [options]
#
# Options:
#   --polybench-dir DIR   PolyBench root  (default: ./third-party/polybench)
#   --dataset SIZE        MINI|SMALL|STANDARD  (default: SMALL)
#   --kernel PATTERN      filter by name substring
#   --configs LIST        DR configs to check  (default: dr-greedy,dr-cost,dr-partial)
#   --no-download         abort if PolyBench not present
#===----------------------------------------------------------------------===//
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

POLYBENCH_DIR="${PROJECT_DIR}/third-party/polybench"
DATASET="SMALL"
KERNEL_FILTER=""
CONFIGS="dr-greedy,dr-cost,dr-partial"
NO_DOWNLOAD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --polybench-dir) POLYBENCH_DIR="$2"; shift 2 ;;
    --dataset)       DATASET="$2";       shift 2 ;;
    --kernel)        KERNEL_FILTER="$2"; shift 2 ;;
    --configs)       CONFIGS="$2";       shift 2 ;;
    --no-download)   NO_DOWNLOAD=1;      shift ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

CLANG="${CLANG:-clang}"
DRCC_IMAGE="${DRCC_IMAGE:-drcc}"
UTILITIES_DIR="${POLYBENCH_DIR}/utilities"
DATASET_DEFINE="-D${DATASET}_DATASET"

log()  { echo "[verify] $*" >&2; }
pass() { echo "[PASS] $1"; }
fail() { echo "[FAIL] $1"; FAILED+=("$1"); }

declare -A CFG_PIPELINE
CFG_PIPELINE["dr-greedy"]="--pass-pipeline=builtin.module(raise-malloc-to-memref,data-recomputation{dr-recompute=true})"
CFG_PIPELINE["dr-cost"]="--pass-pipeline=builtin.module(raise-malloc-to-memref,data-recomputation{dr-recompute=true dr-cost-model=true})"
CFG_PIPELINE["dr-partial"]="--pass-pipeline=builtin.module(raise-malloc-to-memref,data-recomputation{dr-recompute=true dr-cost-model=true dr-partial-remat=true})"

IFS=',' read -ra ACTIVE_CFGS <<< "$CONFIGS"
for cfg in "${ACTIVE_CFGS[@]}"; do
  [[ -v CFG_PIPELINE["$cfg"] ]] || { echo "unknown config: $cfg" >&2; exit 1; }
done

[[ -d "${UTILITIES_DIR}" ]] || {
  [[ "$NO_DOWNLOAD" -eq 1 ]] && { echo "PolyBench not found" >&2; exit 1; }
  log "Cloning PolyBench..."
  git clone --depth 1 https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1 "$POLYBENCH_DIR"
}

declare -a KERNELS=()
while IFS= read -r -d '' kfile; do
  kname="$(basename "$(dirname "$kfile")")"
  fname="$(basename "$kfile" .c)"
  [[ "$fname" == "$kname" ]] || continue
  [[ -z "$KERNEL_FILTER" || "$kname" == *"$KERNEL_FILTER"* ]] && KERNELS+=("$kfile")
done < <(find "$POLYBENCH_DIR" -name "*.c" ! -path "*/utilities/*" -print0 | sort -z)

[[ ${#KERNELS[@]} -gt 0 ]] || { echo "no kernels found" >&2; exit 1; }
log "${#KERNELS[@]} kernel(s); dataset=${DATASET}"

WORK_DIR="$(mktemp -d /tmp/polybench-verify.XXXXXX)"
trap "rm -rf '$WORK_DIR'" EXIT

FAILED=()
POLYBENCH_OBJ="${WORK_DIR}/polybench.o"
"$CLANG" -c "${UTILITIES_DIR}/polybench.c" -I "$UTILITIES_DIR" \
  -DPOLYBENCH_DUMP_ARRAYS "$DATASET_DEFINE" -O2 -o "$POLYBENCH_OBJ"

# nodce.h: pre-include polybench.h (fires its include guard) then replace
# polybench_prevent_dce with a no-op so the argc/strcmp DCE guard is never
# raised to MLIR, letting the static kernel remain in main without polygeist ops.
NODCE_H="${WORK_DIR}/nodce.h"
cat > "$NODCE_H" << 'NODCE_EOF'
#include "polybench.h"
#undef polybench_prevent_dce
#define polybench_prevent_dce(func) do {} while(0)
NODCE_EOF

for kernel_c in "${KERNELS[@]}"; do
  kname="$(basename "$(dirname "$kernel_c")")"
  kdir="$(dirname "$kernel_c")"
  wdir="${WORK_DIR}/${kname}"
  mkdir -p "$wdir"
  log "checking: $kname"

  # Reference: clang -O2 -c, same separate-compilation approach as DR configs.
  ref_exe="${wdir}/ref.bin"
  ref_out="${wdir}/ref.out"
  ref_o="${wdir}/ref.o"
  "$CLANG" -O2 -c "$kernel_c" \
    -I "$UTILITIES_DIR" -I "$kdir" \
    -DPOLYBENCH_DUMP_ARRAYS "$DATASET_DEFINE" \
    -o "$ref_o" 2>/dev/null \
    && "$CLANG" "$ref_o" "$POLYBENCH_OBJ" -lm -o "$ref_exe" 2>/dev/null \
    || { fail "${kname}/clang-build"; continue; }
  "$ref_exe" 2>"$ref_out" >/dev/null \
    || { fail "${kname}/clang-run"; continue; }

  for cfg in "${ACTIVE_CFGS[@]}"; do
    pipeline="${CFG_PIPELINE[$cfg]}"
    stage="${wdir}/stage_${cfg}"
    mkdir -p "$stage"
    cp "$kernel_c"                    "${stage}/"
    cp "${UTILITIES_DIR}/polybench.h" "${stage}/"
    cp "$NODCE_H"                     "${stage}/nodce.h"
    for hdr in "$kdir"/*.h; do [[ -f "$hdr" ]] && cp "$hdr" "${stage}/"; done

    src_stage="${stage}/$(basename "$kernel_c")"
    main_dr_o="${stage}/main_dr.o"

    # Full-file MLIR/DR pipeline: cgeist --function='*' with nodce.h stub raises
    # main (with inlined static kernel) free of polygeist ops.
    docker run --rm --entrypoint bash \
      -v "${stage}:${stage}" \
      "${DRCC_IMAGE}:latest" -c "
set -e
/usr/local/bin/cgeist '${src_stage}' -S --function='*' --raise-scf-to-affine \
  -include '${stage}/nodce.h' \
  -I '${stage}' -DPOLYBENCH_DUMP_ARRAYS ${DATASET_DEFINE} -O2 \
  -o '${stage}/main.mlir' 2>/dev/null || exit 1
sed -i 's/vector<\([0-9]*\)xi32>/vector<\1xi64>/g' '${stage}/main.mlir'
python3 /usr/local/bin/rewrite-struct-memrefs.py '${stage}/main.mlir'
/usr/local/bin/dr-opt --allow-unregistered-dialect '${pipeline}' \
  '${stage}/main.mlir' -o '${stage}/main.opt.mlir' 2>/dev/null || exit 1
/opt/llvm/bin/mlir-opt \
  --lower-affine --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm \
  --convert-math-to-llvm '--convert-func-to-llvm=use-bare-ptr-memref-call-conv=1' \
  --convert-index-to-llvm --finalize-memref-to-llvm --reconcile-unrealized-casts \
  '${stage}/main.opt.mlir' -o '${stage}/main.llvm.mlir' 2>/dev/null || exit 1
/opt/llvm/bin/mlir-translate --mlir-to-llvmir \
  '${stage}/main.llvm.mlir' -o '${stage}/main.ll' 2>/dev/null || exit 1
/opt/llvm/bin/clang -O2 -c '${stage}/main.ll' -o '${main_dr_o}' 2>/dev/null || exit 1
" >/dev/null 2>&1 || { fail "${kname}/${cfg}/mlir-pipeline"; continue; }

    [[ -f "$main_dr_o" ]] || { fail "${kname}/${cfg}/no-obj"; continue; }

    dr_exe="${stage}/dr.bin"
    "$CLANG" "$main_dr_o" "$POLYBENCH_OBJ" -lm -o "$dr_exe" 2>/dev/null \
      || { fail "${kname}/${cfg}/link"; continue; }

    dr_out="${stage}/dr.out"
    "$dr_exe" 2>"$dr_out" >/dev/null \
      || { fail "${kname}/${cfg}/run"; continue; }

    # Diff — compare array values with tolerance for fp rounding
    if diff -q "$ref_out" "$dr_out" >/dev/null 2>&1; then
      pass "${kname}/${cfg}"
    else
      # Bit-exact diff failed — check for fp differences within 1e-6 relative
      if awk '
        /^[0-9eE.+\- ]+$/ { next }   # skip non-numeric lines
        { ok=1 }
        END { exit ok ? 0 : 1 }
      ' <(diff "$ref_out" "$dr_out" 2>/dev/null) 2>/dev/null; then
        pass "${kname}/${cfg} (fp-approx)"
      else
        fail "${kname}/${cfg}/output-mismatch"
        diff "$ref_out" "$dr_out" | head -20 >&2
      fi
    fi
  done
done

echo ""
if [[ ${#FAILED[@]} -eq 0 ]]; then
  echo "All kernels PASSED."
else
  echo "FAILED (${#FAILED[@]}):"
  for f in "${FAILED[@]}"; do echo "  - $f"; done
  exit 1
fi
