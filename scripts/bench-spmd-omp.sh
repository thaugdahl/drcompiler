#!/usr/bin/env bash
# bench-spmd-omp.sh — measured-speedup study for whole-kernel batch SPMD
# (PARALLEL_SPMD_SPEC.md S3).  Generates a large clean batch-shardable kernel
# (gen_spmd_perf_kernel.py), compiles it golden-sequential and via par->omp, and
# times the kernel sequentially vs an OpenMP team across thread counts.  Prints
# per-call seconds, the output checksum (must match across all paths), and the
# speedup vs the sequential golden.
#
# Usage: [N M K P R] as env or args; LLVM_INSTALL_DIR / OMP_LIB overridable.
set -euo pipefail
export LC_ALL=C   # rtclock prints '0.19' (dot); keep printf/awk dot-decimal
N=${1:-64}; M=${2:-4096}; K=${3:-4}; P=${4:-192}; R=${5:-3}; IL=${6:-0}
LL=${LLVM_INSTALL_DIR:-/home/tor/Dev/marco/install/llvm-project}
OMP_LIB=${OMP_LIB:-/usr/lib/libomp.so}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DROPT=${DROPT:-$HERE/build/tools/dr-opt/dr-opt}
OPT=$LL/bin/mlir-opt; RUN=$LL/bin/mlir-runner
D=$(mktemp -d); trap 'rm -rf "$D"' EXIT
THREADS=${THREADS:-"1 2 4 8 16"}

LIBS=(--shared-libs="$LL/lib/libmlir_runner_utils.so"
      --shared-libs="$LL/lib/libmlir_c_runner_utils.so"
      --shared-libs="$OMP_LIB")
LOWER_SEQ=(--lower-affine --convert-scf-to-cf --convert-cf-to-llvm
           --convert-arith-to-llvm --finalize-memref-to-llvm
           --convert-func-to-llvm --reconcile-unrealized-casts)
LOWER_OMP=(--convert-scf-to-cf --convert-openmp-to-llvm --convert-cf-to-llvm
           --convert-arith-to-llvm --finalize-memref-to-llvm
           --convert-func-to-llvm --reconcile-unrealized-casts)

python3 "$HERE/scripts/gen_spmd_perf_kernel.py" "$N" "$M" "$K" "$P" "$R" "$IL" > "$D/k.mlir"
echo "kernel: N=$N M=$M layers=$K steps/elem=$P repeats=$R interleave-allocs=$IL"

# barrier count in the SPMD form (informational)
nb=$($DROPT "$D/k.mlir" --pass-pipeline='builtin.module(dr-par-bubbles{par-spmd})' 2>/dev/null | grep -c 'par.barrier' || true)
nf=$($DROPT "$D/k.mlir" --pass-pipeline='builtin.module(dr-par-bubbles{par-spmd})' 2>/dev/null | grep -c 'par.forall' || true)
echo "SPMD form: $nf forall(s), $nb barrier(s)"

# golden (untransformed, sequential)
$OPT "$D/k.mlir" "${LOWER_SEQ[@]}" -o "$D/golden.ll" 2>/dev/null
$DROPT "$D/k.mlir" --pass-pipeline='builtin.module(dr-par-bubbles{par-spmd},func.func(convert-par-to-omp))' -o "$D/omp.mlir" 2>/dev/null
$OPT "$D/omp.mlir" "${LOWER_OMP[@]}" -o "$D/omp.ll" 2>/dev/null

# best-of-3 wall time per call
best(){ local f=$1 t=$2 b=""; for _ in 1 2 3; do
    out=$(OMP_NUM_THREADS=$t OMP_PROC_BIND=close OMP_PLACES=cores \
          $RUN "$f" -e main -entry-point-result=void "${LIBS[@]}" 2>/dev/null)
    s=$(echo "$out" | head -1); ck=$(echo "$out" | tail -1 | sed 's/.*data =//' | tr -d ' []')
    if [ -z "$b" ] || awk "BEGIN{exit !($s<$b)}"; then b=$s; fi
  done; echo "$b $ck"; }

read gs gck < <(best "$D/golden.ll" 1)
printf "golden  seq          : %.4fs   checksum=%s\n" "$gs" "$gck"
for t in $THREADS; do
  read os ock < <(best "$D/omp.ll" "$t")
  sp=$(awk "BEGIN{printf \"%.2f\", $gs/$os}")
  ckmark=$([ "$ock" = "$gck" ] && echo OK || echo "MISMATCH($ock)")
  printf "par->omp  %2d thread%s : %.4fs   speedup=%5sx  checksum=%s\n" \
    "$t" "$([ "$t" = 1 ] && echo ' ' || echo s)" "$os" "$sp" "$ckmark"
done
