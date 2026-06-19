#!/usr/bin/env bash
# validate-spmd-dyn.sh — execution gate for DYNAMIC-extent whole-kernel SPMD
# (PARALLEL_SPMD_SPEC.md S2, dynamic shard axis).  A two-layer owner-aligned
# kernel over a RUNTIME batch %N (memref<?x...>, par.forall carries %N as its
# dynamic upper bound) is run untransformed-sequential (golden) vs par->omp
# across thread counts; the printed output must be byte-identical every time.
#
# Usage: LLVM_INSTALL_DIR=/path/to/llvm scripts/validate-spmd-dyn.sh
set -euo pipefail
export LC_ALL=C
LL=${LLVM_INSTALL_DIR:-/home/tor/Dev/marco/install/llvm-project}
OMP_LIB=${OMP_LIB:-/usr/lib/libomp.so}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DROPT=${DROPT:-$HERE/build/tools/dr-opt/dr-opt}
OPT=$LL/bin/mlir-opt; RUN=$LL/bin/mlir-runner
D=$(mktemp -d); trap 'rm -rf "$D"' EXIT
N=${1:-48}

LIBS=(--shared-libs="$LL/lib/libmlir_runner_utils.so"
      --shared-libs="$LL/lib/libmlir_c_runner_utils.so"
      --shared-libs="$OMP_LIB")
LOWER_SEQ=(--lower-affine --convert-scf-to-cf --convert-cf-to-llvm
           --convert-arith-to-llvm --finalize-memref-to-llvm
           --convert-func-to-llvm --reconcile-unrealized-casts)
LOWER_OMP=(--convert-scf-to-cf --convert-openmp-to-llvm --convert-cf-to-llvm
           --convert-arith-to-llvm --finalize-memref-to-llvm
           --convert-func-to-llvm --reconcile-unrealized-casts)
norm(){ tr -d '\n' | sed 's/.*data =//'; }

cat > "$D/k.mlir" <<EOF
func.func private @printMemrefF32(memref<*xf32>)
func.func @kernel(%A: memref<?x16xf32>, %D: memref<?x16xf32>, %N: index) {
  %B = memref.alloc(%N) : memref<?x16xf32>
  %two = arith.constant 2.0 : f32
  affine.for %n = 0 to %N { affine.for %j = 0 to 16 {
    %a = affine.load %A[%n, %j] : memref<?x16xf32>
    %m = arith.mulf %a, %two : f32
    affine.store %m, %B[%n, %j] : memref<?x16xf32>
  }}
  affine.for %n = 0 to %N { affine.for %j = 0 to 16 {
    %b = affine.load %B[%n, %j] : memref<?x16xf32>
    %s = arith.addf %b, %b : f32
    affine.store %s, %D[%n, %j] : memref<?x16xf32>
  }}
  memref.dealloc %B : memref<?x16xf32>
  return
}
func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %N = arith.constant $N : index
  %A = memref.alloc(%N) : memref<?x16xf32>
  %D = memref.alloc(%N) : memref<?x16xf32>
  scf.for %n = %c0 to %N step %c1 {
    scf.for %j = %c0 to %c16 step %c1 {
      %nm = arith.muli %n, %c16 : index
      %idx = arith.addi %nm, %j : index
      %ic = arith.index_cast %idx : index to i32
      %f = arith.sitofp %ic : i32 to f32
      memref.store %f, %A[%n, %j] : memref<?x16xf32>
    }
  }
  call @kernel(%A, %D, %N) : (memref<?x16xf32>, memref<?x16xf32>, index) -> ()
  %U = memref.cast %D : memref<?x16xf32> to memref<*xf32>
  call @printMemrefF32(%U) : (memref<*xf32>) -> ()
  memref.dealloc %A : memref<?x16xf32>
  memref.dealloc %D : memref<?x16xf32>
  return
}
EOF

$OPT "$D/k.mlir" "${LOWER_SEQ[@]}" -o "$D/g.ll"
$RUN "$D/g.ll" -e main -entry-point-result=void "${LIBS[@]}" | norm > "$D/g.txt"
$DROPT "$D/k.mlir" --pass-pipeline='builtin.module(dr-par-bubbles{par-spmd},func.func(convert-par-to-omp))' -o "$D/o.mlir"
$OPT "$D/o.mlir" "${LOWER_OMP[@]}" -o "$D/o.ll"

fail=0; n=0
for t in 1 2 4 8 16; do
  for _ in 1 2 3; do
    n=$((n+1))
    o=$(OMP_NUM_THREADS=$t $RUN "$D/o.ll" -e main -entry-point-result=void "${LIBS[@]}" | norm)
    [ "$o" = "$(cat "$D/g.txt")" ] || { echo "MISMATCH threads=$t"; fail=$((fail+1)); }
  done
done
echo "dynamic batch N=$N; omp runs=$n across {1,2,4,8,16} threads, mismatches=$fail"
[ "$fail" = 0 ] && echo "RESULT: PASS — dynamic-extent SPMD execution-validated" || { echo "RESULT: FAIL"; exit 1; }
