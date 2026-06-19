#!/usr/bin/env bash
# validate-spmd-omp.sh — execution gate for the whole-kernel SPMD path
# (PARALLEL_SPMD_SPEC.md S2/S3).  Compiles a clean multi-dim shard kernel three
# ways and checks the outputs are byte-identical:
#
#   golden : untransformed affine kernel, lowered sequentially
#   seq    : dr-par-bubbles{par-spmd} -> convert-par-to-scf  (sequential ref)
#   omp    : dr-par-bubbles{par-spmd} -> convert-par-to-omp  (real OpenMP team)
#
# The kernel has two owner-aligned (ELIDE) bands that fuse into one wsloop and a
# transpose band (a genuine cross-shard dependence) behind the one kept barrier.
# Identical output under many threads => the barrier elision + nowait placement
# is execution-correct.  Requires an LLVM 22 install with mlir-opt/mlir-runner
# and an OpenMP runtime.
#
# Usage: LLVM_INSTALL_DIR=/path/to/llvm scripts/validate-spmd-omp.sh
set -euo pipefail

LL=${LLVM_INSTALL_DIR:-/home/tor/Dev/marco/install/llvm-project}
OMP_LIB=${OMP_LIB:-/usr/lib/libomp.so}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DROPT=${DROPT:-$HERE/build/tools/dr-opt/dr-opt}
OPT=$LL/bin/mlir-opt; RUN=$LL/bin/mlir-runner
D=$(mktemp -d)
trap 'rm -rf "$D"' EXIT

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

cat > "$D/k.mlir" <<'EOF'
func.func private @printMemrefF32(memref<*xf32>)
func.func @kernel(%A: memref<8x8xf32>, %D: memref<8x8xf32>) {
  %B = memref.alloc() : memref<8x8xf32>
  %C = memref.alloc() : memref<8x8xf32>
  %two = arith.constant 2.0 : f32
  %one = arith.constant 1.0 : f32
  affine.for %n = 0 to 8 { affine.for %j = 0 to 8 {
    %a = affine.load %A[%n, %j] : memref<8x8xf32>
    %m = arith.mulf %a, %two : f32
    %p = arith.addf %m, %one : f32
    affine.store %p, %B[%n, %j] : memref<8x8xf32>
  }}
  affine.for %n = 0 to 8 { affine.for %j = 0 to 8 {
    %b = affine.load %B[%n, %j] : memref<8x8xf32>
    %s = arith.mulf %b, %b : f32
    affine.store %s, %C[%n, %j] : memref<8x8xf32>
  }}
  affine.for %n = 0 to 8 { affine.for %j = 0 to 8 {
    %c = affine.load %C[%j, %n] : memref<8x8xf32>
    affine.store %c, %D[%n, %j] : memref<8x8xf32>
  }}
  memref.dealloc %B : memref<8x8xf32>
  memref.dealloc %C : memref<8x8xf32>
  return
}
func.func @main() {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %A = memref.alloc() : memref<8x8xf32>
  %D = memref.alloc() : memref<8x8xf32>
  scf.for %n = %c0 to %c8 step %c1 {
    scf.for %j = %c0 to %c8 step %c1 {
      %n8 = arith.muli %n, %c8 : index
      %idx = arith.addi %n8, %j : index
      %ic = arith.index_cast %idx : index to i32
      %f = arith.sitofp %ic : i32 to f32
      memref.store %f, %A[%n, %j] : memref<8x8xf32>
    }
  }
  call @kernel(%A, %D) : (memref<8x8xf32>, memref<8x8xf32>) -> ()
  %U = memref.cast %D : memref<8x8xf32> to memref<*xf32>
  call @printMemrefF32(%U) : (memref<*xf32>) -> ()
  memref.dealloc %A : memref<8x8xf32>
  memref.dealloc %D : memref<8x8xf32>
  return
}
EOF

SPMD='builtin.module(dr-par-bubbles{par-spmd},func.func(convert-par-to-scf))'
SPMD_OMP='builtin.module(dr-par-bubbles{par-spmd},func.func(convert-par-to-omp))'

$OPT "$D/k.mlir" "${LOWER_SEQ[@]}" -o "$D/golden.ll.mlir"
$RUN "$D/golden.ll.mlir" -e main -entry-point-result=void "${LIBS[@]}" | norm > "$D/golden.txt"

$DROPT "$D/k.mlir" --pass-pipeline="$SPMD" -o "$D/seq.mlir"
$OPT "$D/seq.mlir" "${LOWER_SEQ[@]}" -o "$D/seq.ll.mlir"
$RUN "$D/seq.ll.mlir" -e main -entry-point-result=void "${LIBS[@]}" | norm > "$D/seq.txt"

$DROPT "$D/k.mlir" --pass-pipeline="$SPMD_OMP" -o "$D/omp.mlir"
$OPT "$D/omp.mlir" "${LOWER_OMP[@]}" -o "$D/omp.ll.mlir"

fail=0; n=0
for t in 1 2 3 4 8 16; do
  for _ in 1 2 3 4 5; do
    n=$((n+1))
    o=$(OMP_NUM_THREADS=$t $RUN "$D/omp.ll.mlir" -e main -entry-point-result=void "${LIBS[@]}" | norm)
    [ "$o" = "$(cat "$D/golden.txt")" ] || { echo "MISMATCH threads=$t"; fail=$((fail+1)); }
  done
done

diff -q "$D/golden.txt" "$D/seq.txt" >/dev/null && seqok=ok || seqok=DIFF
echo "golden==seq : $seqok"
echo "omp runs    : $n across threads {1,2,3,4,8,16}, mismatches=$fail"
if [ "$seqok" = ok ] && [ "$fail" = 0 ]; then
  echo "RESULT: PASS — SPMD materialization + par->omp execution-validated"
else
  echo "RESULT: FAIL"; exit 1
fi
