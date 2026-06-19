#!/usr/bin/env bash
# validate-spmd-perband.sh — execution gate for batch-1 within-sample per-band
# SPMD (PARALLEL_SPMD_SPEC.md S7).  A 3-layer kernel whose layers have DIFFERENT
# outer extents (64, 32, 32) -- so each band must shard its own axis -- with a
# cross-band read (layer 2 reads B[63-i]) that the inter-band par.barrier covers.
# Run golden-sequential vs par-spmd-perband -> par->omp across thread counts;
# output must be byte-identical every time.
set -euo pipefail
export LC_ALL=C
LL=${LLVM_INSTALL_DIR:-/home/tor/Dev/marco/install/llvm-project}
OMP_LIB=${OMP_LIB:-/usr/lib/libomp.so}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DROPT=${DROPT:-$HERE/build/tools/dr-opt/dr-opt}
OPT=$LL/bin/mlir-opt; RUN=$LL/bin/mlir-runner
D=$(mktemp -d); trap 'rm -rf "$D"' EXIT

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
func.func @perlayer(%A: memref<64x64xf32>, %D: memref<32x64xf32>) {
  %B = memref.alloc() : memref<64x64xf32>
  %C = memref.alloc() : memref<32x64xf32>
  %one = arith.constant 1.0 : f32
  %three = arith.constant 3.0 : f32
  affine.for %i = 0 to 64 { affine.for %j = 0 to 64 {
    %a = affine.load %A[%i, %j] : memref<64x64xf32>
    %b = arith.addf %a, %one : f32
    affine.store %b, %B[%i, %j] : memref<64x64xf32>
  }}
  affine.for %i = 0 to 32 { affine.for %j = 0 to 64 {
    %b0 = affine.load %B[%i, %j] : memref<64x64xf32>
    %b1 = affine.load %B[63 - %i, %j] : memref<64x64xf32>
    %s = arith.addf %b0, %b1 : f32
    affine.store %s, %C[%i, %j] : memref<32x64xf32>
  }}
  affine.for %i = 0 to 32 { affine.for %j = 0 to 64 {
    %c = affine.load %C[%i, %j] : memref<32x64xf32>
    %d = arith.mulf %c, %three : f32
    affine.store %d, %D[%i, %j] : memref<32x64xf32>
  }}
  memref.dealloc %B : memref<64x64xf32>
  memref.dealloc %C : memref<32x64xf32>
  return
}
func.func @main() {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %A = memref.alloc() : memref<64x64xf32>
  %D = memref.alloc() : memref<32x64xf32>
  scf.for %i = %c0 to %c64 step %c1 { scf.for %j = %c0 to %c64 step %c1 {
    %ix = arith.muli %i, %c64 : index
    %idx = arith.addi %ix, %j : index
    %ic = arith.index_cast %idx : index to i32
    %f = arith.sitofp %ic : i32 to f32
    memref.store %f, %A[%i, %j] : memref<64x64xf32>
  }}
  call @perlayer(%A, %D) : (memref<64x64xf32>, memref<32x64xf32>) -> ()
  %U = memref.cast %D : memref<32x64xf32> to memref<*xf32>
  call @printMemrefF32(%U) : (memref<*xf32>) -> ()
  memref.dealloc %A : memref<64x64xf32>
  memref.dealloc %D : memref<32x64xf32>
  return
}
EOF

$OPT "$D/k.mlir" "${LOWER_SEQ[@]}" -o "$D/g.ll"
$RUN "$D/g.ll" -e main -entry-point-result=void "${LIBS[@]}" | norm > "$D/g.txt"
$DROPT "$D/k.mlir" --pass-pipeline='builtin.module(dr-par-bubbles{par-spmd-perband},func.func(convert-par-to-omp))' -o "$D/o.mlir"
$OPT "$D/o.mlir" "${LOWER_OMP[@]}" -o "$D/o.ll"

fail=0; n=0
for t in 1 2 4 8 16; do
  for _ in 1 2 3; do
    n=$((n+1))
    o=$(OMP_NUM_THREADS=$t $RUN "$D/o.ll" -e main -entry-point-result=void "${LIBS[@]}" | norm)
    [ "$o" = "$(cat "$D/g.txt")" ] || { echo "MISMATCH threads=$t"; fail=$((fail+1)); }
  done
done
echo "per-band (extents 64/32/32) omp runs=$n across {1,2,4,8,16} threads, mismatches=$fail"
[ "$fail" = 0 ] && echo "RESULT: PASS — batch-1 per-band SPMD execution-validated" || { echo "RESULT: FAIL"; exit 1; }
