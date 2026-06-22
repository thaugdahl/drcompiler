#!/usr/bin/env bash
# Spill-point roofline for BLAS-2 (mvt) and stencil (jacobi-2d): scaling vs N
# (seq-1t, in-house par-16t) + the dr-affine-loop-tile decision (tile/reject).
set -uo pipefail
export LC_ALL=C
LL=/home/tor/Dev/marco/install/llvm-project
DROPT=/home/tor/Dev/PhD/DRComp/drcompiler.git/onnx-mlir/build/tools/dr-opt/dr-opt
OPT=$LL/bin/mlir-opt; RUN="$LL/bin/mlir-runner --O3"
D=$(mktemp -d); trap 'rm -rf "$D"' EXIT
LIBS=(--shared-libs="$LL/lib/libmlir_runner_utils.so" --shared-libs="$LL/lib/libmlir_c_runner_utils.so" --shared-libs=/usr/lib/libomp.so)
SEQ=(--lower-affine --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts)
OMPL=(--lower-affine --convert-scf-to-cf --convert-openmp-to-llvm --convert-cf-to-llvm --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts)
med(){ awk '/data =/{g++} g==1' | grep -oE '\[[^]]*\]' | tr -d '[] ' | tr ',' '\n' | grep -E '^[0-9.eE+-]+$' | sort -g | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }
gfe(){ awk -v f="$1" -v t="$2" 'BEGIN{printf "%.1f",(t>0)?f/t/1e9:0}'; }
run1(){ OMP_NUM_THREADS="$1" OMP_PROC_BIND=close OMP_PLACES=cores $RUN "$2" -e main -entry-point-result=void "${LIBS[@]}" 2>/dev/null; }
tiledecision(){ $DROPT --allow-unregistered-dialect --pass-pipeline="builtin.module(func.func(dr-affine-loop-tile{llc-gate-from-model=true emit-rationale=true}))" "$1" 2>&1 | grep -oE 'tile-rationale: (TILE|REJECT|SKIP)[^ ]*( reason=[^ ]*)?' | head -1; }

# ---- mvt: x1 += A*y1 ; x2 += A^T*y2  (BLAS-2, no temporal reuse of A) ----
mvt(){ local N=$1; cat <<EOF
func.func private @printMemrefF64(memref<*xf64>)
func.func private @rtclock() -> f64
func.func @mvt(%A: memref<${N}x${N}xf64>, %x1: memref<${N}xf64>, %x2: memref<${N}xf64>, %y1: memref<${N}xf64>, %y2: memref<${N}xf64>) {
  affine.for %i = 0 to $N { affine.for %j = 0 to $N {
    %a = affine.load %A[%i,%j] : memref<${N}x${N}xf64>
    %y = affine.load %y1[%j] : memref<${N}xf64>
    %x = affine.load %x1[%i] : memref<${N}xf64>
    %p = arith.mulf %a,%y : f64
    %s = arith.addf %x,%p : f64
    affine.store %s, %x1[%i] : memref<${N}xf64>
  }}
  affine.for %i = 0 to $N { affine.for %j = 0 to $N {
    %a = affine.load %A[%j,%i] : memref<${N}x${N}xf64>
    %y = affine.load %y2[%j] : memref<${N}xf64>
    %x = affine.load %x2[%i] : memref<${N}xf64>
    %p = arith.mulf %a,%y : f64
    %s = arith.addf %x,%p : f64
    affine.store %s, %x2[%i] : memref<${N}xf64>
  }}
  return
}
func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %Nn = arith.constant $N : index
  %A = memref.alloc() : memref<${N}x${N}xf64>
  %x1 = memref.alloc() : memref<${N}xf64>
  %x2 = memref.alloc() : memref<${N}xf64>
  %y1 = memref.alloc() : memref<${N}xf64>
  %y2 = memref.alloc() : memref<${N}xf64>
  %v = arith.constant 1.0 : f64
  scf.for %i = %c0 to %Nn step %c1 {
    memref.store %v, %x1[%i] : memref<${N}xf64>
    memref.store %v, %x2[%i] : memref<${N}xf64>
    memref.store %v, %y1[%i] : memref<${N}xf64>
    memref.store %v, %y2[%i] : memref<${N}xf64>
    scf.for %j = %c0 to %Nn step %c1 { memref.store %v, %A[%i,%j] : memref<${N}x${N}xf64> }
  }
  %T = memref.alloc() : memref<5xf64>
  scf.for %r = %c0 to %c1 step %c1 { }
  scf.for %r = %c0 to %Nn step %c1 { }
  %c5 = arith.constant 5 : index
  scf.for %r = %c0 to %c5 step %c1 {
    %t0 = func.call @rtclock() : () -> f64
    func.call @mvt(%A,%x1,%x2,%y1,%y2) : (memref<${N}x${N}xf64>,memref<${N}xf64>,memref<${N}xf64>,memref<${N}xf64>,memref<${N}xf64>) -> ()
    %t1 = func.call @rtclock() : () -> f64
    %dt = arith.subf %t1,%t0 : f64
    memref.store %dt, %T[%r] : memref<5xf64>
  }
  %U = memref.cast %T : memref<5xf64> to memref<*xf64>
  func.call @printMemrefF64(%U) : (memref<*xf64>) -> ()
  return
}
EOF
}

echo "=== BLAS-2 mvt (GFLOP=4N^2, working set A=N^2*8) ==="
printf "%-6s %6s | %-12s | %-14s | %s\n" "N" "A(MB)" "seq 1t" "par 16t" "tile-decision"
for N in 1024 2048 4096; do
  mvt "$N" > "$D/k.mlir"
  $OPT "$D/k.mlir" "${SEQ[@]}" -o "$D/s.ll" 2>/dev/null
  st=$(run1 1 "$D/s.ll" | med)
  $DROPT --allow-unregistered-dialect --pass-pipeline='builtin.module(dr-par-bubbles{par-spmd-perband},func.func(convert-par-to-omp))' "$D/k.mlir" -o "$D/p.mlir" 2>/dev/null
  $OPT "$D/p.mlir" "${OMPL[@]}" -o "$D/p.ll" 2>/dev/null
  pt=$(run1 16 "$D/p.ll" | med)
  fl=$(awk -v n=$N 'BEGIN{print 4.0*n*n}')
  amb=$(awk -v n=$N 'BEGIN{printf "%d",n*n*8/1048576}')
  printf "%-6s %6s | %5ss %4sGF | %7ss %4sGF | %s\n" "$N" "$amb" "$st" "$(gfe $fl $st)" "$pt" "$(gfe $fl $pt)" "$(tiledecision "$D/k.mlir")"
done