#!/usr/bin/env bash
# Spill-point roofline for stencil jacobi-2d: scaling vs N (seq-1t, in-house
# par-16t) + the dr-affine-loop-tile decision on the spatial band.
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
tdec(){ $DROPT --allow-unregistered-dialect --pass-pipeline="builtin.module(func.func(dr-affine-loop-tile{llc-gate-from-model=true emit-rationale=true}))" "$1" 2>&1 | grep -oE 'tile-rationale: (TILE|REJECT|SKIP)[^ ]*( reason=[^ ]*)?' | head -1; }

jac(){ local N=$1 T=$2; cat <<EOF
func.func private @printMemrefF64(memref<*xf64>)
func.func private @rtclock() -> f64
func.func @jac(%A: memref<${N}x${N}xf64>, %B: memref<${N}x${N}xf64>) {
  %c2 = arith.constant 0.2 : f64
  affine.for %t = 0 to $T {
    affine.for %i = 1 to $((N-1)) { affine.for %j = 1 to $((N-1)) {
      %0 = affine.load %A[%i,%j] : memref<${N}x${N}xf64>
      %1 = affine.load %A[%i,%j - 1] : memref<${N}x${N}xf64>
      %2 = affine.load %A[%i,%j + 1] : memref<${N}x${N}xf64>
      %3 = affine.load %A[%i - 1,%j] : memref<${N}x${N}xf64>
      %4 = affine.load %A[%i + 1,%j] : memref<${N}x${N}xf64>
      %5 = arith.addf %0,%1 : f64
      %6 = arith.addf %5,%2 : f64
      %7 = arith.addf %6,%3 : f64
      %8 = arith.addf %7,%4 : f64
      %9 = arith.mulf %8,%c2 : f64
      affine.store %9, %B[%i,%j] : memref<${N}x${N}xf64>
    }}
    affine.for %i = 1 to $((N-1)) { affine.for %j = 1 to $((N-1)) {
      %0 = affine.load %B[%i,%j] : memref<${N}x${N}xf64>
      %1 = affine.load %B[%i,%j - 1] : memref<${N}x${N}xf64>
      %2 = affine.load %B[%i,%j + 1] : memref<${N}x${N}xf64>
      %3 = affine.load %B[%i - 1,%j] : memref<${N}x${N}xf64>
      %4 = affine.load %B[%i + 1,%j] : memref<${N}x${N}xf64>
      %5 = arith.addf %0,%1 : f64
      %6 = arith.addf %5,%2 : f64
      %7 = arith.addf %6,%3 : f64
      %8 = arith.addf %7,%4 : f64
      %9 = arith.mulf %8,%c2 : f64
      affine.store %9, %A[%i,%j] : memref<${N}x${N}xf64>
    }}
  }
  return
}
func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %Nn = arith.constant $N : index
  %A = memref.alloc() : memref<${N}x${N}xf64>
  %B = memref.alloc() : memref<${N}x${N}xf64>
  %v = arith.constant 1.0 : f64
  scf.for %i = %c0 to %Nn step %c1 { scf.for %j = %c0 to %Nn step %c1 {
    memref.store %v, %A[%i,%j] : memref<${N}x${N}xf64>
    memref.store %v, %B[%i,%j] : memref<${N}x${N}xf64>
  }}
  %T = memref.alloc() : memref<3xf64>
  %c3 = arith.constant 3 : index
  scf.for %r = %c0 to %c3 step %c1 {
    %t0 = func.call @rtclock() : () -> f64
    func.call @jac(%A,%B) : (memref<${N}x${N}xf64>,memref<${N}x${N}xf64>) -> ()
    %t1 = func.call @rtclock() : () -> f64
    %dt = arith.subf %t1,%t0 : f64
    memref.store %dt, %T[%r] : memref<3xf64>
  }
  %U = memref.cast %T : memref<3xf64> to memref<*xf64>
  func.call @printMemrefF64(%U) : (memref<*xf64>) -> ()
  return
}
EOF
}

TT=30
echo "=== stencil jacobi-2d (T=$TT, GFLOP=T*2*N^2*5, working set 2N^2*8) ==="
printf "%-6s %6s | %-12s | %-14s | %s\n" "N" "2A(MB)" "seq 1t" "par 16t" "tile-decision (spatial)"
for N in 2048 3072 4096 6144; do
  jac "$N" "$TT" > "$D/k.mlir"
  $OPT "$D/k.mlir" "${SEQ[@]}" -o "$D/s.ll" 2>/dev/null
  st=$(run1 1 "$D/s.ll" | med)
  $DROPT --allow-unregistered-dialect --pass-pipeline='builtin.module(dr-par-bubbles{par-spmd-perband},func.func(convert-par-to-omp))' "$D/k.mlir" -o "$D/p.mlir" 2>/dev/null
  $OPT "$D/p.mlir" "${OMPL[@]}" -o "$D/p.ll" 2>/dev/null
  pt=$(run1 16 "$D/p.ll" | med)
  fl=$(awk -v n=$N -v t=$TT 'BEGIN{print t*2.0*n*n*5}')
  amb=$(awk -v n=$N 'BEGIN{printf "%d",2*n*n*8/1048576}')
  printf "%-6s %6s | %5ss %4sGF | %7ss %4sGF | %s\n" "$N" "$amb" "$st" "$(gfe $fl $st)" "$pt" "$(gfe $fl $pt)" "$(tdec "$D/k.mlir")"
done