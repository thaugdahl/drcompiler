#!/usr/bin/env bash
# Does stencil time-tiling recover the jacobi-2D single-thread L3-spill?
# untiled-seq vs time-tiled-seq at N=2048 (spilled), checksum-verified.
set -uo pipefail
export LC_ALL=C
LL=/home/tor/Dev/marco/install/llvm-project
DROPT=/home/tor/Dev/PhD/DRComp/drcompiler.git/onnx-mlir/build/tools/dr-opt/dr-opt
OPT=$LL/bin/mlir-opt; RUN="$LL/bin/mlir-runner --O3"
D=$(mktemp -d); trap 'rm -rf "$D"' EXIT
LIBS=(--shared-libs="$LL/lib/libmlir_runner_utils.so" --shared-libs="$LL/lib/libmlir_c_runner_utils.so" --shared-libs=/usr/lib/libomp.so)
SEQ=(--lower-affine --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts)
med(){ awk '/data =/{g++} g==1' | grep -oE '\[[^]]*\]' | tr -d '[] ' | tr ',' '\n' | grep -E '^[0-9.eE+-]+$' | sort -g | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }
ckv(){ grep -oE '\[[^]]*\]' | tail -1 | tr -d '[] '; }
gfe(){ awk -v f="$1" -v t="$2" 'BEGIN{printf "%.1f",(t>0)?f/t/1e9:0}'; }
N=2048; M=$((N-1)); TT=30
cat > "$D/k.mlir" <<EOF
func.func private @printMemrefF64(memref<*xf64>)
func.func private @rtclock() -> f64
func.func @jac(%A: memref<${N}x${N}xf64>, %B: memref<${N}x${N}xf64>) {
  %c2 = arith.constant 0.2 : f64
  affine.for %t = 0 to ${TT} {
    affine.for %i = 1 to ${M} { affine.for %j = 1 to ${M} {
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
    affine.for %i = 1 to ${M} { affine.for %j = 1 to ${M} {
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
  %Nn = arith.constant ${N} : index
  %A = memref.alloc() : memref<${N}x${N}xf64>
  %B = memref.alloc() : memref<${N}x${N}xf64>
  scf.for %i = %c0 to %Nn step %c1 { scf.for %j = %c0 to %Nn step %c1 {
    %t = arith.muli %i, %Nn : index
    %t2 = arith.addi %t, %j : index
    %p = arith.constant 7 : index
    %m = arith.remui %t2, %p : index
    %mi = arith.index_cast %m : index to i64
    %mf = arith.sitofp %mi : i64 to f64
    %dd = arith.constant 7.0 : f64
    %v = arith.divf %mf, %dd : f64
    memref.store %v, %A[%i,%j] : memref<${N}x${N}xf64>
    memref.store %v, %B[%i,%j] : memref<${N}x${N}xf64>
  }}
  %Tm = memref.alloc() : memref<3xf64>
  %c3 = arith.constant 3 : index
  scf.for %r = %c0 to %c3 step %c1 {
    %t0 = func.call @rtclock() : () -> f64
    func.call @jac(%A,%B) : (memref<${N}x${N}xf64>,memref<${N}x${N}xf64>) -> ()
    %t1 = func.call @rtclock() : () -> f64
    %dt = arith.subf %t1,%t0 : f64
    memref.store %dt, %Tm[%r] : memref<3xf64>
  }
  %U = memref.cast %Tm : memref<3xf64> to memref<*xf64>
  func.call @printMemrefF64(%U) : (memref<*xf64>) -> ()
  %CK = memref.alloc() : memref<1xf64>
  %zd = arith.constant 0.0 : f64
  memref.store %zd, %CK[%c0] : memref<1xf64>
  scf.for %i = %c0 to %Nn step %c1 { scf.for %j = %c0 to %Nn step %c1 {
    %e = memref.load %A[%i,%j] : memref<${N}x${N}xf64>
    %acc = memref.load %CK[%c0] : memref<1xf64>
    %na = arith.addf %acc,%e : f64
    memref.store %na, %CK[%c0] : memref<1xf64>
  }}
  %CKU = memref.cast %CK : memref<1xf64> to memref<*xf64>
  func.call @printMemrefF64(%CKU) : (memref<*xf64>) -> ()
  return
}
EOF
fl=$(awk -v n=$N -v t=$TT 'BEGIN{print t*2.0*n*n*5}')
$OPT "$D/k.mlir" "${SEQ[@]}" -o "$D/u.ll" 2>/dev/null
o=$(OMP_NUM_THREADS=1 $RUN "$D/u.ll" -e main -entry-point-result=void "${LIBS[@]}" 2>/dev/null); ut=$(echo "$o"|med); base=$(echo "$o"|ckv)
echo "untiled seq 1t : ${ut}s  $(gfe $fl $ut)GF  checksum=$base"
for TS in 32 64 128 256; do
  for TM in 4 8; do
    $DROPT --allow-unregistered-dialect --pass-pipeline="builtin.module(func.func(dr-affine-stencil-time-tile{force-tile=true tile-t=$TM tile-s=$TS}))" "$D/k.mlir" -o "$D/t.mlir" 2>/dev/null
    $OPT "$D/t.mlir" "${SEQ[@]}" -o "$D/t.ll" 2>/dev/null
    o=$(OMP_NUM_THREADS=1 $RUN "$D/t.ll" -e main -entry-point-result=void "${LIBS[@]}" 2>/dev/null); tt=$(echo "$o"|med); tc=$(echo "$o"|ckv)
    cor=$(awk -v a="$base" -v b="$tc" 'BEGIN{d=a-b;if(d<0)d=-d;r=(a!=0)?d/(a<0?-a:a):d;print(r<1e-9)?"ok":"DIFF"}')
    echo "time-tiled(t=$TM,s=$TS) 1t : ${tt}s  $(gfe $fl $tt)GF  $cor  (vs untiled $(awk -v a=$ut -v b=$tt 'BEGIN{printf "%.2fx",(b>0)?a/b:0}'))"
  done
done