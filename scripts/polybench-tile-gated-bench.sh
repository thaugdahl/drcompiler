#!/usr/bin/env bash
# Clean cache-tiling measurement on the i,k,j (streaming) GEMM at L3-spill sizes.
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
ckv(){ grep -oE '\[[^]]*\]' | tail -1 | tr -d '[] '; }
gf(){ awk -v n="$1" -v t="$2" 'BEGIN{printf "%.1f",(t>0)?2.0*n*n*n/t/1e9:0}'; }

drv(){ local N=$1; cat <<EOF
func.func private @printMemrefF64(memref<*xf64>)
func.func private @rtclock() -> f64
func.func @gemm(%C: memref<${N}x${N}xf64>, %A: memref<${N}x${N}xf64>, %B: memref<${N}x${N}xf64>) {
  affine.for %i = 0 to $N { affine.for %k = 0 to $N { affine.for %j = 0 to $N {
    %a = affine.load %A[%i,%k] : memref<${N}x${N}xf64>
    %b = affine.load %B[%k,%j] : memref<${N}x${N}xf64>
    %c = affine.load %C[%i,%j] : memref<${N}x${N}xf64>
    %p = arith.mulf %a,%b : f64
    %s = arith.addf %c,%p : f64
    affine.store %s, %C[%i,%j] : memref<${N}x${N}xf64>
  }}}
  return
}
func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %Nn = arith.constant $N : index
  %K = arith.constant 3 : index
  %Cf = memref.alloc() : memref<${N}x${N}xf64>
  %Af = memref.alloc() : memref<${N}x${N}xf64>
  %Bf = memref.alloc() : memref<${N}x${N}xf64>
  scf.for %i = %c0 to %Nn step %c1 { scf.for %j = %c0 to %Nn step %c1 {
    %t = arith.muli %i, %Nn : index
    %t2 = arith.addi %t, %j : index
    %p = arith.constant 97 : index
    %m = arith.remui %t2, %p : index
    %mi = arith.index_cast %m : index to i64
    %mf = arith.sitofp %mi : i64 to f64
    %d = arith.constant 97.0 : f64
    %v = arith.divf %mf, %d : f64
    memref.store %v, %Af[%i,%j] : memref<${N}x${N}xf64>
    memref.store %v, %Bf[%i,%j] : memref<${N}x${N}xf64>
    %z = arith.constant 0.0 : f64
    memref.store %z, %Cf[%i,%j] : memref<${N}x${N}xf64>
  }}
  %T = memref.alloc() : memref<3xf64>
  scf.for %r = %c0 to %K step %c1 {
    %t0 = func.call @rtclock() : () -> f64
    func.call @gemm(%Cf,%Af,%Bf) : (memref<${N}x${N}xf64>,memref<${N}x${N}xf64>,memref<${N}x${N}xf64>) -> ()
    %t1 = func.call @rtclock() : () -> f64
    %dt = arith.subf %t1,%t0 : f64
    memref.store %dt, %T[%r] : memref<3xf64>
  }
  %U = memref.cast %T : memref<3xf64> to memref<*xf64>
  func.call @printMemrefF64(%U) : (memref<*xf64>) -> ()
  %CK = memref.alloc() : memref<1xf64>
  %zd = arith.constant 0.0 : f64
  memref.store %zd, %CK[%c0] : memref<1xf64>
  scf.for %i = %c0 to %Nn step %c1 { scf.for %j = %c0 to %Nn step %c1 {
    %e = memref.load %Cf[%i,%j] : memref<${N}x${N}xf64>
    %acc = memref.load %CK[%c0] : memref<1xf64>
    %na = arith.addf %acc,%e : f64
    memref.store %na, %CK[%c0] : memref<1xf64>
  }}
  %CKU = memref.cast %CK : memref<1xf64> to memref<*xf64>
  func.call @printMemrefF64(%CKU) : (memref<*xf64>) -> ()
  return
}
EOF
}

run1(){ OMP_NUM_THREADS="$1" OMP_PROC_BIND=close OMP_PLACES=cores $RUN "$2" -e main -entry-point-result=void "${LIBS[@]}" 2>/dev/null; }

for N in 1024 2048; do
  drv "$N" > "$D/g.mlir"
  $OPT "$D/g.mlir" "${SEQ[@]}" -o "$D/su.ll" 2>/dev/null
  o=$(run1 1 "$D/su.ll"); sut=$(echo "$o"|med); base=$(echo "$o"|ckv)
  $DROPT --allow-unregistered-dialect --pass-pipeline='builtin.module(func.func(dr-affine-loop-tile{llc-gate-from-model=true}))' "$D/g.mlir" -o "$D/gt.mlir" 2>/dev/null
  $OPT "$D/gt.mlir" "${SEQ[@]}" -o "$D/st.ll" 2>"$D/e1"
  o=$(run1 1 "$D/st.ll"); stt=$(echo "$o"|med); stc=$(echo "$o"|ckv)
  $DROPT --allow-unregistered-dialect --pass-pipeline='builtin.module(dr-par-bubbles{par-spmd-perband},func.func(convert-par-to-omp))' "$D/g.mlir" -o "$D/pu.mlir" 2>/dev/null
  $OPT "$D/pu.mlir" "${OMPL[@]}" -o "$D/pu.ll" 2>/dev/null
  o=$(run1 16 "$D/pu.ll"); put=$(echo "$o"|med); puc=$(echo "$o"|ckv)
  $DROPT --allow-unregistered-dialect --pass-pipeline='builtin.module(func.func(dr-affine-loop-tile{llc-gate-from-model=true}),dr-par-bubbles{par-spmd-perband},func.func(convert-par-to-omp))' "$D/g.mlir" -o "$D/pt.mlir" 2>"$D/e2"
  $OPT "$D/pt.mlir" "${OMPL[@]}" -o "$D/pt.ll" 2>"$D/e3"
  o=$(run1 16 "$D/pt.ll"); ptt=$(echo "$o"|med); ptc=$(echo "$o"|ckv)
  echo "N=$N (working set $(awk -v n=$N 'BEGIN{printf "%dMB",3*n*n*8/1048576}'))"
  printf "  seq  untiled 1t : %ss  %sGF\n" "${sut:-FAIL}" "$(gf $N ${sut:-0})"
  printf "  seq  TILED   1t : %ss  %sGF  %s\n" "${stt:-FAIL}" "$(gf $N ${stt:-0})" "$([ "${stc:-}" = "$base" ] && echo ok || echo "DIFF/${stc:-empty}")"
  printf "  par  untiled 16t: %ss  %sGF  %s\n" "${put:-FAIL}" "$(gf $N ${put:-0})" "$([ "${puc:-}" = "$base" ] && echo ok || echo DIFF)"
  printf "  par  TILED   16t: %ss  %sGF  %s\n" "${ptt:-FAIL}" "$(gf $N ${ptt:-0})" "$([ "${ptc:-}" = "$base" ] && echo ok || echo "DIFF/${ptc:-empty}")"
  [ -s "$D/e2" ] && echo "  [tile+perband err] $(head -1 $D/e2)"
  [ -s "$D/e3" ] && echo "  [tiled omp lower err] $(head -1 $D/e3)"
done