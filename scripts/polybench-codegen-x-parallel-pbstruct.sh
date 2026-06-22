#!/usr/bin/env bash
# Size-specialized (constant-bound) PolyBench-structured GEMM (beta-scale +
# i-k-j accumulate): register-block + parallel compose. baseline/rb/par/rb+par.
set -uo pipefail
export LC_ALL=C
LL=/home/tor/Dev/marco/install/llvm-project
DROPT=/home/tor/Dev/PhD/DRComp/drcompiler.git/onnx-mlir/build/tools/dr-opt/dr-opt
OPT=$LL/bin/mlir-opt; RUN="$LL/bin/mlir-runner --O3"
N="${1:-1024}"; D=$(mktemp -d); trap 'rm -rf "$D"' EXIT
LIBS=(--shared-libs="$LL/lib/libmlir_runner_utils.so" --shared-libs="$LL/lib/libmlir_c_runner_utils.so" --shared-libs=/usr/lib/libomp.so)
RB='builtin.module(func.func(affine-register-block{cache-tile=true}))'
SEQ=(--lower-affine --convert-vector-to-llvm --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts)
OMP=(--affine-parallelize=max-nested=1 --lower-affine --convert-scf-to-openmp --canonicalize --convert-vector-to-llvm --convert-scf-to-cf --convert-openmp-to-llvm --convert-cf-to-llvm --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts)
med(){ awk '/data =/{g++} g==1' | grep -oE '\[[^]]*\]' | tr -d '[] ' | tr ',' '\n' | grep -E '^[0-9.eE+-]+$' | sort -g | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }
ckv(){ grep -oE '\[[^]]*\]' | tail -1 | tr -d '[] '; }
runit(){ OMP_NUM_THREADS=$1 OMP_PROC_BIND=close OMP_PLACES=cores $RUN "$2" -e main -entry-point-result=void "${LIBS[@]}" 2>/dev/null; }

cat > "$D/g.mlir" <<EOF
func.func private @printMemrefF64(memref<*xf64>)
func.func private @rtclock() -> f64
func.func @kernel_gemm(%alpha: f64, %beta: f64, %C: memref<${N}x${N}xf64>, %A: memref<${N}x${N}xf64>, %B: memref<${N}x${N}xf64>) {
  affine.for %i = 0 to $N {
    affine.for %j = 0 to $N {
      %c = affine.load %C[%i,%j] : memref<${N}x${N}xf64>
      %cb = arith.mulf %c, %beta : f64
      affine.store %cb, %C[%i,%j] : memref<${N}x${N}xf64>
    }
    affine.for %k = 0 to $N { affine.for %j = 0 to $N {
      %a = affine.load %A[%i,%k] : memref<${N}x${N}xf64>
      %ab = arith.mulf %alpha, %a : f64
      %b = affine.load %B[%k,%j] : memref<${N}x${N}xf64>
      %abb = arith.mulf %ab, %b : f64
      %c = affine.load %C[%i,%j] : memref<${N}x${N}xf64>
      %s = arith.addf %c, %abb : f64
      affine.store %s, %C[%i,%j] : memref<${N}x${N}xf64>
    }}
  }
  return
}
func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %Nn = arith.constant $N : index
  %alpha = arith.constant 1.5 : f64
  %beta = arith.constant 1.2 : f64
  %C = memref.alloc() : memref<${N}x${N}xf64>
  %A = memref.alloc() : memref<${N}x${N}xf64>
  %B = memref.alloc() : memref<${N}x${N}xf64>
  scf.for %i = %c0 to %Nn step %c1 { scf.for %j = %c0 to %Nn step %c1 {
    %t = arith.muli %i, %Nn : index
    %t2 = arith.addi %t, %j : index
    %p = arith.constant 97 : index
    %m = arith.remui %t2, %p : index
    %mi = arith.index_cast %m : index to i64
    %mf = arith.sitofp %mi : i64 to f64
    %dd = arith.constant 97.0 : f64
    %v = arith.divf %mf, %dd : f64
    memref.store %v, %A[%i,%j] : memref<${N}x${N}xf64>
    memref.store %v, %B[%i,%j] : memref<${N}x${N}xf64>
    memref.store %v, %C[%i,%j] : memref<${N}x${N}xf64>
  }}
  %T = memref.alloc() : memref<5xf64>
  %c5 = arith.constant 5 : index
  scf.for %r = %c0 to %c5 step %c1 {
    %t0 = func.call @rtclock() : () -> f64
    func.call @kernel_gemm(%alpha,%beta,%C,%A,%B) : (f64,f64,memref<${N}x${N}xf64>,memref<${N}x${N}xf64>,memref<${N}x${N}xf64>) -> ()
    %t1 = func.call @rtclock() : () -> f64
    %dt = arith.subf %t1,%t0 : f64
    memref.store %dt, %T[%r] : memref<5xf64>
  }
  %U = memref.cast %T : memref<5xf64> to memref<*xf64>
  func.call @printMemrefF64(%U) : (memref<*xf64>) -> ()
  %CK = memref.alloc() : memref<1xf64>
  %zd = arith.constant 0.0 : f64
  memref.store %zd, %CK[%c0] : memref<1xf64>
  scf.for %i = %c0 to %Nn step %c1 { scf.for %j = %c0 to %Nn step %c1 {
    %e = memref.load %C[%i,%j] : memref<${N}x${N}xf64>
    %acc = memref.load %CK[%c0] : memref<1xf64>
    %na = arith.addf %acc,%e : f64
    memref.store %na, %CK[%c0] : memref<1xf64>
  }}
  %CKU = memref.cast %CK : memref<1xf64> to memref<*xf64>
  func.call @printMemrefF64(%CKU) : (memref<*xf64>) -> ()
  return
}
EOF

$OPT "$D/g.mlir" "${SEQ[@]}" -o "$D/base.ll" 2>/dev/null
$DROPT --allow-unregistered-dialect --pass-pipeline="$RB" "$D/g.mlir" -o "$D/rb.mlir" 2>/dev/null
$OPT "$D/rb.mlir" "${SEQ[@]}" -o "$D/rb.ll" 2>/dev/null
$OPT "$D/g.mlir" "${OMP[@]}" -o "$D/par.ll" 2>/dev/null
$OPT "$D/rb.mlir" "${OMP[@]}" -o "$D/rbpar.ll" 2>/dev/null
nprb=$($OPT "$D/rb.mlir" --affine-parallelize=max-nested=1 -o - 2>/dev/null | grep -c 'affine.parallel')

bo=$(runit 1 "$D/base.ll"); bt=$(echo "$bo"|med); bk=$(echo "$bo"|ckv)
chk(){ awk -v a="$bk" -v b="$1" 'BEGIN{d=a-b;if(d<0)d=-d;r=(a!=0)?d/(a<0?-a:a):d;print(r<1e-9)?"MATCH":"DIFF"}'; }
echo "PolyBench-structured GEMM (beta + i-k-j), N=$N, constant bounds (size-specialized)"
printf "%-16s %12s %9s  %s\n" "config" "median(s)" "vs-base" "correct"
printf "%-16s %12s %9s  %s\n" "baseline(1t)" "$bt" "1.00x" "MATCH"
for cfg in "rb(1t):1:$D/rb.ll" "par(16t):16:$D/par.ll" "rb+par(16t):16:$D/rbpar.ll"; do
  nm=${cfg%%:*}; rest=${cfg#*:}; th=${rest%%:*}; ll=${rest#*:}
  o=$(runit $th "$ll"); t=$(echo "$o"|med)
  printf "%-16s %12s %8sx  %s\n" "$nm" "$t" "$(awk -v a=$bt -v b=$t 'BEGIN{printf "%.1f",a/b}')" "$(chk "$(echo "$o"|ckv)")"
done
echo "(rb+par parallel bands after register-block: $nprb)"