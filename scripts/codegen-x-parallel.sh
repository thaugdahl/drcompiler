#!/usr/bin/env bash
# codegen-x-parallel.sh — demonstrate that single-thread codegen
# (affine-register-block) COMPOSES with parallelism (affine-parallelize -> omp)
# on a canonical GEMM (k-innermost reduction, the form register-block requires).
# 4 configs: baseline(1t) / rb(1t) / par(16t) / rb+par(16t), checksum-verified.
#
# Honest reading: register-block promotes the k-loop memory accumulator to a
# register + emits an 8x16 vector micro-kernel, so its speedup over the NAIVE
# mlir baseline is large; against clang -O3 (which scalar-promotes) the codegen
# win is ~2.3-2.6x (claude-docs/POLYBENCH_FAMILY_FINDINGS.md). The point here is
# COMPOSITION: both transforms fire together, the result is numerically MATCH,
# and parallelism multiplies the register-blocked single-thread time.
set -uo pipefail
export LC_ALL=C
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LL="${LLVM_INSTALL_DIR:-/home/tor/Dev/marco/install/llvm-project}"
DROPT="${DROPT:-$REPO/build/tools/dr-opt/dr-opt}"
OPT=$LL/bin/mlir-opt; RUN="$LL/bin/mlir-runner --O3"
N="${1:-1024}"
D=$(mktemp -d); trap 'rm -rf "$D"' EXIT
LIBS=(--shared-libs="$LL/lib/libmlir_runner_utils.so" --shared-libs="$LL/lib/libmlir_c_runner_utils.so" --shared-libs=/usr/lib/libomp.so)
RB="builtin.module(func.func(affine-register-block{cache-tile=true}))"
SEQ=(--lower-affine --convert-vector-to-llvm --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts)
OMP=(--affine-parallelize=max-nested=1 --lower-affine --convert-scf-to-openmp --canonicalize --convert-vector-to-llvm --convert-scf-to-cf --convert-openmp-to-llvm --convert-cf-to-llvm --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts)
median(){ awk '/data =/{g++} g==1' | grep -oE '\[[^]]*\]' | tr -d '[] ' | tr ',' '\n' | grep -E '^[0-9.eE+-]+$' | sort -g | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }
checksum(){ grep -oE '\[[^]]*\]' | tail -1 | tr -d '[] '; }
runit(){ OMP_NUM_THREADS=$1 OMP_PROC_BIND=close OMP_PLACES=cores $RUN "$2" -e main -entry-point-result=void "${LIBS[@]}" 2>/dev/null; }

cat > "$D/g.mlir" <<EOF
func.func private @printMemrefF64(memref<*xf64>)
func.func private @rtclock() -> f64
func.func @gemm(%C: memref<${N}x${N}xf64>, %A: memref<${N}x${N}xf64>, %B: memref<${N}x${N}xf64>) {
  affine.for %i = 0 to $N { affine.for %j = 0 to $N { affine.for %k = 0 to $N {
    %a = affine.load %A[%i, %k] : memref<${N}x${N}xf64>
    %b = affine.load %B[%k, %j] : memref<${N}x${N}xf64>
    %c = affine.load %C[%i, %j] : memref<${N}x${N}xf64>
    %p = arith.mulf %a, %b : f64
    %s = arith.addf %c, %p : f64
    affine.store %s, %C[%i, %j] : memref<${N}x${N}xf64>
  }}}
  return
}
func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %Nn = arith.constant $N : index
  %K = arith.constant 7 : index
  %Cf = memref.alloc() : memref<${N}x${N}xf64>
  %Af = memref.alloc() : memref<${N}x${N}xf64>
  %Bf = memref.alloc() : memref<${N}x${N}xf64>
  scf.for %i = %c0 to %Nn step %c1 { scf.for %j = %c0 to %Nn step %c1 {
    %t = arith.muli %i, %Nn : index
    %t2 = arith.addi %t, %j : index
    %c97 = arith.constant 97 : index
    %m = arith.remui %t2, %c97 : index
    %mi = arith.index_cast %m : index to i64
    %mf = arith.sitofp %mi : i64 to f64
    %d = arith.constant 97.0 : f64
    %v = arith.divf %mf, %d : f64
    memref.store %v, %Af[%i, %j] : memref<${N}x${N}xf64>
    %c89 = arith.constant 89 : index
    %m2 = arith.remui %t2, %c89 : index
    %mi2 = arith.index_cast %m2 : index to i64
    %mf2 = arith.sitofp %mi2 : i64 to f64
    %d2 = arith.constant 89.0 : f64
    %v2 = arith.divf %mf2, %d2 : f64
    memref.store %v2, %Bf[%i, %j] : memref<${N}x${N}xf64>
    memref.store %v, %Cf[%i, %j] : memref<${N}x${N}xf64>
  }}
  %T = memref.alloc() : memref<7xf64>
  scf.for %r = %c0 to %K step %c1 {
    %t0 = func.call @rtclock() : () -> f64
    func.call @gemm(%Cf, %Af, %Bf) : (memref<${N}x${N}xf64>, memref<${N}x${N}xf64>, memref<${N}x${N}xf64>) -> ()
    %t1 = func.call @rtclock() : () -> f64
    %dt = arith.subf %t1, %t0 : f64
    memref.store %dt, %T[%r] : memref<7xf64>
  }
  %U = memref.cast %T : memref<7xf64> to memref<*xf64>
  func.call @printMemrefF64(%U) : (memref<*xf64>) -> ()
  %CK = memref.alloc() : memref<1xf64>
  %zd = arith.constant 0.0 : f64
  memref.store %zd, %CK[%c0] : memref<1xf64>
  scf.for %i = %c0 to %Nn step %c1 { scf.for %j = %c0 to %Nn step %c1 {
    %e = memref.load %Cf[%i, %j] : memref<${N}x${N}xf64>
    %acc = memref.load %CK[%c0] : memref<1xf64>
    %na = arith.addf %acc, %e : f64
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

bo=$(runit 1 "$D/base.ll"); bt=$(echo "$bo"|median); bck=$(echo "$bo"|checksum)
chk(){ awk -v a="$bck" -v b="$1" 'BEGIN{d=a-b;if(d<0)d=-d;r=(a!=0)?d/(a<0?-a:a):d;print(r<1e-9)?"MATCH":"DIFF"}'; }
echo "canonical GEMM N=$N, 7 iters, mlir-runner --O3, 16-core Zen4"
printf "%-16s %12s %9s  %s\n" "config" "median(s)" "vs-base" "correct"
printf "%-16s %12s %9s  %s\n" "baseline(1t)" "$bt" "1.00x" "MATCH"
o=$(runit 1 "$D/rb.ll"); t=$(echo "$o"|median)
printf "%-16s %12s %9sx  %s\n" "rb(1t)" "$t" "$(awk -v a=$bt -v b=$t 'BEGIN{printf "%.1f",a/b}')" "$(chk "$(echo "$o"|checksum)")"
o=$(runit 16 "$D/par.ll"); t=$(echo "$o"|median)
printf "%-16s %12s %9sx  %s\n" "par(16t)" "$t" "$(awk -v a=$bt -v b=$t 'BEGIN{printf "%.1f",a/b}')" "$(chk "$(echo "$o"|checksum)")"
o=$(runit 16 "$D/rbpar.ll"); t=$(echo "$o"|median)
printf "%-16s %12s %9sx  %s\n" "rb+par(16t)" "$t" "$(awk -v a=$bt -v b=$t 'BEGIN{printf "%.1f",a/b}')" "$(chk "$(echo "$o"|checksum)")"