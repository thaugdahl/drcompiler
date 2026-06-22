#!/usr/bin/env bash
# Roofline / scaling-vs-N for square GEMM (i,k,j, the cgeist form we parallelize).
# Per N: seq-MLIR-O3 1t, par(affine-parallelize->omp) 16t, clang -O3 AOT 1t.
# Reports GFLOP/s (2N^3/t) + 16t/1t scaling to locate the bandwidth wall and
# distance-to-peak. checksum-verified seq==par.
set -uo pipefail
export LC_ALL=C
LL=/home/tor/Dev/marco/install/llvm-project
OPT=$LL/bin/mlir-opt; RUN="$LL/bin/mlir-runner --O3"
D=$(mktemp -d); trap 'rm -rf "$D"' EXIT
LIBS=(--shared-libs="$LL/lib/libmlir_runner_utils.so" --shared-libs="$LL/lib/libmlir_c_runner_utils.so" --shared-libs=/usr/lib/libomp.so)
SEQ=(--lower-affine --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts)
OMP=(--affine-parallelize=max-nested=1 --lower-affine --convert-scf-to-openmp --canonicalize --convert-scf-to-cf --convert-openmp-to-llvm --convert-cf-to-llvm --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts)
med(){ awk '/data =/{g++} g==1' | grep -oE '\[[^]]*\]' | tr -d '[] ' | tr ',' '\n' | grep -E '^[0-9.eE+-]+$' | sort -g | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }
ck(){ grep -oE '\[[^]]*\]' | tail -1 | tr -d '[] '; }
gflops(){ awk -v n="$1" -v t="$2" 'BEGIN{printf "%.1f", (t>0)?2.0*n*n*n/t/1e9:0}'; }

mlir_drv(){ # $1=N $2=K
local N=$1 K=$2
cat <<EOF
func.func private @printMemrefF64(memref<*xf64>)
func.func private @rtclock() -> f64
func.func @gemm(%C: memref<${N}x${N}xf64>, %A: memref<${N}x${N}xf64>, %B: memref<${N}x${N}xf64>) {
  affine.for %i = 0 to $N { affine.for %k = 0 to $N { affine.for %j = 0 to $N {
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
  %K = arith.constant $K : index
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
    memref.store %v, %Bf[%i, %j] : memref<${N}x${N}xf64>
    %z = arith.constant 0.0 : f64
    memref.store %z, %Cf[%i, %j] : memref<${N}x${N}xf64>
  }}
  %T = memref.alloc() : memref<${K}xf64>
  scf.for %r = %c0 to %K step %c1 {
    %t0 = func.call @rtclock() : () -> f64
    func.call @gemm(%Cf, %Af, %Bf) : (memref<${N}x${N}xf64>, memref<${N}x${N}xf64>, memref<${N}x${N}xf64>) -> ()
    %t1 = func.call @rtclock() : () -> f64
    %dt = arith.subf %t1, %t0 : f64
    memref.store %dt, %T[%r] : memref<${K}xf64>
  }
  %U = memref.cast %T : memref<${K}xf64> to memref<*xf64>
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
}

c_drv(){ # $1=N $2=K
local N=$1 K=$2
cat <<EOF
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define N $N
static double A[N*N],B[N*N],C[N*N];
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(){
  for(int i=0;i<N*N;i++){double v=(i%97)/97.0;A[i]=v;B[i]=v;C[i]=0;}
  double best=1e18;
  for(int r=0;r<$K;r++){
    double t0=now();
    for(int i=0;i<N;i++)for(int k=0;k<N;k++){double a=A[i*N+k];for(int j=0;j<N;j++)C[i*N+j]+=a*B[k*N+j];}
    double t=now()-t0; if(t<best)best=t;
  }
  printf("%.6f\n",best);
  return 0;
}
EOF
}

printf "%-6s %4s | %-22s | %-22s | %-22s | %s\n" "N" "MB" "seq-MLIR-O3 1t" "par-omp 16t" "clang-O3 AOT 1t" "16t/1t"
printf -- "------------------------------------------------------------------------------------------------\n"
for N in 256 512 1024 2048; do
  K=5; [ "$N" -ge 1024 ] && K=3
  mb=$(awk -v n="$N" 'BEGIN{printf "%d", 3.0*n*n*8/1048576}')
  mlir_drv "$N" "$K" > "$D/g.mlir"
  $OPT "$D/g.mlir" "${SEQ[@]}" -o "$D/s.ll" 2>/dev/null
  $OPT "$D/g.mlir" "${OMP[@]}" -o "$D/o.ll" 2>/dev/null
  so=$(OMP_NUM_THREADS=1 $RUN "$D/s.ll" -e main -entry-point-result=void "${LIBS[@]}" 2>/dev/null)
  st=$(echo "$so"|med); sck=$(echo "$so"|ck)
  oo=$(OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores $RUN "$D/o.ll" -e main -entry-point-result=void "${LIBS[@]}" 2>/dev/null)
  ot=$(echo "$oo"|med); ock=$(echo "$oo"|ck)
  cor=$(awk -v a="$sck" -v b="$ock" 'BEGIN{d=a-b;if(d<0)d=-d;r=(a!=0)?d/(a<0?-a:a):d;print(r<1e-9)?"ok":"DIFF"}')
  c_drv "$N" "$K" > "$D/g.c"
  $LL/bin/clang -O3 -march=native -ffast-math "$D/g.c" -o "$D/c.bin" 2>/dev/null
  ct=$("$D/c.bin")
  scal=$(awk -v a="$st" -v b="$ot" 'BEGIN{printf "%.1fx",(b>0)?a/b:0}')
  printf "%-6s %4s | %8ss %6sGF | %8ss %6sGF | %8ss %6sGF | %s %s\n" \
    "$N" "$mb" "$st" "$(gflops $N $st)" "$ot" "$(gflops $N $ot)" "$ct" "$(gflops $N $ct)" "$scal" "$cor"
done
echo "(MB = 3 matrices working set; Zen4 L2=1MB/core, L3=32MB/CCD x2)"