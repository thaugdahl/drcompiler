#!/usr/bin/env bash
# openai-gpt batch=N THROUGHPUT SPMD, end-to-end + native onnx-mlir comparison.
#
# At batch=1 the transformer is sync-bound (4.4x/16t -- ~600 tiny sequential
# bands, §11.21).  At batch>1 the BATCH axis is the universal shard axis: every
# band shards it, each band does N* the work so the inter-band barriers amortize,
# and SPMD scales near-linearly.  VERDICT (batch=16, 16-core Zen4, §11.22):
#   our batch SPMD  0.448s  35.7 smp/s  10.6x  err 0.00e+00 (BYTE-IDENTICAL)
#   onnx-mlir --parallel  1.251s  12.8 smp/s     -> WE WIN 2.8x
#   onnx-mlir --O3 seq    8.756s   1.8 smp/s
#
# The model bakes batch=1 into its Reshape constants, so it is first patched to
# dynamic batch (scripts/patch-onnx-dynamic-batch.py), then compiled at batch=N
# via --shapeInformation.  Recipe = the batch-1 winner (§11.18) + batch axis:
#   register-block{no-cache-tile} -> demote/promote -> par-spmd-perband -> par->omp
#
# Usage: N=16 THREADS="1 2 4 8 16" spmd-throughput-openaigpt.sh
set -uo pipefail
export LC_ALL=C
N="${N:-16}"; THREADS="${THREADS:-1 2 4 8 16}"; ITERS="${ITERS:-7}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
W="$(mktemp -d /tmp/spmd-thru.XXXX)"; trap 'rm -rf "$W"' EXIT
IMG="${ONNX_MLIR_IMAGE:-onnx-mlir-lean:x86_64}"
FULLIMG="${ONNX_MLIR_FULL_IMAGE:-onnx-mlir:x86_64}"
LL="${LLVM_BIN:-/home/tor/Dev/marco/install/llvm-project/bin}"
DROPT="${DROPT:-$REPO/build/tools/dr-opt/dr-opt}"
GM="$REPO/bench/zen4-gemm.json"
OMB=/build/onnx-mlir/build/Release/bin
SRC="$REPO/test/ONNX/openaigpt_Opset18.onnx"
LOWER="--lower-affine --convert-vector-to-llvm --convert-scf-to-cf --convert-math-to-libm --expand-strided-metadata --finalize-memref-to-llvm --convert-cf-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --convert-openmp-to-llvm --reconcile-unrealized-casts"

echo "== patch model to dynamic batch =="
cp "$SRC" "$W/m.onnx"
docker run --rm --entrypoint python3 -v "$W:$W" -v "$REPO:$REPO" "$FULLIMG" \
  "$REPO/scripts/patch-onnx-dynamic-batch.py" "$W/m.onnx" "$W/dyn.onnx" 2>&1 | tail -1

echo "== front: batch=$N =="
docker run --rm --entrypoint "$OMB/onnx-mlir" -v "$W:$W" "$IMG" \
  --O2 --EmitMLIR --shapeInformation="0:${N}x128,1:${N}x128" -o "$W/k" "$W/dyn.onnx" 2>/dev/null
docker run --rm --entrypoint "$OMB/onnx-mlir-opt" -v "$W:$W" "$IMG" \
  --convert-krnl-to-affine "$W/k.onnx.mlir" -o "$W/k.affine.mlir" 2>/dev/null
echo "batch loops (to $N): $(grep -cE "affine.for %[a-z0-9_]+ = 0 to $N$|= 0 to $N " "$W/k.affine.mlir" 2>/dev/null)"

cat > "$W/h.c" <<EOF
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define NB ${N}
struct MI64 { long *a,*al; long off; long sz[2]; long st[2]; };
struct MF   { float *a,*al; long off; long sz[2]; long st[2]; };
struct MO   { float *a,*al; long off; long sz[3]; long st[3]; };
extern void _mlir_ciface_main_graph(struct MO*, struct MI64*, struct MF*);
int main(int argc,char**argv){
  long *ids=malloc((long)NB*128*8); for(long i=0;i<(long)NB*128;i++) ids[i]=(long)(i%40000);
  struct MI64 in_ids={ids,ids,0,{NB,128},{128,1}};
  float *am=malloc((long)NB*128*4); for(long i=0;i<(long)NB*128;i++) am[i]=1.0f;
  struct MF in_am={am,am,0,{NB,128},{128,1}};
  struct MO res; memset(&res,0,sizeof(res));
  _mlir_ciface_main_graph(&res,&in_ids,&in_am);
  FILE*f=argc>1?fopen(argv[1],"w"):NULL;
  for(long i=0;i<(long)NB*128*768;i++){ float v=res.al[res.off+i]; if(f)fprintf(f,"%.9e\n",v);}
  if(f)fclose(f);
  int iters=argc>2?atoi(argv[2]):5;
  for(int r=0;r<iters;r++){ struct timespec a,b; clock_gettime(CLOCK_MONOTONIC,&a);
    struct MO o; memset(&o,0,sizeof(o)); _mlir_ciface_main_graph(&o,&in_ids,&in_am);
    clock_gettime(CLOCK_MONOTONIC,&b);
    printf("%.6f\n",(b.tv_sec-a.tv_sec)+(b.tv_nsec-a.tv_nsec)*1e-9);}
  return 0;}
EOF

CG="func.func(dr-scalar-reduction-demote,affine-register-block{mr=8 nr=16 cpu-cost-model-file=$GM no-cache-tile},dr-scalar-reduction-promote),lower-krnl-global"
build(){ local nm=$1 pipe=$2
  "$DROPT" "$W/k.affine.mlir" -allow-unregistered-dialect --pass-pipeline="$pipe" --emit-bytecode -o - 2>"$W/$nm.r" \
    | "$LL/mlir-opt" -allow-unregistered-dialect $LOWER --emit-bytecode -o - \
    | "$LL/mlir-translate" --mlir-to-llvmir -o - | "$LL/llvm-as" -o - \
    | "$LL/clang" -O2 -march=native -fopenmp -c -x ir - -o "$W/$nm.o" 2>/dev/null
  grep -oE "materialized.*parallel\)" "$W/$nm.r" | head -1
  "$LL/clang" -O2 -no-pie -fopenmp "$W/h.c" "$W/$nm.o" -lm -o "$W/$nm.bin" 2>/dev/null || echo "LINK FAIL $nm"
}
echo "== build seq (no SPMD) =="; build seq "builtin.module($CG)"
echo "== build spmd (batch-axis) =="; build spmd "builtin.module($CG,dr-par-bubbles{par-spmd-perband},func.func(convert-par-to-omp))"

OMP_NUM_THREADS=1 "$W/seq.bin" "$W/seq.logits" 1 >/dev/null 2>&1
med(){ sort -n "$1" | awk -v n="$ITERS" 'NR==int((n+1)/2){print;exit}'; }
echo; echo "=== batch=$N, median of $ITERS, THROUGHPUT (smp/s = $N/time) ==="
for nm in seq spmd; do
  [ -x "$W/$nm.bin" ] || { echo "$nm: no bin"; continue; }
  declare -A M=()
  for t in $THREADS; do
    OMP_NUM_THREADS=$t OMP_PROC_BIND=close OMP_PLACES=cores "$W/$nm.bin" "$W/$nm.s.$t" "$ITERS" >"$W/$nm.t.$t" 2>/dev/null
    M[$t]=$(med "$W/$nm.t.$t")
  done
  err=$(paste "$W/seq.logits" "$W/$nm.s.${THREADS##* }" | awk '{d=$1-$2;if(d<0)d=-d;if(d>ma)ma=d;a=($1<0?-$1:$1);if(a>mx)mx=a}END{printf "%.2e",(mx>0?ma/mx:ma)}')
  printf "%-5s " "$nm"
  for t in $THREADS; do
    sp=$(awk -v r="${M[1]}" -v m="${M[$t]}" 'BEGIN{printf "%.2f",(m>0)?r/m:0}')
    thr=$(awk -v n="$N" -v m="${M[$t]}" 'BEGIN{printf "%.1f",(m>0)?n/m:0}')
    printf "t%s=%.3fs(%.2fx,%ssmp/s) " "$t" "${M[$t]}" "$sp" "$thr"
  done
  printf " err=%s\n" "$err"
done
