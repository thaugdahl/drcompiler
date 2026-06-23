#!/usr/bin/env bash
# validate-spmd-openaigpt.sh — apply whole-function SPMD (the resnet50-winning
# pipeline: demote+promote -> par-spmd-perband -> par->omp) to the openai-gpt
# transformer and measure scaling + correctness.  VERDICT: NO-GO (serial-bound) —
# see PARALLEL_SPMD_SPEC.md §11.13.  Unlike resnet50 (14.5x@16t), openaigpt does
# NOT scale (~1.01x@16t): 519 small bands / 518 barriers, the heavy work is in the
# 85 SERIAL critical bands (softmax max/sum + LayerNorm mean/var reductions) and
# scalar transcendentals (Gelu powf+tanh, softmax exp).  At 16t the OpenMP team
# spins ~700% CPU at the barriers with ZERO wall-time benefit; wall-clock = the
# serial critical path (OMP_WAIT_POLICY=passive leaves it unchanged).
#
# This is the 2-input transformer harness (input_ids i64[1,128] + attention_mask
# f32[1,128] -> hidden f32[1,128,768]), the analogue of validate-spmd-parallel.sh's
# single-f32-input convnet harness.
#
# Usage: validate-spmd-openaigpt.sh [threads...]
set -uo pipefail
export LC_ALL=C
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="$REPO/test/ONNX/openaigpt_Opset18.onnx"
IMG="${ONNX_MLIR_IMAGE:-onnx-mlir-lean:x86_64}"
LL="${LLVM_BIN:-/home/tor/Dev/marco/install/llvm-project/bin}"
DROPT="${DROPT:-$REPO/build/tools/dr-opt/dr-opt}"
OMB=/build/onnx-mlir/build/Release/bin
THREADS="${*:-1 2 4 8 16}"; ITERS="${ITERS:-7}"
W="$(mktemp -d /tmp/spmd-ogpt.XXXX)"; trap 'rm -rf "$W"' EXIT
MA="$(realpath "$MODEL")"
dockt(){ local t=$1; shift; docker run --rm --entrypoint "$OMB/$t" -v "$W:$W" -v "$(dirname "$MA"):$(dirname "$MA")" "$IMG" "$@"; }
LOWER="--lower-affine --convert-scf-to-cf --convert-math-to-libm --expand-strided-metadata --finalize-memref-to-llvm --convert-cf-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --convert-openmp-to-llvm --reconcile-unrealized-casts"

echo "== front: static affine (openaigpt is already 1x128) =="
dockt onnx-mlir --O2 --EmitMLIR -o "$W/m" "$MA" >/dev/null 2>&1
dockt onnx-mlir-opt --convert-krnl-to-affine "$W/m.onnx.mlir" -o "$W/m.affine.mlir"

cat > "$W/h.c" <<'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
struct MI64 { long *a,*al; long off; long sz[2]; long st[2]; };
struct MF   { float *a,*al; long off; long sz[2]; long st[2]; };
struct MO   { float *a,*al; long off; long sz[3]; long st[3]; };
extern void _mlir_ciface_main_graph(struct MO*, struct MI64*, struct MF*);
int main(int argc,char**argv){
  long *ids=malloc(128*8); for(int i=0;i<128;i++) ids[i]=(long)(i%40000);
  struct MI64 in_ids={ids,ids,0,{1,128},{128,1}};
  float *am=malloc(128*4); for(int i=0;i<128;i++) am[i]=1.0f;
  struct MF in_am={am,am,0,{1,128},{128,1}};
  struct MO res; memset(&res,0,sizeof(res));
  _mlir_ciface_main_graph(&res,&in_ids,&in_am);
  FILE*f=argc>1?fopen(argv[1],"w"):NULL;
  for(int i=0;i<128*768;i++){ float v=res.al[res.off+i]; if(f)fprintf(f,"%.9e\n",v);}
  if(f)fclose(f);
  int iters=argc>2?atoi(argv[2]):5;
  for(int r=0;r<iters;r++){ struct timespec a,b; clock_gettime(CLOCK_MONOTONIC,&a);
    struct MO o; memset(&o,0,sizeof(o)); _mlir_ciface_main_graph(&o,&in_ids,&in_am);
    clock_gettime(CLOCK_MONOTONIC,&b);
    printf("%.6f\n",(b.tv_sec-a.tv_sec)+(b.tv_nsec-a.tv_nsec)*1e-9);}
  return 0;}
EOF

build_cfg(){ local nm=$1 pipe=$2
  $DROPT "$W/m.affine.mlir" -allow-unregistered-dialect --pass-pipeline="$pipe" --emit-bytecode -o - 2>"$W/$nm.r" \
    | "$LL/mlir-opt" -allow-unregistered-dialect $LOWER --emit-bytecode -o - \
    | "$LL/mlir-translate" --mlir-to-llvmir -o - \
    | "$LL/llvm-as" -o - \
    | "$LL/clang" -O2 -march=native -fopenmp -c -x ir - -o "$W/$nm.o" 2>/dev/null
  grep -iE 'perband' "$W/$nm.r" | head -1
  "$LL/clang" -O2 -no-pie -fopenmp "$W/h.c" "$W/$nm.o" -lm -o "$W/$nm.bin"
}
echo "== seq reference (krnl-free, no SPMD) =="
build_cfg seq "builtin.module(lower-krnl-global)"
echo "== spmd (demote+promote -> par-spmd-perband -> par->omp) =="
build_cfg spmd "builtin.module(func.func(dr-affine-loop-distribute,dr-scalar-reduction-demote,dr-scalar-reduction-promote),lower-krnl-global,dr-par-bubbles{par-spmd-perband},func.func(convert-par-to-omp))"

OMP_NUM_THREADS=1 "$W/seq.bin" "$W/seq.logits" 1 >/dev/null 2>&1
med(){ sort -n "$1" | awk -v n="$ITERS" 'NR==int((n+1)/2){print;exit}'; }
echo "== scaling + correctness (spmd vs seq) =="
SREF=""
for t in $THREADS; do
  OMP_NUM_THREADS=$t OMP_PROC_BIND=close OMP_PLACES=cores "$W/spmd.bin" "$W/s.$t" "$ITERS" >"$W/t.$t" 2>/dev/null
  m=$(med "$W/t.$t")
  err=$(paste "$W/seq.logits" "$W/s.$t" | awk '{d=$1-$2;if(d<0)d=-d;if(d>ma)ma=d;a=($1<0?-$1:$1);if(a>mx)mx=a}END{printf "%.2e",(mx>0?ma/mx:ma)}')
  [ -z "$SREF" ] && SREF=$m
  spd=$(awk -v r="$SREF" -v m="$m" 'BEGIN{printf "%.2f",r/m}')
  printf "  threads=%-3s median=%.3fs speedup=%sx norm_rel_err=%s\n" "$t" "$m" "$spd" "$err"
done
