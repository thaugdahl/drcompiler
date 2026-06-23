#!/usr/bin/env bash
# onnx-mlir NATIVE EmitObj (--O3) seq vs --parallel on openai-gpt batch-1, 1/16t.
# Our SPMD: plain-seq 20.5s (scalar), 16t=1.09s (18x over our scalar baseline).
set -uo pipefail
cd /home/tor/Dev/PhD/DRComp/drcompiler.git/onnx-mlir
W=/home/tor/.claude/jobs/3a6d8e8d/tmp
MA=$(realpath test/ONNX/openaigpt_Opset18.onnx); MD=$(dirname "$MA")
IMG=onnx-mlir:x86_64
OMB=/build/onnx-mlir/build/Release/bin
INC=/build/onnx-mlir/include
CRT=/build/onnx-mlir/build/Release/lib/libcruntime.a
CLANG=/opt/llvm/bin/clang
ITERS=${ITERS:-7}

echo "== compile onnx-mlir --O3 (native seq) =="
docker run --rm --entrypoint "$OMB/onnx-mlir" -v "$W:$W" -v "$MD:$MD" "$IMG" \
  --O3 --EmitObj -o "$W/ogpt_o3" "$MA" 2>&1 | tail -1
echo "== compile onnx-mlir --O3 --parallel =="
docker run --rm --entrypoint "$OMB/onnx-mlir" -v "$W:$W" -v "$MD:$MD" "$IMG" \
  --O3 --parallel --EmitObj -o "$W/ogpt_par" "$MA" 2>&1 | tail -1

cat > "$W/hrt_gpt.c" <<EOF
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "OnnxMlirRuntime.h"
extern OMTensorList *run_main_graph(OMTensorList *);
int main(int argc,char**argv){
  int iters=argc>1?atoi(argv[1]):7;
  long*ids=malloc(128*8); for(int i=0;i<128;i++) ids[i]=(long)(i%40000);
  float*am=malloc(128*4); for(int i=0;i<128;i++) am[i]=1.0f;
  int64_t sh[2]={1,128};
  OMTensor*t0=omTensorCreate(ids,sh,2,ONNX_TYPE_INT64);
  OMTensor*t1=omTensorCreate(am,sh,2,ONNX_TYPE_FLOAT);
  OMTensor*ins[2]={t0,t1};
  OMTensorList*il=omTensorListCreate(ins,2);
  run_main_graph(il); run_main_graph(il);
  for(int i=0;i<iters;i++){ struct timespec a,b; clock_gettime(CLOCK_MONOTONIC,&a);
    run_main_graph(il); clock_gettime(CLOCK_MONOTONIC,&b);
    printf("%.6f\n",(b.tv_sec-a.tv_sec)+(b.tv_nsec-a.tv_nsec)*1e-9);}
  return 0;}
EOF

LOMP=/usr/lib/libomp.so   # host LLVM OpenMP (image has none); provides __kmpc_*
for cfg in o3 par; do
  omp=""; [ "$cfg" = par ] && omp="$LOMP"
  docker run --rm --entrypoint "$CLANG" -v "$W:$W" -v "$LOMP:$LOMP:ro" "$IMG" \
    -O2 -I"$INC" "$W/hrt_gpt.c" "$W/ogpt_$cfg.o" "$CRT" $omp -lm -lpthread -o "$W/ogpt_$cfg.bin" 2>"$W/ogpt_$cfg.lerr" \
    || { echo "link fail $cfg"; tail -3 "$W/ogpt_$cfg.lerr"; }
done

med(){ sort -n | awk -v n="$ITERS" 'NR==int((n+1)/2){print;exit}'; }
echo; echo "=== openai-gpt batch-1, onnx-mlir NATIVE EmitObj, median of $ITERS ==="
for cfg in o3 par; do
  [ -x "$W/ogpt_$cfg.bin" ] || { echo "$cfg: no bin"; continue; }
  declare -A M=()
  for t in 1 16; do
    out=$(docker run --rm --entrypoint "$W/ogpt_$cfg.bin" -e OMP_NUM_THREADS=$t -e OMP_PROC_BIND=close -e OMP_PLACES=cores -v "$W:$W" -v "$LOMP:$LOMP:ro" "$IMG" "$ITERS" 2>/dev/null)
    M[$t]=$(echo "$out" | med)
  done
  sc=$(awk -v a="${M[1]}" -v b="${M[16]}" 'BEGIN{printf "%.2f",(b>0)?a/b:0}')
  printf "onnx-mlir --%-9s 1t=%ss  16t=%ss  scaling=%sx\n" "$cfg" "${M[1]}" "${M[16]}" "$sc"
done
echo "--- our SPMD (krnl-free, scalar per-thread): plain-seq 20.5s, 16t=1.09s ---"
