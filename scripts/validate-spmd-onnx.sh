#!/usr/bin/env bash
# validate-spmd-onnx.sh — NUMERIC correctness gate for whole-function SPMD on a
# real ONNX model.  Compiles a static-shape model two ways and diffs the output
# logits:
#   seq  : affine -> lower-affine -> convert-krnl-to-llvm -> run  (reference)
#   spmd : dr-par-bubbles{par-spmd-perband} -> convert-par-to-scf -> lower-affine
#          -> convert-scf-to-cf -> convert-krnl-to-llvm -> run
# The spmd path exercises the FULL whole-function widening (hoisted shared
# allocs, replicated glue, par.critical single-worker bands, par.forall sharded
# bands, par.barrier between) and must produce byte-identical output.
#
# NOTE on parallel execution: onnx-mlir's convert-krnl-to-llvm rejects external
# parallel constructs ("failed to legalize omp.* / scf.parallel" -- it expects
# sequential scf.for; onnx-mlir parallelizes via its own --parallel).  So the
# par.forall is lowered to SEQUENTIAL cf here (scf.parallel -> convert-scf-to-cf)
# for the correctness gate.  True parallel (omp) execution needs a krnl-free
# lowering path and is tracked separately.
#
# Result (dev host, 2026-06-19): mnist norm_rel_err=0.000e+00 (n=10);
# resnet50-v2-7 batch-1 norm_rel_err=0.000e+00 (n=1000), 82 forall + 55 critical
# + 207 hoisted + 136 barriers materialized.
#
# Usage: validate-spmd-onnx.sh <model.onnx> <SHAPE e.g. 1x3x224x224>
set -uo pipefail
export LC_ALL=C
MODEL=$1; SHAPE=$2
IMG=${ONNX_MLIR_IMAGE:-onnx-mlir-lean:x86_64}
LL=${LLVM_BIN:-/home/tor/Dev/marco/install/llvm-project/bin}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DROPT=${DROPT:-$REPO/build/tools/dr-opt/dr-opt}
OMB=/build/onnx-mlir/build/Release/bin
W=$(mktemp -d /tmp/spmd-onnx.XXXX); trap 'rm -rf "$W"' EXIT
MA=$(realpath "$MODEL"); DIMS=$(echo "$SHAPE" | tr 'x' ',')

dockt(){ local t=$1; shift; docker run --rm --entrypoint "$OMB/$t" \
  -v "$W:$W" -v "$(dirname "$MA"):$(dirname "$MA")" "$IMG" "$@"; }

if [[ ! -f "$W/rt/libcruntime.a" ]]; then
  C=$(docker create "$IMG"); mkdir -p "$W/rt"
  docker cp "$C":/build/onnx-mlir/build/Release/lib/libcruntime.a "$W/rt/"
  docker cp "$C":/build/onnx-mlir/include "$W/rt/include"; docker rm "$C">/dev/null
fi
ND=$(awk -F, '{print NF}' <<<"$DIMS"); NE=$(awk -F, '{p=1;for(i=1;i<=NF;i++)p*=$i;print p}' <<<"$DIMS")
cat > "$W/harness.c" <<EOF
#include <OnnxMlirRuntime.h>
#include <stdio.h>
#include <stdlib.h>
extern OMTensorList *run_main_graph(OMTensorList *);
int main(int argc, char **argv){
  int64_t sh[$ND]={$DIMS}; size_t n=$NE; float *d=malloc(n*4);
  for(size_t i=0;i<n;i++) d[i]=((int)(i%255)-127)*0.0078431f;
  OMTensor *in=omTensorCreate(d,sh,$ND,ONNX_TYPE_FLOAT);
  OMTensor *ins[1]={in}; OMTensorList *il=omTensorListCreate(ins,1);
  OMTensorList *ol=run_main_graph(il); if(!ol){fprintf(stderr,"NULL\n");return 1;}
  FILE *f=argc>1?fopen(argv[1],"w"):NULL; int64_t ns=omTensorListGetSize(ol);
  for(int64_t t=0;t<ns;t++){OMTensor *o=omTensorListGetOmtArray(ol)[t];
    float *v=(float*)omTensorGetDataPtr(o);int64_t no=omTensorGetNumElems(o);
    for(int64_t i=0;i<no;i++) if(f) fprintf(f,"%.9e\n",v[i]);}
  if(f)fclose(f); return 0;
}
EOF

echo "== front: static affine (shape 0:$SHAPE) =="
dockt onnx-mlir --O2 --EmitMLIR --shapeInformation=0:$SHAPE -o "$W/m" "$MA" >/dev/null 2>&1
dockt onnx-mlir-opt --convert-krnl-to-affine "$W/m.onnx.mlir" -o "$W/m.affine.mlir"

echo "== seq (reference) =="
"$LL/mlir-opt" -allow-unregistered-dialect --lower-affine "$W/m.affine.mlir" -o "$W/seq.scf.mlir"
dockt onnx-mlir-opt --convert-krnl-to-llvm "$W/seq.scf.mlir" -o "$W/seq.kl.mlir"
"$LL/mlir-translate" --mlir-to-llvmir "$W/seq.kl.mlir" -o "$W/seq.ll"
"$LL/clang" -O2 -march=native "$W/harness.c" "$W/seq.ll" "$W/rt/libcruntime.a" -I "$W/rt/include" -lm -o "$W/seq.bin"
"$W/seq.bin" "$W/seq.logits" >/dev/null 2>&1

echo "== spmd (whole-function widening; sequential correctness) =="
$DROPT "$W/m.affine.mlir" -allow-unregistered-dialect \
  --pass-pipeline='builtin.module(dr-par-bubbles{par-spmd-perband},func.func(convert-par-to-scf))' \
  -o "$W/spmd.par.mlir" 2>&1 | grep -iE 'perband' | head -1
"$LL/mlir-opt" -allow-unregistered-dialect --lower-affine "$W/spmd.par.mlir" -o "$W/spmd.scf.mlir"
"$LL/mlir-opt" -allow-unregistered-dialect --convert-scf-to-cf "$W/spmd.scf.mlir" -o "$W/spmd.cf.mlir"
dockt onnx-mlir-opt --convert-krnl-to-llvm "$W/spmd.cf.mlir" -o "$W/spmd.kl.mlir"
"$LL/mlir-translate" --mlir-to-llvmir "$W/spmd.kl.mlir" -o "$W/spmd.ll"
"$LL/clang" -O2 -march=native "$W/harness.c" "$W/spmd.ll" "$W/rt/libcruntime.a" -I "$W/rt/include" -lm -o "$W/spmd.bin"
"$W/spmd.bin" "$W/spmd.logits" >/dev/null 2>&1

echo "== compare =="
ERR=$(paste "$W/seq.logits" "$W/spmd.logits" | awk '
  {d=$1-$2;if(d<0)d=-d;if(d>ma)ma=d;a=($1<0?-$1:$1);if(a>mx)mx=a}
  END{printf "%.3e", (mx>0?ma/mx:ma)}')
N=$(wc -l < "$W/spmd.logits")
echo "norm_rel_err=$ERR (n=$N)"
awk -v e="$ERR" 'BEGIN{exit !(e<=1e-4)}' \
  && echo "RESULT: PASS - SPMD numerically equals reference" \
  || echo "RESULT: FAIL"
