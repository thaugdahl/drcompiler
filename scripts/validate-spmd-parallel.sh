#!/usr/bin/env bash
# validate-spmd-parallel.sh — krnl-FREE parallel SPMD: lower a static-shape ONNX
# model to a REAL OpenMP binary (bypassing onnx-mlir's convert-krnl-to-llvm) and
# check (a) numeric correctness vs the sequential reference and (b) speedup.
#
#   lower-krnl-global (krnl.global->memref.global) + par-spmd-perband +
#   convert-par-to-omp  -> host mlir-opt full lowering (omp intact) ->
#   convert-openmp-to-llvm -> mlir-translate -> clang -fopenmp.
#   Entry via a hand-written C harness calling _mlir_ciface_main_graph.
#
# Usage: validate-spmd-parallel.sh <model.onnx> <INSHAPE> <OUTSHAPE> [threads...]
#   INSHAPE/OUTSHAPE like 1x3x224x224 / 1x1000   (input 0, single f32 in/out)
set -uo pipefail
export LC_ALL=C
MODEL=$1; INSHAPE=$2; OUTSHAPE=$3; shift 3
THREADS=${*:-1 2 4 8 16}
IMG=${ONNX_MLIR_IMAGE:-onnx-mlir-lean:x86_64}
LL=${LLVM_BIN:-/home/tor/Dev/marco/install/llvm-project/bin}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DROPT=${DROPT:-$REPO/build/tools/dr-opt/dr-opt}
OMB=/build/onnx-mlir/build/Release/bin
W=$(mktemp -d /tmp/spmd-par.XXXX); trap 'rm -rf "$W"' EXIT
MA=$(realpath "$MODEL")
dockt(){ local t=$1; shift; docker run --rm --entrypoint "$OMB/$t" -v "$W:$W" -v "$(dirname "$MA"):$(dirname "$MA")" "$IMG" "$@"; }

# --- harness gen: row-major descriptors for one f32 input + one f32 output ---
gen_harness(){ # $1 inshape  $2 outshape  -> stdout C
  python3 - "$1" "$2" <<'PY'
import sys
ind=[int(x) for x in sys.argv[1].split('x')]; outd=[int(x) for x in sys.argv[2].split('x')]
def strides(d):
  s=[1]*len(d)
  for i in range(len(d)-2,-1,-1): s[i]=s[i+1]*d[i+1]
  return s
ist,ost=strides(ind),strides(outd); ne=1
for x in ind: ne*=x
oe=1
for x in outd: oe*=x
def mref(n,r): return f"struct M{n} {{ float *a,*al; long off; long sz[{r}]; long st[{r}]; }};"
print('#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#include <time.h>')
print(mref('I',len(ind))); print(mref('O',len(outd)))
print('extern void _mlir_ciface_main_graph(struct MO*, struct MI*);')
print('int main(int argc,char**argv){')
print(f'  size_t n={ne}; float*d=malloc(n*4); for(size_t i=0;i<n;i++) d[i]=((int)(i%255)-127)*0.0078431f;')
print(f'  struct MI in={{d,d,0,{{{",".join(map(str,ind))}}},{{{",".join(map(str,ist))}}}}};')
print('  struct MO res; memset(&res,0,sizeof(res));')
print('  _mlir_ciface_main_graph(&res,&in); /* warmup+correctness */')
print('  FILE*f=argc>1?fopen(argv[1],"w"):NULL;')
print(f'  for(int i=0;i<{oe};i++){{ float v=res.al[res.off+i]; if(f)fprintf(f,"%.9e\\n",v);}}')
print('  if(f)fclose(f);')
print('  int iters=argc>2?atoi(argv[2]):5;')
print('  for(int r=0;r<iters;r++){ struct timespec a,b; clock_gettime(CLOCK_MONOTONIC,&a);')
print('    struct MO o; memset(&o,0,sizeof(o)); _mlir_ciface_main_graph(&o,&in);')
print('    clock_gettime(CLOCK_MONOTONIC,&b);')
print('    printf("%.6f\\n",(b.tv_sec-a.tv_sec)+(b.tv_nsec-a.tv_nsec)*1e-9);}')
print('  return 0;}')
PY
}
gen_harness "$INSHAPE" "$OUTSHAPE" > "$W/h.c"

LOWER="--lower-affine --convert-scf-to-cf --convert-math-to-libm --expand-strided-metadata --finalize-memref-to-llvm --convert-cf-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --convert-openmp-to-llvm --reconcile-unrealized-casts"

echo "== front: static affine (0:$INSHAPE) =="
dockt onnx-mlir --O2 --EmitMLIR --shapeInformation=0:$INSHAPE -o "$W/m" "$MA" >/dev/null 2>&1
dockt onnx-mlir-opt --convert-krnl-to-affine "$W/m.onnx.mlir" -o "$W/m.affine.mlir"

build_cfg(){ # $1=name  $2=extra dr-opt pipeline (or empty)
  local nm=$1 pipe=$2
  if [[ -n "$pipe" ]]; then
    $DROPT "$W/m.affine.mlir" -allow-unregistered-dialect --pass-pipeline="$pipe" -o "$W/$nm.mlir" 2>&1 | grep -iE 'perband' | head -1
  else cp "$W/m.affine.mlir" "$W/$nm.mlir"; fi
  "$LL/mlir-opt" -allow-unregistered-dialect $LOWER "$W/$nm.mlir" -o "$W/$nm.llvm.mlir"
  "$LL/mlir-translate" --mlir-to-llvmir "$W/$nm.llvm.mlir" -o "$W/$nm.ll"
  "$LL/clang" -O2 -march=native -fopenmp "$W/h.c" "$W/$nm.ll" -lm -o "$W/$nm.bin"
}
echo "== seq reference (krnl-free, no SPMD) =="
build_cfg seq "builtin.module(lower-krnl-global)"
echo "== spmd (perfect reductions + lower-krnl-global + par-spmd-perband + par->omp) =="
# dr-affine-loop-distribute + dr-scalar-reduction-demote PERFECT the conv/gemm
# reduction bands (fission init out, single-level memref-accumulator) so they
# materialize as par.forall (sharded) instead of par.critical (serial) -- the
# heavy convs become parallel.  Both passes are exact (semantics-preserving).
build_cfg spmd "builtin.module(func.func(dr-affine-loop-distribute,dr-scalar-reduction-demote),lower-krnl-global,dr-par-bubbles{par-spmd-perband},func.func(convert-par-to-omp))"

OMP_NUM_THREADS=1 "$W/seq.bin" "$W/seq.logits" 1 >/dev/null 2>&1
echo "== correctness (spmd vs seq, per thread count) + median time =="
SREF=""
for t in $THREADS; do
  IT=${ITERS:-7}
  OMP_NUM_THREADS=$t OMP_PROC_BIND=close OMP_PLACES=cores "$W/spmd.bin" "$W/spmd.logits" "$IT" >"$W/t.$t" 2>/dev/null
  med=$(sort -n "$W/t.$t" | awk -v n="$IT" 'NR==int((n+1)/2){print;exit}')
  err=$(paste "$W/seq.logits" "$W/spmd.logits" | awk '{d=$1-$2;if(d<0)d=-d;if(d>ma)ma=d;a=($1<0?-$1:$1);if(a>mx)mx=a}END{printf "%.2e",(mx>0?ma/mx:ma)}')
  [[ -z "$SREF" ]] && SREF=$med
  spd=$(awk -v r="$SREF" -v m="$med" 'BEGIN{printf "%.2f", r/m}')
  printf "  threads=%-3s median=%.4fs  speedup=%sx  norm_rel_err=%s\n" "$t" "$med" "$spd" "$err"
done
