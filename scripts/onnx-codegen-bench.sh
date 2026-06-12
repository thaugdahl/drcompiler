#!/usr/bin/env bash
#===----------------------------------------------------------------------===//
# onnx-codegen-bench.sh — WP-O0 end-to-end ONNX inference benchmark for the
# codegen campaign (demote + register-block + promote).
#
# Configs (all linked against the same C harness calling run_main_graph):
#   none    — onnx-mlir --O2 --EmitMLIR → krnl-to-affine → [no dr-opt] →
#             lower-affine → krnl-to-llvm → translate → clang.  Isolates the
#             transform effect (same backend, dr-opt skipped).
#   codegen — same pipeline with host dr-opt:
#             dr-scalar-reduction-demote, affine-register-block{mr,nr,vl},
#             dr-scalar-reduction-promote.
#   o3      — onnx-mlir --O3 --EmitObj (their optimized krnl path).  The
#             honest bar: beating only `none` is a finding about naive
#             lowering, not about onnx-mlir.
#
# Correctness: logits compared against `none` — criterion is max abs error
# normalized by max |logit| <= 1e-4 (vectorized reductions reassociate FP, so
# bit-identity is not the bar) plus top-1 class agreement.
# Timing: median of --iters single-inference latencies (untimed warmup first).
#
# Usage:
#   ./scripts/onnx-codegen-bench.sh <model.onnx> [options]
# Options:
#   --iters N        timed iterations per config (default 5)
#   --shape D,D,...  input tensor shape (default 1,3,224,224, f32)
#   --configs LIST   comma-separated subset of none,codegen,o3
#   --mr/--nr/--vl N register-block parameters (default 8/16/16)
#   --workdir DIR    keep intermediates here (default: mktemp, kept)
# Environment:
#   ONNX_MLIR_IMAGE  docker image with onnx-mlir + libcruntime
#                    (default onnx-mlir-lean:x86_64)
#   LLVM_BIN         host LLVM-22 bin dir with mlir-opt/mlir-translate/clang
#                    (default /home/tor/Dev/marco/install/llvm-project/bin)
#   DR_OPT           host dr-opt (default <repo>/build/tools/dr-opt/dr-opt)
#===----------------------------------------------------------------------===//
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

ONNX_MLIR_IMAGE="${ONNX_MLIR_IMAGE:-onnx-mlir-lean:x86_64}"
LLVM_BIN="${LLVM_BIN:-/home/tor/Dev/marco/install/llvm-project/bin}"
DR_OPT="${DR_OPT:-$REPO_ROOT/build/tools/dr-opt/dr-opt}"
OM_BIN=/build/onnx-mlir/build/Release/bin

ITERS=5; SHAPE="1,3,224,224"; CONFIGS="none,codegen,o3"
MR=8; NR=16; VL=16; WORKDIR=""
MODEL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --iters) ITERS=$2; shift 2;;
    --shape) SHAPE=$2; shift 2;;
    --configs) CONFIGS=$2; shift 2;;
    --mr) MR=$2; shift 2;;
    --nr) NR=$2; shift 2;;
    --vl) VL=$2; shift 2;;
    --workdir) WORKDIR=$2; shift 2;;
    *) MODEL=$1; shift;;
  esac
done
[[ -f "$MODEL" ]] || { echo "usage: $0 <model.onnx> [options]" >&2; exit 1; }
MODEL_ABS="$(realpath "$MODEL")"
[[ -n "$WORKDIR" ]] || WORKDIR="$(mktemp -d /tmp/onnx-codegen-bench.XXXX)"
mkdir -p "$WORKDIR"
echo "workdir: $WORKDIR" >&2

dock() { # run a tool from the image with workdir + model dir mounted
  local tool=$1; shift
  docker run --rm -v "$WORKDIR:$WORKDIR" -v "$(dirname "$MODEL_ABS"):$(dirname "$MODEL_ABS")" \
    --entrypoint "$OM_BIN/$tool" "$ONNX_MLIR_IMAGE" "$@"
}

# ---- runtime bits from the image (once) ----
if [[ ! -f "$WORKDIR/rt/libcruntime.a" ]]; then
  C=$(docker create "$ONNX_MLIR_IMAGE")
  mkdir -p "$WORKDIR/rt"
  docker cp "$C":/build/onnx-mlir/build/Release/lib/libcruntime.a "$WORKDIR/rt/"
  docker cp "$C":/build/onnx-mlir/include "$WORKDIR/rt/include"
  docker rm "$C" >/dev/null
fi

# ---- harness (shape baked in) ----
NDIM=$(awk -F, '{print NF}' <<<"$SHAPE")
NELEM=$(awk -F, '{p=1; for(i=1;i<=NF;i++) p*=$i; print p}' <<<"$SHAPE")
cat > "$WORKDIR/harness.c" <<EOF
#include <OnnxMlirRuntime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
extern OMTensorList *run_main_graph(OMTensorList *);
int main(int argc, char **argv) {
  const char *logits_path = argc > 1 ? argv[1] : NULL;
  int iters = argc > 2 ? atoi(argv[2]) : 5;
  int64_t shape[$NDIM] = {${SHAPE}};
  size_t n = $NELEM;
  float *data = malloc(n * sizeof(float));
  for (size_t i = 0; i < n; i++)
    data[i] = ((int)(i % 255) - 127) * 0.0078431f;
  OMTensor *in = omTensorCreate(data, shape, $NDIM, ONNX_TYPE_FLOAT);
  OMTensor *ins[1] = {in};
  OMTensorList *inl = omTensorListCreate(ins, 1);
  OMTensorList *outl = run_main_graph(inl); /* warmup + correctness */
  if (!outl) { fprintf(stderr, "run_main_graph NULL\n"); return 1; }
  OMTensor *out = omTensorListGetOmtArray(outl)[0];
  float *logits = (float *)omTensorGetDataPtr(out);
  int64_t nout = omTensorGetNumElems(out);
  if (logits_path) {
    FILE *f = fopen(logits_path, "w");
    for (int64_t i = 0; i < nout; i++) fprintf(f, "%.9e\n", logits[i]);
    fclose(f);
  }
  int64_t best = 0;
  for (int64_t i = 1; i < nout; i++) if (logits[i] > logits[best]) best = i;
  fprintf(stderr, "top1=%lld nout=%lld\n", (long long)best, (long long)nout);
  omTensorListDestroy(outl);
  for (int r = 0; r < iters; r++) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    OMTensorList *o = run_main_graph(inl);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (!o) return 1;
    omTensorListDestroy(o);
    printf("%.6f\n", (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9);
  }
  omTensorListDestroy(inl);
  free(data);
  return 0;
}
EOF

link_bin() { # $1 = object, $2 = binary
  "$LLVM_BIN/clang" -O2 "$WORKDIR/harness.c" "$1" "$WORKDIR/rt/libcruntime.a" \
    -I "$WORKDIR/rt/include" -lm -o "$2"
}

# ---- shared front half (none + codegen) ----
need_affine() {
  [[ -f "$WORKDIR/model.affine.mlir" ]] && return 0
  echo "[front] onnx-mlir --O2 --EmitMLIR" >&2
  dock onnx-mlir --O2 --EmitMLIR -o "$WORKDIR/model" "$MODEL_ABS"
  echo "[front] convert-krnl-to-affine" >&2
  dock onnx-mlir-opt --convert-krnl-to-affine "$WORKDIR/model.onnx.mlir" \
    -o "$WORKDIR/model.affine.mlir"
}

back_half() { # $1 = input affine mlir, $2 = config name
  echo "[$2] lower-affine (host)" >&2
  "$LLVM_BIN/mlir-opt" -allow-unregistered-dialect --lower-affine "$1" \
    -o "$WORKDIR/$2.scf.mlir"
  echo "[$2] convert-krnl-to-llvm (image)" >&2
  dock onnx-mlir-opt --convert-krnl-to-llvm "$WORKDIR/$2.scf.mlir" \
    -o "$WORKDIR/$2.kl.mlir"
  echo "[$2] translate + clang (host)" >&2
  "$LLVM_BIN/mlir-translate" --mlir-to-llvmir "$WORKDIR/$2.kl.mlir" -o "$WORKDIR/$2.ll"
  "$LLVM_BIN/clang" -O2 -march=native -c "$WORKDIR/$2.ll" -o "$WORKDIR/$2.o"
  link_bin "$WORKDIR/$2.o" "$WORKDIR/$2.bin"
}

build_none() { need_affine; back_half "$WORKDIR/model.affine.mlir" none; }

build_codegen() {
  need_affine
  echo "[codegen] dr-opt demote + register-block + promote (host)" >&2
  "$DR_OPT" "$WORKDIR/model.affine.mlir" -allow-unregistered-dialect \
    --pass-pipeline="builtin.module(func.func(dr-scalar-reduction-demote,affine-register-block{mr=$MR nr=$NR vl=$VL},dr-scalar-reduction-promote))" \
    -o "$WORKDIR/codegen.dr.mlir"
  back_half "$WORKDIR/codegen.dr.mlir" codegen
}

build_o3() {
  echo "[o3] onnx-mlir --O3 --EmitObj" >&2
  dock onnx-mlir --O3 --EmitObj -o "$WORKDIR/o3" "$MODEL_ABS"
  link_bin "$WORKDIR/o3.o" "$WORKDIR/o3.bin"
}

IFS=, read -ra CFGS <<<"$CONFIGS"
for c in "${CFGS[@]}"; do "build_$c"; done

echo
echo "== run (iters=$ITERS) =="
declare -A MEDIAN TOP1
for c in "${CFGS[@]}"; do
  "$WORKDIR/$c.bin" "$WORKDIR/$c.logits" "$ITERS" >"$WORKDIR/$c.times" 2>"$WORKDIR/$c.top1"
  MEDIAN[$c]=$(sort -n "$WORKDIR/$c.times" | awk -v n="$ITERS" 'NR==int((n+1)/2){print; exit}')
  TOP1[$c]=$(grep -o "top1=[0-9]*" "$WORKDIR/$c.top1" | cut -d= -f2)
done

echo
printf "%-9s %12s %8s %14s %10s\n" config median_s top1 norm_rel_err speedup
REF=${MEDIAN[none]:-${MEDIAN[${CFGS[0]}]}}
for c in "${CFGS[@]}"; do
  ERR="-"
  if [[ "$c" != none && -f "$WORKDIR/none.logits" ]]; then
    ERR=$(paste "$WORKDIR/none.logits" "$WORKDIR/$c.logits" | awk '
      {d=$1-$2; if(d<0)d=-d; if(d>ma)ma=d; a=($1<0?-$1:$1); if(a>mx)mx=a}
      END{printf "%.3e", (mx>0? ma/mx : ma)}')
  fi
  SPD=$(awk -v r="$REF" -v m="${MEDIAN[$c]}" 'BEGIN{printf "%.2fx", r/m}')
  printf "%-9s %12s %8s %14s %10s\n" "$c" "${MEDIAN[$c]}" "${TOP1[$c]}" "$ERR" "$SPD"
done
echo
echo "criterion: norm_rel_err <= 1e-4 and identical top1" >&2
