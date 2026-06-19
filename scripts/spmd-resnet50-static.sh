#!/usr/bin/env bash
# spmd-resnet50-static.sh — generate a STATIC batch-1 resnet50 affine via the
# onnx-mlir image, then materialize whole-function SPMD (par-spmd-perband) and
# lower par->omp, reporting the structure.  This is the batch-1 LATENCY target:
# a fixed input shape drops onnx-mlir's dynamic-shape glue (data-dependent
# allocs) so the whole function widens into ONE par.region / omp.parallel team.
#
# Result (dev host, 2026-06-19): materialized foralls=82 critical=55 moved=207
# barriers=136 (137 top-level bands, 82 parallel); par->omp => omp.parallel=1
# omp.wsloop=82 omp.single=55 omp.barrier=136, par leftover=0.
#
# Requires: onnx-mlir-lean docker image + host dr-opt.  Env: ONNX_MLIR_IMAGE,
# MODEL (path to resnet50-v2-7.onnx), SHAPE (default 1x3x224x224).
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IMG=${ONNX_MLIR_IMAGE:-onnx-mlir-lean:x86_64}
MODEL=${MODEL:-$REPO/test/ONNX/resnet50-v2-7.onnx}
SHAPE=${SHAPE:-1x3x224x224}
DROPT=${DROPT:-$REPO/build/tools/dr-opt/dr-opt}
OUT=$(mktemp -d)
BIN=/build/onnx-mlir/build/Release/bin

echo "==> static affine (shape 0:$SHAPE) via $IMG"
docker run --rm --entrypoint bash \
  -v "$(dirname "$MODEL"):/work:ro" -v "$OUT:/out" "$IMG" -c \
  "$BIN/onnx-mlir --O2 --EmitMLIR --shapeInformation=0:$SHAPE /work/$(basename "$MODEL") -o /out/m && \
   $BIN/onnx-mlir-opt --convert-krnl-to-affine /out/m.onnx.mlir -o /out/m.affine.mlir" 2>&1 | tail -2
[ -f "$OUT/m.affine.mlir" ] || { echo "affine gen FAILED"; exit 1; }
echo "    batch dim: $(grep -oE 'memref<[0-9?]+x3x224x224' "$OUT/m.affine.mlir" | head -1)"

echo "==> par-spmd-perband (whole-function widening)"
$DROPT "$OUT/m.affine.mlir" -allow-unregistered-dialect \
  --pass-pipeline='builtin.module(dr-par-bubbles{par-spmd-perband})' -o /dev/null 2>&1 | grep perband

echo "==> + convert-par-to-omp  (op counts)"
$DROPT "$OUT/m.affine.mlir" -allow-unregistered-dialect \
  --pass-pipeline='builtin.module(dr-par-bubbles{par-spmd-perband},func.func(convert-par-to-omp))' \
  -o "$OUT/m.omp.mlir" 2>/dev/null
for k in 'omp.parallel' 'omp.wsloop' 'omp.single' 'omp.barrier'; do
  echo "    $k: $(grep -c "$k" "$OUT/m.omp.mlir")"
done
echo "    par leftover: $(grep -cE '\bpar\.(region|forall|barrier|critical)' "$OUT/m.omp.mlir")"
rm -rf "$OUT"
