#!/usr/bin/env bash
#===----------------------------------------------------------------------===//
# onnx-to-mlir.sh — Lower an ONNX model to MLIR via the onnx-mlir Docker image.
#
# Usage:
#   ./scripts/onnx-to-mlir.sh <model.onnx> [-o output.mlir] [--level LEVEL]
#
# Options:
#   -o FILE     Output path (default: <model>.mlir next to input)
#   --level     Emission level: onnx | onnx-inferred | mlir (default: mlir)
#               onnx          → --EmitONNXBasic  (pre-shape-inference ONNX dialect)
#               onnx-inferred → --EmitONNXIR     (ONNX dialect + shape inference)
#               mlir          → --EmitMLIR       (Krnl/Affine/MemRef, ready for dr-opt)
#
# Images:
#   $ONNX_MLIR_IMAGE  (default: onnx-mlir)
#===----------------------------------------------------------------------===//
set -euo pipefail

ONNX_MLIR_IMAGE="${ONNX_MLIR_IMAGE:-onnx-mlir}"
ONNX_MLIR_BIN="/build/onnx-mlir/build/Release/bin/onnx-mlir"

MODEL=""
OUTPUT=""
LEVEL="mlir"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o)        OUTPUT="$2"; shift 2 ;;
    --level)   LEVEL="$2";  shift 2 ;;
    -*)        echo "onnx-to-mlir: unknown option: $1" >&2; exit 1 ;;
    *)         MODEL="$1";  shift ;;
  esac
done

[[ -z "$MODEL" ]] && {
  echo "Usage: onnx-to-mlir.sh <model.onnx> [-o output.mlir] [--level onnx|onnx-inferred|mlir]" >&2
  exit 1
}
[[ -f "$MODEL" ]] || { echo "onnx-to-mlir: not found: $MODEL" >&2; exit 1; }

case "$LEVEL" in
  onnx)          EMIT_FLAG="--EmitONNXBasic" ;;
  onnx-inferred) EMIT_FLAG="--EmitONNXIR"    ;;
  mlir)          EMIT_FLAG="--EmitMLIR"       ;;
  *) echo "onnx-to-mlir: unknown level '$LEVEL' (use: onnx | onnx-inferred | mlir)" >&2; exit 1 ;;
esac

MODEL_ABS="$(realpath "$MODEL")"
MODEL_DIR="$(dirname "$MODEL_ABS")"

if [[ -z "$OUTPUT" ]]; then
  OUTPUT="${MODEL_DIR}/$(basename "$MODEL_ABS" .onnx).mlir"
fi
OUTPUT_ABS="$(realpath -m "$OUTPUT")"
OUTPUT_DIR="$(dirname "$OUTPUT_ABS")"
mkdir -p "$OUTPUT_DIR"

# -o takes base path; onnx-mlir appends .mlir
OUTPUT_BASE="${OUTPUT_ABS%.mlir}"

# Collect unique mount dirs
mounts=()
seen=""
for d in "$MODEL_DIR" "$OUTPUT_DIR"; do
  if [[ ":$seen:" != *":$d:"* ]]; then
    mounts+=(-v "$d:$d")
    seen="$seen:$d"
  fi
done

echo "[onnx-to-mlir] $MODEL_ABS  →  $OUTPUT_ABS  (${EMIT_FLAG})" >&2
exec docker run --rm \
  "${mounts[@]}" \
  --entrypoint "$ONNX_MLIR_BIN" \
  "$ONNX_MLIR_IMAGE" \
  "$EMIT_FLAG" -o "$OUTPUT_BASE" "$MODEL_ABS"
