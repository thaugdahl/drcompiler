#!/usr/bin/env bash
#===----------------------------------------------------------------------===//
# onnx-dr-pipeline.sh — ONNX model → MLIR → dr-opt (DataRecomputation).
#
# Runs three Docker containers in sequence:
#   1. onnx-mlir image  — lowers .onnx to Krnl/Affine/MemRef MLIR
#   2. onnx-mlir image  — lowers Krnl → Affine (--convert-krnl-to-affine)
#   3. drcc image       — runs dr-opt DataRecomputation pass pipeline
#
# Usage:
#   ./scripts/onnx-dr-pipeline.sh <model.onnx> [-o output.mlir] [options]
#
# Options:
#   -o FILE            Output MLIR path (default: <model>.dr.mlir next to input)
#   --keep-mlir FILE   Also save the pre-dr-opt MLIR to FILE
#   --dr-flags FLAGS   Override dr-opt pass pipeline
#                      (default: builtin.module(data-recomputation{dr-recompute=true}))
#   --diagnostics      Add dr-test-diagnostics (emit per-load classification)
#   --dot FILE         Emit provenance GraphViz dot file
#   -v / --verbose     Print each step to stderr
#
# Images:
#   $ONNX_MLIR_IMAGE  (default: onnx-mlir)
#   $DRCC_IMAGE       (default: drcc)
#===----------------------------------------------------------------------===//
set -euo pipefail

ONNX_MLIR_IMAGE="${ONNX_MLIR_IMAGE:-onnx-mlir}"
DRCC_IMAGE="${DRCC_IMAGE:-drcc}"
ONNX_MLIR_BIN="/build/onnx-mlir/build/Release/bin/onnx-mlir"
ONNX_MLIR_OPT_BIN="/build/onnx-mlir/build/Release/bin/onnx-mlir-opt"
DR_OPT_BIN="/usr/local/bin/dr-opt"

MODEL=""
OUTPUT=""
KEEP_MLIR=""
DR_FLAGS="--pass-pipeline=builtin.module(data-recomputation{dr-recompute=true})"
DIAGNOSTICS=0
DOT_FILE=""
VERBOSE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o)             OUTPUT="$2";     shift 2 ;;
    --keep-mlir)    KEEP_MLIR="$2";  shift 2 ;;
    --dr-flags)     DR_FLAGS="$2";   shift 2 ;;
    --diagnostics)  DIAGNOSTICS=1;   shift   ;;
    --dot)          DOT_FILE="$2";   shift 2 ;;
    -v|--verbose)   VERBOSE=1;       shift   ;;
    -*)             echo "onnx-dr-pipeline: unknown option: $1" >&2; exit 1 ;;
    *)              MODEL="$1";      shift   ;;
  esac
done

[[ -z "$MODEL" ]] && {
  echo "Usage: onnx-dr-pipeline.sh <model.onnx> [-o out.mlir] [--keep-mlir raw.mlir]" >&2
  echo "       [--dr-flags '...'] [--diagnostics] [--dot out.dot] [-v]" >&2
  exit 1
}
[[ -f "$MODEL" ]] || { echo "onnx-dr-pipeline: not found: $MODEL" >&2; exit 1; }

log() { [[ "$VERBOSE" -eq 1 ]] && echo "[onnx-dr-pipeline] $*" >&2 || true; }
die() { echo "onnx-dr-pipeline: error: $*" >&2; exit 1; }

# --- Inject diagnostics / dot into DR_FLAGS ---
if [[ "$DIAGNOSTICS" -eq 1 || -n "$DOT_FILE" ]]; then
  EXTRA=""
  [[ "$DIAGNOSTICS" -eq 1 ]] && EXTRA="${EXTRA} dr-test-diagnostics"
  [[ -n "$DOT_FILE" ]]        && EXTRA="${EXTRA} dr-dot-file=$(realpath -m "$DOT_FILE")"
  if [[ "$DR_FLAGS" == *"data-recomputation{"* ]]; then
    DR_FLAGS="${DR_FLAGS/data-recomputation\{/data-recomputation\{${EXTRA} }"
  elif [[ "$DR_FLAGS" == *"data-recomputation"* ]]; then
    DR_FLAGS="${DR_FLAGS/data-recomputation/data-recomputation\{${EXTRA}\}}"
  fi
fi

# --- Resolve absolute paths ---
MODEL_ABS="$(realpath "$MODEL")"
MODEL_DIR="$(dirname "$MODEL_ABS")"
BASE="$(basename "$MODEL_ABS" .onnx)"

[[ -z "$OUTPUT" ]] && OUTPUT="${MODEL_DIR}/${BASE}.dr.mlir"
OUTPUT_ABS="$(realpath -m "$OUTPUT")"
OUTPUT_DIR="$(dirname "$OUTPUT_ABS")"
mkdir -p "$OUTPUT_DIR"

# All intermediate files live in a temp dir (mounted into each container)
TMPDIR="$(mktemp -d "/tmp/onnx-dr.${BASE}.XXXXXX")"
trap "rm -rf '$TMPDIR'" EXIT
# onnx-mlir appends .onnx.mlir (not .mlir) for --EmitMLIR
RAW_MLIR="${TMPDIR}/${BASE}.onnx.mlir"
AFFINE_MLIR="${TMPDIR}/${BASE}.affine.mlir"

# --keep-mlir: resolve to abs path so we can mount its dir
if [[ -n "$KEEP_MLIR" ]]; then
  KEEP_MLIR_ABS="$(realpath -m "$KEEP_MLIR")"
  KEEP_MLIR_DIR="$(dirname "$KEEP_MLIR_ABS")"
  mkdir -p "$KEEP_MLIR_DIR"
else
  KEEP_MLIR_ABS=""
  KEEP_MLIR_DIR=""
fi

# --- Build mount list helper ---
# Collect unique directories into mounts array
add_mount() {
  local d="$1"
  if [[ ":$_SEEN:" != *":$d:"* ]]; then
    _MOUNTS+=(-v "$d:$d")
    _SEEN="$_SEEN:$d"
  fi
}

# ===== Step 1: ONNX → MLIR (onnx-mlir image) =====
_MOUNTS=(); _SEEN=""
add_mount "$MODEL_DIR"
add_mount "$TMPDIR"

log "docker run $ONNX_MLIR_IMAGE  --EmitMLIR  $MODEL_ABS  →  $RAW_MLIR"
docker run --rm \
  "${_MOUNTS[@]}" \
  --entrypoint "$ONNX_MLIR_BIN" \
  "$ONNX_MLIR_IMAGE" \
  --EmitMLIR -o "${RAW_MLIR%.onnx.mlir}" "$MODEL_ABS" \
  || die "onnx-mlir step failed"

# ===== Step 2: Krnl → Affine (onnx-mlir-opt, same image) =====
_MOUNTS=(); _SEEN=""
add_mount "$TMPDIR"

log "docker run $ONNX_MLIR_IMAGE  onnx-mlir-opt --convert-krnl-to-affine  $RAW_MLIR  →  $AFFINE_MLIR"
docker run --rm \
  "${_MOUNTS[@]}" \
  --entrypoint "$ONNX_MLIR_OPT_BIN" \
  "$ONNX_MLIR_IMAGE" \
  --convert-krnl-to-affine "$RAW_MLIR" -o "$AFFINE_MLIR" \
  || die "krnl-to-affine step failed"

if [[ -n "$KEEP_MLIR_ABS" ]]; then
  cp "$AFFINE_MLIR" "$KEEP_MLIR_ABS"
  log "saved pre-dr-opt MLIR → $KEEP_MLIR_ABS"
fi

# ===== Step 3: Affine MLIR → dr-opt (drcc image) =====
_MOUNTS=(); _SEEN=""
add_mount "$TMPDIR"
add_mount "$OUTPUT_DIR"
[[ -n "$DOT_FILE" ]] && add_mount "$(dirname "$(realpath -m "$DOT_FILE")")"

log "docker run $DRCC_IMAGE  dr-opt  $DR_FLAGS  $AFFINE_MLIR  →  $OUTPUT_ABS"
docker run --rm \
  "${_MOUNTS[@]}" \
  --entrypoint "$DR_OPT_BIN" \
  "$DRCC_IMAGE" \
  --allow-unregistered-dialect $DR_FLAGS "$AFFINE_MLIR" -o "$OUTPUT_ABS" \
  || die "dr-opt step failed"

echo "[onnx-dr-pipeline] done → $OUTPUT_ABS" >&2
