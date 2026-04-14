#!/usr/bin/env bash
#===----------------------------------------------------------------------===//
# onnx-compile.sh — ONNX model → object file via dr-opt DataRecomputation.
#
# Pipeline (6 Docker steps):
#   1. onnx-mlir image  — ONNX → Krnl/Affine/MemRef MLIR  (--EmitMLIR)
#   2. onnx-mlir image  — Krnl loops → Affine             (--convert-krnl-to-affine)
#   3. drcc image       — DataRecomputation pass           (dr-opt)
#   4. onnx-mlir image  — Krnl globals/entry → LLVM       (--convert-krnl-to-llvm)
#   5. drcc image       — Affine/SCF/CF/MemRef → LLVM dialect (mlir-opt)
#   6. drcc image       — LLVM dialect → IR → object file  (mlir-translate + clang)
#
# Usage:
#   ./scripts/onnx-compile.sh <model.onnx> [-o output.o] [options]
#
# Options:
#   -o FILE            Output object path (default: <model>.o next to input)
#   --keep-mlir DIR    Save all intermediate .mlir files to DIR
#   --dr-flags FLAGS   Override dr-opt pass pipeline
#   --diagnostics      Add dr-test-diagnostics to DR pass
#   --dot FILE         Emit provenance GraphViz dot file
#   --emit-llvm        Stop after LLVM IR (.ll) instead of compiling to .o
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
MLIR_OPT_BIN="/opt/llvm/bin/mlir-opt"
MLIR_TRANSLATE_BIN="/opt/llvm/bin/mlir-translate"
CLANG_BIN="/opt/llvm/bin/clang"

MODEL=""
OUTPUT=""
KEEP_MLIR_DIR=""
DR_FLAGS="--pass-pipeline=builtin.module(data-recomputation{dr-recompute=true})"
DIAGNOSTICS=0
DOT_FILE=""
EMIT_LLVM=0
VERBOSE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o)             OUTPUT="$2";        shift 2 ;;
    --keep-mlir)    KEEP_MLIR_DIR="$2"; shift 2 ;;
    --dr-flags)     DR_FLAGS="$2";      shift 2 ;;
    --diagnostics)  DIAGNOSTICS=1;      shift   ;;
    --dot)          DOT_FILE="$2";      shift 2 ;;
    --emit-llvm)    EMIT_LLVM=1;        shift   ;;
    -v|--verbose)   VERBOSE=1;          shift   ;;
    -*)             echo "onnx-compile: unknown option: $1" >&2; exit 1 ;;
    *)              MODEL="$1";         shift   ;;
  esac
done

[[ -z "$MODEL" ]] && {
  echo "Usage: onnx-compile.sh <model.onnx> [-o out.o] [--keep-mlir dir]" >&2
  echo "       [--dr-flags '...'] [--diagnostics] [--dot out.dot]" >&2
  echo "       [--emit-llvm] [-v]" >&2
  exit 1
}
[[ -f "$MODEL" ]] || { echo "onnx-compile: not found: $MODEL" >&2; exit 1; }

log() { [[ "$VERBOSE" -eq 1 ]] && echo "[onnx-compile] $*" >&2 || true; }
die() { echo "onnx-compile: error: $*" >&2; exit 1; }

# --- Inject diagnostics / dot into DR_FLAGS ---
if [[ "$DIAGNOSTICS" -eq 1 || -n "$DOT_FILE" ]]; then
  EXTRA=""
  [[ "$DIAGNOSTICS" -eq 1 ]] && EXTRA="${EXTRA}dr-test-diagnostics "
  [[ -n "$DOT_FILE" ]]        && EXTRA="${EXTRA}dr-dot-file=$(realpath -m "$DOT_FILE") "
  EXTRA="${EXTRA% }"  # trim trailing space
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

EXT=".o"; [[ "$EMIT_LLVM" -eq 1 ]] && EXT=".ll"
[[ -z "$OUTPUT" ]] && OUTPUT="${MODEL_DIR}/${BASE}${EXT}"
OUTPUT_ABS="$(realpath -m "$OUTPUT")"
OUTPUT_DIR="$(dirname "$OUTPUT_ABS")"
mkdir -p "$OUTPUT_DIR"

TMPDIR="$(mktemp -d "/tmp/onnx-compile.${BASE}.XXXXXX")"
trap "rm -rf '$TMPDIR'" EXIT

# Intermediate file paths (all in TMPDIR)
RAW_MLIR="${TMPDIR}/${BASE}.onnx.mlir"        # onnx-mlir appends .onnx.mlir
AFFINE_MLIR="${TMPDIR}/${BASE}.affine.mlir"
DR_MLIR="${TMPDIR}/${BASE}.dr.mlir"
KRNL_LLVM_MLIR="${TMPDIR}/${BASE}.krnl-llvm.mlir"
LLVM_MLIR="${TMPDIR}/${BASE}.llvm.mlir"
LLVM_IR="${TMPDIR}/${BASE}.ll"

# --keep-mlir dir
if [[ -n "$KEEP_MLIR_DIR" ]]; then
  KEEP_DIR="$(realpath -m "$KEEP_MLIR_DIR")"
  mkdir -p "$KEEP_DIR"
else
  KEEP_DIR=""
fi

# --- Mount helper ---
add_mount() {
  local d="$1"
  if [[ ":$_SEEN:" != *":$d:"* ]]; then
    _MOUNTS+=(-v "$d:$d")
    _SEEN="$_SEEN:$d"
  fi
}

# ===== Step 1: ONNX → Krnl/Affine MLIR =====
_MOUNTS=(); _SEEN=""
add_mount "$MODEL_DIR"
add_mount "$TMPDIR"

log "Step 1: onnx-mlir --EmitMLIR  $MODEL_ABS → $RAW_MLIR"
docker run --rm \
  "${_MOUNTS[@]}" \
  --entrypoint "$ONNX_MLIR_BIN" \
  "$ONNX_MLIR_IMAGE" \
  --EmitMLIR -o "${RAW_MLIR%.onnx.mlir}" "$MODEL_ABS" \
  || die "Step 1 (onnx-mlir --EmitMLIR) failed"

[[ -n "$KEEP_DIR" ]] && cp "$RAW_MLIR" "$KEEP_DIR/${BASE}.01-raw.mlir"

# ===== Step 2: Krnl loops → Affine =====
_MOUNTS=(); _SEEN=""
add_mount "$TMPDIR"

log "Step 2: onnx-mlir-opt --convert-krnl-to-affine  $RAW_MLIR → $AFFINE_MLIR"
docker run --rm \
  "${_MOUNTS[@]}" \
  --entrypoint "$ONNX_MLIR_OPT_BIN" \
  "$ONNX_MLIR_IMAGE" \
  --convert-krnl-to-affine "$RAW_MLIR" -o "$AFFINE_MLIR" \
  || die "Step 2 (convert-krnl-to-affine) failed"

[[ -n "$KEEP_DIR" ]] && cp "$AFFINE_MLIR" "$KEEP_DIR/${BASE}.02-affine.mlir"

# ===== Step 3: DataRecomputation (dr-opt) =====
_MOUNTS=(); _SEEN=""
add_mount "$TMPDIR"
[[ -n "$DOT_FILE" ]] && add_mount "$(dirname "$(realpath -m "$DOT_FILE")")"

log "Step 3: dr-opt $DR_FLAGS  $AFFINE_MLIR → $DR_MLIR"
docker run --rm \
  "${_MOUNTS[@]}" \
  --entrypoint "$DR_OPT_BIN" \
  "$DRCC_IMAGE" \
  --allow-unregistered-dialect $DR_FLAGS "$AFFINE_MLIR" -o "$DR_MLIR" \
  || die "Step 3 (dr-opt data-recomputation) failed"

[[ -n "$KEEP_DIR" ]] && cp "$DR_MLIR" "$KEEP_DIR/${BASE}.03-dr.mlir"

# ===== Step 4: krnl.global / krnl.entry_point → LLVM dialect =====
_MOUNTS=(); _SEEN=""
add_mount "$TMPDIR"

log "Step 4: onnx-mlir-opt --convert-krnl-to-llvm  $DR_MLIR → $KRNL_LLVM_MLIR"
docker run --rm \
  "${_MOUNTS[@]}" \
  --entrypoint "$ONNX_MLIR_OPT_BIN" \
  "$ONNX_MLIR_IMAGE" \
  --convert-krnl-to-llvm "$DR_MLIR" -o "$KRNL_LLVM_MLIR" \
  || die "Step 4 (convert-krnl-to-llvm) failed"

[[ -n "$KEEP_DIR" ]] && cp "$KRNL_LLVM_MLIR" "$KEEP_DIR/${BASE}.04-krnl-llvm.mlir"

# ===== Step 5: Affine/SCF/CF/MemRef → LLVM dialect =====
_MOUNTS=(); _SEEN=""
add_mount "$TMPDIR"

LOWER_PIPELINE="builtin.module(\
lower-affine,\
convert-scf-to-cf,\
convert-cf-to-llvm,\
convert-func-to-llvm,\
finalize-memref-to-llvm,\
reconcile-unrealized-casts)"

log "Step 5: mlir-opt (affine→scf→cf→llvm)  $KRNL_LLVM_MLIR → $LLVM_MLIR"
docker run --rm \
  "${_MOUNTS[@]}" \
  --entrypoint "$MLIR_OPT_BIN" \
  "$DRCC_IMAGE" \
  --allow-unregistered-dialect \
  --pass-pipeline="$LOWER_PIPELINE" \
  "$KRNL_LLVM_MLIR" -o "$LLVM_MLIR" \
  || die "Step 5 (mlir-opt lowering) failed"

[[ -n "$KEEP_DIR" ]] && cp "$LLVM_MLIR" "$KEEP_DIR/${BASE}.05-llvm.mlir"

# ===== Step 6: LLVM dialect → LLVM IR =====
_MOUNTS=(); _SEEN=""
add_mount "$TMPDIR"

log "Step 6: mlir-translate --mlir-to-llvmir  $LLVM_MLIR → $LLVM_IR"
docker run --rm \
  "${_MOUNTS[@]}" \
  --entrypoint "$MLIR_TRANSLATE_BIN" \
  "$DRCC_IMAGE" \
  --mlir-to-llvmir "$LLVM_MLIR" -o "$LLVM_IR" \
  || die "Step 6 (mlir-translate) failed"

[[ -n "$KEEP_DIR" ]] && cp "$LLVM_IR" "$KEEP_DIR/${BASE}.06.ll"

if [[ "$EMIT_LLVM" -eq 1 ]]; then
  cp "$LLVM_IR" "$OUTPUT_ABS"
  echo "[onnx-compile] done (LLVM IR) → $OUTPUT_ABS" >&2
  exit 0
fi

# ===== Step 7: LLVM IR → object file =====
_MOUNTS=(); _SEEN=""
add_mount "$TMPDIR"
add_mount "$OUTPUT_DIR"

log "Step 7: clang -O2 -c  $LLVM_IR → $OUTPUT_ABS"
docker run --rm \
  "${_MOUNTS[@]}" \
  --entrypoint "$CLANG_BIN" \
  "$DRCC_IMAGE" \
  -O2 -c "$LLVM_IR" -o "$OUTPUT_ABS" \
  || die "Step 7 (clang -c) failed"

echo "[onnx-compile] done → $OUTPUT_ABS" >&2