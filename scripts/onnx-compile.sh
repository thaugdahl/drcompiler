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
#   -o FILE                 Output object path (default: <model>.o next to input)
#   --keep-mlir DIR         Save all intermediate .mlir files to DIR
#   --dr-flags FLAGS        Override entire dr-opt pass pipeline (escape hatch)
#
#   DR pass options (translated to data-recomputation{...}):
#   --no-recompute          Disable dr-recompute (default: on)
#   --cost-model            Enable dr-cost-model (cache-aware gating)
#   --partial-remat         Enable dr-partial-remat (implies --cost-model)
#   --full                  Preset: enable all input-less DR bool flags
#                           (recompute, cost-model, partial-remat,
#                            footprint-analysis, diagnostics)
#   --partial-max-leaves N  Cap on partial remat leaves (default 4)
#   --footprint-analysis    Enable dr-footprint-analysis
#   --diagnostics           Add dr-test-diagnostics
#   --dot FILE              Emit provenance GraphViz dot file
#   --cpu-cost-model FILE   JSON per-op cycle costs (mounted into container)
#   --l1-size N, --l2-size N, --l3-size N            Cache sizes (bytes)
#   --l1-lat N, --l2-lat N, --l3-lat N, --mem-lat N  Latencies (cycles)
#   --cache-line N          Cache line size in bytes (default 64)
#
#   --emit-llvm             Stop after LLVM IR (.ll) instead of .o
#   -v / --verbose          Print each step to stderr
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
DR_FLAGS_OVERRIDE=""
DIAGNOSTICS=0
DOT_FILE=""
EMIT_LLVM=0
VERBOSE=0

# DR pass options (assembled into data-recomputation{...} unless --dr-flags)
DR_RECOMPUTE=1
DR_COST_MODEL=0
DR_PARTIAL_REMAT=0
DR_PARTIAL_MAX_LEAVES=""
DR_FOOTPRINT=0
DR_CPU_COST_FILE=""
DR_L1_SIZE=""; DR_L2_SIZE=""; DR_L3_SIZE=""
DR_L1_LAT="";  DR_L2_LAT="";  DR_L3_LAT="";  DR_MEM_LAT=""
DR_CACHE_LINE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o)                    OUTPUT="$2";              shift 2 ;;
    --keep-mlir)           KEEP_MLIR_DIR="$2";       shift 2 ;;
    --dr-flags)            DR_FLAGS_OVERRIDE="$2";   shift 2 ;;
    --no-recompute)        DR_RECOMPUTE=0;           shift   ;;
    --cost-model)          DR_COST_MODEL=1;          shift   ;;
    --partial-remat)       DR_PARTIAL_REMAT=1; DR_COST_MODEL=1; shift ;;
    --full)                DR_RECOMPUTE=1; DR_COST_MODEL=1; DR_PARTIAL_REMAT=1
                           DR_FOOTPRINT=1; DIAGNOSTICS=1;       shift ;;
    --partial-max-leaves)  DR_PARTIAL_MAX_LEAVES="$2"; shift 2 ;;
    --footprint-analysis)  DR_FOOTPRINT=1;           shift   ;;
    --cpu-cost-model)      DR_CPU_COST_FILE="$2";    shift 2 ;;
    --l1-size)             DR_L1_SIZE="$2";          shift 2 ;;
    --l2-size)             DR_L2_SIZE="$2";          shift 2 ;;
    --l3-size)             DR_L3_SIZE="$2";          shift 2 ;;
    --l1-lat)              DR_L1_LAT="$2";           shift 2 ;;
    --l2-lat)              DR_L2_LAT="$2";           shift 2 ;;
    --l3-lat)              DR_L3_LAT="$2";           shift 2 ;;
    --mem-lat)             DR_MEM_LAT="$2";          shift 2 ;;
    --cache-line)          DR_CACHE_LINE="$2";       shift 2 ;;
    --diagnostics)         DIAGNOSTICS=1;            shift   ;;
    --dot)                 DOT_FILE="$2";            shift 2 ;;
    --emit-llvm)           EMIT_LLVM=1;              shift   ;;
    -v|--verbose)          VERBOSE=1;                shift   ;;
    -*)                    echo "onnx-compile: unknown option: $1" >&2; exit 1 ;;
    *)                     MODEL="$1";               shift   ;;
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

# --- Assemble DR pass options ---
# Resolve dot/cost-model file paths early (need absolute for in-container access)
DOT_ABS=""
[[ -n "$DOT_FILE" ]] && DOT_ABS="$(realpath -m "$DOT_FILE")"
CPU_COST_ABS=""
if [[ -n "$DR_CPU_COST_FILE" ]]; then
  [[ -f "$DR_CPU_COST_FILE" ]] || die "cpu cost model file not found: $DR_CPU_COST_FILE"
  CPU_COST_ABS="$(realpath "$DR_CPU_COST_FILE")"
fi

if [[ -n "$DR_FLAGS_OVERRIDE" ]]; then
  DR_FLAGS="$DR_FLAGS_OVERRIDE"
else
  DR_OPTS=()
  [[ "$DR_RECOMPUTE"     -eq 1 ]] && DR_OPTS+=("dr-recompute")
  [[ "$DR_COST_MODEL"    -eq 1 ]] && DR_OPTS+=("dr-cost-model")
  [[ "$DR_PARTIAL_REMAT" -eq 1 ]] && DR_OPTS+=("dr-partial-remat")
  [[ "$DR_FOOTPRINT"     -eq 1 ]] && DR_OPTS+=("dr-footprint-analysis")
  [[ "$DIAGNOSTICS"      -eq 1 ]] && DR_OPTS+=("dr-test-diagnostics")
  [[ -n "$DOT_ABS"             ]] && DR_OPTS+=("dr-dot-file=$DOT_ABS")
  [[ -n "$CPU_COST_ABS"        ]] && DR_OPTS+=("cpu-cost-model-file=$CPU_COST_ABS")
  [[ -n "$DR_PARTIAL_MAX_LEAVES" ]] && DR_OPTS+=("dr-partial-max-leaves=$DR_PARTIAL_MAX_LEAVES")
  [[ -n "$DR_L1_SIZE"          ]] && DR_OPTS+=("dr-l1-size=$DR_L1_SIZE")
  [[ -n "$DR_L2_SIZE"          ]] && DR_OPTS+=("dr-l2-size=$DR_L2_SIZE")
  [[ -n "$DR_L3_SIZE"          ]] && DR_OPTS+=("dr-l3-size=$DR_L3_SIZE")
  [[ -n "$DR_L1_LAT"           ]] && DR_OPTS+=("dr-l1-latency=$DR_L1_LAT")
  [[ -n "$DR_L2_LAT"           ]] && DR_OPTS+=("dr-l2-latency=$DR_L2_LAT")
  [[ -n "$DR_L3_LAT"           ]] && DR_OPTS+=("dr-l3-latency=$DR_L3_LAT")
  [[ -n "$DR_MEM_LAT"          ]] && DR_OPTS+=("dr-mem-latency=$DR_MEM_LAT")
  [[ -n "$DR_CACHE_LINE"       ]] && DR_OPTS+=("dr-cache-line-size=$DR_CACHE_LINE")
  DR_FLAGS="--pass-pipeline=builtin.module(data-recomputation{${DR_OPTS[*]}})"
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
[[ -n "$DOT_ABS"      ]] && add_mount "$(dirname "$DOT_ABS")"
[[ -n "$CPU_COST_ABS" ]] && add_mount "$(dirname "$CPU_COST_ABS")"

log "Step 3: dr-opt $DR_FLAGS  $AFFINE_MLIR → $DR_MLIR"
docker run --rm \
  "${_MOUNTS[@]}" \
  --entrypoint "$DR_OPT_BIN" \
  "$DRCC_IMAGE" \
  --allow-unregistered-dialect "$DR_FLAGS" "$AFFINE_MLIR" -o "$DR_MLIR" \
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
