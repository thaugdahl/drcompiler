#!/usr/bin/env bash
#===----------------------------------------------------------------------===//
# onnx-bench.sh — Benchmark ONNX model: baseline vs DR-recomputed.
#
# Compiles two versions of the model and runs inference timing:
#   baseline  — onnx-mlir native pipeline  (no data-recomputation)
#   dr        — onnx-compile.sh pipeline   (data-recomputation{dr-recompute=true})
#
# Outputs median inference latency and, with --perf, hardware cache counters.
#
# Usage:
#   ./scripts/onnx-bench.sh <model.onnx> [options]
#
# Options:
#   --input-shape SHAPE   Comma-separated tensor shape, e.g. 1,1,28,28 (required
#                         unless --input-shape-f32 / --detect-shape is used)
#   --iters N             Number of inference iterations  (default: 500)
#   --warmup N            Warmup iterations before timing (default: 50)
#   --perf                Run with perf stat for cache counters (needs --privileged
#                         or cap SYS_ADMIN / PERFMON)
#   --out-dir DIR         Directory for compiled objects and executables
#                         (default: /tmp/onnx-bench.<model>)
#   --skip-baseline       Skip baseline compilation (reuse existing)
#   --skip-dr             Skip DR compilation (reuse existing)
#   -v / --verbose        Print each step to stderr
#
# Images:
#   $ONNX_MLIR_IMAGE  (default: onnx-mlir)
#   $DRCC_IMAGE       (default: drcc)
#===----------------------------------------------------------------------===//
set -euo pipefail

ONNX_MLIR_IMAGE="${ONNX_MLIR_IMAGE:-onnx-mlir}"
DRCC_IMAGE="${DRCC_IMAGE:-drcc}"

ONNX_MLIR_BIN="/build/onnx-mlir/build/Release/bin/onnx-mlir"
ONNX_MLIR_INCLUDE="/build/onnx-mlir/include"
CRUNTIME_LIB="/build/onnx-mlir/build/Release/lib/libcruntime.a"
CLANG_BIN="/opt/llvm/bin/clang"

MODEL=""
INPUT_SHAPE=""
ITERS=500
WARMUP=50
WITH_PERF=0
OUT_DIR=""
SKIP_BASELINE=0
SKIP_DR=0
VERBOSE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-shape)  INPUT_SHAPE="$2";   shift 2 ;;
    --iters)        ITERS="$2";         shift 2 ;;
    --warmup)       WARMUP="$2";        shift 2 ;;
    --perf)         WITH_PERF=1;        shift   ;;
    --out-dir)      OUT_DIR="$2";       shift 2 ;;
    --skip-baseline) SKIP_BASELINE=1;   shift   ;;
    --skip-dr)      SKIP_DR=1;          shift   ;;
    -v|--verbose)   VERBOSE=1;          shift   ;;
    -*)             echo "onnx-bench: unknown option: $1" >&2; exit 1 ;;
    *)              MODEL="$1";         shift   ;;
  esac
done

[[ -z "$MODEL" ]] && {
  echo "Usage: onnx-bench.sh <model.onnx> --input-shape 1,1,28,28 [--iters N] [--perf]" >&2
  exit 1
}
[[ -f "$MODEL" ]] || { echo "onnx-bench: not found: $MODEL" >&2; exit 1; }
[[ -z "$INPUT_SHAPE" ]] && {
  echo "onnx-bench: --input-shape required (e.g. --input-shape 1,1,28,28)" >&2
  exit 1
}

log() { [[ "$VERBOSE" -eq 1 ]] && echo "[onnx-bench] $*" >&2 || true; }
die() { echo "onnx-bench: error: $*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_ABS="$(realpath "$MODEL")"
MODEL_DIR="$(dirname "$MODEL_ABS")"
BASE="$(basename "$MODEL_ABS" .onnx)"

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="/tmp/onnx-bench.${BASE}"
fi
OUT_DIR="$(realpath -m "$OUT_DIR")"
mkdir -p "$OUT_DIR"

BASELINE_OBJ="${OUT_DIR}/${BASE}_baseline.o"
DR_OBJ="${OUT_DIR}/${BASE}_dr.o"
HARNESS_SRC="${OUT_DIR}/bench_harness.c"
BASELINE_BIN="${OUT_DIR}/bench_baseline"
DR_BIN="${OUT_DIR}/bench_dr"

# --- Convert INPUT_SHAPE "1,1,28,28" → C array initializer "{1,1,28,28}" ---
SHAPE_C_INIT="{$(echo "$INPUT_SHAPE" | tr ',' ',')}"
SHAPE_RANK=$(echo "$INPUT_SHAPE" | tr -cd ',' | wc -c)
SHAPE_RANK=$((SHAPE_RANK + 1))
TOTAL_ELEMS=$(echo "$INPUT_SHAPE" | tr ',' '\n' | awk 'BEGIN{p=1} {p*=$1} END{print p}')

# ===== Step 1: Compile baseline =====
if [[ "$SKIP_BASELINE" -eq 0 ]]; then
  echo "[onnx-bench] Compiling baseline (onnx-mlir --EmitObj)..." >&2
  docker run --rm \
    -v "$MODEL_DIR:$MODEL_DIR" \
    -v "$OUT_DIR:$OUT_DIR" \
    --entrypoint "$ONNX_MLIR_BIN" \
    "$ONNX_MLIR_IMAGE" \
    --EmitObj -o "${BASELINE_OBJ%.o}" "$MODEL_ABS" \
    || die "baseline compilation failed"
  log "baseline object: $BASELINE_OBJ"
else
  [[ -f "$BASELINE_OBJ" ]] || die "--skip-baseline set but $BASELINE_OBJ not found"
  log "reusing baseline object: $BASELINE_OBJ"
fi

# ===== Step 2: Compile DR version =====
if [[ "$SKIP_DR" -eq 0 ]]; then
  echo "[onnx-bench] Compiling DR version (onnx-compile.sh)..." >&2
  bash "$SCRIPT_DIR/onnx-compile.sh" "$MODEL_ABS" -o "$DR_OBJ" \
    ${VERBOSE:+"-v"} \
    || die "DR compilation failed"
  log "DR object: $DR_OBJ"
else
  [[ -f "$DR_OBJ" ]] || die "--skip-dr set but $DR_OBJ not found"
  log "reusing DR object: $DR_OBJ"
fi

# ===== Step 3: Generate harness =====
log "Generating harness: $HARNESS_SRC"
cat > "$HARNESS_SRC" << HARNESS_EOF
/* Auto-generated benchmark harness — do not edit.
 * Model: ${BASE}  Input shape: ${INPUT_SHAPE}
 * Iterations: ${ITERS}  Warmup: ${WARMUP}
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "OnnxMlirRuntime.h"

extern OMTensorList *run_main_graph(OMTensorList *);

#define ITERS   ${ITERS}
#define WARMUP  ${WARMUP}
#define RANK    ${SHAPE_RANK}
#define N_ELEMS ${TOTAL_ELEMS}

static int64_t shape[RANK] = ${SHAPE_C_INIT};

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

int main(void) {
    float *input_data = (float *)calloc(N_ELEMS, sizeof(float));
    if (!input_data) { fputs("OOM\n", stderr); return 1; }

    OMTensor *in_t = omTensorCreate(input_data, shape, RANK, ONNX_TYPE_FLOAT);
    OMTensorList *inputs = omTensorListCreate(&in_t, 1);

    /* Warmup — discard output (model may return static pointer; do not free) */
    for (int i = 0; i < WARMUP; i++)
        run_main_graph(inputs);

    double samples[ITERS];
    for (int i = 0; i < ITERS; i++) {
        double t0 = now_ns();
        run_main_graph(inputs);
        double t1 = now_ns();
        samples[i] = t1 - t0;
    }

    /* Compute stats */
    double sum = 0, min_v = samples[0], max_v = samples[0];
    for (int i = 0; i < ITERS; i++) {
        sum += samples[i];
        if (samples[i] < min_v) min_v = samples[i];
        if (samples[i] > max_v) max_v = samples[i];
    }
    double mean_us = sum / ITERS / 1000.0;
    double min_us  = min_v / 1000.0;
    double max_us  = max_v / 1000.0;

    /* Simple median via partial sort (nth_element substitute) */
    double *s2 = (double *)malloc(ITERS * sizeof(double));
    memcpy(s2, samples, ITERS * sizeof(double));
    /* Insertion sort for small N (<=500 is fine) */
    for (int i = 1; i < ITERS; i++) {
        double key = s2[i]; int j = i - 1;
        while (j >= 0 && s2[j] > key) { s2[j+1] = s2[j]; j--; }
        s2[j+1] = key;
    }
    double median_us = s2[ITERS/2] / 1000.0;
    free(s2);

    printf("median=%.2f us  mean=%.2f us  min=%.2f us  max=%.2f us  iters=%d\n",
           median_us, mean_us, min_us, max_us, ITERS);
    return 0;  /* short-lived process; OS reclaims */
}
HARNESS_EOF

# ===== Step 4: Compile harness against each object =====
compile_harness() {
  local obj="$1" out="$2" label="$3"
  log "Compiling $label harness: $out"
  docker run --rm \
    -v "$OUT_DIR:$OUT_DIR" \
    --entrypoint "$CLANG_BIN" \
    "$ONNX_MLIR_IMAGE" \
    -O2 \
    -I"$ONNX_MLIR_INCLUDE" \
    "$HARNESS_SRC" "$obj" "$CRUNTIME_LIB" \
    -lm -lpthread \
    -o "$out" \
    || die "harness link failed for $label"
}

compile_harness "$BASELINE_OBJ" "$BASELINE_BIN" "baseline"
compile_harness "$DR_OBJ"       "$DR_BIN"       "dr"

# ===== Step 5: Run and report =====
run_bench() {
  local bin="$1" label="$2"
  if [[ "$WITH_PERF" -eq 1 ]]; then
    echo "--- $label ---"
    docker run --rm \
      --privileged \
      -v "$OUT_DIR:$OUT_DIR" \
      --entrypoint perf \
      "$ONNX_MLIR_IMAGE" \
      stat -e cache-misses,cache-references,cycles,instructions \
      "$bin" 2>&1
  else
    local result
    result=$(docker run --rm \
      -v "$OUT_DIR:$OUT_DIR" \
      --entrypoint "$bin" \
      "$ONNX_MLIR_IMAGE" \
      2>/dev/null)
    echo "$label: $result"
  fi
}

echo "" >&2
echo "===== Results: ${BASE} (input=${INPUT_SHAPE}) =====" >&2
BASELINE_RESULT=$(docker run --rm \
  -v "$OUT_DIR:$OUT_DIR" \
  --entrypoint "$BASELINE_BIN" \
  "$ONNX_MLIR_IMAGE" 2>/dev/null)
DR_RESULT=$(docker run --rm \
  -v "$OUT_DIR:$OUT_DIR" \
  --entrypoint "$DR_BIN" \
  "$ONNX_MLIR_IMAGE" 2>/dev/null)

echo "baseline : $BASELINE_RESULT"
echo "dr       : $DR_RESULT"

# Compute speedup from median fields
BL_MED=$(echo "$BASELINE_RESULT" | grep -oP 'median=\K[0-9.]+')
DR_MED=$(echo "$DR_RESULT"       | grep -oP 'median=\K[0-9.]+')
if [[ -n "$BL_MED" && -n "$DR_MED" ]]; then
  SPEEDUP=$(awk "BEGIN{printf \"%.3f\", $BL_MED / $DR_MED}")
  echo "speedup  : ${SPEEDUP}x  (baseline/dr median latency)"
fi

if [[ "$WITH_PERF" -eq 1 ]]; then
  echo "" >&2
  echo "=== perf stat: baseline ===" >&2
  run_bench "$BASELINE_BIN" "baseline"
  echo "" >&2
  echo "=== perf stat: dr ===" >&2
  run_bench "$DR_BIN" "dr"
fi

echo "[onnx-bench] artifacts in $OUT_DIR" >&2
