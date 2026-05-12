#!/usr/bin/env bash
#===----------------------------------------------------------------------===//
# onnx-fair-bench.sh — Fair-baseline runtime benchmark for an ONNX model.
#
# Compiles the same pipeline twice, toggling only the data-recomputation
# transform, links a timing harness against each object, runs both, and
# prints a results table.
#
#   baseline :  onnx-compile.sh --no-recompute
#   dr       :  onnx-compile.sh <DR_FLAGS>            (default: --cost-model
#                                                     --partial-remat)
#
# Usage:
#   ./scripts/onnx-fair-bench.sh <model.onnx> --shape D0,D1,... [options]
#
# Shape:
#   --shape SHAPE         Explicit input shape, comma-separated (e.g. 1,3,224,224).
#                         Overrides auto-detection.
#   --batch-size N        When auto-detecting, substitute N for any dynamic
#                         dim (default: 1). Ignored if --shape is set.
#
# Options:
#   --iters N             Timed iterations           (default: 30)
#   --warmup N            Warmup iterations          (default: 5)
#   --dr-flags "FLAGS"    Override DR flags          (default: "--cost-model
#                                                     --partial-remat")
#   --work-dir DIR        Use DIR instead of mktemp; not removed on exit
#   --keep                Don't delete the temp dir on exit (print path)
#   -v / --verbose        Forward -v to onnx-compile.sh
#
# Environment:
#   ONNX_MLIR_IMAGE  (default: onnx-mlir)
#   DRCC_IMAGE       (default: drcc)
#===----------------------------------------------------------------------===//
set -euo pipefail

ONNX_MLIR_IMAGE="${ONNX_MLIR_IMAGE:-onnx-mlir}"

ONNX_MLIR_INCLUDE="/build/onnx-mlir/include"
CRUNTIME_LIB="/build/onnx-mlir/build/Release/lib/libcruntime.a"
CLANG_BIN="/opt/llvm/bin/clang"

MODEL=""
SHAPE=""
BATCH_SIZE=1
ITERS=30
WARMUP=5
DR_FLAGS="--cost-model --partial-remat"
WORK_DIR=""
KEEP=0
VERBOSE=0

usage() { sed -n '2,32p' "$0" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --shape)      SHAPE="$2";      shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --iters)      ITERS="$2";      shift 2 ;;
    --warmup)     WARMUP="$2";     shift 2 ;;
    --dr-flags)   DR_FLAGS="$2";   shift 2 ;;
    --work-dir)   WORK_DIR="$2";   shift 2 ;;
    --keep)       KEEP=1;          shift   ;;
    -v|--verbose) VERBOSE=1;       shift   ;;
    -h|--help)    usage ;;
    -*)           echo "unknown option: $1" >&2; usage ;;
    *)            MODEL="$1";      shift   ;;
  esac
done

[[ -n "$MODEL" ]] || usage
[[ -f "$MODEL" ]] || { echo "model not found: $MODEL" >&2; exit 1; }

MODEL_ABS="$(realpath "$MODEL")"
MODEL_DIR="$(dirname "$MODEL_ABS")"
BASE="$(basename "$MODEL_ABS" .onnx)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "$WORK_DIR" ]]; then
  WORK_DIR="$(mktemp -d -t "onnx-fair-bench.${BASE}.XXXXXX")"
  if [[ "$KEEP" -eq 0 ]]; then
    trap 'rm -rf "$WORK_DIR"' EXIT
  fi
else
  mkdir -p "$WORK_DIR"
fi
WORK_DIR="$(realpath "$WORK_DIR")"
echo "[bench] work dir: $WORK_DIR"

META_FILE="$WORK_DIR/inputs.meta"
CPU_COST_FILE="$WORK_DIR/cpu-cost-model.json"

# ---- Build per-input metadata: NAME|ENUM|CTYPE|D0,D1,... -----------------
if [[ -n "$SHAPE" ]]; then
  # Manual override: single float32 input named "input".
  echo "input|ONNX_TYPE_FLOAT|float|$SHAPE" > "$META_FILE"
  echo "[bench] using override shape: $SHAPE (float32, single input)"
else
  echo "[bench] auto-detecting inputs (batch=${BATCH_SIZE})..."
  DETECT_PY="$WORK_DIR/detect_inputs.py"
  cat > "$DETECT_PY" <<'PY'
import onnx, sys
model_path, batch, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
TYPE_MAP = {
    1:  ("float",    "ONNX_TYPE_FLOAT"),
    2:  ("uint8_t",  "ONNX_TYPE_UINT8"),
    3:  ("int8_t",   "ONNX_TYPE_INT8"),
    4:  ("uint16_t", "ONNX_TYPE_UINT16"),
    5:  ("int16_t",  "ONNX_TYPE_INT16"),
    6:  ("int32_t",  "ONNX_TYPE_INT32"),
    7:  ("int64_t",  "ONNX_TYPE_INT64"),
    9:  ("bool",     "ONNX_TYPE_BOOL"),
    11: ("double",   "ONNX_TYPE_DOUBLE"),
}
m = onnx.load(model_path)
inits = {i.name for i in m.graph.initializer}
inputs = [i for i in m.graph.input if i.name not in inits]
if not inputs:
    sys.stderr.write("no graph inputs\n"); sys.exit(2)
lines = []
for inp in inputs:
    et = inp.type.tensor_type.elem_type
    if et not in TYPE_MAP:
        sys.stderr.write(f"unsupported dtype {et} for input {inp.name}\n")
        sys.exit(3)
    ctype, enum = TYPE_MAP[et]
    dims = []
    for d in inp.type.tensor_type.shape.dim:
        dims.append(str(d.dim_value) if d.dim_value > 0 else str(batch))
    lines.append(f"{inp.name}|{enum}|{ctype}|{','.join(dims)}")
with open(out_path, "w") as f:
    f.write("\n".join(lines) + "\n")
PY
  docker run --rm \
    -v "$MODEL_DIR:$MODEL_DIR" -v "$WORK_DIR:$WORK_DIR" \
    --entrypoint python3 "$ONNX_MLIR_IMAGE" \
    "$DETECT_PY" "$MODEL_ABS" "$BATCH_SIZE" "$META_FILE" \
    || { echo "input detection failed" >&2; exit 1; }
  echo "[bench] detected inputs:"
  awk -F'|' '{printf "  %-24s dtype=%s shape=[%s]\n", $1, $3, $4}' "$META_FILE"
fi

mapfile -t INPUT_LINES < "$META_FILE"
NUM_INPUTS=${#INPUT_LINES[@]}

BASELINE_OBJ="$WORK_DIR/${BASE}_baseline.o"
DR_OBJ="$WORK_DIR/${BASE}_dr.o"
HARNESS_SRC="$WORK_DIR/bench_harness.c"
BASELINE_BIN="$WORK_DIR/bench_baseline"
DR_BIN="$WORK_DIR/bench_dr"
BASELINE_MLIR_DIR="$WORK_DIR/baseline-mlir"
DR_MLIR_DIR="$WORK_DIR/dr-mlir"

VFLAG=()
[[ "$VERBOSE" -eq 1 ]] && VFLAG=(-v)

echo "[bench] probing CPU cost model..."
bash "$SCRIPT_DIR/probe-cost-model-docker.sh" "$CPU_COST_FILE"

# ---- Compile both configurations -----------------------------------------
echo "[bench] compiling baseline (--no-recompute)..."
bash "$SCRIPT_DIR/onnx-compile.sh" "$MODEL_ABS" --no-recompute \
  --keep-mlir "$BASELINE_MLIR_DIR" -o "$BASELINE_OBJ" \
  "${VFLAG[@]}"

echo "[bench] compiling dr ($DR_FLAGS)..."
# shellcheck disable=SC2086
bash "$SCRIPT_DIR/onnx-compile.sh" "$MODEL_ABS" $DR_FLAGS --cpu-cost-model "$CPU_COST_FILE" \
  --keep-mlir "$DR_MLIR_DIR" -o "$DR_OBJ" \
  "${VFLAG[@]}"

BASELINE_DR_MLIR="$BASELINE_MLIR_DIR/${BASE}.03-dr.mlir"
DR_DR_MLIR="$DR_MLIR_DIR/${BASE}.03-dr.mlir"
if [[ -f "$BASELINE_DR_MLIR" && -f "$DR_DR_MLIR" ]] && \
   cmp -s "$BASELINE_DR_MLIR" "$DR_DR_MLIR"; then
  echo "[bench] baseline and full-DR emitted MLIR are identical; skipping runtime benchmark."
  [[ "$KEEP" -eq 1 ]] && echo "[bench] artifacts kept in $WORK_DIR"
  exit 0
fi

# ---- Generate timing harness ---------------------------------------------
{
cat <<HARNESS_HDR
/* onnx-fair-bench harness — model: ${BASE}, inputs: ${NUM_INPUTS},
 * iters: ${ITERS}, warmup: ${WARMUP}.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include "OnnxMlirRuntime.h"

extern OMTensorList *run_main_graph(OMTensorList *);

#define ITERS       ${ITERS}
#define WARMUP      ${WARMUP}
#define NUM_INPUTS  ${NUM_INPUTS}

HARNESS_HDR

# Per-input static decls.
idx=0
for line in "${INPUT_LINES[@]}"; do
  IFS='|' read -r NAME ENUM CTYPE DIMS <<< "$line"
  RANK=$(($(echo "$DIMS" | tr -cd ',' | wc -c) + 1))
  ELEMS=$(echo "$DIMS" | tr ',' '\n' | awk 'BEGIN{p=1}{p*=$1}END{print p}')
  cat <<DECL
/* input $idx: $NAME ($CTYPE [$DIMS]) */
#define RANK_${idx}   ${RANK}
#define ELEMS_${idx}  ${ELEMS}
static int64_t shape_${idx}[RANK_${idx}] = {${DIMS}};
static ${CTYPE} *buf_${idx};

DECL
  idx=$((idx+1))
done

cat <<'HARNESS_MID'
static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(void) {
    OMTensor *tensors[NUM_INPUTS];
HARNESS_MID

# Per-input alloc + tensor build.
idx=0
for line in "${INPUT_LINES[@]}"; do
  IFS='|' read -r NAME ENUM CTYPE DIMS <<< "$line"
  cat <<INIT
    buf_${idx} = (${CTYPE} *)calloc(ELEMS_${idx}, sizeof(${CTYPE}));
    if (!buf_${idx}) { fputs("OOM\n", stderr); return 1; }
    tensors[${idx}] = omTensorCreate(buf_${idx}, shape_${idx}, RANK_${idx}, ${ENUM});
INIT
  idx=$((idx+1))
done

cat <<'HARNESS_TAIL'
    OMTensorList *inputs = omTensorListCreate(tensors, NUM_INPUTS);

HARNESS_TAIL
} > "$HARNESS_SRC"

cat >> "$HARNESS_SRC" <<HARNESS_EOF
    for (int i = 0; i < WARMUP; i++)
        run_main_graph(inputs);

    double *samples = (double *)malloc(ITERS * sizeof(double));
    for (int i = 0; i < ITERS; i++) {
        double t0 = now_ns();
        run_main_graph(inputs);
        double t1 = now_ns();
        samples[i] = t1 - t0;
    }

    double sum = 0, mn = samples[0], mx = samples[0];
    for (int i = 0; i < ITERS; i++) {
        sum += samples[i];
        if (samples[i] < mn) mn = samples[i];
        if (samples[i] > mx) mx = samples[i];
    }
    double mean_ms = (sum / ITERS) / 1.0e6;
    double min_ms  = mn / 1.0e6;
    double max_ms  = mx / 1.0e6;

    qsort(samples, ITERS, sizeof(double), cmp_double);
    double median_ms = samples[ITERS / 2] / 1.0e6;

    /* tab-separated for easy parsing */
    printf("median_ms=%.4f\tmean_ms=%.4f\tmin_ms=%.4f\tmax_ms=%.4f\titers=%d\n",
           median_ms, mean_ms, min_ms, max_ms, ITERS);
    return 0;
}
HARNESS_EOF

# ---- Link harness against each object via the onnx-mlir image ------------
link_harness() {
  local obj="$1" out="$2" label="$3"
  echo "[bench] linking $label..."
  docker run --rm \
    -v "$WORK_DIR:$WORK_DIR" \
    --entrypoint "$CLANG_BIN" \
    "$ONNX_MLIR_IMAGE" \
    -O2 -I"$ONNX_MLIR_INCLUDE" \
    "$HARNESS_SRC" "$obj" "$CRUNTIME_LIB" \
    -lm -lpthread -o "$out"
}
link_harness "$BASELINE_OBJ" "$BASELINE_BIN" "baseline"
link_harness "$DR_OBJ"       "$DR_BIN"       "dr"

# ---- Run both and capture metrics ----------------------------------------
run_bin() {
  docker run --rm \
    -v "$WORK_DIR:$WORK_DIR" \
    --entrypoint "$1" \
    "$ONNX_MLIR_IMAGE" 2>/dev/null
}
echo "[bench] running baseline..."
BL_OUT=$(run_bin "$BASELINE_BIN")
echo "[bench] running dr..."
DR_OUT=$(run_bin "$DR_BIN")

extract() { grep -oP "$2=\K[0-9.]+" <<<"$1"; }

BL_MED=$(extract "$BL_OUT" median_ms)
BL_MEAN=$(extract "$BL_OUT" mean_ms)
BL_MIN=$(extract "$BL_OUT" min_ms)
BL_MAX=$(extract "$BL_OUT" max_ms)
DR_MED=$(extract "$DR_OUT" median_ms)
DR_MEAN=$(extract "$DR_OUT" mean_ms)
DR_MIN=$(extract "$DR_OUT" min_ms)
DR_MAX=$(extract "$DR_OUT" max_ms)

DELTA_PCT=$(awk "BEGIN{printf \"%+.2f\", ($DR_MED - $BL_MED) * 100.0 / $BL_MED}")
SPEEDUP=$(awk   "BEGIN{printf \"%.3f\",  $BL_MED / $DR_MED}")

SHAPE_SUMMARY=$(awk -F'|' '{printf "%s%s[%s]", (NR>1?",":""), $1, $4}' "$META_FILE")

echo
printf '===== %s   inputs=%s   iters=%d   warmup=%d =====\n' \
       "$BASE" "$SHAPE_SUMMARY" "$ITERS" "$WARMUP"
printf '%-10s  %10s  %10s  %10s  %10s\n' \
       config median_ms mean_ms min_ms max_ms
printf '%-10s  %10s  %10s  %10s  %10s\n' \
       baseline "$BL_MED" "$BL_MEAN" "$BL_MIN" "$BL_MAX"
printf '%-10s  %10s  %10s  %10s  %10s\n' \
       dr       "$DR_MED" "$DR_MEAN" "$DR_MIN" "$DR_MAX"
printf '%-10s  %+10s%%  speedup=%sx\n' \
       'delta'  "$DELTA_PCT" "$SPEEDUP"

[[ "$KEEP" -eq 1 ]] && echo "[bench] artifacts kept in $WORK_DIR"
