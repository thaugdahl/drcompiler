#!/usr/bin/env bash
#===----------------------------------------------------------------------===//
# run-onnx-bench.sh — Timestamped ONNX benchmark runner
#
# Creates benchmarks/results/YYYYMMDD-HHMMSS-onnx/ and writes:
#   run.log      — full stdout/stderr from onnx-run-bench.sh
#   results.csv  — per-model median/avg/stddev for all configs (us)
#   meta.txt     — date, host, configs, iters, warmup, cost-model, ref-cfg, git, kernel
#
# All onnx-run-bench.sh flags are forwarded verbatim (only --csv is injected).
#
# Usage:
#   ./scripts/run-onnx-bench.sh model.onnx [model2.onnx ...] [options]
#   ./scripts/run-onnx-bench.sh --model-dir DIR [options]
#   ./scripts/run-onnx-bench.sh --model-dir DIR --iters 50 --configs baseline,dr-partial
#===----------------------------------------------------------------------===//
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_ROOT="${PROJECT_DIR}/benchmarks/results"

STAMP="$(date '+%Y%m%d-%H%M%S')-onnx"
RUN_DIR="${RESULTS_ROOT}/${STAMP}"
mkdir -p "$RUN_DIR"

LOG_FILE="${RUN_DIR}/run.log"
CSV_FILE="${RUN_DIR}/results.csv"
META_FILE="${RUN_DIR}/meta.txt"

# --- Sniff relevant flags from forwarded args for metadata only ---
ITERS=100
WARMUP=10
CONFIGS="baseline,dr-recompute,dr-cost,dr-partial"
COST_MODEL=""
REF_CFG=""

_prev=""
for _a in "$@"; do
  case "$_prev" in
    --iters)       ITERS="$_a"       ;;
    --warmup)      WARMUP="$_a"      ;;
    --configs)     CONFIGS="$_a"     ;;
    --cost-model)  COST_MODEL="$_a"  ;;
    --ref-cfg)     REF_CFG="$_a"     ;;
  esac
  _prev="$_a"
done
unset _prev _a

# --- Write metadata ---
GIT_HASH="$(git -C "$PROJECT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
GIT_BRANCH="$(git -C "$PROJECT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
{
  printf "date:        %s\n" "$(date --iso-8601=seconds)"
  printf "host:        %s\n" "$(hostname)"
  printf "iters:       %s\n" "$ITERS"
  printf "warmup:      %s\n" "$WARMUP"
  printf "configs:     %s\n" "$CONFIGS"
  printf "cost-model:  %s\n" "${COST_MODEL:-<builtin>}"
  printf "ref-cfg:     %s\n" "${REF_CFG:-<auto>}"
  printf "git:         %s (%s)\n" "$GIT_HASH" "$GIT_BRANCH"
  printf "kernel:      %s\n" "$(uname -r)"
} > "$META_FILE"

echo "[run-onnx-bench] results dir: ${RUN_DIR}" >&2
cat "$META_FILE" >&2

# --- Run benchmark, tee to log (only --csv is injected; all user args forwarded) ---
"${SCRIPT_DIR}/onnx-run-bench.sh" \
  --csv "$CSV_FILE" \
  "$@" \
  2>&1 | tee "$LOG_FILE"

echo "[run-onnx-bench] done → ${RUN_DIR}" >&2
