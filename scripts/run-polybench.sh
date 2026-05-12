#!/usr/bin/env bash
#===----------------------------------------------------------------------===//
# run-polybench.sh — Timestamped PolyBench benchmark runner
#
# Creates benchmarks/results/YYYYMMDD-HHMMSS/ and writes:
#   run.log    — full stdout/stderr from polybench-bench.sh
#   results.csv— per-kernel median/avg/stddev for all configs
#   meta.txt   — dataset, iters, configs, git hash, hostname, date
#
# All polybench-bench.sh flags are forwarded verbatim (only --csv is injected).
#
# Usage:
#   ./scripts/run-polybench.sh [polybench-bench options]
#   ./scripts/run-polybench.sh --dataset LARGE --iters 5
#   ./scripts/run-polybench.sh --dataset STANDARD --iters 15 --configs clang,dr-partial
#===----------------------------------------------------------------------===//
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_ROOT="${PROJECT_DIR}/benchmarks/results"

STAMP="$(date '+%Y%m%d-%H%M%S')"
RUN_DIR="${RESULTS_ROOT}/${STAMP}"
mkdir -p "$RUN_DIR"

LOG_FILE="${RUN_DIR}/run.log"
CSV_FILE="${RUN_DIR}/results.csv"
META_FILE="${RUN_DIR}/meta.txt"

# --- Sniff dataset/iters/configs/cost-model from forwarded args for metadata only ---
DATASET="LARGE"
ITERS=5
CONFIGS="clang,cgeist-base,dr-greedy,dr-cost,dr-partial"
COST_MODEL=""
REF_CFG=""

_prev=""
for _a in "$@"; do
  case "$_prev" in
    --dataset)     DATASET="$_a"     ;;
    --iters)       ITERS="$_a"       ;;
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
  printf "date:       %s\n" "$(date --iso-8601=seconds)"
  printf "host:       %s\n" "$(hostname)"
  printf "dataset:    %s\n" "$DATASET"
  printf "iters:      %s\n" "$ITERS"
  printf "configs:    %s\n" "$CONFIGS"
  printf "cost-model: %s\n" "${COST_MODEL:-<builtin>}"
  printf "ref-cfg:    %s\n" "${REF_CFG:-<auto>}"
  printf "git:        %s (%s)\n" "$GIT_HASH" "$GIT_BRANCH"
  printf "kernel:     %s\n" "$(uname -r)"
} > "$META_FILE"

echo "[run-polybench] results dir: ${RUN_DIR}" >&2
cat "$META_FILE" >&2

# --- Run benchmark, tee to log (only --csv is injected; all user args forwarded) ---
"${SCRIPT_DIR}/polybench-bench.sh" \
  --csv "$CSV_FILE" \
  "$@" \
  2>&1 | tee "$LOG_FILE"

echo "[run-polybench] done → ${RUN_DIR}" >&2
