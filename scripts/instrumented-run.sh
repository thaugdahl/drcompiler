#!/usr/bin/env bash
#===----------------------------------------------------------------------===//
# instrumented-run.sh — SPEC submit wrapper for runtime metrics
#
# Called by SPEC's submit mechanism (see config/drcc.cfg):
#   submit = /opt/drcc/scripts/instrumented-run.sh <results-dir> $command
#
# Shell redirections in $command (stdin/stdout/stderr) are processed by the
# invoking shell BEFORE this script runs, so "$@" contains only the
# executable + arguments.  The benchmark inherits the redirected FDs.
#
# Collects:
#   - Wall time + peak RSS via /usr/bin/time -v
#   - Cache references/misses via perf stat (if available)
#
# Output files:
#   <results-dir>/timing/<TAG>.<SEQ>.time   — /usr/bin/time output
#   <results-dir>/timing/<TAG>.<SEQ>.perf   — perf stat output (if available)
#   <results-dir>/timing/<TAG>.<SEQ>.args   — command-line arguments (for workload ID)
#
# SEQ is a zero-padded monotonic counter (000, 001, ...) so files sort in
# execution order.  With 3 sub-workloads × N iterations, you get 3*N files.
#===----------------------------------------------------------------------===//
set -uo pipefail

RESULTS_DIR="$1"; shift

TIMING_DIR="${RESULTS_DIR}/timing"
mkdir -p "$TIMING_DIR"

# Derive a human-readable tag from the executable name.
EXE_NAME="$(basename "$1")"
TAG="${EXE_NAME%%_base*}"          # e.g. "mcf_r" from "mcf_r_base.drcc"
TAG="${TAG:-${EXE_NAME}}"

# Monotonic sequence number.  Use a lock file to avoid races if SPEC ever
# runs copies in parallel (rate benchmarks with copies > 1).
COUNTER_FILE="${TIMING_DIR}/.${TAG}.counter"
(
    flock 9
    SEQ=$(cat "$COUNTER_FILE" 2>/dev/null || echo 0)
    printf '%d' $((SEQ + 1)) > "$COUNTER_FILE"
    printf '%03d' "$SEQ"
) 9>"${COUNTER_FILE}.lock" > "${TIMING_DIR}/.${TAG}.seq"
SEQ=$(cat "${TIMING_DIR}/.${TAG}.seq")

TIME_LOG="${TIMING_DIR}/${TAG}.${SEQ}.time"
PERF_LOG="${TIMING_DIR}/${TAG}.${SEQ}.perf"
ARGS_LOG="${TIMING_DIR}/${TAG}.${SEQ}.args"

# Save command-line arguments so we can identify which sub-workload this was.
printf '%s\n' "$@" > "$ARGS_LOG"

# Check if perf is available and functional (needs kernel support + perms).
use_perf=0
if command -v perf &>/dev/null; then
    if perf stat -e cache-references -- true 2>/dev/null; then
        use_perf=1
    fi
fi

if [[ "$use_perf" -eq 1 ]]; then
    perf stat \
        -e cache-references,cache-misses,instructions,cycles \
        -o "$PERF_LOG" \
        -- /usr/bin/time -v -o "$TIME_LOG" -- "$@"
else
    /usr/bin/time -v -o "$TIME_LOG" -- "$@"
fi
