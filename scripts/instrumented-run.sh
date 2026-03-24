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
#===----------------------------------------------------------------------===//
set -uo pipefail

RESULTS_DIR="$1"; shift
mkdir -p "$RESULTS_DIR"

# Derive a human-readable tag from the executable name.
EXE_NAME="$(basename "$1")"
TAG="${EXE_NAME%%_base*}"          # e.g. "mcf_r" from "mcf_r_base.drcc"
TAG="${TAG:-${EXE_NAME}}"

TIME_LOG="${RESULTS_DIR}/${TAG}.time"
PERF_LOG="${RESULTS_DIR}/${TAG}.perf"

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
