#!/usr/bin/env bash
# crossthread-roofline-bench.sh — compile + run the CROSSCUTTING.md III roofline
# validation bench (scripts/crossthread-roofline-bench.c).
#
# Measures, per compute intensity (fops) and thread count, whether MATERIALIZE
# (keep a >LLC buffer, reload per consumer) or RECOMPUTE (per-thread ALU, no
# buffer) is faster.  The cost-model claim is that the verdict FLIPS with thread
# count once bandwidth saturates; this measures the flip on the host.
#
# Env: CC (default: marco LLVM clang), NELEM (default 1048576 = 8 MiB/array; must
# exceed activeThreads-share-of-LLC to expose the reversal -- see the .c header).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CC="${CC:-/home/tor/Dev/marco/install/llvm-project/bin/clang}"
NELEM="${NELEM:-1048576}"
BIN="$(mktemp /tmp/ctbench.XXXX)"
echo "compiling ($CC, NELEM=$NELEM)..." >&2
"$CC" -O2 -march=native -fopenmp -DNELEM="$NELEM" \
  "$SCRIPT_DIR/crossthread-roofline-bench.c" -lm -o "$BIN"
"$BIN"
rm -f "$BIN"
