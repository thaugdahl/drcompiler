#!/usr/bin/env bash
#
# Smoke test for the data-recomputation CLI flags exposed in marco.
#
# Compiles one .mo model three ways:
#   1. no DR              (default)
#   2. -fdata-recomputation
#   3. -fdata-recomputation -dr-cost-model
#
# Verifies each binary runs and prints the simulation CSV.  Also greps
# stderr for DRCOMP: / DRSUM: lines to confirm the pass actually ran.
#
# Usage:  bash scripts/marco-dr-smoke.sh [path-to.mo] [model-name]
# Default: marco's test/Simulation/euler-forward/cycle-with-derivative.mo

set -euo pipefail

MARCO_BIN="${MARCO_BIN:-/home/tor/Dev/marco/install/marco/bin/marco}"
RT_LIB="${RT_LIB:-/home/tor/Dev/marco/install/marco-runtime/lib}"

MO_FILE="${1:-/home/tor/Dev/marco/source/marco/test/Simulation/euler-forward/cycle-with-derivative.mo}"
MODEL="${2:-CycleWithDerivative}"

WORK="$(mktemp -d -t marco-dr-smoke.XXXXXX)"
trap "rm -rf '$WORK'" EXIT

SRC="$WORK/in.mo"
cp "$MO_FILE" "$SRC"

COMMON=(
  --omc-bypass
  --model="$MODEL"
  --solver=euler-forward
  -L "$RT_LIB"
  -Wl,-rpath,"$RT_LIB"
  "$SRC"
)

run_variant() {
  local label="$1"; shift
  local out="$WORK/$label.bin"
  local log="$WORK/$label.log"
  echo "================================================================"
  echo "==  $label"
  echo "================================================================"
  echo "+ marco $* ${COMMON[*]} -o $out"
  if ! "$MARCO_BIN" "$@" "${COMMON[@]}" -o "$out" 2>"$log"; then
    echo "COMPILE FAILED:"
    tail -30 "$log" >&2
    return 1
  fi
  local drcomp drsum
  drcomp=$(grep -c '^DRCOMP:' "$log" || true)
  drsum=$(grep -c '^DRSUM:'  "$log" || true)
  echo "DRCOMP lines: $drcomp    DRSUM lines: $drsum"
  echo "--- binary md5 ---"
  md5sum "$out" | awk '{print $1}'
  echo "--- simulation output ---"
  "$out" --end-time=0.2 --time-step=0.1 --precision=6
  echo
}

run_variant nodr
run_variant withdr   -fdata-recomputation
run_variant withdrcm -fdata-recomputation -dr-cost-model

echo "================================================================"
echo "all variants compiled + ran"
echo "work dir kept at: $WORK   (rm to clean)"
trap - EXIT
