#!/usr/bin/env bash
#===----------------------------------------------------------------------===//
# probe_cost_model.sh — Generate a hardware-specific CPU cost model for drcc.
#
# Uses llvm-mca for ALU operation latencies and a pointer-chasing benchmark
# (or lmbench if available) for cache hierarchy latencies.
#
# Usage:
#   # Run inside the drcc Docker container:
#   docker run --rm -v "$PWD":/work --entrypoint /usr/local/share/drcompiler/scripts/probe_cost_model.sh \
#       drcc /work/costs.json
#
#   # Or run locally (requires llvm-mca and a C compiler on PATH):
#   ./scripts/probe_cost_model.sh costs.json
#
#   # Then compile with the probed model:
#   docker run --rm -v "$PWD":/work drcc \
#       --cpu-cost-model-file /work/costs.json -c /work/foo.c -o /work/foo.o
#===----------------------------------------------------------------------===//
set -euo pipefail

OUTPUT="${1:-}"
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"

# Resolve the Python script — check local checkout first, then Docker path.
PROBE_PY=""
for candidate in \
    "$SCRIPTS_DIR/gen_cpu_cost_model.py" \
    /usr/local/share/drcompiler/scripts/gen_cpu_cost_model.py; do
    if [[ -f "$candidate" ]]; then
        PROBE_PY="$candidate"
        break
    fi
done

if [[ -z "$PROBE_PY" ]]; then
    echo "error: gen_cpu_cost_model.py not found" >&2
    exit 1
fi

# Resolve llvm-mca.
LLVM_MCA="${LLVM_MCA:-}"
if [[ -z "$LLVM_MCA" ]]; then
    for candidate in \
        /opt/llvm/bin/llvm-mca \
        "$(command -v llvm-mca 2>/dev/null || true)"; do
        if [[ -x "$candidate" ]]; then
            LLVM_MCA="$candidate"
            break
        fi
    done
fi

# Resolve C compiler (for the built-in cache benchmark).
CC="${CC:-}"
if [[ -z "$CC" ]]; then
    for candidate in \
        /opt/llvm/bin/clang \
        "$(command -v cc 2>/dev/null || true)" \
        "$(command -v gcc 2>/dev/null || true)"; do
        if [[ -x "$candidate" ]]; then
            CC="$candidate"
            break
        fi
    done
fi

# Build args.
ARGS=(--probe)
[[ -n "$LLVM_MCA" ]] && ARGS+=(--llvm-mca "$LLVM_MCA")
[[ -n "$CC" ]]       && ARGS+=(--cc "$CC")
[[ -n "$OUTPUT" ]]   && ARGS+=(-o "$OUTPUT")

exec python3 "$PROBE_PY" "${ARGS[@]}"
