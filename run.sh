#!/usr/bin/env bash
#===----------------------------------------------------------------------===//
# run.sh — Build Docker images and run SPEC CPU 2017 benchmarks with drcc.
#
# This is the single entry point.  It handles the full workflow:
#   1. Build the Docker image stack (drcc-base → drcc → drcc-spec)
#   2. Mount your SPEC2017 installation into the container
#   3. Run benchmarks and collect metrics to ./results/
#
# Usage:
#   ./run.sh                          # tier 1 (lbm, mcf, nab, xz)
#   ./run.sh --tier 2                 # + C++ benchmarks
#   ./run.sh --tier all               # everything plausible (13 benchmarks)
#   ./run.sh 505.mcf_r 519.lbm_r     # specific benchmarks
#   ./run.sh --size ref --iterations 3 505.mcf_r   # reference workload
#   ./run.sh --shell                  # interactive shell in container
#   ./run.sh --rebuild                # force rebuild all images
#===----------------------------------------------------------------------===//
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- Defaults (override via env) ----
SPEC_DIR="${SPEC_DIR:-$HOME/Dev/SPEC2017}"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"
RUNTIME="${RUNTIME:-docker}"                    # docker or podman
PRIVILEGED=0
SHELL_MODE=0
REBUILD=0
REBUILD_BASE=0
RESET_SPEC=0
DR_FLAGS=""
DRCC_LABEL=""
CONTAINER_ARGS=()
CONTAINER_ENV=()

# ---- Detect container runtime ----
if ! command -v "$RUNTIME" &>/dev/null; then
    if command -v podman &>/dev/null; then
        RUNTIME=podman
    elif command -v docker &>/dev/null; then
        RUNTIME=docker
    else
        echo "error: neither docker nor podman found" >&2
        exit 1
    fi
fi

# ---- Usage ----
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [BENCHMARKS...]

Build Docker images and run SPEC CPU 2017 benchmarks with drcc.

Options:
  --spec-dir DIR       SPEC2017 install path     (default: \$SPEC_DIR or ~/Dev/SPEC2017)
  --output DIR         Results directory          (default: ./results)
  --runtime RT         docker or podman           (default: auto-detect)
  --privileged         Enable perf stat for cache metrics
  --shell              Drop into interactive shell instead of running benchmarks
  --rebuild            Force rebuild drcc + drcc-spec images
  --rebuild-all        Force rebuild everything including drcc-base (slow — rebuilds LLVM)
  --reset-spec         Delete and re-install SPEC into Docker volume (re-runs install.sh)
  --dr-flags FLAGS     Override DR_PASS_FLAGS in the container (pipeline configuration)
  --env KEY=VALUE      Pass additional environment variable to the container

  All other options are forwarded to run-spec.sh inside the container:
  --tier TIER          1, 2, 3, or all            (default: 1)
  --size SIZE          test, train, or ref         (default: test)
  --iterations N       Timing iterations           (default: 1)
  --build-only         Only compile, don't run
  --run-only           Only run (assumes prior build)

Benchmark tiers:
  Tier 1 — Clean C, near-certain to compile through cgeist:
    519.lbm_r  505.mcf_r  544.nab_r  557.xz_r

  Tier 2 — Clean C/C++, good chance:
    531.deepsjeng_r  541.leela_r  508.namd_r  520.omnetpp_r

  Tier 3 — Large / tricky, may need workarounds:
    538.imagick_r  511.povray_r  510.parest_r  523.xalancbmk_r
    500.perlbench_r

Examples:
  $(basename "$0")                                # quick smoke test (tier 1, test size)
  $(basename "$0") --tier 2 --size ref            # serious run with C++ benchmarks
  $(basename "$0") --privileged 519.lbm_r         # single benchmark with cache metrics
  $(basename "$0") --shell                        # debug inside container
EOF
    exit 0
}

# ---- Parse arguments ----
# Split into our flags vs. flags forwarded to the container.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --spec-dir)     SPEC_DIR="$2";      shift 2 ;;
        --output)       RESULTS_DIR="$2";   shift 2 ;;
        --runtime)      RUNTIME="$2";       shift 2 ;;
        --privileged)   PRIVILEGED=1;       shift ;;
        --shell)        SHELL_MODE=1;       shift ;;
        --rebuild)      REBUILD=1;          shift ;;
        --rebuild-all)  REBUILD=1; REBUILD_BASE=1; shift ;;
        --reset-spec)   RESET_SPEC=1;             shift ;;
        --dr-flags)     DR_FLAGS="$2";            shift 2 ;;
        --label)        DRCC_LABEL="$2";          shift 2 ;;
        --env)          CONTAINER_ENV+=("$1" "$2"); shift 2 ;;
        -h|--help)      usage ;;
        *)              CONTAINER_ARGS+=("$1"); shift ;;
    esac
done

# ---- Validate SPEC installation ----
if [[ ! -d "$SPEC_DIR/benchspec/CPU" ]]; then
    echo "error: SPEC CPU 2017 not found at $SPEC_DIR" >&2
    echo "Set SPEC_DIR or use --spec-dir /path/to/SPEC2017" >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR"

# ---- Image / volume names ----
IMG_BASE="drcc-base"
IMG_DRCC="drcc"
IMG_SPEC="drcc-spec"
SPEC_VOL="drcc-spec2017"            # persistent volume for installed SPEC

# ---- Build helpers ----
image_exists() {
    "$RUNTIME" image inspect "$1" &>/dev/null
}

build_image() {
    local tag="$1" dockerfile="$2"
    shift 2
    echo ">>> Building $tag ..."
    "$RUNTIME" build \
        -f "$SCRIPT_DIR/$dockerfile" \
        -t "$tag" \
        "$@" \
        "$SCRIPT_DIR"
}

# ---- Build image stack ----
# Layer 1: drcc-base (LLVM 22 + cgeist) — very slow, only rebuild if missing or forced.
if ! image_exists "$IMG_BASE" || [[ $REBUILD_BASE -eq 1 ]]; then
    build_image "$IMG_BASE" docker/base.Dockerfile
else
    echo ">>> $IMG_BASE image exists (use --rebuild-all to force)"
fi

# Layer 2: drcc (dr-opt + drcc wrapper) — moderate, rebuild when source changes.
if ! image_exists "$IMG_DRCC" || [[ $REBUILD -eq 1 ]]; then
    build_image "$IMG_DRCC" docker/drcc.Dockerfile
else
    echo ">>> $IMG_DRCC image exists (use --rebuild to force)"
fi

# Layer 3: drcc-spec (scripts + config + instrumentation tools) — fast.
if ! image_exists "$IMG_SPEC" || [[ $REBUILD -eq 1 ]]; then
    build_image "$IMG_SPEC" docker/spec.Dockerfile
else
    echo ">>> $IMG_SPEC image exists (use --rebuild to force)"
fi

# ---- Manage SPEC volume ----
if [[ $RESET_SPEC -eq 1 ]]; then
    echo ">>> Removing SPEC volume ($SPEC_VOL) ..."
    "$RUNTIME" volume rm "$SPEC_VOL" 2>/dev/null || true
fi
# Create the named volume if it doesn't exist (idempotent).
"$RUNTIME" volume inspect "$SPEC_VOL" &>/dev/null \
    || "$RUNTIME" volume create "$SPEC_VOL" >/dev/null

# ---- Compose docker run ----
# Host SPEC is read-only at /spec2017-src; the volume at /spec2017 is writable.
# On first run, run-spec.sh rsyncs the source to the volume and runs install.sh.
RUN_ARGS=(
    "$RUNTIME" run --rm
    -v "$SPEC_DIR":/spec2017-src:ro
    -v "$SPEC_VOL":/spec2017
    -v "$RESULTS_DIR":/results
)

if [[ -n "$DR_FLAGS" ]]; then
    RUN_ARGS+=(-e "DR_PASS_FLAGS=$DR_FLAGS")
fi

if [[ -n "$DRCC_LABEL" ]]; then
    RUN_ARGS+=(-e "DRCC_LABEL=$DRCC_LABEL")
fi

for ((i=0; i<${#CONTAINER_ENV[@]}; i+=2)); do
    RUN_ARGS+=(-e "${CONTAINER_ENV[i+1]}")
done

if [[ $PRIVILEGED -eq 1 ]]; then
    RUN_ARGS+=(--privileged)
fi

if [[ $SHELL_MODE -eq 1 ]]; then
    RUN_ARGS+=(-it --entrypoint bash "$IMG_SPEC")
    echo ""
    echo ">>> Entering interactive shell (SPEC at /spec2017, results at /results)"
    echo ">>>   source /spec2017/shrc"
    echo ">>>   runcpu --config=drcc.cfg 505.mcf_r"
    echo ""
    exec "${RUN_ARGS[@]}"
fi

# Normal mode: forward remaining args to run-spec.sh (the entrypoint).
RUN_ARGS+=("$IMG_SPEC" "${CONTAINER_ARGS[@]}")

echo ""
echo ">>> SPEC dir:  $SPEC_DIR"
echo ">>> Results:   $RESULTS_DIR"
echo ">>> Runtime:   $RUNTIME"
echo ">>> Args:      ${CONTAINER_ARGS[*]:-<defaults>}"
echo ""

exec "${RUN_ARGS[@]}"
