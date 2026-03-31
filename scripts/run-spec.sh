#!/usr/bin/env bash
#===----------------------------------------------------------------------===//
# run-spec.sh — Build & benchmark SPEC CPU 2017 with drcc
#
# Collects six metrics per benchmark:
#   Code Size  |  Time-to-sol  |  Cache Hit Rate  |  Peak Mem Usage
#   #DR Candidates  |  #DR Recomputed
#
# Usage (Docker):
#   docker run --rm \
#     -v /path/to/SPEC2017:/spec2017 \
#     -v ./results:/results \
#     [--privileged]               # needed for perf stat
#     drcc-spec [OPTIONS] [BENCHMARKS...]
#
# Usage (host, with SPEC already sourced):
#   ./scripts/run-spec.sh --spec-dir /path/to/SPEC2017 505.mcf_r
#===----------------------------------------------------------------------===//
set -euo pipefail

# ---- Defaults ----
SPEC_SRC="${SPEC_SRC:-/spec2017-src}"   # read-only source mount (host SPEC)
SPEC_DIR="${SPEC_DIR:-/spec2017}"       # writable installed copy (Docker volume)
RESULTS_DIR="${DR_RESULTS_DIR:-/results}"
SIZE="test"
ITERATIONS=1
COPIES=1
BUILD_ONLY=0
RUN_ONLY=0
BENCH_LIST=()
LABEL="${DRCC_LABEL:-drcc}"

# ---- Helpers ----
usage() {
    cat <<'EOF'
Usage: run-spec.sh [OPTIONS] [BENCHMARKS...]

Build & benchmark SPEC CPU 2017 with drcc (DataRecomputation Compiler).

Options:
  --spec-dir DIR       SPEC2017 install path  (default: /spec2017)
  --output DIR         Results directory       (default: /results)
  --size SIZE          Workload: test|train|ref (default: test)
  --iterations N       Timing iterations       (default: 1)
  --copies N           Rate copies             (default: 1)
  --build-only         Build only, skip run
  --run-only           Run only (must already be built)
  --tier TIER          Benchmark tier: 1, 2, 3, or all (default: 1)
  -h, --help           Show this help

Benchmarks:
  SPEC names like 505.mcf_r 519.lbm_r.  Defaults to --tier 1.

  Tier 1 — Clean C, near-certain to compile through cgeist:
    519.lbm_r  505.mcf_r  544.nab_r  557.xz_r

  Tier 2 — Clean C/C++, good chance:
    531.deepsjeng_r  541.leela_r  508.namd_r  520.omnetpp_r

  Tier 3 — Large / tricky features, may need workarounds:
    538.imagick_r  511.povray_r  510.parest_r  523.xalancbmk_r
    500.perlbench_r

  Avoid (inline asm, extreme size):
    502.gcc_r  525.x264_r  526.blender_r
EOF
    exit 0
}

# Benchmark tiers by cgeist survival likelihood.
TIER1=(519.lbm_r 505.mcf_r 544.nab_r 557.xz_r)
TIER2=(531.deepsjeng_r 541.leela_r 508.namd_r 520.omnetpp_r)
TIER3=(538.imagick_r 511.povray_r 510.parest_r 523.xalancbmk_r 500.perlbench_r)
TIER="1"

die()  { echo "run-spec: error: $*" >&2; exit 1; }
info() { echo ">>> $*"; }

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --spec-dir)    SPEC_DIR="$2";    shift 2 ;;
        --output)      RESULTS_DIR="$2"; shift 2 ;;
        --size)        SIZE="$2";        shift 2 ;;
        --iterations)  ITERATIONS="$2";  shift 2 ;;
        --copies)      COPIES="$2";      shift 2 ;;
        --build-only)  BUILD_ONLY=1;     shift ;;
        --run-only)    RUN_ONLY=1;       shift ;;
        --tier)        TIER="$2";        shift 2 ;;
        --label)       LABEL="$2";       shift 2 ;;
        -h|--help)     usage ;;
        -*)            die "unknown option: $1" ;;
        *)             BENCH_LIST+=("$1"); shift ;;
    esac
done

# Expand --tier into benchmark list if no explicit benchmarks given.
if [[ ${#BENCH_LIST[@]} -eq 0 ]]; then
    case "$TIER" in
        1)   BENCH_LIST=("${TIER1[@]}") ;;
        2)   BENCH_LIST=("${TIER1[@]}" "${TIER2[@]}") ;;
        3|all) BENCH_LIST=("${TIER1[@]}" "${TIER2[@]}" "${TIER3[@]}") ;;
        *)   die "unknown tier: $TIER (use 1, 2, 3, or all)" ;;
    esac
fi

# ---- Ensure SPEC is installed inside the container ----
# The host's SPEC is mounted read-only at SPEC_SRC.  A writable Docker
# volume is at SPEC_DIR.  On first run we rsync + install.sh; the volume
# persists so subsequent runs skip this step.
ensure_spec_installed() {
    # Already installed and tools functional?
    if [[ -x "$SPEC_DIR/bin/specperl" ]] \
       && "$SPEC_DIR/bin/specperl" -e1 2>/dev/null; then
        return 0
    fi

    # Need the source mount.
    if [[ ! -d "$SPEC_SRC/benchspec/CPU" ]]; then
        die "SPEC CPU 2017 source not found at $SPEC_SRC — mount it with -v /path/to/SPEC2017:$SPEC_SRC:ro"
    fi

    info "Populating SPEC volume from $SPEC_SRC (first run only) ..."
    rsync -a --delete "$SPEC_SRC/" "$SPEC_DIR/"

    info "Running SPEC install.sh (-f non-interactive) ..."
    cd "$SPEC_DIR"
    # -f = non-interactive, -d = destination (in-place).
    ./install.sh -f -d "$SPEC_DIR" \
        || die "SPEC install.sh failed.  Check that gcc/make work in the container."

    # Sanity check.
    if ! "$SPEC_DIR/bin/specperl" -e1 2>/dev/null; then
        die "SPEC tools not functional after install.sh"
    fi
    info "SPEC installation OK"
}

mkdir -p "$RESULTS_DIR"
export DR_RESULTS_DIR="$RESULTS_DIR"

ensure_spec_installed

# ---- Source SPEC environment ----
cd "$SPEC_DIR"
# shellcheck disable=SC1091
source shrc

# Install drcc config into SPEC's config dir.
DRCC_CFG_SRC="/opt/drcc/config/drcc.cfg"
if [[ ! -f "$DRCC_CFG_SRC" ]]; then
    # Running outside Docker — look relative to this script.
    SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
    DRCC_CFG_SRC="$SCRIPT_DIR/config/drcc.cfg"
fi
cp "$DRCC_CFG_SRC" "$SPEC_DIR/config/drcc.cfg"

# ---- Banner ----
cat <<EOF

================================================================
  SPEC CPU 2017 + drcc (DataRecomputation Compiler)
================================================================
  SPEC dir     : $SPEC_DIR
  Results      : $RESULTS_DIR
  Workload     : $SIZE
  Iterations   : $ITERATIONS
  Benchmarks   : ${BENCH_LIST[*]}
================================================================

EOF

# Common runcpu flags (array).
RUNCPU_COMMON=(
    --config=drcc.cfg
    --size="$SIZE"
    --iterations="$ITERATIONS"
    --copies="$COPIES"
    --define build_ncpus="$(nproc)"
    --define results_dir="$RESULTS_DIR"
    --define label="$LABEL"
    --tune=base
    --noreportable
)

# Track which benchmarks succeed at each stage.
declare -A BUILD_OK RUN_OK
for b in "${BENCH_LIST[@]}"; do BUILD_OK[$b]=0; RUN_OK[$b]=0; done

# ============================================================
# BUILD PHASE
# ============================================================
if [[ $RUN_ONLY -eq 0 ]]; then
    info "BUILD PHASE"
    for bench in "${BENCH_LIST[@]}"; do
        info "Building $bench ..."

        # Per-benchmark DR stats directory.
        export DR_STATS_DIR="$RESULTS_DIR/dr-stats/$bench"
        mkdir -p "$DR_STATS_DIR"

        if runcpu "${RUNCPU_COMMON[@]}" \
                --define stats_dir="$RESULTS_DIR/dr-stats/$bench" \
                --action=build \
                "$bench"; then
            BUILD_OK[$bench]=1
            info "$bench — build OK"
        else
            info "$bench — BUILD FAILED (continuing)"
        fi
    done
    unset DR_STATS_DIR
fi

# ============================================================
# RUN PHASE
# ============================================================
if [[ $BUILD_ONLY -eq 0 ]]; then
    info "RUN PHASE"

    # Clear timing counters from any previous run so sequence numbers start
    # at 000.  The timing/ directory itself is preserved for inspection.
    rm -f "$RESULTS_DIR"/timing/.*.counter "$RESULTS_DIR"/timing/.*.seq \
          "$RESULTS_DIR"/timing/.*.lock 2>/dev/null || true

    for bench in "${BENCH_LIST[@]}"; do
        # Skip if build failed (unless --run-only).
        if [[ $RUN_ONLY -eq 0 && "${BUILD_OK[$bench]}" -eq 0 ]]; then
            info "$bench — skipping run (build failed)"
            continue
        fi

        info "Running $bench ..."

        if runcpu "${RUNCPU_COMMON[@]}" \
                --define results_dir="$RESULTS_DIR" \
                --action=run \
                "$bench"; then
            RUN_OK[$bench]=1
            info "$bench — run OK"
        else
            info "$bench — RUN FAILED (continuing)"
        fi
    done
fi

# ============================================================
# COLLECT METRICS
# ============================================================
info "COLLECTING METRICS"
echo ""

# Helper: find the built executable for a benchmark.
find_exe() {
    local bench="$1"
    local exe
    exe=$(find "$SPEC_DIR/benchspec/CPU/$bench/exe/" -name "*${LABEL}*" -type f -executable 2>/dev/null | head -1)
    if [[ -z "$exe" ]]; then
        exe=$(find "$SPEC_DIR/benchspec/CPU/$bench/exe/" -type f -executable 2>/dev/null | head -1)
    fi
    echo "$exe"
}

# Helper: extract short benchmark name (e.g. "mcf_r" from "505.mcf_r").
short_name() {
    local b="$1"
    echo "${b#*.}"   # strip leading "NNN."
}

# Helper: extract wall-clock seconds from a /usr/bin/time -v output file.
extract_wall_time() {
    local tf="$1"
    local wall
    wall=$(grep "Elapsed (wall clock)" "$tf" 2>/dev/null | tail -1 | awk '{print $NF}')
    if [[ -z "$wall" ]]; then echo ""; return; fi
    if [[ "$wall" == *:*:* ]]; then
        IFS=: read -r h m s <<< "$wall"
        awk "BEGIN {printf \"%.2f\", $h*3600 + $m*60 + $s}"
    else
        IFS=: read -r m s <<< "$wall"
        awk "BEGIN {printf \"%.2f\", $m*60 + $s}"
    fi
}

# Helper: extract peak RSS (KB) from a /usr/bin/time -v output file.
extract_peak_rss_kb() {
    local tf="$1"
    grep "Maximum resident" "$tf" 2>/dev/null | tail -1 | awk '{print $NF}'
}

# Helper: compute median of a whitespace-separated list of numbers.
median_of() {
    echo "$@" | tr ' ' '\n' | sort -g | awk '{a[NR]=$1} END {
        if (NR==0) { print "—"; exit }
        if (NR%2==1) printf "%.2f", a[(NR+1)/2]
        else printf "%.2f", (a[NR/2] + a[NR/2+1]) / 2
    }'
}

# Helper: identify the sub-workload from the .args file.
# Builds a compact signature from args, filtering out the executable (line 1)
# and long hex hashes (>20 chars, likely SHA checksums).
# E.g. "cpu2006docs.tar.xz/4/1548636/1555348/0" for xz_r.
workload_id() {
    local af="$1"
    tail -n+2 "$af" 2>/dev/null \
        | awk 'length <= 20 {printf "%s/", $0} END {print ""}' \
        | sed 's|/$||; s|^/||'
}

TIMING_DIR="$RESULTS_DIR/timing"

for bench in "${BENCH_LIST[@]}"; do
    sn="$(short_name "$bench")"

    # ---- Code Size ----
    code_size="—"
    exe="$(find_exe "$bench")"
    if [[ -n "$exe" ]]; then
        code_size=$(size "$exe" 2>/dev/null | tail -1 | awk '{print $4}')
        [[ -z "$code_size" ]] && code_size="—"
    fi

    # ---- DR Candidates & Recomputed ----
    candidates="—"; recomputed="—"
    stats_dir="$RESULTS_DIR/dr-stats/$bench"
    if [[ -d "$stats_dir" ]] && compgen -G "$stats_dir"/*.dr-stats >/dev/null 2>&1; then
        single=$(grep -rch "load: SINGLE" "$stats_dir"/ 2>/dev/null | awk '{s+=$1}END{print s+0}')
        multi=$(grep -rch  "load: MULTI"  "$stats_dir"/ 2>/dev/null | awk '{s+=$1}END{print s+0}')
        leaked=$(grep -rch "load: LEAKED" "$stats_dir"/ 2>/dev/null | awk '{s+=$1}END{print s+0}')
        killed=$(grep -rch "load: KILLED" "$stats_dir"/ 2>/dev/null | awk '{s+=$1}END{print s+0}')
        candidates="$single"
        recomp_bufs=$(grep -rch "cost-model: RECOMPUTE" "$stats_dir"/ 2>/dev/null | awk '{s+=$1}END{print s+0}')
        keep_bufs=$(grep -rch   "cost-model: KEEP"      "$stats_dir"/ 2>/dev/null | awk '{s+=$1}END{print s+0}')
        if (( recomp_bufs + keep_bufs > 0 )); then
            recomputed="${recomp_bufs}buf"
        else
            recomputed="$single"
        fi
    fi

    # ---- Timing data (from timing/ directory) ----
    # Each invocation produces <TAG>.<SEQ>.time + .args.
    # Group by sub-workload (from .args) to get per-workload medians.
    if [[ ! -d "$TIMING_DIR" ]] || ! compgen -G "$TIMING_DIR/${sn}".*.time >/dev/null 2>&1; then
        # No timing data — print summary line only.
        echo "$bench  (code=$code_size  DR_SINGLE=$candidates)"
        echo "  No timing data (build-only run?)"
        echo ""
        continue
    fi

    # Collect all invocations into parallel arrays.
    declare -a ALL_TIMES=() ALL_RSS=() ALL_WKLD=()
    for tf in "$TIMING_DIR/${sn}".*.time; do
        [[ -f "$tf" ]] || continue
        local_seq="${tf%.time}"
        local_seq="${local_seq##*.}"
        af="${tf%.time}.args"

        t=$(extract_wall_time "$tf")
        rss=$(extract_peak_rss_kb "$tf")
        wid=$(workload_id "$af")

        [[ -z "$t" ]] && continue
        ALL_TIMES+=("$t")
        ALL_RSS+=("${rss:-0}")
        ALL_WKLD+=("$wid")
    done

    n_invocations=${#ALL_TIMES[@]}

    # Identify distinct sub-workloads.
    declare -A WKLD_SEEN=()
    declare -a WKLD_ORDER=()
    for w in "${ALL_WKLD[@]}"; do
        if [[ -z "${WKLD_SEEN[$w]:-}" ]]; then
            WKLD_SEEN[$w]=1
            WKLD_ORDER+=("$w")
        fi
    done
    n_workloads=${#WKLD_ORDER[@]}
    n_iters=$(( n_invocations / (n_workloads > 0 ? n_workloads : 1) ))

    # Compute total wall time per iteration (sum of sub-workloads).
    declare -a ITER_TOTALS=()
    for (( iter=0; iter<n_iters; iter++ )); do
        total=0
        for (( wi=0; wi<n_workloads; wi++ )); do
            idx=$(( iter * n_workloads + wi ))
            if (( idx < n_invocations )); then
                total=$(awk "BEGIN {printf \"%.2f\", $total + ${ALL_TIMES[$idx]}}")
            fi
        done
        ITER_TOTALS+=("$total")
    done

    # Print benchmark header.
    echo "$bench  (code=$code_size  DR_SINGLE=$candidates  DR_MULTI=${multi:-0}  DR_LEAKED=${leaked:-0})"
    echo ""

    # Per-workload table.
    if (( n_workloads > 1 )); then
        printf "  %-40s" "Sub-workload"
        for (( iter=0; iter<n_iters; iter++ )); do
            printf "  %10s" "Iter$((iter+1))"
        done
        printf "  %10s  %10s\n" "Median" "PeakRSS"
        printf "  %-40s" "────────────────────────────────────────"
        for (( iter=0; iter<n_iters; iter++ )); do
            printf "  %10s" "──────────"
        done
        printf "  %10s  %10s\n" "──────────" "──────────"

        for (( wi=0; wi<n_workloads; wi++ )); do
            wid="${WKLD_ORDER[$wi]}"
            # Truncate long workload names.
            display_wid="${wid:0:38}"
            printf "  %-40s" "$display_wid"
            times_for_wkld=""
            max_rss=0
            for (( iter=0; iter<n_iters; iter++ )); do
                idx=$(( iter * n_workloads + wi ))
                if (( idx < n_invocations )); then
                    printf "  %9ss" "${ALL_TIMES[$idx]}"
                    times_for_wkld+="${ALL_TIMES[$idx]} "
                    rss_val=${ALL_RSS[$idx]}
                    (( rss_val > max_rss )) && max_rss=$rss_val
                else
                    printf "  %10s" "—"
                fi
            done
            med=$(median_of $times_for_wkld)
            rss_mb=$(awk "BEGIN {printf \"%.1f\", $max_rss / 1024}")
            printf "  %9ss  %8sMB\n" "$med" "$rss_mb"
        done
        echo ""
    fi

    # Total / iteration summary.
    printf "  %-40s" "Total (all sub-workloads)"
    for (( iter=0; iter<n_iters; iter++ )); do
        printf "  %9ss" "${ITER_TOTALS[$iter]}"
    done
    total_med=$(median_of "${ITER_TOTALS[@]}")
    # Peak RSS across all invocations.
    max_rss_all=0
    for r in "${ALL_RSS[@]}"; do
        (( r > max_rss_all )) && max_rss_all=$r
    done
    rss_mb_all=$(awk "BEGIN {printf \"%.1f\", $max_rss_all / 1024}")
    printf "  %9ss  %8sMB\n" "$total_med" "$rss_mb_all"
    echo ""

    # Clean up associative arrays for next benchmark.
    unset ALL_TIMES ALL_RSS ALL_WKLD WKLD_SEEN WKLD_ORDER ITER_TOTALS
done

# ---- Detailed DR breakdown ----
echo "DR load classification breakdown:"
for bench in "${BENCH_LIST[@]}"; do
    stats_dir="$RESULTS_DIR/dr-stats/$bench"
    if [[ -d "$stats_dir" ]] && compgen -G "$stats_dir"/*.dr-stats >/dev/null 2>&1; then
        single=$(grep -rch "load: SINGLE" "$stats_dir"/ 2>/dev/null | awk '{s+=$1}END{print s+0}')
        multi=$(grep -rch  "load: MULTI"  "$stats_dir"/ 2>/dev/null | awk '{s+=$1}END{print s+0}')
        leaked=$(grep -rch "load: LEAKED" "$stats_dir"/ 2>/dev/null | awk '{s+=$1}END{print s+0}')
        killed=$(grep -rch "load: KILLED" "$stats_dir"/ 2>/dev/null | awk '{s+=$1}END{print s+0}')
        total=$((single + multi + leaked + killed))
        nfiles=$(ls "$stats_dir"/*.dr-stats 2>/dev/null | wc -l)
        echo "  $bench ($nfiles files, $total loads): SINGLE=$single  MULTI=$multi  LEAKED=$leaked  KILLED=$killed"
    else
        echo "  $bench: no DR stats (build may have failed)"
    fi
done

echo ""
echo "Results written to: $RESULTS_DIR/"
