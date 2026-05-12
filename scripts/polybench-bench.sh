#!/usr/bin/env bash
#===----------------------------------------------------------------------===//
# polybench-bench.sh — Benchmark drcompiler recomputation configs on PolyBench/C
#
# Downloads PolyBench/C 4.2.1 if not present, compiles each kernel under
# five configurations, and reports median POLYBENCH_TIME execution times.
#
# Configurations:
#   clang       — clang -O2 (reference, no MLIR)
#   cgeist-base — cgeist → lower → clang (MLIR roundtrip, no DR pass)
#   dr-greedy   — cgeist → data-recomputation{dr-recompute=true} → lower → clang
#   dr-cost     — cgeist → DR{recompute, cost-model} → lower → clang
#   dr-partial  — cgeist → DR{recompute, cost-model, partial-remat} → lower → clang
#
# Requirements:
#   - docker with drcc image (all compilation and linking runs inside the container)
#
# Usage:
#   ./scripts/polybench-bench.sh [options]
#
# Options:
#   --polybench-dir DIR    PolyBench source root   (default: ./third-party/polybench)
#   --iters N              runs per kernel          (default: 3)
#   --dataset SIZE         MINI|SMALL|STANDARD|LARGE|EXTRALARGE  (default: LARGE)
#   --kernel PATTERN       filter kernels by name substring
#   --configs LIST         comma-separated subset of: clang,cgeist-base,dr-greedy,dr-cost,dr-partial
#   --cost-model FILE      JSON cost model (from probe-cost-model.sh) for dr-cost and dr-partial configs
#   --ref-cfg NAME         config to use as speedup reference (default: cgeist-base if present, else clang)
#   --csv FILE             write results to CSV in addition to stdout
#   --no-download          abort if PolyBench not already present
#===----------------------------------------------------------------------===//
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

POLYBENCH_DIR="${PROJECT_DIR}/third-party/polybench"
ITERS=3
DATASET="LARGE"
KERNEL_FILTER=""
CONFIGS="clang,cgeist-base,dr-greedy,dr-cost,dr-partial"
COST_MODEL_FILE=""
REF_CFG_OVERRIDE=""
CSV_FILE=""
NO_DOWNLOAD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --polybench-dir) POLYBENCH_DIR="$2"; shift 2 ;;
    --iters)         ITERS="$2";         shift 2 ;;
    --dataset)       DATASET="$2";       shift 2 ;;
    --kernel)        KERNEL_FILTER="$2"; shift 2 ;;
    --configs)       CONFIGS="$2";       shift 2 ;;
    --cost-model)    COST_MODEL_FILE="$2"; shift 2 ;;
    --ref-cfg)       REF_CFG_OVERRIDE="$2"; shift 2 ;;
    --csv)           CSV_FILE="$2";      shift 2 ;;
    --no-download)   NO_DOWNLOAD=1;      shift ;;
    -h|--help)
      sed -n '2,/^#===/p' "$0" | sed 's/^# \?//'
      exit 0 ;;
    *) echo "polybench-bench: unknown arg: $1" >&2; exit 1 ;;
  esac
done

DRCC_IMAGE="${DRCC_IMAGE:-drcc}"
DRCC_TAG="${DRCC_TAG:-latest}"

if [[ -n "$COST_MODEL_FILE" ]]; then
  [[ -f "$COST_MODEL_FILE" ]] || die "cost model file not found: $COST_MODEL_FILE"
  COST_MODEL_FILE="$(realpath "$COST_MODEL_FILE")"
fi

log()  { echo "[polybench] $*" >&2; }
warn() { echo "[polybench] WARN: $*" >&2; }
die()  { echo "polybench-bench: error: $*" >&2; exit 1; }

# --- Validate tools ---
docker image inspect "${DRCC_IMAGE}:${DRCC_TAG}" >/dev/null 2>&1 \
  || die "${DRCC_IMAGE}:${DRCC_TAG} Docker image not found. Build with:\n  docker build -f docker/drcc.Dockerfile -t ${DRCC_IMAGE} ."

# --- Acquire PolyBench ---
POLYBENCH_REPO="https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1"
if [[ ! -d "${POLYBENCH_DIR}/utilities" ]]; then
  [[ "$NO_DOWNLOAD" -eq 1 ]] \
    && die "PolyBench not found at ${POLYBENCH_DIR} (--no-download set)"
  log "Cloning PolyBench/C 4.2.1 → ${POLYBENCH_DIR}"
  mkdir -p "$(dirname "$POLYBENCH_DIR")"
  git clone --depth 1 "$POLYBENCH_REPO" "$POLYBENCH_DIR" >&2 \
    || die "git clone failed"
fi

UTILITIES_DIR="${POLYBENCH_DIR}/utilities"
[[ -f "${UTILITIES_DIR}/polybench.h" ]] || die "polybench.h missing at ${UTILITIES_DIR}"
[[ -f "${UTILITIES_DIR}/polybench.c" ]] || die "polybench.c missing at ${UTILITIES_DIR}"

# --- Build kernel list ---
declare -a KERNELS=()
while IFS= read -r -d '' kfile; do
  kname="$(basename "$(dirname "$kfile")")"
  fname="$(basename "$kfile" .c)"
  # Only accept the canonical kernel file: <dir>/<dirname>.c
  [[ "$fname" == "$kname" ]] || continue
  if [[ -z "$KERNEL_FILTER" || "$kname" == *"$KERNEL_FILTER"* ]]; then
    KERNELS+=("$kfile")
  fi
done < <(find "$POLYBENCH_DIR" -name "*.c" ! -path "*/utilities/*" -print0 | sort -z)

[[ ${#KERNELS[@]} -gt 0 ]] \
  || die "No kernels found under ${POLYBENCH_DIR} (filter='${KERNEL_FILTER}')"
log "${#KERNELS[@]} kernel(s); dataset=${DATASET}, iters=${ITERS}"

DATASET_DEFINE="-D${DATASET}_DATASET"

# --- Config → DR pipeline string (CLANG_ONLY = skip MLIR) ---
declare -A CFG_PIPELINE
CFG_PIPELINE["clang"]="CLANG_ONLY"
CFG_PIPELINE["cgeist-base"]="--pass-pipeline=builtin.module(raise-malloc-to-memref)"
CFG_PIPELINE["dr-greedy"]="--pass-pipeline=builtin.module(raise-malloc-to-memref,data-recomputation{dr-recompute=true})"
CFG_PIPELINE["dr-cost"]="--pass-pipeline=builtin.module(raise-malloc-to-memref,data-recomputation{dr-recompute=true dr-cost-model=true})"
CFG_PIPELINE["dr-partial"]="--pass-pipeline=builtin.module(raise-malloc-to-memref,data-recomputation{dr-recompute=true dr-cost-model=true dr-partial-remat=true})"

IFS=',' read -ra ACTIVE_CFGS <<< "$CONFIGS"
for cfg in "${ACTIVE_CFGS[@]}"; do
  [[ -v CFG_PIPELINE["$cfg"] ]] || die "Unknown config: $cfg. Available: ${!CFG_PIPELINE[*]}"
done

# --- Shared work dir (cleaned on exit) ---
WORK_DIR="$(mktemp -d /tmp/polybench-bench.XXXXXX)"
DRSUM_DIR="${WORK_DIR}/drsum"
mkdir -p "$DRSUM_DIR"
trap "rm -rf '$WORK_DIR'" EXIT

# Compile polybench.c once: timing harness for MLIR config linking (container clang).
# Skipped when only the clang config is active (it compiles polybench.c in its own run).
POLYBENCH_OBJ="${WORK_DIR}/polybench.o"
_need_polybench_obj=0
for _cfg in "${ACTIVE_CFGS[@]}"; do
  [[ "${CFG_PIPELINE[$_cfg]}" != "CLANG_ONLY" ]] && { _need_polybench_obj=1; break; }
done
if [[ "$_need_polybench_obj" -eq 1 ]]; then
  log "Compiling polybench.c..."
  docker run --rm --entrypoint bash \
    -v "${WORK_DIR}:${WORK_DIR}" \
    -v "${UTILITIES_DIR}:${UTILITIES_DIR}:ro" \
    "${DRCC_IMAGE}:${DRCC_TAG}" -c "
/opt/llvm/bin/clang -c '${UTILITIES_DIR}/polybench.c' \
  -I '${UTILITIES_DIR}' -DPOLYBENCH_TIME ${DATASET_DEFINE} -O2 \
  -o '${POLYBENCH_OBJ}' 2>/dev/null
" || die "polybench.c compilation failed"
fi

# nodce.h stub written once per WORK_DIR.
# Including polybench.h here first sets its include guard; the subsequent
# #undef+#define overrides polybench_prevent_dce so cgeist never sees the
# argc/strcmp/print_array DCE guard.  With -O2 the now-unreachable static
# print_array is dropped before MLIR is emitted.
NODCE_H="${WORK_DIR}/nodce.h"
cat > "$NODCE_H" << 'NODCE_EOF'
#include "polybench.h"
#undef polybench_prevent_dce
#define polybench_prevent_dce(func) do {} while(0)
NODCE_EOF

# --- compile_kernel <kernel.c> <cfg> <outdir>
# Prints path to executable on success, nothing on failure.
#
# clang config  — container clang -O2, compile+link fully inside drcc image
# MLIR configs  — full-file cgeist --function='*' with nodce.h to suppress the
#                 argv/strcmp/print_array DCE guard (which would generate
#                 polygeist ops that block the lowering pipeline).
#                 main() contains the inlined static kernel — DR fires there.
#                 Output: mlir-compiled main.o linked with polybench.o only.
compile_kernel() {
  local kernel_c="$1" cfg="$2" outdir="$3"
  local kname kdir pipeline cfg_safe exe

  kname="$(basename "$(dirname "$kernel_c")")"
  kdir="$(dirname "$kernel_c")"
  pipeline="${CFG_PIPELINE[$cfg]}"
  cfg_safe="${cfg//[^a-zA-Z0-9_-]/_}"
  exe="${outdir}/${cfg_safe}.bin"

  if [[ "$pipeline" == "CLANG_ONLY" ]]; then
    # Fair baseline: compile+link entirely inside the drcc container so the
    # same toolchain (container clang) is used for both kernel and timing harness.
    local stage="${outdir}/stage_${cfg_safe}"
    mkdir -p "$stage"
    cp "$kernel_c"                    "${stage}/"
    cp "${UTILITIES_DIR}/polybench.h" "${stage}/"
    cp "${UTILITIES_DIR}/polybench.c" "${stage}/"
    for hdr in "$kdir"/*.h; do [[ -f "$hdr" ]] && cp "$hdr" "${stage}/"; done

    local src_name="${stage}/$(basename "$kernel_c")"
    local stage_exe="${stage}/clang.bin"

    docker run --rm --entrypoint bash \
      -v "${stage}:${stage}" \
      "${DRCC_IMAGE}:${DRCC_TAG}" -c "
set -e
CLANG=/opt/llvm/bin/clang
\$CLANG -O2 -c '${src_name}' \
  -I '${stage}' -DPOLYBENCH_TIME ${DATASET_DEFINE} \
  -o '${stage}/kernel.o' || exit 1
\$CLANG -O2 -c '${stage}/polybench.c' \
  -I '${stage}' -DPOLYBENCH_TIME ${DATASET_DEFINE} \
  -o '${stage}/polybench.o' || exit 1
\$CLANG '${stage}/kernel.o' '${stage}/polybench.o' -lm \
  -o '${stage_exe}' || exit 1
" 2>&1 | head -20 >&2 \
      || { warn "${kname}/${cfg}: clang compilation failed"; return; }

    mv "$stage_exe" "$exe" && echo "$exe" \
      || warn "${kname}/${cfg}: clang binary move failed"
    return
  fi

  # Stage kernel source + headers + polybench.h + nodce.h into a flat dir.
  local stage="${outdir}/stage_${cfg_safe}"
  mkdir -p "$stage"
  cp "$kernel_c"                    "${stage}/"
  cp "${UTILITIES_DIR}/polybench.h" "${stage}/"
  cp "$NODCE_H"                     "${stage}/nodce.h"
  for hdr in "$kdir"/*.h; do [[ -f "$hdr" ]] && cp "$hdr" "${stage}/"; done

  local src_name="${stage}/$(basename "$kernel_c")"
  local main_o="${stage}/main_dr.o"
  local effective_pipeline="$pipeline"
  if [[ "$pipeline" == *"dr-recompute=true"* ]]; then
    effective_pipeline="${effective_pipeline/dr-recompute=true/dr-recompute=true dr-summary=true}"
  fi
  if [[ -n "$COST_MODEL_FILE" && "$pipeline" == *"dr-cost-model=true"* ]]; then
    cp "$COST_MODEL_FILE" "${stage}/cost_model.json"
    effective_pipeline="${effective_pipeline/dr-cost-model=true/dr-cost-model=true cpu-cost-model-file=${stage}/cost_model.json}"
  fi

  # Run full-file MLIR/DR pipeline inside the drcc container.
  # -include nodce.h pre-includes polybench.h then overrides polybench_prevent_dce
  # so the argc/strcmp DCE guard is never raised to MLIR.  --function='*' raises
  # all external functions (= main, since polybench kernels are static and get
  # inlined into main by cgeist).
  docker run --rm --entrypoint bash \
    -v "${stage}:${stage}" \
    "${DRCC_IMAGE}:${DRCC_TAG}" -c "
set -e
CGEIST=/usr/local/bin/cgeist
DR_OPT=/usr/local/bin/dr-opt
MLIR_OPT=/opt/llvm/bin/mlir-opt
MLIR_TRANSLATE=/opt/llvm/bin/mlir-translate
CLANG=/opt/llvm/bin/clang

\$CGEIST '${src_name}' -S --function='*' --raise-scf-to-affine \\
  -target aarch64-linux-gnu \\
  -include '${stage}/nodce.h' \\
  -I '${stage}' -DPOLYBENCH_TIME ${DATASET_DEFINE} -O2 \\
  -o '${stage}/main.mlir' || exit 1

sed -i 's/vector<\([0-9]*\)xi32>/vector<\1xi64>/g' '${stage}/main.mlir'
python3 /usr/local/bin/rewrite-struct-memrefs.py '${stage}/main.mlir'

\$DR_OPT --allow-unregistered-dialect '${effective_pipeline}' \\
  '${stage}/main.mlir' -o '${stage}/main.opt.mlir' 2>'${stage}/drsum.log' || exit 1

\$MLIR_OPT \\
  --lower-affine --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm \\
  --convert-math-to-llvm '--convert-func-to-llvm=use-bare-ptr-memref-call-conv=1' \\
  --convert-index-to-llvm --finalize-memref-to-llvm --reconcile-unrealized-casts \\
  '${stage}/main.opt.mlir' -o '${stage}/main.llvm.mlir' 2>/dev/null || exit 1

\$MLIR_TRANSLATE --mlir-to-llvmir \\
  '${stage}/main.llvm.mlir' -o '${stage}/main.ll' 2>/dev/null || exit 1

\$CLANG -O2 -c '${stage}/main.ll' -o '${main_o}' || exit 1
" 2>&1 | head -20 >&2 \
    || { warn "${kname}/${cfg}: MLIR pipeline failed"; return; }

  [[ -f "$main_o" ]] || { warn "${kname}/${cfg}: main_dr.o not produced"; return; }
  if [[ -f "${stage}/drsum.log" ]]; then
    mkdir -p "${DRSUM_DIR}/${cfg}"
    cp "${stage}/drsum.log" "${DRSUM_DIR}/${cfg}/${kname}.log"
  fi

  # Link: DR-compiled main.o + polybench.o (timing harness only), in container.
  docker run --rm --entrypoint bash \
    -v "${stage}:${stage}" \
    -v "${POLYBENCH_OBJ}:${POLYBENCH_OBJ}:ro" \
    "${DRCC_IMAGE}:${DRCC_TAG}" -c "
/opt/llvm/bin/clang '${main_o}' '${POLYBENCH_OBJ}' -lm -o '${stage}/linked.bin'
" 2>&1 | head -20 >&2 \
    && mv "${stage}/linked.bin" "$exe" && echo "$exe" \
    || warn "${kname}/${cfg}: link failed"
}

# --- time_kernel <exe> <n> → "MEDIAN AVG STDDEV" space-separated, or "ERR ERR ERR" ---
time_kernel() {
  local exe="$1" n="$2"
  local -a times=()
  local t i

  for ((i = 0; i < n; i++)); do
    t=$("$exe" 2>/dev/null) || { echo "ERR ERR ERR"; return; }
    if ! [[ "$t" =~ ^[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?$ ]]; then
      echo "ERR ERR ERR"; return
    fi
    times+=("$t")
  done

  local -a sorted
  IFS=$'\n' sorted=($(printf '%s\n' "${times[@]}" | sort -g)); unset IFS
  local median="${sorted[$(( n / 2 ))]}"
  # Compute avg and sample stddev in one awk pass
  local avg stddev
  read -r avg stddev < <(awk '
    BEGIN { s=0; s2=0; n=0 }
    { s+=$1; s2+=$1*$1; n++ }
    END {
      avg=s/n
      variance=(n>1) ? (s2 - s*s/n)/(n-1) : 0
      printf "%.6f %.6f\n", avg, sqrt(variance)
    }' <(printf '%s\n' "${times[@]}"))
  echo "${median} ${avg} ${stddev}"
}

_print_table_header() {
  local label="$1"
  printf "\n=== PolyBench/C — dataset=%s  iters=%s  [%s] ===\n\n" \
    "$DATASET" "$ITERS" "$label"
  printf "%-22s" "Kernel"
  for cfg in "${ACTIVE_CFGS[@]}"; do printf "%14s" "$cfg"; done
  printf "%10s\n" "speedup"
  printf "%-22s" "$(printf '%0.s-' {1..22})"
  for cfg in "${ACTIVE_CFGS[@]}"; do printf "%14s" "$(printf '%0.s-' {1..14})"; done
  printf "%10s\n" "$(printf '%0.s-' {1..10})"
}

_print_drsum_table() {
  local cfg="$1"
  local dir="${DRSUM_DIR}/${cfg}"
  [[ -d "$dir" ]] || return
  local logs=("${dir}/"*.log)
  [[ -e "${logs[0]}" ]] || return

  printf "\n=== Recomputation Decisions — %s ===\n\n" "$cfg"
  printf "%-22s %9s %9s %9s %9s %9s\n" \
    "Kernel" "accepted" "rej:type" "rej:unsafe" "rej:cost" "skipped"
  printf "%-22s %9s %9s %9s %9s %9s\n" \
    "----------------------" "---------" "---------" "----------" "---------" "---------"

  local total_acc=0 total_type=0 total_unsafe=0 total_cost=0 total_skip=0
  for log in "${dir}/"*.log; do
    local kname="${log##*/}"; kname="${kname%.log}"
    local acc type unsafe cost skip
    read -r acc type unsafe cost skip < <(awk '
      /DRSUM: load .* ACCEPT/           { acc++ }
      /REJECT_TYPE/                     { type++ }
      /REJECT_UNSAFE/                   { unsafe++ }
      /REJECT_COST/                     { cost++ }
      /SKIP_COST|SKIP_AMBIGUOUS/        { skip++ }
      END { printf "%d %d %d %d %d\n", acc+0, type+0, unsafe+0, cost+0, skip+0 }
    ' "$log")
    printf "%-22s %9d %9d %9d %9d %9d\n" \
      "$kname" "$acc" "$type" "$unsafe" "$cost" "$skip"
    (( total_acc    += acc    )) || true
    (( total_type   += type   )) || true
    (( total_unsafe += unsafe )) || true
    (( total_cost   += cost   )) || true
    (( total_skip   += skip   )) || true
  done

  printf "%-22s %9s %9s %9s %9s %9s\n" \
    "----------------------" "---------" "---------" "----------" "---------" "---------"
  printf "%-22s %9d %9d %9d %9d %9d\n" \
    "TOTAL" "$total_acc" "$total_type" "$total_unsafe" "$total_cost" "$total_skip"
}

_speedup() {
  local ref="$1" val="$2"
  if [[ -n "$ref" && "$ref" != ERR && "$ref" != FAIL \
                   && "$val" != ERR && "$val" != FAIL ]]; then
    awk "BEGIN{ d=${val}+0; printf \"%.2fx\", (d>0 ? ${ref}/d : 0) }"
  else
    echo "—"
  fi
}

# CSV header
if [[ -n "$CSV_FILE" ]]; then
  { printf "kernel"
    for cfg in "${ACTIVE_CFGS[@]}"; do
      printf ",%s_median,%s_avg" "$cfg" "$cfg"
    done
    printf "\n"
  } > "$CSV_FILE"
fi

# Collect all results first, then print two tables.
declare -A ALL_MED ALL_AVG ALL_STD

# --- Per-kernel benchmark ---
for kernel_c in "${KERNELS[@]}"; do
  kname="$(basename "$(dirname "$kernel_c")")"
  kdir="${WORK_DIR}/${kname}"
  mkdir -p "$kdir"
  log "kernel: ${kname}"

  for cfg in "${ACTIVE_CFGS[@]}"; do
    exe="$(compile_kernel "$kernel_c" "$cfg" "$kdir")" || exe=""
    if [[ -z "$exe" || ! -x "$exe" ]]; then
      ALL_MED["${kname}:${cfg}"]="FAIL"
      ALL_AVG["${kname}:${cfg}"]="FAIL"
      ALL_STD["${kname}:${cfg}"]="FAIL"
      continue
    fi
    read -r med avg std < <(time_kernel "$exe" "$ITERS")
    ALL_MED["${kname}:${cfg}"]="$med"
    ALL_AVG["${kname}:${cfg}"]="$avg"
    ALL_STD["${kname}:${cfg}"]="$std"
  done

  # CSV row (written immediately)
  if [[ -n "$CSV_FILE" ]]; then
    { printf "%s" "$kname"
      for cfg in "${ACTIVE_CFGS[@]}"; do
        printf ",%s,%s,%s" \
          "${ALL_MED[${kname}:${cfg}]:-FAIL}" \
          "${ALL_AVG[${kname}:${cfg}]:-FAIL}" \
          "${ALL_STD[${kname}:${cfg}]:-FAIL}"
      done
      printf "\n"
    } >> "$CSV_FILE"
  fi
done

last_cfg="${ACTIVE_CFGS[-1]}"

# Speedup reference: explicit --ref-cfg > cgeist-base (if present) > clang.
if [[ -n "$REF_CFG_OVERRIDE" ]]; then
  [[ -v CFG_PIPELINE["$REF_CFG_OVERRIDE"] ]] \
    || die "Unknown --ref-cfg: $REF_CFG_OVERRIDE. Available: ${!CFG_PIPELINE[*]}"
  REF_CFG="$REF_CFG_OVERRIDE"
else
  REF_CFG="clang"
  for _c in "${ACTIVE_CFGS[@]}"; do
    [[ "$_c" == "cgeist-base" ]] && { REF_CFG="cgeist-base"; break; }
  done
fi

# --- Table 1: Median ---
_print_table_header "MEDIAN"
for kernel_c in "${KERNELS[@]}"; do
  kname="$(basename "$(dirname "$kernel_c")")"
  printf "%-22s" "$kname"
  for cfg in "${ACTIVE_CFGS[@]}"; do
    printf "%14s" "${ALL_MED[${kname}:${cfg}]:-FAIL}"
  done
  ref="${ALL_MED[${kname}:${REF_CFG}]:-}"
  val="${ALL_MED[${kname}:${last_cfg}]:-FAIL}"
  [[ "$last_cfg" != "$REF_CFG" ]] \
    && printf "%10s\n" "$(_speedup "$ref" "$val")" \
    || printf "%10s\n" "—"
done

# --- Table 2: Average ---
_print_table_header "AVERAGE"
for kernel_c in "${KERNELS[@]}"; do
  kname="$(basename "$(dirname "$kernel_c")")"
  printf "%-22s" "$kname"
  for cfg in "${ACTIVE_CFGS[@]}"; do
    printf "%14s" "${ALL_AVG[${kname}:${cfg}]:-FAIL}"
  done
  ref="${ALL_AVG[${kname}:${REF_CFG}]:-}"
  val="${ALL_AVG[${kname}:${last_cfg}]:-FAIL}"
  [[ "$last_cfg" != "$REF_CFG" ]] \
    && printf "%10s\n" "$(_speedup "$ref" "$val")" \
    || printf "%10s\n" "—"
done

# --- Table 3: Signal-to-noise — is the speedup distinguishable from noise?
# Computes pooled noise floor (quadrature sum of ref and last-config stddevs)
# vs |avg_ref - avg_cfg|.  SNR > 2 → likely real;  SNR < 1 → noise.
if [[ "$last_cfg" != "$REF_CFG" ]]; then
  printf "\n=== Signal-to-Noise (avg speedup vs pooled σ) — %s vs %s ===\n\n" \
    "$REF_CFG" "$last_cfg"
  printf "%-22s %10s %10s %10s %10s %8s\n" \
    "Kernel" "${REF_CFG}-avg" "cfg-avg" "delta" "pooled-σ" "SNR"
  printf "%-22s %10s %10s %10s %10s %8s\n" \
    "----------------------" "----------" "----------" "----------" "----------" "--------"
  for kernel_c in "${KERNELS[@]}"; do
    kname="$(basename "$(dirname "$kernel_c")")"
    ref_avg="${ALL_AVG[${kname}:${REF_CFG}]:-FAIL}"
    cfg_avg="${ALL_AVG[${kname}:${last_cfg}]:-FAIL}"
    ref_std="${ALL_STD[${kname}:${REF_CFG}]:-FAIL}"
    cfg_std="${ALL_STD[${kname}:${last_cfg}]:-FAIL}"
    if [[ "$ref_avg" == FAIL || "$cfg_avg" == FAIL ]]; then
      printf "%-22s %10s\n" "$kname" "FAIL"
      continue
    fi
    awk -v kname="$kname" \
        -v ra="$ref_avg" -v ca="$cfg_avg" \
        -v rs="$ref_std" -v cs="$cfg_std" '
    BEGIN {
      delta = ra - ca
      pooled = sqrt(rs*rs + cs*cs)
      snr = (pooled > 0) ? delta/pooled : 999
      flag = (snr > 2) ? "SIGNAL" : (snr < -2) ? "REGRESS" : "noise"
      printf "%-22s %10.6f %10.6f %+10.6f %10.6f %8.2f  %s\n",
        kname, ra, ca, delta, pooled, snr, flag
    }'
  done
fi

for _cfg in "${ACTIVE_CFGS[@]}"; do
  [[ "${CFG_PIPELINE[$_cfg]}" == *"dr-recompute=true"* ]] && _print_drsum_table "$_cfg"
done

echo ""
[[ -n "$CSV_FILE" ]] && log "CSV written: $CSV_FILE"
log "Done."
