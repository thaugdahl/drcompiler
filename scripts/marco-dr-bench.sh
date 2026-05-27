#!/usr/bin/env bash
#
# Bench harness for the DataRecomputation pass through marco.
#
# For each .mo in the corpus (default: marco's test/Simulation/euler-forward),
# compiles three variants:
#   1. baseline  — no DR pass
#   2. dr        — -fdata-recomputation
#   3. drcm      — -fdata-recomputation -dr-cost-model
#
# Each binary is run N times (default 5).  Median wall-clock is reported.
# Output: per-model table + speedups vs baseline.
#
# Usage:
#   bash scripts/marco-dr-bench.sh [corpus-dir] [iterations]
#
# Env overrides:
#   MARCO_BIN    — marco driver       (default /home/tor/Dev/marco/install/marco/bin/marco)
#   RT_LIB       — runtime lib dir    (default /home/tor/Dev/marco/install/marco-runtime/lib)
#   END_TIME     — simulation end-time (default scale from RUN line × SCALE)
#   TIME_STEP    — simulation step    (default keep RUN line)
#   SCALE        — end-time multiplier so models run long enough to time (default 100)
#   PRECISION    — output precision   (default 6)
#   SKIP_RUN     — set to 1 to skip execution (compile-only timing)
#   VERBOSE      — set to 1 to print per-iteration timings
#   MODEL        — override extracted model name (e.g. ThermalChipSimpleBoundary)
#   SOLVER       — override solver (default euler-forward)
#   MARCO_EXTRA  — extra args passed to marco at compile time (space-separated)
#   SIM_EXTRA    — extra args passed to the simulation binary at run time

set -euo pipefail

MARCO_BIN="${MARCO_BIN:-/home/tor/Dev/marco/install/marco/bin/marco}"
RT_LIB="${RT_LIB:-/home/tor/Dev/marco/install/marco-runtime/lib}"
SCALE="${SCALE:-100}"
PRECISION="${PRECISION:-6}"
SKIP_RUN="${SKIP_RUN:-0}"
VERBOSE="${VERBOSE:-0}"

CORPUS="${1:-/home/tor/Dev/marco/source/marco/test/Simulation/euler-forward}"
ITERS="${2:-5}"

if [[ ! -x "$MARCO_BIN" ]]; then
  echo "marco not found at $MARCO_BIN" >&2; exit 1
fi
if [[ ! -e "$CORPUS" ]]; then
  echo "corpus not found: $CORPUS" >&2; exit 1
fi

WORK="$(mktemp -d -t marco-dr-bench.XXXXXX)"
trap "rm -rf '$WORK'" EXIT

LOG="$WORK/bench.log"
TABLE="$WORK/table.tsv"
printf "model\tbaseline_s\tdr_s\tdrcm_s\tdr_speedup\tdrcm_speedup\tnotes\n" > "$TABLE"

# ----- helpers -----

extract_model_solver() {
  # Resolution order:
  #   1. MODEL / SOLVER env overrides
  #   2. --model= / --solver= from a lit RUN line
  #   3. The last `model X` (possibly quoted) declared in the file
  # Default solver=euler-forward.
  local f="$1"
  if [[ -n "${MODEL:-}" ]]; then
    printf '%s\t%s\n' "$MODEL" "${SOLVER:-euler-forward}"
    return
  fi
  local out
  out=$(awk '
    /^\/\/ RUN: marco / {
      for (i = 1; i <= NF; i++) {
        if (match($i, /--model=([^ ]+)/, m))   model = m[1]
        if (match($i, /--solver=([^ ]+)/, s))  solver = s[1]
      }
      if (model != "") { print model "\t" (solver == "" ? "euler-forward" : solver); exit }
    }' "$f")
  if [[ -z "$out" ]]; then
    local model
    model=$(awk '
      /^[[:space:]]*model[[:space:]]+/ {
        if (match($0, /model[[:space:]]+'\''([^'\'']+)'\''/, m)) { name = m[1] }
        else if (match($0, /model[[:space:]]+([A-Za-z_][A-Za-z0-9_]*)/, m)) { name = m[1] }
      }
      END { if (name != "") print name }' "$f")
    [[ -n "$model" ]] && out="${model}\t${SOLVER:-euler-forward}"
  fi
  printf '%b\n' "$out"
}

extract_end_time_step() {
  # Pull --end-time= and --time-step= from the second RUN line; otherwise defaults.
  local f="$1"
  local out
  out=$(awk '
    /^\/\/ RUN: \.\// {
      for (i = 1; i <= NF; i++) {
        if (match($i, /--end-time=([^ |]+)/, m))  et = m[1]
        if (match($i, /--time-step=([^ |]+)/, m)) ts = m[1]
      }
      print (et == "" ? "" : et) "\t" (ts == "" ? "" : ts); exit
    }' "$f")
  local et="${out%%$'\t'*}"
  local ts="${out##*$'\t'}"
  [[ -z "$et" ]] && et="1.0"
  [[ -z "$ts" ]] && ts="0.01"
  printf '%s\t%s\n' "$et" "$ts"
}

median() {
  # stdin: one float per line.  Outputs the median.
  sort -g | awk '{ a[NR]=$1 } END { n=NR; if (n%2) print a[(n+1)/2]; else print (a[n/2]+a[n/2+1])/2 }'
}

time_one() {
  # Run "$@" $ITERS times; print median wall-clock seconds.
  local times=()
  for ((i = 0; i < ITERS; i++)); do
    local start end
    start=$(date +%s.%N)
    "$@" >/dev/null 2>&1
    end=$(date +%s.%N)
    local dt
    dt=$(awk -v s="$start" -v e="$end" 'BEGIN { printf "%.6f", e - s }')
    times+=("$dt")
    [[ "$VERBOSE" == "1" ]] && echo "    iter $i: ${dt}s" >&2
  done
  printf '%s\n' "${times[@]}" | median
}

speedup() {
  # speedup of $2 vs $1 — bigger means faster.
  awk -v a="$1" -v b="$2" 'BEGIN { if (b == 0) print "NaN"; else printf "%.3f", a / b }'
}

bench_one() {
  local mo="$1"
  local base; base="$(basename "$mo" .mo)"
  local ms; ms="$(extract_model_solver "$mo")"
  local model="${ms%%$'\t'*}"
  local solver="${ms##*$'\t'}"
  if [[ -z "$model" ]]; then
    printf '%s\t-\t-\t-\t-\t-\tno RUN line\n' "$base" >> "$TABLE"
    return
  fi

  local ets; ets="$(extract_end_time_step "$mo")"
  local et="${ets%%$'\t'*}"
  local ts="${ets##*$'\t'}"
  local end_time; end_time="${END_TIME:-$(awk -v et="$et" -v s="$SCALE" 'BEGIN { printf "%g", et*s }')}"
  local time_step; time_step="${TIME_STEP:-$ts}"

  echo "==== $base  (model=$model solver=$solver end=$end_time step=$time_step) ====" >&2

  local extra=()
  if [[ -n "${MARCO_EXTRA:-}" ]]; then
    # shellcheck disable=SC2206
    extra=(${MARCO_EXTRA})
  fi
  local common=(
    --omc-bypass
    --model="$model"
    --solver="$solver"
    "${extra[@]}"
    -L "$RT_LIB"
    -Wl,-rpath,"$RT_LIB"
    "$mo"
  )

  local sim_extra=()
  if [[ -n "${SIM_EXTRA:-}" ]]; then
    # shellcheck disable=SC2206
    sim_extra=(${SIM_EXTRA})
  fi

  local nodr="$WORK/$base.nodr"
  local dr="$WORK/$base.dr"
  local drcm="$WORK/$base.drcm"

  if ! "$MARCO_BIN" "${common[@]}" -o "$nodr" >>"$LOG" 2>&1; then
    printf '%s\t-\t-\t-\t-\t-\tbaseline compile failed\n' "$base" >> "$TABLE"; return
  fi
  if ! "$MARCO_BIN" -fdata-recomputation "${common[@]}" -o "$dr" >>"$LOG" 2>&1; then
    printf '%s\t-\t-\t-\t-\t-\tdr compile failed\n' "$base" >> "$TABLE"; return
  fi
  if ! "$MARCO_BIN" -fdata-recomputation -dr-cost-model "${common[@]}" -o "$drcm" >>"$LOG" 2>&1; then
    printf '%s\t-\t-\t-\t-\t-\tdrcm compile failed\n' "$base" >> "$TABLE"; return
  fi

  if [[ "$SKIP_RUN" == "1" ]]; then
    printf '%s\tcompiled\tcompiled\tcompiled\t-\t-\tcompile-only\n' "$base" >> "$TABLE"
    return
  fi

  local b_t d_t c_t
  if ! b_t=$(time_one "$nodr" --end-time="$end_time" --time-step="$time_step" --precision="$PRECISION" "${sim_extra[@]}"); then
    printf '%s\t-\t-\t-\t-\t-\tbaseline run failed\n' "$base" >> "$TABLE"; return
  fi
  if ! d_t=$(time_one "$dr"   --end-time="$end_time" --time-step="$time_step" --precision="$PRECISION" "${sim_extra[@]}"); then
    printf '%s\t%s\t-\t-\t-\t-\tdr run failed\n' "$base" "$b_t" >> "$TABLE"; return
  fi
  if ! c_t=$(time_one "$drcm" --end-time="$end_time" --time-step="$time_step" --precision="$PRECISION" "${sim_extra[@]}"); then
    printf '%s\t%s\t%s\t-\t-\t-\tdrcm run failed\n' "$base" "$b_t" "$d_t" >> "$TABLE"; return
  fi

  local d_sp c_sp
  d_sp=$(speedup "$b_t" "$d_t")
  c_sp=$(speedup "$b_t" "$c_t")

  printf '%s\t%s\t%s\t%s\t%s\t%s\tok\n' "$base" "$b_t" "$d_t" "$c_t" "$d_sp" "$c_sp" >> "$TABLE"
}

# ----- main -----

if [[ -f "$CORPUS" ]]; then
  MO_FILES=("$CORPUS")
else
  mapfile -t MO_FILES < <(find "$CORPUS" -maxdepth 1 -name '*.mo' | sort)
fi
echo "corpus: $CORPUS  (${#MO_FILES[@]} models, $ITERS iter, scale=${SCALE}x)" >&2

for mo in "${MO_FILES[@]}"; do
  bench_one "$mo"
done

echo
echo "================================================================"
column -ts $'\t' < "$TABLE"
echo "================================================================"
echo "compile log: $LOG"
echo "raw table:   $TABLE"
trap - EXIT
