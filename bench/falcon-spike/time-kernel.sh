#!/usr/bin/env bash
# time-kernel.sh -- wall-clock time a bare affine MLIR kernel (no main required).
#
# The kernels in results-shapes/ are Polygeist output: a single
# `func.func @kernel_<name>` with llvm.linkage<external> and no entry point, in
# the LLVM-18 DLTI form. This script:
#   1. rewrites the DLTI spec to the LLVM-22 form (vector<Nxi32> -> vector<Nxi64>)
#   2. splices in a synthesized @drv_main (gen-driver.py) that allocates and
#      initializes the memref arguments, calls the kernel `reps` times with
#      @rtclock around each call, and prints per-rep times plus a checksum
#   3. lowers affine -> scf -> cf -> llvm with mlir-opt
#   4. JITs it with mlir-runner --O3 and reports the MEDIAN per-rep time
#
# Initialization happens inside the rep loop but OUTSIDE the timed region, so
# every rep sees identical inputs and the reported time is the kernel only.
#
# Usage: ./time-kernel.sh [opts] <file.mlir> [kernel_name]
#   -k NAME   kernel function to time (default: the kernel_* / first definition)
#   -r N      repetitions, median reported (default 5)
#   -n N      extent for dynamic '?' memref dims (default: max constant bound)
#   -c        also print the output checksum (for cross-config verification)
#   -a        print all per-rep times instead of just the median
#   -K DIR    keep intermediates in DIR
# Prints the median wall-clock seconds on stdout.

set -uo pipefail
export LC_ALL=C

LL="${LLVM_INSTALL_DIR:-/home/tor/Dev/marco/install/llvm-project}"
OPT="$LL/bin/mlir-opt"
RUN="$LL/bin/mlir-runner"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN="$HERE/gen-driver.py"

KERNEL=""; REPS=5; MAXDIM=0; SHOW_CK=0; SHOW_ALL=0; KEEP=""
while getopts "k:r:n:caK:" o; do
  case "$o" in
    k) KERNEL="$OPTARG" ;;
    r) REPS="$OPTARG" ;;
    n) MAXDIM="$OPTARG" ;;
    c) SHOW_CK=1 ;;
    a) SHOW_ALL=1 ;;
    K) KEEP="$OPTARG" ;;
    *) sed -n '3,25p' "$0" >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
[ $# -ge 1 ] || { sed -n '3,25p' "$0" >&2; exit 2; }
SRC="$1"
# optional positional kernel name (equivalent to -k)
[ $# -ge 2 ] && [ -n "$2" ] && KERNEL="$2"
[ -r "$SRC" ] || { echo "time-kernel: cannot read $SRC" >&2; exit 2; }
for t in "$OPT" "$RUN" "$GEN"; do
  [ -e "$t" ] || { echo "time-kernel: missing $t" >&2; exit 2; }
done

if [ -n "$KEEP" ]; then mkdir -p "$KEEP"; D="$KEEP"; else
  D=$(mktemp -d); trap 'rm -rf "$D"' EXIT; fi

LIBS=(--shared-libs="$LL/lib/libmlir_runner_utils.so"
      --shared-libs="$LL/lib/libmlir_c_runner_utils.so")
LOWER=(--lower-affine --convert-scf-to-cf
       --convert-math-to-llvm --convert-math-to-libm
       --convert-cf-to-llvm --convert-arith-to-llvm
       --finalize-memref-to-llvm --convert-func-to-llvm
       --reconcile-unrealized-casts)

# 1. LLVM-18 -> LLVM-22 DLTI form
sed 's/vector<\([0-9]*\)xi32>/vector<\1xi64>/g' "$SRC" > "$D/in.mlir"

# 2. splice in the synthesized driver
GENARGS=(--reps "$REPS")
[ -n "$KERNEL" ] && GENARGS+=(--kernel "$KERNEL")
[ "$MAXDIM" -gt 0 ] && GENARGS+=(--max-dim "$MAXDIM")
if ! python3 "$GEN" "$D/in.mlir" "${GENARGS[@]}" > "$D/drv.mlir" 2>"$D/gen.err"; then
  echo "time-kernel: driver synthesis failed" >&2; cat "$D/gen.err" >&2; exit 1
fi

# 3. lower to the LLVM dialect
if ! "$OPT" "$D/drv.mlir" "${LOWER[@]}" -o "$D/low.mlir" 2>"$D/opt.err"; then
  echo "time-kernel: lowering failed" >&2; head -20 "$D/opt.err" >&2; exit 1
fi

# 3b. mlir-runner does not register the DLTI dialect, so drop the (already
#     consumed) dlti.dl_spec module attribute. llvm.data_layout / target_triple
#     are kept -- they belong to the LLVM dialect, which mlir-runner does load.
python3 - "$D/low.mlir" "$D/run.mlir" <<'PY'
import re, sys
t = open(sys.argv[1]).read()
key = 'dlti.dl_spec = #dlti.dl_spec<'
i = t.find(key)
if i != -1:
    j, d = i + len(key) - 1, 0
    while j < len(t):
        if t[j] == '<':
            d += 1
        elif t[j] == '>':
            d -= 1
            if d == 0:
                break
        j += 1
    e = j + 1
    if t[e:e + 2] == ', ':
        e += 2
    t = t[:i] + t[e:]
    t = re.sub(r'\{\s*,\s*', '{', t)
    t = re.sub(r',\s*\}', '}', t)
    t = t.replace('module attributes {} {', 'module {')
open(sys.argv[2], 'w').write(t)
PY
[ -s "$D/run.mlir" ] || cp "$D/low.mlir" "$D/run.mlir"

# 4. JIT and run
if ! "$RUN" "$D/run.mlir" --O3 -e drv_main -entry-point-result=void \
     "${LIBS[@]}" > "$D/out.txt" 2>"$D/run.err"; then
  echo "time-kernel: execution failed" >&2; head -20 "$D/run.err" >&2; exit 1
fi

# printMemrefF64 emits a one-line header ending in "data = " followed by the
# values. Skip the header itself (it carries sizes = [..] strides = [..] which
# would otherwise be parsed as data). First block = per-rep times, last = checksum.
mapfile -t TIMES < <(awk '/data =/{g++} g==1 && !/data =/' "$D/out.txt" \
  | grep -oE '\[[^]]*\]' | tr -d '[] ' | tr ',' '\n' \
  | grep -E '^[0-9.eE+-]+$')
CK=$(grep -oE '\[[^]]*\]' "$D/out.txt" | tail -1 | tr -d '[] ')

if [ "${#TIMES[@]}" -eq 0 ]; then
  echo "time-kernel: could not parse timings" >&2
  head -20 "$D/out.txt" >&2; exit 1
fi

MED=$(printf '%s\n' "${TIMES[@]}" | sort -g \
  | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')

if [ "$SHOW_ALL" = 1 ]; then
  echo "reps: ${TIMES[*]}" >&2
fi
if [ "$SHOW_CK" = 1 ]; then
  printf '%s\tchecksum=%s\n' "$MED" "$CK"
else
  printf '%s\n' "$MED"
fi
