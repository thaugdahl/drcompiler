#!/usr/bin/env bash
# polybench-gen-mlir.sh — generate per-kernel affine MLIR for all PolyBench
# kernels via the drcc docker cgeist, cached in bench/polybench-mlir/.
# Each kernel's `static` is stripped so cgeist keeps it as a separate,
# non-inlined external function (clean per-kernel loop nest for analysis).
# Usage: polybench-gen-mlir.sh [kernel-name-filter]
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PB="$REPO/third-party/polybench"; UTIL="$PB/utilities"
OUT="$REPO/bench/polybench-mlir"
IMG="${DRCC_IMAGE:-drcc}:${DRCC_TAG:-x86_64}"
DATASET="${DATASET:-LARGE}"
mkdir -p "$OUT"

NODCE="$OUT/nodce.h"
cat > "$NODCE" <<'EOF'
#include "polybench.h"
#undef polybench_prevent_dce
#define polybench_prevent_dce(func) do {} while(0)
EOF

FILTER="${1:-}"
while IFS= read -r -d '' kfile; do
  kname="$(basename "$(dirname "$kfile")")"
  [[ "$(basename "$kfile" .c)" == "$kname" ]] || continue
  [[ -n "$FILTER" && "$kname" != *"$FILTER"* ]] && continue
  class="$(basename "$(dirname "$(dirname "$kfile")")")"
  kdir="$(dirname "$kfile")"
  kfn="$(grep -hoE 'kernel_[a-z0-9_]+' "$kfile" | head -1)"
  [[ -z "$kfn" ]] && { echo "SKIP $kname"; continue; }
  stage="$OUT/.stage_${kname}"; rm -rf "$stage"; mkdir -p "$stage"
  cp "$kfile" "$stage/"
  perl -0pi -e 's/\bstatic\b(\s*\n?\s*void\s+kernel_)/$1/g' "$stage/$(basename "$kfile")"
  cp "$UTIL/polybench.h" "$stage/"; cp "$NODCE" "$stage/nodce.h"
  for hdr in "$kdir"/*.h; do [[ -f "$hdr" ]] && cp "$hdr" "$stage/"; done
  src="$stage/$(basename "$kfile")"; out="$OUT/${class}__${kname}.mlir"
  docker run --rm --entrypoint bash -v "$stage:$stage" "$IMG" -c "
set -e
/usr/local/bin/cgeist '$src' -S --function='$kfn' --raise-scf-to-affine \
  -target x86_64-linux-gnu -include '$stage/nodce.h' \
  -I '$stage' -DPOLYBENCH_TIME -D${DATASET}_DATASET -O2 -o '$stage/k.mlir' 2>/dev/null
sed -i 's/vector<\([0-9]*\)xi32>/vector<\1xi64>/g' '$stage/k.mlir'
python3 /usr/local/bin/rewrite-struct-memrefs.py '$stage/k.mlir' 2>/dev/null || true
cat '$stage/k.mlir'
" > "$out" 2>/dev/null
  if [[ -s "$out" ]] && grep -q func "$out"; then
    echo "OK   $class/$kname ($kfn, $(grep -c 'affine.for\|scf.for' "$out") loops)"
  else echo "FAIL $class/$kname"; rm -f "$out"; fi
  rm -rf "$stage"
done < <(find "$PB" -name "*.c" ! -path "*/utilities/*" -print0 | sort -z)
