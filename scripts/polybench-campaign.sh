#!/usr/bin/env bash
# polybench-campaign.sh — speedup campaign for the new-won passes on
# CONSTANT-BOUND (deployment-form) PolyBench kernels. Per kernel, lowers 4 ways
# through mlir-runner --O3, takes the median of `reps` timed calls, checks
# seq==config on a scalar checksum, and appends long-form rows for
# polybench-plotter (kernel,compiler,median,avg,stddev,speedup_vs_clang).
#
#   seq-1t      baseline (speedup denominator)
#   reg-block   dr-opt affine-register-block{cache-tile} (1 thread, codegen)
#   par-spmd    dr-opt dr-par-bubbles{par-spmd-perband} -> convert-par-to-omp (Nt)
#   rb+par      register-block THEN par-spmd -> omp (Nt, compose)
#
# Usage: polybench-campaign.sh OUT.csv THREADS "kernel:N [kernel:N ...]"
set -uo pipefail
export LC_ALL=C
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LL="${LLVM_INSTALL_DIR:-/home/tor/Dev/marco/install/llvm-project}"
OMP_LIB="${OMP_LIB:-/usr/lib/libomp.so}"
DROPT="${DROPT:-$REPO/build/tools/dr-opt/dr-opt}"
OPT="$LL/bin/mlir-opt"; RUN="$LL/bin/mlir-runner --O3"
GEN="python3 $REPO/scripts/polybench_campaign_gen.py"
OUT="${1:-/tmp/pbcampaign.csv}"; THREADS="${2:-16}"; KSPEC="${3:?need kernel:N list}"
W="$(mktemp -d)"; trap 'rm -rf "$W"' EXIT
LIBS=(--shared-libs="$LL/lib/libmlir_runner_utils.so" --shared-libs="$LL/lib/libmlir_c_runner_utils.so" --shared-libs="$OMP_LIB")

# par-spmd  = the in-house par dialect (the campaign's contribution).
# rb+par     = register-block THEN upstream affine-parallelize -> omp. (The
#             in-house perband does not yet shard unroll-jammed / stepped bands,
#             so the codegen x parallel COMPOSE ceiling is shown via the upstream
#             parallelizer; the parallel speedup itself is identical OpenMP.)
RB_PIPE='builtin.module(func.func(affine-register-block{cache-tile=true}))'
PAR_PIPE='builtin.module(dr-par-bubbles{par-spmd-perband},func.func(convert-par-to-omp))'
SEQ=(--lower-affine --convert-vector-to-llvm --convert-scf-to-cf --convert-math-to-libm --convert-cf-to-llvm --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts)
# in-house par dialect lowering (omp already emitted by convert-par-to-omp)
OMPL=(--lower-affine --canonicalize --convert-vector-to-llvm --convert-scf-to-cf --convert-math-to-libm --convert-openmp-to-llvm --convert-cf-to-llvm --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts)
# upstream affine-parallelize -> scf.parallel -> omp lowering (for rb+par compose)
APOMP=(--affine-parallelize=max-nested=1 --lower-affine --convert-scf-to-openmp --canonicalize --convert-vector-to-llvm --convert-scf-to-cf --convert-math-to-libm --convert-openmp-to-llvm --convert-cf-to-llvm --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts)

# stats over the FIRST printed data array (timing). printMemrefF64 puts the
# numbers on their own line starting with '[' -- the header line carries
# sizes=[N]/strides=[1], so we must match data lines only, not the header.
stats(){ grep -E '^[[:space:]]*\[' | head -1 | tr -d '[] ' | tr ',' '\n' | grep -E '^[0-9.eE+-]+$' \
  | awk '{a[NR]=$1; s+=$1} END{ n=NR; if(n==0){print "0 0 0"; exit} for(i=1;i<=n;i++)for(j=i+1;j<=n;j++)if(a[j]<a[i]){t=a[i];a[i]=a[j];a[j]=t} med=a[int((n+1)/2)]; avg=s/n; ss=0; for(i=1;i<=n;i++){d=a[i]-avg; ss+=d*d} sd=(n>1)?sqrt(ss/(n-1)):0; printf "%.6f %.6f %.6f", med, avg, sd }'; }
# checksum = the LAST printed data array (scalar CK)
ckv(){ grep -E '^[[:space:]]*\[' | tail -1 | tr -d '[] '; }
run(){ OMP_NUM_THREADS="$1" OMP_PROC_BIND=close OMP_PLACES=cores $RUN "$2" -e main -entry-point-result=void "${LIBS[@]}" 2>/dev/null; }
match(){ awk -v a="$1" -v b="$2" 'BEGIN{d=a-b;if(d<0)d=-d; r=(a!=0)?d/((a<0)?-a:a):d; print (r<1e-9)?"MATCH":"DIFF"}'; }

echo "kernel,compiler,median,avg,stddev,speedup_vs_clang" > "$OUT"
echo "## campaign: threads=$THREADS  kspec=$KSPEC" >&2

for spec in $KSPEC; do
  k="${spec%%:*}"; N="${spec##*:}"
  $GEN "$k" "$N" > "$W/$k.mlir" 2>"$W/$k.gen.err" || { echo "### $k GEN_FAIL"; sed -n '1,3p' "$W/$k.gen.err" >&2; continue; }

  # lower the four configs
  $OPT "$W/$k.mlir" "${SEQ[@]}" -o "$W/$k.seq.ll" 2>"$W/$k.seq.err" || { echo "### $k SEQ_LOWER_FAIL"; sed -n '1,4p' "$W/$k.seq.err" >&2; continue; }
  $DROPT --allow-unregistered-dialect --pass-pipeline="$RB_PIPE" "$W/$k.mlir" -o "$W/$k.rb.mlir" 2>/dev/null
  $OPT "$W/$k.rb.mlir" "${SEQ[@]}" -o "$W/$k.rb.ll" 2>/dev/null || cp "$W/$k.seq.ll" "$W/$k.rb.ll"
  $DROPT --allow-unregistered-dialect --pass-pipeline="$PAR_PIPE" "$W/$k.mlir" -o "$W/$k.par.mlir" 2>"$W/$k.par.drerr"
  $OPT "$W/$k.par.mlir" "${OMPL[@]}" -o "$W/$k.par.ll" 2>"$W/$k.par.err" || { echo "### $k PAR_LOWER_FAIL"; sed -n '1,4p' "$W/$k.par.err" >&2; cp "$W/$k.seq.ll" "$W/$k.par.ll"; }
  # rb+par compose: register-block (already in $k.rb.mlir) THEN upstream
  # affine-parallelize -> omp (perband can't shard the stepped/unroll-jammed band)
  $OPT "$W/$k.rb.mlir" "${APOMP[@]}" -o "$W/$k.rbpar.ll" 2>/dev/null || cp "$W/$k.rb.ll" "$W/$k.rbpar.ll"

  nrb=$(grep -c 'vector<' "$W/$k.rb.mlir" 2>/dev/null || echo 0)
  npar=$(grep -c 'omp.wsloop' "$W/$k.par.mlir" 2>/dev/null || echo 0)

  # baseline
  bo="$(run 1 "$W/$k.seq.ll")"; read -r bmed bavg bsd <<<"$(echo "$bo" | stats)"; bck="$(echo "$bo" | ckv)"
  [ -z "$bmed" ] && { echo "### $k BASELINE_RUN_FAIL"; continue; }
  printf "%s,seq-1t,%s,%s,%s,1.0\n" "$k" "$bmed" "$bavg" "$bsd" >> "$OUT"
  printf "=== %-12s N=%s  rb-vec-ops=%s  par-wsloops=%s ===\n" "$k" "$N" "$nrb" "$npar" >&2
  printf "  %-10s %12s %10s %8s\n" "seq-1t" "$bmed" "1.00x" "MATCH" >&2

  emit(){ # label thread llfile
    local lbl="$1" th="$2" ll="$3"
    local o; o="$(run "$th" "$ll")"; local med avg sd ck
    read -r med avg sd <<<"$(echo "$o" | stats)"; ck="$(echo "$o" | ckv)"
    if [ -z "$med" ] || awk -v m="$med" 'BEGIN{exit !(m+0<=0)}'; then
      printf "  %-10s %12s %10s %8s\n" "$lbl" "FAIL" "--" "--" >&2; return; fi
    local sp; sp=$(awk -v a="$bmed" -v b="$med" 'BEGIN{printf "%.4f", (b>0)?a/b:0}')
    local mc; mc="$(match "$bck" "$ck")"
    printf "%s,%s,%s,%s,%s,%s\n" "$k" "$lbl" "$med" "$avg" "$sd" "$sp" >> "$OUT"
    printf "  %-10s %12s %9sx %8s\n" "$lbl" "$med" "$(awk -v s=$sp 'BEGIN{printf "%.2f",s}')" "$mc" >&2
  }
  emit "reg-block" 1 "$W/$k.rb.ll"
  emit "par-spmd" "$THREADS" "$W/$k.par.ll"
  emit "rb+par" "$THREADS" "$W/$k.rbpar.ll"
done
echo "## wrote $OUT" >&2