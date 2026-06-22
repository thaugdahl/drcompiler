#!/usr/bin/env bash
# polybench-par-bench.sh — measured OpenMP thread-scaling + correctness for the
# representative PolyBench kernels. Generates a self-contained timing+checksum
# driver (polybench_par_driver.py), lowers two ways (sequential / upstream
# affine-parallelize -> scf-to-openmp), runs both through mlir-runner --O3, and
# reports speedup (seq 1t / omp Nt) and a seq==omp checksum check.
# Usage: polybench-par-bench.sh <kernel> [kernel...]   (e.g. gemm 2mm jacobi-2d lu)
set -uo pipefail
export LC_ALL=C
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LL="${LLVM_INSTALL_DIR:-/home/tor/Dev/marco/install/llvm-project}"
OMP_LIB="${OMP_LIB:-/usr/lib/libomp.so}"
OPT="$LL/bin/mlir-opt"; RUN="$LL/bin/mlir-runner --O3"
M="$REPO/bench/polybench-mlir"
GEN="python3 $REPO/scripts/polybench_par_driver.py"
W="$(mktemp -d)"; trap 'rm -rf "$W"' EXIT
LIBS=(--shared-libs="$LL/lib/libmlir_runner_utils.so" --shared-libs="$LL/lib/libmlir_c_runner_utils.so" --shared-libs="$OMP_LIB")
SEQ=(--lower-affine --convert-scf-to-cf --convert-math-to-libm --convert-cf-to-llvm --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts)
OMP=(--affine-parallelize=max-nested=1 --lower-affine --convert-scf-to-openmp --canonicalize --convert-scf-to-cf --convert-math-to-libm --convert-openmp-to-llvm --convert-cf-to-llvm --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts)
median(){ awk '/data =/{g++} g==1' | grep -oE '\[[^]]*\]' | tr -d '[] ' | tr ',' '\n' | grep -E '^[0-9.eE+-]+$' | sort -g | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }
checksum(){ grep -oE '\[[^]]*\]' | tail -1 | tr -d '[] '; }

# per-kernel spec: scalars (value,type in arg order) + memref full shapes + elem + K [+ diag]
spec(){ case "$1" in
  gemm)       echo '{"elem":"f64","K":7,"scalars":[[1000,"i32"],[1100,"i32"],[1200,"i32"],[1.5,"f64"],[1.2,"f64"]],"memrefs":[[1000,1100],[1000,1200],[1200,1100]]}';;
  2mm)        echo '{"elem":"f64","K":7,"scalars":[[800,"i32"],[900,"i32"],[1100,"i32"],[1200,"i32"],[1.5,"f64"],[1.2,"f64"]],"memrefs":[[800,900],[800,1100],[1100,900],[900,1200],[800,1200]]}';;
  syrk)       echo '{"elem":"f64","K":7,"scalars":[[1000,"i32"],[1200,"i32"],[1.5,"f64"],[1.2,"f64"]],"memrefs":[[1200,1200],[1200,1000]]}';;
  covariance) echo '{"elem":"f64","K":5,"scalars":[[1200,"i32"],[1400,"i32"],[1400.0,"f64"]],"memrefs":[[1400,1200],[1200,1200],[1200]]}';;
  doitgen)    echo '{"elem":"f64","K":7,"scalars":[[220,"i32"],[140,"i32"],[160,"i32"]],"memrefs":[[220,140,160],[160,160],[160]]}';;
  jacobi-2d)  echo '{"elem":"f64","K":5,"scalars":[[40,"i32"],[1300,"i32"]],"memrefs":[[1300,1300],[1300,1300]]}';;
  heat-3d)    echo '{"elem":"f64","K":5,"scalars":[[40,"i32"],[120,"i32"]],"memrefs":[[120,120,120],[120,120,120]]}';;
  lu)         echo '{"elem":"f64","K":5,"diag":true,"scalars":[[1000,"i32"]],"memrefs":[[1000,2000]]}';;
  deriche)    echo '{"elem":"f32","K":7,"scalars":[[4096,"i32"],[2160,"i32"],[0.25,"f32"]],"memrefs":[[4096,2160],[4096,2160],[4096,2160],[4096,2160]]}';;
  mvt)        echo '{"elem":"f64","K":7,"scalars":[[2000,"i32"]],"memrefs":[[2000],[2000],[2000],[2000],[2000,2000]]}';;
  atax)       echo '{"elem":"f64","K":7,"scalars":[[1800,"i32"],[2100,"i32"]],"memrefs":[[1800,2100],[2100],[2100],[1800]]}';;
  *) echo ""; return 1;;
 esac; }
declare -A MLIROF=( [gemm]=blas__gemm [2mm]=kernels__2mm [syrk]=blas__syrk [covariance]=datamining__covariance [doitgen]=kernels__doitgen [jacobi-2d]=stencils__jacobi-2d [heat-3d]=stencils__heat-3d [lu]=solvers__lu [deriche]=medley__deriche [mvt]=kernels__mvt [atax]=kernels__atax )

for k in "$@"; do
  sp="$(spec "$k")" || { echo "### $k: no spec"; continue; }
  echo "{\"mlir\":\"$M/${MLIROF[$k]}.mlir\",${sp:1}" > "$W/$k.json"
  $GEN "$W/$k.json" > "$W/$k.mlir" 2>"$W/$k.generr" || { echo "### $k GEN FAIL"; head -3 "$W/$k.generr"; continue; }
  $OPT "$W/$k.mlir" "${SEQ[@]}" -o "$W/$k.s.ll" 2>/dev/null || { echo "### $k SEQ_LOWER_FAIL"; continue; }
  $OPT "$W/$k.mlir" "${OMP[@]}" -o "$W/$k.o.ll" 2>/dev/null || { echo "### $k OMP_LOWER_FAIL"; continue; }
  np=$($OPT "$W/$k.mlir" --affine-parallelize=max-nested=1 -o - 2>/dev/null | grep -c 'affine.parallel')
  so=$(OMP_NUM_THREADS=1 $RUN "$W/$k.s.ll" -e main -entry-point-result=void "${LIBS[@]}" 2>/dev/null)
  st=$(echo "$so"|median); sck=$(echo "$so"|checksum)
  printf "=== %s (par bands=%s, seq O3 1t=%ss) ===\n" "$k" "$np" "$st"
  printf "%-8s %12s %9s %s\n" "threads" "median(s)" "speedup" "correct"
  for t in 1 2 4 8 16; do
    oo=$(OMP_NUM_THREADS=$t OMP_PROC_BIND=close OMP_PLACES=cores $RUN "$W/$k.o.ll" -e main -entry-point-result=void "${LIBS[@]}" 2>/dev/null)
    ot=$(echo "$oo"|median); ock=$(echo "$oo"|checksum)
    sp2=$(awk -v s="$st" -v o="$ot" 'BEGIN{printf (o>0)?"%.2fx":"--",s/o}')
    ok=$(awk -v a="$sck" -v b="$ock" 'BEGIN{d=a-b;if(d<0)d=-d;r=(a!=0)?d/(a<0?-a:a):d;print(r<1e-9)?"MATCH":"DIFF"}')
    printf "%-8s %12s %9s %s\n" "$t" "$ot" "$sp2" "$ok"
  done; echo
done
