#!/usr/bin/env bash
#===----------------------------------------------------------------------===//
# onnx-run-bench.sh — Benchmark ONNX models with/without DR recomputation.
#
# Compiles each model under multiple configs, links a timing harness,
# and reports median/avg inference latency and speedup tables.
#
# Configs:
#   baseline     — same 7-step pipeline as DR configs, DR pass skipped (fair backend)
#   dr-recompute — DR: data-recomputation{dr-recompute}
#   dr-cost      — DR: data-recomputation{dr-recompute, cost-model}
#   dr-partial   — DR: data-recomputation{dr-recompute, cost-model, partial-remat}
#
# Usage:
#   ./scripts/onnx-run-bench.sh model.onnx [model2.onnx ...] [options]
#   ./scripts/onnx-run-bench.sh --model-dir DIR [options]
#
# Options:
#   --iters N            inference iterations          (default: 100)
#   --warmup N           warmup iterations             (default: 10)
#   --configs LIST       comma-separated configs       (default: baseline,dr-recompute,dr-cost,dr-partial)
#   --ref-cfg NAME       speedup reference             (default: baseline)
#   --cost-model FILE    JSON cost model for dr-cost and dr-partial
#   --input-shape SHAPE  override input shape, e.g. 1,3,224,224
#   --csv FILE           write CSV results
#   --model-dir DIR      scan directory for .onnx files
#   --out-dir DIR        compiled objects + binaries   (default: /tmp/onnx-run-bench)
#   --no-recompile       reuse existing compiled objects
#   -v / --verbose
#
# Environment:
#   ONNX_MLIR_IMAGE  (default: onnx-mlir)  — image for onnx-mlir steps
#   ONNX_MLIR_TAG    (default: latest)
#   DRCC_IMAGE       (default: drcc)        — image for dr-opt / mlir-opt steps
#   DRCC_TAG         (default: latest)
#===----------------------------------------------------------------------===//
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ONNX_MLIR_IMAGE="${ONNX_MLIR_IMAGE:-onnx-mlir}"
ONNX_MLIR_TAG="${ONNX_MLIR_TAG:-latest}"
DRCC_IMAGE="${DRCC_IMAGE:-drcc}"
DRCC_TAG="${DRCC_TAG:-latest}"

ONNX_MLIR_BIN="/build/onnx-mlir/build/Release/bin/onnx-mlir"
ONNX_MLIR_OPT_BIN="/build/onnx-mlir/build/Release/bin/onnx-mlir-opt"
DR_OPT_BIN="/usr/local/bin/dr-opt"
MLIR_OPT_BIN="/opt/llvm/bin/mlir-opt"
MLIR_TRANSLATE_BIN="/opt/llvm/bin/mlir-translate"
CLANG_BIN="/opt/llvm/bin/clang"

ITERS=100
WARMUP=10
CONFIGS="baseline,dr-recompute,dr-cost,dr-partial"
REF_CFG_OVERRIDE=""
COST_MODEL_FILE=""
INPUT_SHAPE_OVERRIDE=""
CSV_FILE=""
MODEL_DIR_ARG=""
OUT_DIR="${HOME}/onnx-bench-staging"
NO_RECOMPILE=0
VERBOSE=0
declare -a MODEL_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --iters)        ITERS="$2";              shift 2 ;;
    --warmup)       WARMUP="$2";            shift 2 ;;
    --configs)      CONFIGS="$2";           shift 2 ;;
    --ref-cfg)      REF_CFG_OVERRIDE="$2";  shift 2 ;;
    --cost-model)   COST_MODEL_FILE="$2";   shift 2 ;;
    --input-shape)  INPUT_SHAPE_OVERRIDE="$2"; shift 2 ;;
    --csv)          CSV_FILE="$2";          shift 2 ;;
    --model-dir)    MODEL_DIR_ARG="$2";     shift 2 ;;
    --out-dir)      OUT_DIR="$2";           shift 2 ;;
    --no-recompile) NO_RECOMPILE=1;         shift   ;;
    -v|--verbose)   VERBOSE=1;              shift   ;;
    -h|--help)
      sed -n '2,/^#===/p' "$0" | sed 's/^# \?//'
      exit 0 ;;
    -*)  echo "onnx-run-bench: unknown option: $1" >&2; exit 1 ;;
    *)   MODEL_ARGS+=("$1");               shift   ;;
  esac
done

log()  { echo "[onnx-run-bench] $*" >&2; }
warn() { echo "[onnx-run-bench] WARN: $*" >&2; }
die()  { echo "onnx-run-bench: error: $*" >&2; exit 1; }

# --- Validate cost model ---
if [[ -n "$COST_MODEL_FILE" ]]; then
  [[ -f "$COST_MODEL_FILE" ]] || die "cost model file not found: $COST_MODEL_FILE"
  COST_MODEL_FILE="$(realpath "$COST_MODEL_FILE")"
fi

# --- Collect models ---
declare -a MODELS=()
if [[ -n "$MODEL_DIR_ARG" ]]; then
  [[ -d "$MODEL_DIR_ARG" ]] || die "model dir not found: $MODEL_DIR_ARG"
  while IFS= read -r -d '' f; do
    MODELS+=("$f")
  done < <(find "$MODEL_DIR_ARG" -maxdepth 1 -name "*.onnx" -print0 | sort -z)
fi
for m in "${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"}"; do
  [[ -f "$m" ]] || die "model not found: $m"
  MODELS+=("$(realpath "$m")")
done
[[ ${#MODELS[@]} -gt 0 ]] || die "No .onnx models specified. Use positional args or --model-dir DIR."

# --- Validate configs ---
declare -A VALID_CFGS
VALID_CFGS["baseline"]=1
VALID_CFGS["dr-recompute"]=1
VALID_CFGS["dr-cost"]=1
VALID_CFGS["dr-partial"]=1

IFS=',' read -ra ACTIVE_CFGS <<< "$CONFIGS"
for cfg in "${ACTIVE_CFGS[@]}"; do
  [[ -v VALID_CFGS["$cfg"] ]] || die "Unknown config: $cfg. Available: baseline,dr-recompute,dr-cost,dr-partial"
done

# --- Validate images ---
docker image inspect "${ONNX_MLIR_IMAGE}:${ONNX_MLIR_TAG}" >/dev/null 2>&1 \
  || die "${ONNX_MLIR_IMAGE}:${ONNX_MLIR_TAG} not found. Pull or build the image first."
docker image inspect "${DRCC_IMAGE}:${DRCC_TAG}" >/dev/null 2>&1 \
  || die "${DRCC_IMAGE}:${DRCC_TAG} not found. Build with: docker/build.sh --arch aarch64"

# --- Work dir ---
WORK_DIR="$(mktemp -d /tmp/onnx-run-bench.XXXXXX)"
DRSUM_DIR="${WORK_DIR}/drsum"
mkdir -p "$DRSUM_DIR" "$OUT_DIR"
trap "rm -rf '$WORK_DIR'" EXIT

# ===----------------------------------------------------------------------===
# _onnx_dtype_to_c  — map ONNX TensorProto.DataType integer to C type string
# ===----------------------------------------------------------------------===
_onnx_dtype_to_c() {
  case "$1" in
    1)  echo "float" ;;
    2)  echo "uint8_t" ;;
    3)  echo "int8_t" ;;
    4)  echo "uint16_t" ;;
    5)  echo "int16_t" ;;
    6)  echo "int32_t" ;;
    7)  echo "int64_t" ;;
    11) echo "double" ;;
    12) echo "uint32_t" ;;
    13) echo "uint64_t" ;;
    *)  echo "float" ;;   # fallback
  esac
}

# ===----------------------------------------------------------------------===
# detect_input_spec  — run python3 inside ONNX_MLIR_IMAGE to get JSON input spec
# Returns JSON array:  [{"dtype": N, "shape": [d0,d1,...]}, ...]
# ===----------------------------------------------------------------------===
detect_input_spec() {
  local model_abs="$1"
  local model_dir; model_dir="$(dirname "$model_abs")"

  docker run --rm \
    -v "${model_dir}:${model_dir}:ro" \
    --entrypoint python3 \
    "${ONNX_MLIR_IMAGE}:${ONNX_MLIR_TAG}" \
    -c "
import onnx, json, sys
m = onnx.load('${model_abs}')
m = onnx.shape_inference.infer_shapes(m)
inputs = []
for inp in m.graph.input:
    t = inp.type.tensor_type
    shape = [max(1, d.dim_value) for d in t.shape.dim]
    inputs.append({'dtype': t.elem_type, 'shape': shape})
print(json.dumps(inputs))
" 2>/dev/null
}

# ===----------------------------------------------------------------------===
# generate_harness  — write a C timing harness to harness_c given input_spec JSON
#
# The generated harness:
#   - allocates input tensors, fills with 1s cast to each input type
#   - calls run_main_graph (the onnx-mlir entry point)
#   - runs WARMUP+ITERS iterations, measures wall time with clock_gettime
#   - prints "MEDIAN MEAN STDDEV" (all in microseconds, 6 decimal places)
# ===----------------------------------------------------------------------===
generate_harness() {
  local harness_c="$1"
  local input_spec="$2"   # JSON string from detect_input_spec
  local mname="$3"
  local iters="$4"
  local warmup="$5"

  # Parse input_spec with python3 to extract dtype/shape per input
  local n_inputs
  n_inputs=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(len(d))" "$input_spec")

  # Build per-input declarations and alloc+free code via python3
  local input_decls alloc_code free_code call_args
  read -r -d '' py_harness_gen <<'PYEOF' || true
import json, sys, math

spec   = json.loads(sys.argv[1])
mname  = sys.argv[2]
iters  = int(sys.argv[3])
warmup = int(sys.argv[4])

dtype_map = {
    1:  "float",    2:  "uint8_t",  3:  "int8_t",
    4:  "uint16_t", 5:  "int16_t",  6:  "int32_t",
    7:  "int64_t",  11: "double",   12: "uint32_t",
    13: "uint64_t",
}

decls = []
allocs = []
frees  = []
args   = []

for i, inp in enumerate(spec):
    ctype  = dtype_map.get(inp["dtype"], "float")
    shape  = inp["shape"]
    nelems = 1
    for d in shape:
        nelems *= max(1, d)
    rank   = len(shape)
    dims_arr = "{" + ",".join(str(d) for d in shape) + "}"
    strides_arr = "{" + ",".join(str(1) for _ in shape) + "}"  # unused but required

    vname = f"inp{i}"
    decls.append(f"  OMTensor *{vname};")
    allocs.append(f"  {{")
    allocs.append(f"    {ctype} *data{i} = ({ctype}*)malloc({nelems} * sizeof({ctype}));")
    allocs.append(f"    if (!data{i}) {{ fprintf(stderr, \"malloc failed\\n\"); return 1; }}")
    allocs.append(f"    for (int64_t k=0; k<{nelems}; k++) data{i}[k] = ({ctype})1;")
    allocs.append(f"    int64_t shape{i}[] = {dims_arr};")
    allocs.append(f"    {vname} = omTensorCreate(data{i}, shape{i}, {rank}, ONNX_TYPE_{'FLOAT' if inp['dtype']==1 else 'INT32' if inp['dtype']==6 else 'INT64' if inp['dtype']==7 else 'DOUBLE' if inp['dtype']==11 else 'UINT8' if inp['dtype']==2 else 'INT8' if inp['dtype']==3 else 'UINT16' if inp['dtype']==4 else 'INT16' if inp['dtype']==5 else 'UINT32' if inp['dtype']==12 else 'UINT64' if inp['dtype']==13 else 'FLOAT'});")
    allocs.append(f"  }}")
    frees.append(f"  free(omTensorGetDataPtr({vname}));")
    frees.append(f"  omTensorDestroy({vname});")
    args.append(vname)

n = len(spec)
args_list = ", ".join(args)

total = warmup + iters

print(f"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include "OnnxMlirRuntime.h"

OMTensorList *run_main_graph(OMTensorList *);

static double now_us(void) {{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}}

static int cmp_double(const void *a, const void *b) {{
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}}

int main(void) {{
    /* Tensor pointer declarations */
{chr(10).join('  ' + l.strip() for l in decls)}
    /* Allocate input tensors */
{chr(10).join('  ' + l.strip() for l in allocs)}

    OMTensor *inputs[] = {{ {args_list} }};
    OMTensorList *input_list = omTensorListCreate(inputs, {n});

    int total = {total};
    int warmup = {warmup};
    int iters  = {iters};
    double *times = (double*)malloc(iters * sizeof(double));
    if (!times) {{ fprintf(stderr, "malloc times failed\\n"); return 1; }}

    for (int i = 0; i < total; i++) {{
        double t0 = now_us();
        OMTensorList *out = run_main_graph(input_list);
        double t1 = now_us();
        if (!out) {{ fprintf(stderr, "inference returned NULL\\n"); return 1; }}
        omTensorListDestroy(out);
        if (i >= warmup) times[i - warmup] = t1 - t0;
    }}

    /* Statistics */
    qsort(times, iters, sizeof(double), cmp_double);
    double median = (iters % 2 == 1)
        ? times[iters / 2]
        : (times[iters/2 - 1] + times[iters/2]) / 2.0;
    double sum = 0.0;
    for (int i = 0; i < iters; i++) sum += times[i];
    double mean = sum / iters;
    double var  = 0.0;
    for (int i = 0; i < iters; i++) {{ double d = times[i] - mean; var += d*d; }}
    double stddev = (iters > 1) ? sqrt(var / (iters - 1)) : 0.0;

    printf("%.6f %.6f %.6f\\n", median, mean, stddev);

    free(times);
    omTensorListDestroy(input_list);
    return 0;
}}
""")
PYEOF

  python3 -c "$py_harness_gen" "$input_spec" "$mname" "$iters" "$warmup" > "$harness_c" \
    || { warn "harness generation failed for $mname"; return 1; }
}

# ===----------------------------------------------------------------------===
# compile_model  — compile one ONNX model under one config
#
# args: $1=model_abs  $2=cfg  $3=outdir
# echoes path to linked binary on success, nothing on failure
# ===----------------------------------------------------------------------===
compile_model() {
  local model_abs="$1" cfg="$2" outdir="$3"
  local mname; mname="$(basename "$model_abs" .onnx)"
  local model_dir; model_dir="$(dirname "$model_abs")"
  local cfg_safe; cfg_safe="${cfg//[^a-zA-Z0-9_-]/_}"
  local staging="${outdir}/stage_${cfg_safe}"
  local binary="${outdir}/${cfg_safe}.bin"

  # Reuse existing binary if --no-recompile
  if [[ "$NO_RECOMPILE" -eq 1 && -x "$binary" ]]; then
    [[ "$VERBOSE" -eq 1 ]] && log "  reusing ${mname}/${cfg}"
    echo "$binary"
    return
  fi

  mkdir -p "$staging"

  # ---------------------------------------------------------- all configs (7-step)
  # baseline skips step 3 (DR pass) — identical backend for fair comparison.
  local DR_PIPELINE=""
  case "$cfg" in
    baseline)     DR_PIPELINE="" ;;
    dr-recompute) DR_PIPELINE="data-recomputation{dr-recompute=true dr-summary=true}" ;;
    dr-cost)      DR_PIPELINE="data-recomputation{dr-recompute=true dr-cost-model=true dr-summary=true}" ;;
    dr-partial)   DR_PIPELINE="data-recomputation{dr-recompute=true dr-cost-model=true dr-partial-remat=true dr-summary=true}" ;;
  esac

  # Inject cost model file path for configs that use it
  if [[ -n "$COST_MODEL_FILE" ]] && [[ "$cfg" == "dr-cost" || "$cfg" == "dr-partial" ]]; then
    local cost_in_staging="${staging}/cost_model.json"
    cp "$COST_MODEL_FILE" "$cost_in_staging"
    DR_PIPELINE="${DR_PIPELINE%\}}"
    DR_PIPELINE="${DR_PIPELINE} cpu-cost-model-file=${cost_in_staging}}"
  fi

  # Intermediate file paths
  local raw_mlir="${staging}/${mname}.onnx.mlir"
  local affine_mlir="${staging}/${mname}.affine.mlir"
  local dr_mlir="${staging}/${mname}.dr.mlir"
  local krnl_llvm_mlir="${staging}/${mname}.krnl-llvm.mlir"
  local llvm_mlir="${staging}/${mname}.llvm.mlir"
  local llvm_ir="${staging}/${mname}.ll"
  local obj="${staging}/${mname}.o"

  # Step 1: ONNX → Krnl/Affine MLIR
  [[ "$VERBOSE" -eq 1 ]] && log "  ${mname}/${cfg}: step 1 — onnx-mlir --EmitMLIR"
  docker run --rm \
    -v "${model_dir}:${model_dir}:ro" \
    -v "${staging}:${staging}" \
    --entrypoint "${ONNX_MLIR_BIN}" \
    "${ONNX_MLIR_IMAGE}:${ONNX_MLIR_TAG}" \
    --O2 --EmitMLIR -o "${raw_mlir%.onnx.mlir}" "$model_abs" \
    >/dev/null 2>&1 \
    || { warn "${mname}/${cfg}: step 1 (EmitMLIR) failed"; return; }

  [[ -f "$raw_mlir" ]] || { warn "${mname}/${cfg}: step 1 produced no output (expected ${raw_mlir})"; return; }

  # Step 2: Krnl loops → Affine
  [[ "$VERBOSE" -eq 1 ]] && log "  ${mname}/${cfg}: step 2 — convert-krnl-to-affine"
  docker run --rm \
    -v "${staging}:${staging}" \
    --entrypoint "${ONNX_MLIR_OPT_BIN}" \
    "${ONNX_MLIR_IMAGE}:${ONNX_MLIR_TAG}" \
    --convert-krnl-to-affine "$raw_mlir" -o "$affine_mlir" \
    >/dev/null 2>&1 \
    || { warn "${mname}/${cfg}: step 2 (convert-krnl-to-affine) failed"; return; }
  rm -f "$raw_mlir"

  # Step 3: DataRecomputation (dr-opt) — skipped for baseline (copy through)
  if [[ "$cfg" == "baseline" ]]; then
    cp "$affine_mlir" "$dr_mlir"
  else
    [[ "$VERBOSE" -eq 1 ]] && log "  ${mname}/${cfg}: step 3 — dr-opt ${DR_PIPELINE}"
    docker run --rm \
      -v "${staging}:${staging}" \
      --entrypoint "${DR_OPT_BIN}" \
      "${DRCC_IMAGE}:${DRCC_TAG}" \
      --allow-unregistered-dialect \
      "--pass-pipeline=builtin.module(${DR_PIPELINE})" \
      "$affine_mlir" -o "$dr_mlir" \
      >/dev/null 2>"${staging}/drsum.log" \
      || { warn "${mname}/${cfg}: step 3 (dr-opt) failed"; return; }

    if [[ -f "${staging}/drsum.log" ]]; then
      mkdir -p "${DRSUM_DIR}/${cfg}"
      cp "${staging}/drsum.log" "${DRSUM_DIR}/${cfg}/${mname}.log"
    fi
  fi
  rm -f "$affine_mlir"

  # Step 4: krnl.global / krnl.entry_point → LLVM dialect
  [[ "$VERBOSE" -eq 1 ]] && log "  ${mname}/${cfg}: step 4 — convert-krnl-to-llvm"
  docker run --rm \
    -v "${staging}:${staging}" \
    --entrypoint "${ONNX_MLIR_OPT_BIN}" \
    "${ONNX_MLIR_IMAGE}:${ONNX_MLIR_TAG}" \
    --convert-krnl-to-llvm "$dr_mlir" -o "$krnl_llvm_mlir" \
    >/dev/null 2>&1 \
    || { warn "${mname}/${cfg}: step 4 (convert-krnl-to-llvm) failed"; return; }
  rm -f "$dr_mlir"

  # Step 5: Affine/SCF/CF/MemRef → LLVM dialect
  local LOWER_PIPELINE="builtin.module(lower-affine,convert-scf-to-cf,convert-cf-to-llvm,convert-func-to-llvm,finalize-memref-to-llvm,reconcile-unrealized-casts)"
  [[ "$VERBOSE" -eq 1 ]] && log "  ${mname}/${cfg}: step 5 — mlir-opt lowering"
  docker run --rm \
    -v "${staging}:${staging}" \
    --entrypoint "${MLIR_OPT_BIN}" \
    "${DRCC_IMAGE}:${DRCC_TAG}" \
    --allow-unregistered-dialect \
    "--pass-pipeline=${LOWER_PIPELINE}" \
    "$krnl_llvm_mlir" -o "$llvm_mlir" \
    >/dev/null 2>&1 \
    || { warn "${mname}/${cfg}: step 5 (mlir-opt lowering) failed"; return; }
  rm -f "$krnl_llvm_mlir"

  # Step 6: LLVM dialect → LLVM IR
  [[ "$VERBOSE" -eq 1 ]] && log "  ${mname}/${cfg}: step 6 — mlir-translate"
  docker run --rm \
    -v "${staging}:${staging}" \
    --entrypoint "${MLIR_TRANSLATE_BIN}" \
    "${DRCC_IMAGE}:${DRCC_TAG}" \
    --mlir-to-llvmir "$llvm_mlir" -o "$llvm_ir" \
    >/dev/null 2>&1 \
    || { warn "${mname}/${cfg}: step 6 (mlir-translate) failed"; return; }
  rm -f "$llvm_mlir"

  # Step 7: LLVM IR → object file
  [[ "$VERBOSE" -eq 1 ]] && log "  ${mname}/${cfg}: step 7 — clang -O2 -c"
  docker run --rm \
    -v "${staging}:${staging}" \
    --entrypoint "${CLANG_BIN}" \
    "${DRCC_IMAGE}:${DRCC_TAG}" \
    -O2 -c "$llvm_ir" -o "$obj" \
    >/dev/null 2>&1 \
    || { warn "${mname}/${cfg}: step 7 (clang -c) failed"; return; }
  rm -f "$llvm_ir"

  [[ -f "$obj" ]] || { warn "${mname}/${cfg}: step 7 produced no .o"; return; }

  _link_harness "$model_abs" "$obj" "$staging" "$binary" "$mname" \
    || return
  rm -f "$obj"
  echo "$binary"
}

# ===----------------------------------------------------------------------===
# _link_harness  — generate C harness, compile+link, produce runnable binary
# args: model_abs obj staging binary mname
# ===----------------------------------------------------------------------===
_link_harness() {
  local model_abs="$1" obj="$2" staging="$3" binary="$4" mname="$5"

  # Detect input shapes
  [[ "$VERBOSE" -eq 1 ]] && log "  ${mname}: detecting input spec..."
  local input_spec
  input_spec="$(detect_input_spec "$model_abs")" \
    || { warn "${mname}: input spec detection failed"; return 1; }
  [[ -n "$input_spec" && "$input_spec" != "[]" ]] \
    || { warn "${mname}: empty input spec from model"; return 1; }

  # Apply shape override (replace first input's shape if specified)
  if [[ -n "$INPUT_SHAPE_OVERRIDE" ]]; then
    input_spec="$(python3 -c "
import json,sys
spec = json.loads(sys.argv[1])
override = [int(x) for x in sys.argv[2].split(',')]
if spec:
    spec[0]['shape'] = override
print(json.dumps(spec))
" "$input_spec" "$INPUT_SHAPE_OVERRIDE")"
  fi

  # Generate C harness
  local harness_c="${staging}/harness.c"
  generate_harness "$harness_c" "$input_spec" "$mname" "$ITERS" "$WARMUP" \
    || { warn "${mname}: harness generation failed"; return 1; }

  # Compile harness + link against obj and libcruntime.a (inside onnx-mlir-lean image)
  [[ "$VERBOSE" -eq 1 ]] && log "  ${mname}: linking harness..."
  local outdir; outdir="$(dirname "$staging")"
  docker run --rm \
    -v "${outdir}:${outdir}" \
    --entrypoint gcc \
    "${ONNX_MLIR_IMAGE}:${ONNX_MLIR_TAG}" \
    -O2 \
    -I/build/onnx-mlir/include \
    "$harness_c" "$obj" \
    /build/onnx-mlir/build/Release/lib/libcruntime.a \
    -lm -lpthread -lstdc++ \
    -o "$binary" \
    >/dev/null 2>&1 \
    || { warn "${mname}: harness link failed"; return 1; }

  [[ -x "$binary" ]] || { warn "${mname}: binary not executable after link"; return 1; }
}

# ===----------------------------------------------------------------------===
# time_binary  — run binary, collect timing samples
# echoes "MEDIAN MEAN STDDEV" (microseconds, 6 decimal places) or "ERR ERR ERR"
# ===----------------------------------------------------------------------===
time_binary() {
  local binary="$1"
  local result
  result="$("$binary" 2>/dev/null)" || { echo "ERR ERR ERR"; return; }
  local med avg std
  read -r med avg std <<< "$result"
  # Validate: must be numeric
  if ! [[ "$med" =~ ^[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?$ ]]; then
    echo "ERR ERR ERR"; return
  fi
  echo "$med $avg $std"
}

# ===----------------------------------------------------------------------===
# Table helpers
# ===----------------------------------------------------------------------===
_print_table_header() {
  local label="$1"
  printf "\n=== ONNX Inference — iters=%s  [%s us] ===\n\n" "$ITERS" "$label"
  printf "%-22s" "Model"
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
    "Model" "accepted" "rej:type" "rej:unsafe" "rej:cost" "skipped"
  printf "%-22s %9s %9s %9s %9s %9s\n" \
    "----------------------" "---------" "---------" "----------" "---------" "---------"

  local total_acc=0 total_type=0 total_unsafe=0 total_cost=0 total_skip=0
  for logf in "${dir}/"*.log; do
    local mname="${logf##*/}"; mname="${mname%.log}"
    local acc type unsafe cost skip
    read -r acc type unsafe cost skip < <(awk '
      /DRSUM: load .* ACCEPT/           { acc++ }
      /REJECT_TYPE/                     { type++ }
      /REJECT_UNSAFE/                   { unsafe++ }
      /REJECT_COST/                     { cost++ }
      /SKIP_COST|SKIP_AMBIGUOUS/        { skip++ }
      END { printf "%d %d %d %d %d\n", acc+0, type+0, unsafe+0, cost+0, skip+0 }
    ' "$logf")
    printf "%-22s %9d %9d %9d %9d %9d\n" \
      "$mname" "$acc" "$type" "$unsafe" "$cost" "$skip"
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

# ===----------------------------------------------------------------------===
# Main benchmark loop
# ===----------------------------------------------------------------------===
log "${#MODELS[@]} model(s); iters=${ITERS}, warmup=${WARMUP}"

# Speedup reference resolution: explicit --ref-cfg > first config > baseline
if [[ -n "$REF_CFG_OVERRIDE" ]]; then
  [[ -v VALID_CFGS["$REF_CFG_OVERRIDE"] ]] \
    || die "Unknown --ref-cfg: $REF_CFG_OVERRIDE"
  REF_CFG="$REF_CFG_OVERRIDE"
else
  REF_CFG="baseline"
  _found_ref=0
  for _c in "${ACTIVE_CFGS[@]}"; do
    [[ "$_c" == "baseline" ]] && { _found_ref=1; break; }
  done
  if [[ "$_found_ref" -eq 0 ]]; then
    REF_CFG="${ACTIVE_CFGS[0]}"
  fi
fi
last_cfg="${ACTIVE_CFGS[-1]}"

# CSV header
if [[ -n "$CSV_FILE" ]]; then
  { printf "model"
    for cfg in "${ACTIVE_CFGS[@]}"; do
      printf ",%s_median_us,%s_avg_us,%s_stddev_us" "$cfg" "$cfg" "$cfg"
    done
    printf "\n"
  } > "$CSV_FILE"
fi

declare -A ALL_MED ALL_AVG ALL_STD

# Per-model compile + time
for model_abs in "${MODELS[@]}"; do
  mname="$(basename "$model_abs" .onnx)"
  mdir="${OUT_DIR}/${mname}"
  mkdir -p "$mdir"
  log "model: ${mname}"

  for cfg in "${ACTIVE_CFGS[@]}"; do
    binary="$(compile_model "$model_abs" "$cfg" "$mdir")" || binary=""
    if [[ -z "$binary" || ! -x "$binary" ]]; then
      ALL_MED["${mname}:${cfg}"]="FAIL"
      ALL_AVG["${mname}:${cfg}"]="FAIL"
      ALL_STD["${mname}:${cfg}"]="FAIL"
      continue
    fi
    read -r med avg std < <(time_binary "$binary")
    ALL_MED["${mname}:${cfg}"]="$med"
    ALL_AVG["${mname}:${cfg}"]="$avg"
    ALL_STD["${mname}:${cfg}"]="$std"
  done

  # CSV row
  if [[ -n "$CSV_FILE" ]]; then
    { printf "%s" "$mname"
      for cfg in "${ACTIVE_CFGS[@]}"; do
        printf ",%s,%s,%s" \
          "${ALL_MED[${mname}:${cfg}]:-FAIL}" \
          "${ALL_AVG[${mname}:${cfg}]:-FAIL}" \
          "${ALL_STD[${mname}:${cfg}]:-FAIL}"
      done
      printf "\n"
    } >> "$CSV_FILE"
  fi
done

# ===----------------------------------------------------------------------===
# Print tables
# ===----------------------------------------------------------------------===

# --- Table 1: Median ---
_print_table_header "MEDIAN"
for model_abs in "${MODELS[@]}"; do
  mname="$(basename "$model_abs" .onnx)"
  printf "%-22s" "$mname"
  for cfg in "${ACTIVE_CFGS[@]}"; do
    val="${ALL_MED[${mname}:${cfg}]:-FAIL}"
    if [[ "$val" == FAIL || "$val" == ERR ]]; then
      printf "%14s" "$val"
    else
      printf "%14.2f" "$val"
    fi
  done
  ref="${ALL_MED[${mname}:${REF_CFG}]:-}"
  val="${ALL_MED[${mname}:${last_cfg}]:-FAIL}"
  [[ "$last_cfg" != "$REF_CFG" ]] \
    && printf "%10s\n" "$(_speedup "$ref" "$val")" \
    || printf "%10s\n" "—"
done

# --- Table 2: Average ---
_print_table_header "AVERAGE"
for model_abs in "${MODELS[@]}"; do
  mname="$(basename "$model_abs" .onnx)"
  printf "%-22s" "$mname"
  for cfg in "${ACTIVE_CFGS[@]}"; do
    val="${ALL_AVG[${mname}:${cfg}]:-FAIL}"
    if [[ "$val" == FAIL || "$val" == ERR ]]; then
      printf "%14s" "$val"
    else
      printf "%14.2f" "$val"
    fi
  done
  ref="${ALL_AVG[${mname}:${REF_CFG}]:-}"
  val="${ALL_AVG[${mname}:${last_cfg}]:-FAIL}"
  [[ "$last_cfg" != "$REF_CFG" ]] \
    && printf "%10s\n" "$(_speedup "$ref" "$val")" \
    || printf "%10s\n" "—"
done

# --- Table 3: Signal-to-noise ---
if [[ "$last_cfg" != "$REF_CFG" ]]; then
  printf "\n=== Signal-to-Noise (avg speedup vs pooled σ) — %s vs %s ===\n\n" \
    "$REF_CFG" "$last_cfg"
  printf "%-22s %12s %12s %12s %12s %8s\n" \
    "Model" "${REF_CFG}-avg" "cfg-avg" "delta" "pooled-σ" "SNR"
  printf "%-22s %12s %12s %12s %12s %8s\n" \
    "----------------------" "------------" "------------" "------------" "------------" "--------"
  for model_abs in "${MODELS[@]}"; do
    mname="$(basename "$model_abs" .onnx)"
    ref_avg="${ALL_AVG[${mname}:${REF_CFG}]:-FAIL}"
    cfg_avg="${ALL_AVG[${mname}:${last_cfg}]:-FAIL}"
    ref_std="${ALL_STD[${mname}:${REF_CFG}]:-FAIL}"
    cfg_std="${ALL_STD[${mname}:${last_cfg}]:-FAIL}"
    if [[ "$ref_avg" == FAIL || "$cfg_avg" == FAIL ]]; then
      printf "%-22s %12s\n" "$mname" "FAIL"
      continue
    fi
    awk -v mname="$mname" \
        -v ra="$ref_avg" -v ca="$cfg_avg" \
        -v rs="$ref_std" -v cs="$cfg_std" '
    BEGIN {
      delta  = ra - ca
      pooled = sqrt(rs*rs + cs*cs)
      snr    = (pooled > 0) ? delta/pooled : 999
      flag   = (snr > 2) ? "SIGNAL" : (snr < -2) ? "REGRESS" : "noise"
      printf "%-22s %12.2f %12.2f %+12.2f %12.2f %8.2f  %s\n",
        mname, ra, ca, delta, pooled, snr, flag
    }'
  done
fi

# --- DR summary tables ---
for _cfg in "${ACTIVE_CFGS[@]}"; do
  [[ "$_cfg" != "baseline" ]] && _print_drsum_table "$_cfg"
done

echo ""
[[ -n "$CSV_FILE" ]] && log "CSV written: $CSV_FILE"
log "Done."
