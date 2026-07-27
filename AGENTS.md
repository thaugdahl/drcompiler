# Agent Instructions

<!-- lean-ctx -->
## lean-ctx

Prefer lean-ctx MCP tools over native equivalents for token savings.
Full rules: @LEAN-CTX.md
<!-- /lean-ctx -->

## Benchmarking — drcc-benchmarks ONNX harness

Use the harness in `/home/tor/Dev/PhD/DRComp/drcc-benchmarks` to run ONNX model
benchmarks. The relevant runner is
`drcc-benchmarks/onnx/scripts/onnx-run-bench.sh` (wrapped by
`drcc-benchmarks/onnx/run.sh` for timestamped CSV/meta output). Models live in
`drcc-benchmarks/onnx/models/` (`mnist`, `resnet50-v2-7`, `openaigpt_Opset18`,
`gptneox_Opset18`). It compiles each model under one or more configs, links an
auto-generated timing harness (inputs auto-detected from the ONNX graph, filled
with 1s — safe for openai-gpt: token id 1 valid, attention_mask 1.0 ⇒ no NaN
softmax), and prints median/avg/stddev + a speedup + correctness table.

### Configs

- **DR campaign** (in-image `dr-opt`, 7-step pipeline, clang `-O2` no march):
  `baseline`, `dr-recompute`, `dr-cost`, `dr-partial`, `dr-footprint`.
- **Codegen campaign** (added for the o3 gap analysis — vector-aware lowering +
  `clang -march`): `none`, `codegen`, `o3host`.
  - `none`   — onnx-mlir `--O2 --EmitMLIR`, dr-opt skipped. Fair reference.
  - `codegen`— our `dr-opt` pipeline `func.func(dr-scalar-reduction-demote,
    affine-register-block{mr=8 nr=16 [vl=…] [cpu-cost-model-file=…]},
    dr-scalar-reduction-promote)`. A GEMM-model JSON via `--cost-model` sets
    `hasExplicitGemmModel` ⇒ `canonicalizeAllocaGemm` + gemmBlocking + cache-tile
    (needed for the transformer FFN win; use `onnx-mlir/bench/zen4-gemm.json`).
  - `o3host` — onnx-mlir `--O3 --EmitMLIR` (their optimized krnl path) lowered
    through the SAME backend, dr-opt skipped. The fair "do our transforms beat
    --O3's" number (no backend confound).

### Codegen-campaign gotchas (baked into the script — do not re-discover)

- **HOST dr-opt:** the in-image `dr-opt` (drcc-lean) predates the codegen passes.
  The `codegen` config runs `$DR_OPT_HOST`
  (default `…/drcompiler.git/onnx-mlir/build/tools/dr-opt/dr-opt`, override via
  env). Rebuild it with `ninja -C build dr-opt` before benchmarking.
- **Lowering order:** register-block emits `affine.vector_load`, which
  onnx-mlir's `convert-krnl-to-llvm` cannot lower → these configs run
  `mlir-opt --lower-affine` BEFORE `convert-krnl-to-llvm` (the DR configs do the
  opposite). Branched in `compile_codegen_cfg`.
- **`-march=native`** is THE "vectorization backend enabled" knob (znver4 ⇒
  AVX-512+FMA). Without it clang defaults to generic SSE2. Set via `--march`.
- `LC_ALL=C` is forced (a comma-locale breaks printf/awk on dot-decimals).

### Run it

```bash
cd drcc-benchmarks/onnx/scripts
ONNX_MLIR_IMAGE=onnx-mlir-lean ONNX_MLIR_TAG=x86_64 \
DRCC_IMAGE=drcc-lean DRCC_TAG=x86_64 \
DR_OPT_HOST=/…/onnx-mlir/build/tools/dr-opt/dr-opt \
./onnx-run-bench.sh /…/onnx/models/openaigpt_Opset18.onnx \
  --configs none,codegen,o3host \
  --cost-model /…/onnx-mlir/bench/zen4-gemm.json \
  --iters 15 --warmup 3 --keep-ir --out-dir /tmp/cg-ogpt -v
```

Useful flags: `--keep-ir` (preserve every intermediate `.mlir`/`.ll` under
`<out-dir>/<model>/stage_<cfg>/` for IR inspection), `--vl N` (pin
register-block VL; default from machine model), `--no-recompile`, `--march
ARCH`. Correctness (`norm-rel-err ≤ 1e-4` vs `none`, plus a top1 checksum) is
checked automatically for codegen-campaign configs. Logits dumped to
`stage_<cfg>/logits.txt`.

The legacy single-purpose codegen gate is
`onnx-mlir/scripts/onnx-codegen-bench.sh` (host-tool variant of the same
recipe; same numbers).
