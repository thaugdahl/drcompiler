# Falcon on ONNX models — the transformer gap has an L1 signature

Question: is the known performance gap versus `onnx-mlir --O3` on transformers
(`ONNX_O3_GAP_OPENAIGPT.md`) a cache-behaviour gap?

Method: run Falcon's `lazystack` (PLDI 2024, see `FALCON_ORACLE_SPIKE.md` for
the tool and its integration) on onnx-mlir affine IR for real models.

## Verdict: yes, and it is an L1 traversal-order gap, not a capacity gap

Cache config `--cs=512 -a 512 --cs=8192 -a 8192 --line-size=64` = 32 KiB L1 /
512 KiB L2, both fully associative.

| model | accesses | misses L1 | L1 rate | misses L2 | L2-of-L1 | analysis |
|---|---:|---:|---:|---:|---:|---:|
| mnist (1x1x28x28) | 5.35e4 | 1,803 | 3.4% | 1,798 | — | 0.02 s |
| **resnet50-v2-7** (1x3x224x224) | 7.961e9 | 2.618e9 | **32.9%** | 1.440e8 | 5.5% | 6.3 s |
| **openaigpt_Opset18** (1x128, d=768, 12L) | 2.247e10 | 1.160e10 | **51.6%** | 6.856e8 | 5.9% | 2.2 s |
| *control:* polybench gemm L | 5.282e9 | 1.659e8 | 3.1% | 1.659e8 | — | 0.03 s |
| *control:* Falcon's own resnet50 (torch-mlir, batch 64) | 1.051e12 | 2.869e11 | 27.3% | 1.105e10 | 3.9% | 3.5 s |

**f32 with 64 B lines means the ideal streaming L1 miss rate is 6.25%** (16
floats per line, one miss per line).  openaigpt sits at **51.6% — 8x worse than
streaming**: most accesses touch a fresh line.  resnet50 at 32.9% is 5x worse.

Crucially, **L2 catches nearly everything that misses L1** (5.5% / 5.9% of L1
misses go on to miss L2).  So the working set is not spilling to DRAM — this is
an **L1 blocking / traversal-order** problem, not a capacity or bandwidth
problem.  That is consistent with the register-blocking-depth account in
`ONNX_O3_GAP_OPENAIGPT.md` §3 and points at the same fix direction.

Both controls behave, so none of this is a tool artefact.

**Cross-validation.** Falcon's own corpus ships a resnet50 built by
*torch-mlir* at batch 64.  Per image that is 1.64e10 accesses against our
7.96e9 from onnx-mlir — only 2.06x apart for two independent compilers, with
onnx-mlir emitting fewer.  The resnet50 figure is therefore trustworthy.

## Getting onnx-mlir IR into lazystack

`lazystack` is built on **LLVM 16**; onnx-mlir emits LLVM 20+.  Exactly two
mechanical incompatibilities, plus one real blocker.

1. **Properties syntax `<{...}>`** (new in LLVM 17) fails to parse:
   ```
   error: expected ':' followed by operation type
       %0 = "krnl.global"() <{name = "constant_1", shape = [128, 196], ...
                            ^
   ```
   Not a krnl-is-unregistered problem — generic-form krnl ops *without*
   properties parse fine as unregistered ops.
2. **`arith.maxnumf`** (LLVM 20 rename of `arith.maxf`):
   `error: custom op 'arith.maxnumf' is unknown`.

`bench/falcon-spike/krnl2memref.py` handles both: rewrites `krnl.global` to
`memref.global "private" constant ... = dense_resource<__elided__>` plus
`memref.get_global` (the form Falcon's own corpus uses), drops
`krnl.entry_point`, and renames `maxnumf`.  Weight elision shrinks resnet50
from 204 MB to 141 KB.

**Hard blocker: dynamic shapes.**  Dynamic-batch IR produces thousands of
`isl_space.c:1515: parameters need to match` and then **SIGSEGV** inside
`CacheModel::computeAccessSets()`.  Minimal repro is 14 lines: a
`memref<?x100xf32>` with `affine.for 0 to %n` where `%n = memref.dim`.  Not
fixable by massaging the input — compile with `--shapeInformation`.

Generation (recipe from `scripts/spmd-resnet50-static.sh`), in the
`onnx-mlir-lean:x86_64` image, and *not* a bottleneck — mnist 0.8 s, resnet50
1.4 s, openaigpt 3.8 s:

```
onnx-mlir --O2 --EmitMLIR --shapeInformation=0:<SHAPE> model.onnx
onnx-mlir-opt --convert-krnl-to-affine model.onnx.mlir
python3 bench/falcon-spike/krnl2memref.py in.mlir > out.mlir
```

Note `scripts/onnx-to-mlir.sh` is the wrong entry point: no
`--shapeInformation`, no krnl→affine step.  Models are on disk at
`/home/tor/Dev/PhD/DRComp/drcc-benchmarks/onnx/models/`.

Ready-to-analyze inputs are committed under `bench/falcon-spike/nn/` (512 KB
total, weights elided); the table above reproduces from them directly.

## Caveats — read before trusting a number

1. **lazystack silently drops accesses it cannot model.**  Non-affine
   `memref.load`/`store`, rank-0 (scalar) memref accesses, and unregistered ops
   are ignored with **no diagnostic**.  Demonstrated: a file with 4
   accesses/iteration x 100 iterations reports `accesses: 100`.  For these
   models the loss is small — resnet50 1 `memref.load`, openaigpt 36
   `krnl.memcpy` (~0.03% of accesses) — but **audit per model, never assume**.
   mnist was hand-verified: 53,514 is exactly the count excluding the 784
   maxpool `memref.load`s and all scalar-alloca traffic.
2. **`gptneox_Opset18` on disk is a toy** — hidden size 32, weights `32x32` /
   `32x96`.  Its numbers (1.7% L1) are worthless for a cache study.  openaigpt
   (d=768, 12 layers, 208 weight tensors) is the genuine transformer.
3. **Falcon predicts misses, not time.**  The JSON `time` field is analysis
   wall-clock; `calculated_serial_time` is a serial-analysis estimate.  Neither
   is a program-runtime prediction — converting misses to cycles needs our own
   latency model.
4. **These are `--O2` + `convert-krnl-to-affine` numbers.**  Attributing the
   `--O3` gap specifically requires the matching `--O3` IR *and* our
   post-codegen IR through the same pipeline.  This run establishes that the
   oracle accepts the input class and that the L1 signature is real — it does
   not yet apportion the gap between the two compilers.

## Next

The obvious follow-up is (4): run the same measurement on the three IRs that
matter — onnx-mlir `--O2`, onnx-mlir `--O3`, and our post-codegen output — and
see which one's L1 miss rate is closest to the 6.25% streaming floor.  That
would convert "the gap has an L1 signature" into "the gap is *here*".
