# ONNX_CODEGEN_FINDINGS — campaign results

Companion to ONNX_CODEGEN_SPEC.md. Spike-first, honest verdicts. One local
commit per WP (`onnx_codegen_N`). NEVER pushed.

---

## WP-O1 — Reduction de-promotion (the keystone): **WIN**, committed `onnx_codegen_1`

### Spike (go/no-go for the whole campaign): PASS, decisively

Hand-built a 1x1-conv GEMM (M=oc=256, N=p=196, K=ic=256) in onnx-mlir's exact
emitted shape (SSA `affine.for ... iter_args`, batch trip-1 promoted to const 0)
and three lowerings, timed (best-of-5, REPS=2000, marco LLVM 22 `clang -O2`):

| variant | time | vs onnx-mlir original |
|---|---|---|
| iter_args original (onnx-mlir form) | 12.40 s | 1.00x |
| de-promoted, NO register-block (LLVM SLP) | 13.30 s | 0.93x (slower) |
| **de-promoted + register-block vector kernel** | **1.31 s** | **9.4x** |

All 15 runs checksum-identical (`-154.283519`) → bit-correct.
**Risk R1 killed**: LLVM does NOT already vectorize the iter_args inner-product
form, and de-promotion *alone* does nothing — only the explicit broadcast vector
micro-kernel wins, and it wins ~9.4x on a layer class that is ~45-50% of
resnet50's conv FLOPs.

### The design fact the spike forced

A naive in-place de-promotion (`store 0, %C` then `for k { load %C; fma; store
%C }`) reaches only the SCALAR register-block path (mr*nr scalar accumulators +
LLVM SLP), 0 vector ops. Cause: the vectorizer's perfect-body check
(`vectorizeBroadcastBand`, Vectorize.cpp) requires the inner-spatial loop body
to contain ONLY the reduction loop + pure ops; a leading `affine.store` (the
zero-init) has a memory effect and bails it. **The zero-init must live in its
OWN spatial nest** (the cgeist -O0 shape, where `%C` is pre-zeroed elsewhere).
`dr-affine-loop-distribute` does NOT do this fission (it needs ≥2 sibling
*loops*; an init-store + one loop isn't that), so the demote pass fissions the
init nest itself.

### The pass: `dr-scalar-reduction-demote`

`lib/Transforms/ScalarReductionDemote.cpp` (+ td/header/CMake). v1 contract:
single-level f32/f64 `addf` reduction, 1 iter_arg, result stored DIRECTLY,
accumulator address independent of the reduction IV, enclosing spatial band
perfect. Emits: (1) a fresh init nest mirroring the spatial loops that stores
the seed at the accumulator subscript; (2) the reduction band rewritten to a
memref accumulator (load %C / fma / store %C, no iter_args). Bails (leaves
intact) on: nested conv reductions (3x3 — inner addf feeds an outer yield, not a
store), `maxnumf` pool/softmax chains, bias-epilogue stores (Case B, deferred),
multiple iter_args. lit: `test/DataRecomputation/scalar-reduction-demote.mlir`.

### resnet50 coverage

Demotes **33 of 93** iter_args reductions (the single-level 1x1 GEMMs); the
remaining 60 are nested 3x3-conv reductions (WP-O2) + `maxnumf` pools (out of
scope) — all correctly left intact. Standalone correctness re-verified through
the pass-driven pipeline: 12.33 s → 1.31 s, checksum-identical.

---

## Per-band family selection (WP-O4.1, landed with O1 — it was load-bearing)

### Bug found while wiring O1 onto real resnet50

After demote, register-block gave **0 vector ops on resnet50** even though 33
perfect GEMM bands were present. Root cause: `familySelect` in
AffineRegisterBlock.cpp sets a single function-wide `anyDot` flag — if ANY
reduction in the function is classified Dot (rank-k), the WHOLE function switches
to dot mode (scalar 4x4 tile + reassoc), disabling the broadcast vector kernel
for every band. One PolyBench kernel = one family, so the global flag was fine
there; onnx-mlir emits 33+ contractions per function, and a few mis/genuinely
classify as Dot → all 33 fell to the scalar tile. (`family-select=false` forced
broadcast and vectorized all 33: 264 broadcasts — confirming the diagnosis.)

### Fix: per-band family in Stage 3

Family is now detected per band (from the reduction under each `sOut`, before
jamming) and drives that band's jam factor + broadcast-vs-dot kernel choice. For
a single-kernel function this is identical to the old global decision (all bands
share a family), so **PolyBench is unaffected** — full lit 210 pass / 0 fail,
every register-block test (gemm/syrk-family/bmm/blas2) byte-identical vector
counts. The global mode is retained only for the triangular-peel stages and
function-wide fast-math (no-op / tolerable on onnx; correct on single-kernel
PolyBench).

Result: resnet50 now vectorizes the broadcast main loops of **all 33** demoted
bands (136 `vector.broadcast` = 17 main loops × mr 8), up from 0.

### The "17 of 33" is not a coverage gap (instrumented to be sure)

Tagging every family verdict and every `vectorizeBroadcastBand` exit on resnet50:
**all 33 demoted reductions classify BROADCAST (0 Dot)** — detectFamily is not
misclassifying anything. 17 vectorize their main loop; the other 16 exit at
`trip < VL` with **trip = 4** — these are the **vl-remainder tails of the N=196
layers** (196 mod 16 = 4), which the vectorizer deliberately peels off and
leaves scalar (4 of every 196 columns ≈ 2 % of those layers' work). LLVM masks
that tail. So there is no misclassification and no real coverage loss — every
demoted GEMM's bulk vectorizes. (An earlier draft of this doc wrongly guessed
the 16 were Dot-misclassified channels-last layouts; the instrumentation refuted
it. No axis-selection fix is needed.)

---

## WP-O2 — Direct-conv register-block (the 3x3 convs): spike PASS, building

### Spike gate (≥2x required before any matcher work): PASS, 11.7x

Hand-wrote two MLIR kernels for a representative resnet50 interior 3x3 layer
(oc=64, oh=56, ow=64, ic=64, constant bounds — the padded border excluded):
(A) scalar de-promoted memref form (what LLVM -O2 autovectorizes); (B) an
ow-vectorized micro-kernel (vectorize ow by VL=16, broadcast `w[oc,ic,kh,kw]`,
stream `in[ic,oh+kh,ow+kw:+16]` as a contiguous `affine.vector_load`, carry a
`vector<16>` accumulator through the ic/kh/kw band). Anti-hoist dependency in
the driver (each rep perturbs `in` from the prior `Y` — else the REPS loop is
hoisted to one call), REPS=100, best-of-5:

| variant | per-call | GFLOP/s |
|---|---|---|
| scalar interior (LLVM -O2) | 72.6 ms | 3.6 |
| **ow-vectorized micro-kernel** | **6.2 ms** | **42.6** |

**11.7x, checksums identical** (264 MFLOP/call). The kernel is the 1x1 broadcast
kernel's shape — broadcast the weight, stream the stride-1 spatial operand —
but over a 3-loop reduction *band* (ic/kh/kw) instead of a single k. Well above
the gate; im2col (the non-goal fallback) is not needed.

### Landed

**Part 1 — nested-band de-promotion** (commit `onnx_codegen_2a`).
`dr-scalar-reduction-demote` generalized from a single-level reduction to a
nested add-reduction BAND (ic→kh→kw threading one accumulator). The matcher
descends the band; the rewrite rebuilds every level as a plain loop with the
accumulator in memory + the separate zero-init nest. On the interior 3x3 spike
the demoted form is byte-for-byte the hand-written scalar kernel and runs
checksum-identical. Single-level (1x1/GEMM) unchanged.

**Part 2 — direct-conv band vectorizer** (commit `onnx_codegen_2b`).
New Stage 1d in affine-register-block detects the conv band (innermost
accumulator loop grown upward through reduction loops to the spatial loop ow)
and `vectorizeConvBand` re-steps ow by VL, rebuilding the band with a
`vector<VL>` accumulator: weight→broadcast, input (stride-1 in ow)→vector load,
accumulator→vector iter_arg. `innermostStrideOne` generalized to "coefficient of
the IV in the innermost result == 1" (so `in[..][ow+kw]` is stride-1 in ow) — a
strict superset, full lit unchanged. **Measured 11.4x** on the interior spike
(matches the hand-written 11.7x), max rel-err 1e-6, lit 211/0, PolyBench
byte-identical, resnet50 1x1 path unchanged.

### Part 3 (remaining) — make it fire on real resnet50 3x3

The interior spike works; real resnet50 3x3 convs do NOT yet, for two reasons:
- **ow not vl-divisible** (ow ∈ {56,28,14,7}): needs the vl-remainder ow-peel
  (main vl-divisible loop + scalar tail), like the broadcast kernel already has.
- **padded kw bounds depend on ow** (`kw: max(0,1-ow)..min(3,57-ow)`): the 16
  ow-lanes would need different kw bounds — unvectorizable. Requires the
  interior/border split of ow into `[0,pad) ∪ [pad,W-pad) ∪ [W-pad,W)`; the
  interior strip gets constant `kw ∈ [0,K)` and vectorizes (kh may keep its
  oh-dependent bounds — same across all ow-lanes, so it does not block), borders
  stay scalar.

Spike files: `/tmp/onnx_spike/conv3x3_{scalar,vec,itargs}.mlir`, `driver3x3.c`.

## Validation harness (built this session)

- `/tmp/onnx_spike/conv1x1_itargs.mlir` — onnx-mlir iter_args form (the input).
- `/tmp/onnx_spike/driver.c` — C driver via `_mlir_ciface_`, REPS loop,
  checksum guard (anti-DCE + correctness).
- `/tmp/onnx_spike/verify.sh` — lowers (marco LLVM 22:
  lower-affine→…→llvm→`clang -O2`), times orig vs demote+register-block, checks
  checksum equality.
- Correctness criterion here is rel-err / checksum, NOT bit-identical (vector
  reductions reassociate) — but the 1x1 GEMM happens to stay bit-identical at
  this size.
- Real IR: `/tmp/onnx_spike/{mnist,resnet50-v2-7}.affine.mlir` (host dr-opt runs
  with `-allow-unregistered-dialect`: the krnl.global weight constants are
  unregistered but the pass only touches affine ops).

## Next

1. **WP-O0 productionize**: end-to-end resnet50 latency harness (lower the whole
   model + run `run_main_graph`) to measure the real inference win and the
   onnx-mlir `--O3` bar — the standalone-kernel 9.4x is per-layer, not the model.
2. **WP-O2**: direct-conv register-block for the 60 nested 3x3 reductions (the
   other ~half of conv FLOPs) — the largest remaining lever.
