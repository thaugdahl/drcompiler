# Risk Probe — "Why not Linalg?" and Affine-Path Audience

**Date:** 2026-06-03
**Author:** Claude (Opus 4.8)
**Purpose:** Before committing weeks to Phase 1 of `CODEGEN_CAMPAIGN.md`, de-risk the two reviewer-killers identified at the Phase-0 gate. User directive: *"probe risks before building — if either fails, the campaign is dead."*

**Verdict: Neither risk kills the campaign, but together they narrow and reframe it. The contribution is defensible ONLY if scoped precisely to the memref/affine path and explicitly cedes the tensor path to IREE/Linalg.**

---

## Risk A — Are the target frontends actually not cheap-Linalg-raisable?

**Result: holds (audience is real).** There is **no stock `affine→linalg` raising pass** in MLIR (LLVM 22 / Marco build). `mlir-opt --help` exposes only the *lowering* direction (`--convert-linalg-to-affine-loops`, `--convert-linalg-to-loops`) and narrow raisers (`--tosa-to-linalg`, `--convert-tensor-to-linalg`, `--convert-elementwise-to-linalg`) — none lifts a general affine loop nest (cgeist's gemm) to `linalg.matmul`. Robust affine→linalg raising is an open problem (only special-case pattern matching exists). The named audiences emit memref/affine, not Linalg:
- **cgeist/Polygeist** (C→MLIR): affine + memref.
- **ONNX-MLIR**: Krnl → **Affine** → memref (its *own* lowering; it does **not** route through Linalg/IREE).
- **MARCO** `bmodelica`: affine/scf.

Verified: `linalg.matmul --convert-linalg-to-affine-loops` produces **byte-identical** IR to cgeist's naive gemm (the 0.48-GFLOPs kernel from `PHASE0_FINDINGS.md`). So Linalg-without-a-schedule is no faster than what the frontends already emit.

## Risk B — Does Linalg get register-blocking "for free"?

**Result: only on the *tensor* path — not on memref. This is the crux.** Measured the standard MLIR transform-dialect schedule (`tile_using_for [8,16,1]` → `vectorize_children_and_apply_patterns` → `hoist_redundant_vector_transfers`) on a `linalg.matmul`, both input forms, lowered identically and run (N=1024, core8, checksums verified):

| input form | `iter_args` after tiling | register-blocks | GFLOPs | why |
|---|---|---|---|---|
| **memref** linalg.matmul | **0** | **no** | **4.2–5.5** | C updated in place; auto-hoist can't prove C ⊥ A,B on bare memrefs → accumulator stays in memory, read/written every k |
| **tensor** linalg.matmul (DPS) | **3** | **yes** | ~50* | C threaded as loop-carried `iter_args` through all 3 loops by construction; bufferizes to a register accumulator (IREE's exact mechanism) |
| **affine `iter_args`** (Phase-0, ours) | n/a (hand-written) | **yes** | **51** | accumulator asserted private via `iter_args`, identical effect |

\*Structurally confirmed (DPS threading visible in tiled IR); same mechanism as the affine `iter_args` path that measured 51. Perf not separately harnessed (tensor return-value vs in-place harness mismatch).

Adding canonicalize + CSE + LICM before the hoist did **not** change the memref result (still 0 `iter_args`, 4.2 GFLOPs) — so it is **not** a pass-ordering artifact; it is fundamental to in-place memref semantics + aliasing conservatism.

### What this means
- **Tensor-domain frontends** (anything that hands MLIR `tensor`/`linalg` — IREE, JAX, ONNX *routed through tensor*) already get register-blocking via DPS + bufferization. For them, **"why not Linalg" is correct — do not compete here.**
- **Memref/affine frontends** (cgeist, ONNX-MLIR's native Krnl→Affine, MARCO, hand-affine, polyhedral output) do **not** get it: there's no raiser (Risk A), and wrapping in `linalg.matmul` on memrefs lands at 4–5 GFLOPs (Risk B). The Phase-0 affine `iter_args` approach (51) is the thing that reaches register-blocked performance for this path.

---

## Net assessment

**The contribution survives, but is now precisely pinned:** *bring DPS-style register-blocking accumulators (`iter_args`) to the memref/affine path — where tensor-Linalg's bufferization-based mechanism cannot reach — with analytically chosen register tiles.* Audience: C-frontend HPC (cgeist/Polygeist), ONNX-MLIR Krnl→Affine, MARCO.

**Residual risks the probe could NOT retire (judgment calls for the user/advisor):**
1. **"Is this just porting a known Linalg technique to memref?"** A reviewer may frame the register-block-via-`iter_args` step as the memref analog of tensor DPS — engineering, not novelty. Defense leans on (a) the analytical cost-model tile selection and (b) the underserved-but-large memref/affine audience. This is a framing/novelty judgment, not an empirical one.
2. **"Then just go memref→tensor and use IREE."** Counter: un-bufferization (memref→tensor) is not generally sound; no robust pass exists. True, but a determined reviewer can still push.
3. The cost model remains **weakly leveraged** (register-*fit* selection is near-deterministic), unchanged from the Phase-0 caveat.

**Recommendation:** The two existential risks (A, B) are retired *for the scoped memref/affine audience*. The decision to proceed to Phase 1 now hinges on the **novelty/framing** judgment (residual risk 1), which is the user's/advisor's call, not an empirical gate. If "register-blocking for the memref/affine path + analytical tiles" is considered a contribution, build it; if a reviewer would dismiss it as a known-technique port, the stronger move is to either (a) make the cost model load-bearing first, or (b) bank Phase 0 + this scoping as a motivation/related-work result.

### Reproduce
`/tmp/claude/mm_sched.mlir` (memref schedule), `mm_tensor.mlir` (tensor schedule), run with `mlir-opt --transform-interpreter --test-transform-dialect-erase-schedule`; lower via the vector-aware pipeline (adds `--convert-vector-to-scf --convert-vector-to-llvm --convert-ub-to-llvm`).
