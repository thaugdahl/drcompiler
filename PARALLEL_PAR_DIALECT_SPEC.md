# PARALLEL_PAR_DIALECT_SPEC.md — the `par` dialect & its lowerings (Phase B)

Status: design spec (Phase B of the parallel-codegen track, 2026-06-18). Front-end:
`PARALLEL_BUBBLE_SPEC.md`. Back-end runtime/cost: `PARALLEL_CODEGEN_SPEC.md`.

`par` is the **runtime-agnostic contract** between bubble-widening (which produces
maximal parallel regions) and the executors. It names *what* is parallel and *how
data maps to workers* — never a concrete runtime. This spec gives the ODS op
definitions and three lowerings: `par → libdrpar` (primary), `par → omp` (oracle /
portability), `par → scf` (testing / semantic reference).

---

## 1. Design invariants

1. **Runtime-agnostic.** No op mentions threads, OpenMP, or a pool. Parallelism is a
   *mapping* attribute on a structured iteration space; concretization is the
   lowering's job.
2. **Structured, verifiable.** Every region is single-block with an implicit
   `par.yield` terminator. Legality is checked by op verifiers, not by convention.
3. **Sequential refinement.** A `par.region` executed by one worker, in iteration
   order, with redistributes and barriers as no-ops, is *exactly* the original
   program. This is what makes `par → scf` a correctness oracle and keeps the whole
   track default-OFF / byte-identical (`cores == 1` ⇒ the front-end never emits `par`).
4. **Implicit capture.** `par.region`/`par.forall` are *not* `IsolatedFromAbove`;
   they capture SSA values by dominance like `scf.parallel`/`omp.parallel`. Outlining
   + ctx-packing happens only in the `libdrpar` lowering, reusing register-block's
   existing packer (`PARALLEL_CODEGEN_SPEC.md` §10) — we do not invent a second one.

---

## 2. Attributes & types

A single distribution attribute drives mapping; the figures use the shorthand
`#par.block` for `#par.dist<block>`.

```tablegen
def Par_DistKind : I32EnumAttr<"DistKind", "worker→iteration distribution", [
    I32EnumAttrCase<"Block",     0, "block">,      // contiguous shard per worker (default)
    I32EnumAttrCase<"Cyclic",    1, "cyclic">,     // round-robin
    I32EnumAttrCase<"Collapsed", 2, "collapsed">   // flatten multi-dim, then block
  ]> { let cppNamespace = "::mlir::par"; }

def Par_DistAttr : AttrDef<Par_Dialect, "Dist"> {
  let mnemonic = "dist";
  let parameters = (ins "DistKind":$kind, OptionalParameter<"int64_t">:$dim);
  // #par.dist<block>           — block over the (single / collapsed) space
  // #par.dist<block, dim = 1>  — block over axis 1 (used by redistribute, §3.4)
  let assemblyFormat = "`<` $kind (`,` `dim` `=` $dim^)? `>`";
}
```

No new SSA types in v1; `par` operates on the existing `index` / `memref` / element
types. (Phase C may add a `!par.token` for async regions — out of scope.)

---

## 3. Operations

```tablegen
def Par_Dialect : Dialect {
  let name = "par";
  let summary = "Runtime-agnostic parallel regions for drcompiler";
  let cppNamespace = "::mlir::par";
  let useDefaultAttributePrinterParser = 1;
  let dependentDialects = ["affine::AffineDialect", "memref::MemRefDialect",
                           "arith::ArithDialect", "scf::SCFDialect"];
}
class Par_Op<string mnemonic, list<Trait> traits = []> : Op<Par_Dialect, mnemonic, traits>;
```

### 3.1 `par.region` — maximal parallel region

```tablegen
def Par_RegionOp : Par_Op<"region",
    [SingleBlockImplicitTerminator<"YieldOp">, RecursiveMemoryEffects,
     AutomaticAllocationScope]> {
  let summary = "A maximal region executed concurrently by a worker team";
  let arguments = (ins OptionalAttr<Par_DistAttr>:$mapping);
  let regions   = (region SizedRegion<1>:$body);
  let assemblyFormat = "($mapping^)? $body attr-dict";
  let hasVerifier = 1;   // body's only "team-structured" ops are forall/barrier/
                         // critical/redistribute at the top level
}
```

Semantics: the body executes once per region entry; the worker team is established at
region entry and torn down at exit. Nested `par.forall`s share the team (one
fork/join per region, not per loop — the amortization §5.5 of the back-end spec).

### 3.2 `par.forall` — distributed loop nest

```tablegen
def Par_ForallOp : Par_Op<"forall",
    [AttrSizedOperandSegments, SingleBlockImplicitTerminator<"YieldOp">,
     RecursiveMemoryEffects, DeclareOpInterfaceMethods<LoopLikeOpInterface>]> {
  let summary = "Distribute a rectangular iteration space across the team";
  let arguments = (ins Variadic<Index>:$lowerBounds,
                       Variadic<Index>:$upperBounds,
                       Variadic<Index>:$steps,
                       OptionalAttr<Par_DistAttr>:$mapping);
  let regions   = (region SizedRegion<1>:$body);   // 1 block arg per dim (the IVs)
  let hasVerifier = 1;   // #operands(lb)==#operands(ub)==#operands(step)==#blockArgs
  // Pretty form (custom parser/printer):
  //   par.forall (%i, %j) in (%lb0, %lb1) to (%ub0, %ub1) step (%s0, %s1)
  //     { ... par.yield }  { mapping = #par.dist<block> }
}
```

Semantics: the cross product of the per-dim ranges is partitioned across workers per
`mapping` (defaults to the enclosing region's mapping, else `#par.dist<block>` on dim
0). Iterations assigned to one worker run in lexicographic order; no ordering is
guaranteed *across* workers. The verifier requires the body to carry no dependence on
a distributed axis (the front-end guarantees this; the verifier is a backstop that
rejects hand-written illegal IR by re-running the affine check when accesses are
affine).

### 3.3 `par.barrier`

```tablegen
def Par_BarrierOp : Par_Op<"barrier", [HasParent<"RegionOp">]> {
  let summary = "Team barrier: all workers synchronize here";
  let assemblyFormat = "attr-dict";
  // Side-effecting (writes a dedicated `Barrier` resource) so it is never reordered
  // or DCE'd across parallel work.
}
```

### 3.4 `par.redistribute` — transitory mapping change

```tablegen
def Par_RedistributeOp : Par_Op<"redistribute", [HasParent<"RegionOp">]> {
  let summary = "Re-block a memref's worker↔data mapping between two sub-regions";
  let arguments = (ins AnyMemRef:$value, Par_DistAttr:$from, Par_DistAttr:$to);
  let assemblyFormat = "$value `:` type($value) `from` $from `to` $to attr-dict";
  // Implies a team barrier; the data in $value is repartitioned from $from to $to.
}
```

### 3.5 `par.critical` — peeled single-worker slab

```tablegen
def Par_CriticalOp : Par_Op<"critical",
    [SingleBlockImplicitTerminator<"YieldOp">, RecursiveMemoryEffects,
     HasParent<"RegionOp">]> {
  let summary = "A sub-region executed by exactly one worker, in order";
  let arguments = (ins UnitAttr:$ordered);   // ordered ⇒ runs at its lexical position
  let regions   = (region SizedRegion<1>:$body);
  let assemblyFormat = "(`ordered` $ordered^)? $body attr-dict";
}
```

Used for the peeled boundary slabs of §4.1 (front-end): one worker runs the whole
slab while the team waits (a `single` with an implicit barrier, *not* a per-iteration
critical section).

### 3.6 `par.reduce` — cross-worker reduction (M5 / stretch)

Modelled on `scf.reduce`: a terminator of a `par.forall` that contributes a
per-iteration value through an associative combinator region; the `par.forall` then
gains result values and matching `inits`.

```tablegen
def Par_ReduceOp : Par_Op<"reduce", [Terminator, HasParent<"ForallOp">]> {
  let summary = "Associative cross-worker reduction contribution";
  let arguments = (ins Variadic<AnyType>:$contributions);
  let regions   = (region VariadicRegion<SizedRegion<1>>:$combiners);  // (%lhs,%rhs)->%out
  // Requires the combiner to be associative+commutative (FP ⇒ fastmath<reassoc>),
  // matching the back-end's __drpar_reduce + the dot-family reassoc already in use.
}
```

### 3.7 `par.yield` — terminator

```tablegen
def Par_YieldOp : Par_Op<"yield", [Pure, Terminator, ReturnLike,
    ParentOneOf<["RegionOp","ForallOp","CriticalOp"]>]> {
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
}
```

### 3.8 `par.call` — not an op

A "consumed call" (front-end §5) is a plain `func.call` inside a region body, with the
disjointness verdict recorded as an analysis fact (a discardable
`par.consumed` unit attr on the call for debugging). No dedicated op; the figures'
`par.call` is sugar for "an ordinary call the analysis proved per-iteration disjoint".

---

## 4. Lowerings

Three conversion passes, each a `Pass<"...", "mlir::ModuleOp">` registered the same
way as the transform passes (`-gen-pass-decls -name DRCompPasses`).

### 4.1 `convert-par-to-libdrpar` (primary)

Reuses everything from `PARALLEL_CODEGEN_SPEC.md`: `decideShard()`, the ctx-struct
packer, the pinned pool ABI, `Topology` placement.

| `par` op | libdrpar lowering |
|----------|-------------------|
| `par.region` | Compute live-ins → pack into a `ctx` struct (register-block's packer). For each top-level `par.forall`, outline its body to `@__parbody_N(i64 lo, i64 hi, i32 tid, !llvm.ptr ctx)` and replace with `func.call @__drpar_for(lo, hi, grain, @__parbody_N, ctx)`. One team for the whole region (pool is persistent). |
| `par.forall` | `decideShard(trip, perIterCycles, perIterBytes, mm)` → `{cores, grain, domain, padBoundary}`. Multi-dim: `#par.dist<collapsed>` flattens to one linear space; otherwise the outermost dim is sharded and inner dims become sequential loops in the body. `cores == 1` ⇒ emit a plain `scf.for` (no runtime call). |
| `par.barrier` | Between two `__drpar_for` calls: implicit (each call joins). A barrier *inside* one outlined body ⇒ `func.call @__drpar_barrier(ctx)` (new ABI entry). |
| `par.redistribute` | `@__drpar_alloc_local(bytes, tid)` first-touched under the `to` mapping + a parallel all-to-all copy loop + free of the old residency; a team barrier. The `domain` of the new placement comes from the same `decideShard().domain`. |
| `par.critical` | Guard the slab with `if (tid == 0)` inside the outlined body (`ordered` ⇒ at its lexical position, preceded by a barrier so prior parallel writes are visible); a trailing barrier so other workers wait. |
| `par.reduce` | Per-thread cache-line-padded partials via `@__drpar_alloc_local`; final `@__drpar_reduce_f64/_f32(perThread, n, out, kind)` tree-combine (`PARALLEL_CODEGEN_SPEC.md` §3.3), under `fastmath<reassoc>` / `--par-deterministic`. |
| `par.yield` | Erased (or `scf.yield` of carried values inside the outlined loop). |

ABI delta vs the existing `libdrpar`: add `__drpar_barrier(void *ctx)` (intra-region
barrier) — everything else (`__drpar_for`, `__drpar_alloc_local`,
`__drpar_reduce_*`) already exists.

### 4.2 `convert-par-to-openmp` (correctness oracle + portability)

Targets the upstream `omp` dialect (lowered by `--convert-openmp-to-llvm` against
libomp). The back-end spec already keeps OpenMP as its oracle (§7 there); this is
where it is produced.

| `par` op | omp lowering |
|----------|-------------|
| `par.region` | `omp.parallel { ... omp.terminator }` |
| `par.forall` | `omp.wsloop { omp.loop_nest (%i,…) : index = (lb) to (ub) step (st) { ... omp.yield } }`. `mapping` → `schedule` clause (`block`→static, `cyclic`→`static = 1`, `collapsed`→`collapse(n)`). |
| `par.barrier` | `omp.barrier` |
| `par.redistribute` | `omp.barrier` + a parallel copy loop (OpenMP has no first-touch/placement control — *correctness only*, placement is the libdrpar path's job). |
| `par.critical` | `omp.single { ... }` (implicit barrier; `ordered` ⇒ `omp.ordered` region). |
| `par.reduce` | `omp.declare_reduction` + a `reduction` clause on the enclosing `omp.wsloop`. |
| `par.yield` | `omp.yield` / `omp.terminator` as appropriate. |

### 4.3 `convert-par-to-scf` (testing / semantic reference)

Lowers to plain structured control flow — **runs sequentially**, used for FileCheck
and execution-correctness diffs, *not* for performance.

| `par` op | scf lowering |
|----------|-------------|
| `par.region` | inlined into the parent block (or `scf.execute_region` to preserve scoping) |
| `par.forall` | `scf.forall (%i,…) in (…) { ... }` (or `scf.parallel` when a `par.reduce` is present, paired with `scf.reduce`) |
| `par.barrier` | erased — sequential execution already orders the sub-regions |
| `par.redistribute` | erased — data is in place under one worker |
| `par.critical` | inlined region |
| `par.reduce` | `scf.reduce` / `scf.forall ... in_parallel` |

Because of invariant #3, `par → scf` reproduces the original program's results
exactly; diffing `par → omp` and `par → libdrpar` outputs against it catches
outlining / packing / reduction-reassociation bugs.

---

## 5. Build & registration wiring

Mirror the (fixed) dialect convention from `PARALLEL_BUBBLE_SPEC.md` §10.2:

- `include/drcompiler/Dialect/Par/IR/{Par.td, ParOps.td, ParAttrs.td, CMakeLists.txt}`
  — use `add_mlir_dialect(ParOps par)`; emit attr defs with `-gen-attrdef-{decls,defs}`;
  decls→`ParOps.h.inc`, defs→`ParOps.cpp.inc` (distinct files — the Topology bug).
- `lib/Dialect/Par/IR/{ParDialect.cpp, ParOps.cpp, ParAttrs.cpp}` →
  `add_mlir_library(DRCompParDialect ... DEPENDS MLIRParOpsIncGen)`.
- `lib/Conversion/{ParToLibdrpar,ParToOpenMP,ParToSCF}/` → conversion libraries
  (`add_mlir_conversion_library` or a plain `add_mlir_library`), declared in a
  `include/drcompiler/Conversion/Passes.td` with its own `-gen-pass-decls -name DRCompConversion`.
- Top-level `CMakeLists.txt`: add the `add_subdirectory` chain for
  `include/drcompiler/Dialect`, `lib/Dialect`, `include/drcompiler/Conversion`,
  `lib/Conversion` (none are currently wired).
- `tools/dr-opt/dr-opt.cpp`: `registry.insert<par::ParDialect, omp::OpenMPDialect>();`
  and `registerDRCompConversionPasses();` alongside the existing
  `registerDRCompPassesPasses()` at `:20`.

---

## 6. Verification & test plan

1. **Round-trip** (`dr-opt %s | dr-opt | diff`) for every op + the `#par.dist` attr,
   including the multi-dim `par.forall` custom syntax.
2. **Verifiers** (`-verify-diagnostics`): forall operand-count mismatch; a
   `par.barrier`/`par.redistribute` outside a `par.region`; a non-associative
   `par.reduce` combiner without `fastmath<reassoc>`.
3. **Lowering FileCheck** per target: a 2-region + barrier + redistribute + critical
   kernel through each of the three passes.
4. **Oracle diff**: compile a small kernel via `par → scf` (sequential),
   `par → omp`, and `par → libdrpar`; assert bit-identical (no-reassoc path) and
   `norm-rel-err ≤ 1e-4` (with reductions reassociated), reusing the ONNX bench gate.

---

## 7. Open questions (Phase B)

- **Multi-dim shard strategy in libdrpar** — always collapse, or shard the largest
  parallel dim and keep the rest sequential? (Default: collapse when the outer dim
  alone underfills the pool — the back-end's §3.4 batch-1 rule.)
- **Mapping richness** — should `#par.dist` carry an explicit `domain`/`coreClass`
  hint for heterogeneous placement, or leave all placement to `decideShard()`?
  (Default: leave it to `decideShard()`; the attr stays a pure distribution kind.)
- **`par.reduce` shape** — terminator-of-forall (scf.reduce style, chosen here) vs a
  standalone op with init operands. Revisit if nested reductions need it.
- **Async / token regions** — a `!par.token` for overlapping a redistribute with
  compute is deferred to Phase C.
```