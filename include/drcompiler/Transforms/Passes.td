#ifndef DRCOMPILER_TRANSFORMS_PASSES_TD
#define DRCOMPILER_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def DataRecomputationPass : Pass<"data-recomputation", "mlir::ModuleOp">
{
  let summary = "Applies data recomputation where likely to be profitable";

  let description = [{
    Load-store provenance analysis: for each memref.load (and llvm.load),
    determines which store operations could have been the last writer.

    Loads are classified into four categories:
      SINGLE  - exactly one known store (recomputation candidate)
      MULTI   - multiple possible stores
      LEAKED  - provenance includes an external/unknown write (nullptr)
      KILLED  - all reaching stores were killed (empty provenance set)

    Both memref and LLVM dialect ops are supported:
      memref.alloc / llvm.call @malloc  → allocation roots
      memref.store / llvm.store         → tracked stores
      memref.load  / llvm.load          → tracked loads
    LLVM loads/stores are tracked with flat (no index) coverage, so they
    are more likely to be classified as MULTI.

    When --dr-recompute is set, SINGLE loads are replaced by rematerializing
    the stored value at the load site (memref loads only).
  }];

  let options = [
    Option<"drTestDiagnostics", "dr-test-diagnostics", "bool", "false",
           "Enables test output">,
    Option<"drDotFile", "dr-dot-file", "std::string", /*default=*/"",
           "Output file path for load-provenance DOT graph">,
    Option<"drRecompute", "dr-recompute", "bool", "false",
           "Replace SINGLE loads with the reaching store's value">,
    Option<"drCostModel", "dr-cost-model", "bool", "false",
           "Gate recomputation on a cache-aware cost model">,
    Option<"drL1Size", "dr-l1-size", "unsigned", "32768",
           "L1 data cache size in bytes">,
    Option<"drL2Size", "dr-l2-size", "unsigned", "1048576",
           "Private (per-core) L2 cache size in bytes. Matches MemoryFission/"
           "AffineRegisterBlock so the three passes judge one machine on one "
           "geometry (was 262144, a stale/shared value 4x too small for Zen4).">,
    Option<"drL3Size", "dr-l3-size", "unsigned", "33554432",
           "Shared last-level (L3) cache size in bytes. Was 0, which made the "
           "`l3Size>0` guard SKIP the LLC tier entirely: every buffer 256KB-32MB "
           "was priced straight at mem latency and dr-llc-sharers was dead "
           "(effectiveL3=0/sharers=0). Setting it revives the contention model on "
           "the DataRecomputation path.">,
    Option<"drL1Latency", "dr-l1-latency", "unsigned", "4",
           "L1 cache hit latency in cycles">,
    Option<"drL2Latency", "dr-l2-latency", "unsigned", "12",
           "L2 cache hit latency in cycles">,
    Option<"drL3Latency", "dr-l3-latency", "unsigned", "40",
           "L3 cache hit latency in cycles">,
    Option<"drMemLatency", "dr-mem-latency", "unsigned", "200",
           "DRAM access latency in cycles">,
    Option<"drCacheLineSize", "dr-cache-line-size", "unsigned", "64",
           "Cache line size in bytes (used for stride-aware partial remat cost)">,
    Option<"drLlcSharers", "dr-llc-sharers", "unsigned", "1",
           "Co-tenants of the shared LLC. Effective L3 a kept buffer can count on "
           "is dr-l3-size/this; a reuse beyond it is priced as a memory access "
           "(the residency a co-tenant can evict). 1 = isolated; set to the LLC "
           "core count for a contention-aware estimate.">,
    Option<"cpuCostModelFile", "cpu-cost-model-file", "std::string", /*default=*/"",
           "Path to a JSON file with per-op cycle costs (see scripts/gen_cpu_cost_model.py)">,
    Option<"drFootprintAnalysis", "dr-footprint-analysis", "bool", "false",
           "Enable intervening memory footprint analysis for cache eviction estimation">,
    Option<"drPartialRemat", "dr-partial-remat", "bool", "false",
           "Allow rematerialization when operand tree contains non-chainable "
           "leaf loads, subject to cache-aware cost gate (requires dr-cost-model)">,
    Option<"drPartialMaxLeaves", "dr-partial-max-leaves", "unsigned", "4",
           "Maximum number of non-chainable leaf loads permitted per "
           "partial rematerialization">,
    Option<"drDebug", "dr-debug", "bool", "false",
           "Enable verbose DRDBG (DRCOMP:) trace output to llvm::dbgs()">,
    Option<"drSummary", "dr-summary", "bool", "false",
           "Print compact decision summary (buffer cost-model + per-load "
           "remat outcomes with reasons) to stderr as DRSUM: lines">,
    Option<"drBufferElim", "dr-buffer-elim", "bool", "false",
           "Run whole-buffer elimination feasibility analysis (escape + "
           "remaining-loads + classification rollup). Diagnostic-only "
           "unless dr-erase-eliminated-buffers is also set.">,
    Option<"drEraseElimBuffers", "dr-erase-eliminated-buffers", "bool",
           "false",
           "When a buffer is deemed eliminable by dr-buffer-elim, erase its "
           "alloc/stores/dealloc/view chain explicitly instead of relying "
           "on downstream DCE.">,
    Option<"drBufferElimDrivesStrategies",
           "dr-buffer-elim-drives-strategies", "bool", "false",
           "Let the whole-buffer rollup override the per-load cost-model "
           "veto. When the rollup says eliminate < keep, the per-load "
           "strategies are allowed to fire on that buffer's loads even if "
           "dr-cost-model would have kept them. Requires dr-buffer-elim.">,
    Option<"drIcacheSoftBudget", "dr-icache-soft-budget", "unsigned", "128",
           "Per-buffer soft budget on cloned op count; excess ops contribute "
           "to the elim cost as code-bloat penalty (1 cycle per excess op).">,
    // Unified cost-model — register-pressure parameters (Phase 3 of
    // REGISTER_PRESSURE_PLAN.md).  JSON file (`cpu-cost-model-file`)
    // overrides any of these when present.
    Option<"drArchHandler", "dr-arch-handler", "std::string", /*default=*/"",
           "ArchHandler name (generic | x86-64-avx2 | x86-64-avx512 | arm-neon). "
           "Empty = inferred from cost-model JSON or 'generic'.">,
    Option<"drSpillStrategy", "dr-spill-strategy", "std::string",
           /*default=*/"\"excess-hot\"",
           "Spill aggregation: excess-hot | sum-excess | graph-color">,
    Option<"drRegBudgetGp", "dr-reg-budget-gp", "unsigned", "16",
           "General-purpose register budget (overridden by JSON/handler).">,
    Option<"drRegBudgetFp", "dr-reg-budget-fp", "unsigned", "16",
           "Scalar FP register budget (overridden by JSON/handler).">,
    Option<"drRegBudgetVec", "dr-reg-budget-vec", "unsigned", "16",
           "Vector register budget (overridden by JSON/handler).">,
    Option<"drRegBudgetPred", "dr-reg-budget-pred", "unsigned", "0",
           "Predicate/mask register budget (overridden by JSON/handler).">,
    Option<"drSpillReload", "dr-spill-reload", "unsigned", "5",
           "Cycles per spill reload (overridden by JSON/handler).">,
    Option<"drSpillStore", "dr-spill-store", "unsigned", "1",
           "Cycles per spill store (overridden by JSON/handler).">,
    Option<"drCostWeightMem", "dr-cost-weight-mem", "double", "1.0",
           "Memory cycle weight (alpha) in unified cost combiner.">,
    Option<"drCostWeightReg", "dr-cost-weight-reg", "double", "1.0",
           "Register cycle weight (beta) in unified cost combiner.">,
    Option<"drCostWeightAlu", "dr-cost-weight-alu", "double", "1.0",
           "ALU cycle weight (gamma) in unified cost combiner.">
  ];

  let dependentDialects = [
    "memref::MemRefDialect", "vector::VectorDialect",
    "affine::AffineDialect", "ptr::PtrDialect",
    "LLVM::LLVMDialect"
  ];

  let constructor = "mlir::createDataRecomputationPass()";
}

def RaiseMallocToMemRefPass
    : Pass<"raise-malloc-to-memref", "mlir::ModuleOp">
{
  let summary = "Raise malloc/free + LLVM load/store/GEP to memref ops";

  let description = [{
    Converts patterns produced by Polygeist cgeist for malloc-based allocations
    back to memref dialect operations.  This is a pre-pass for
    DataRecomputation.

    Canonical input pattern (per malloc site):
      %raw = call @malloc(%bytes) : (i64) -> memref<?xi8>
      %ptr = "polygeist.memref2pointer"(%raw) : (memref<?xi8>) -> !llvm.ptr
      %gep = llvm.getelementptr %ptr[%idx] : (!llvm.ptr, ...) -> !llvm.ptr, T
      llvm.store %val, %gep : T, !llvm.ptr
      %v   = llvm.load %gep : !llvm.ptr -> T
      call @free(%ptr)

    Target output:
      %alloc = memref.alloc(%count) : memref<?xT>
      memref.store %val, %alloc[%idx] : memref<?xT>
      %v = memref.load %alloc[%idx] : memref<?xT>
      memref.dealloc %alloc : memref<?xT>

    Malloc sites whose pointer escapes (passed to a call other than @free,
    stored to memory, cast to integer) are left untouched.
  }];

  let dependentDialects = [
    "memref::MemRefDialect", "arith::ArithDialect",
    "LLVM::LLVMDialect"
  ];

  let constructor = "mlir::createRaiseMallocToMemRefPass()";
}

def DrAffineLoopDistributePass
    : Pass<"dr-affine-loop-distribute", "mlir::func::FuncOp">
{
  let summary = "Fission imperfect affine loops into perfectly nestable bands";

  let description = [{
    Classical loop distribution: an affine.for whose body contains two or
    more top-level affine.for children (and otherwise only pure scalar ops)
    is split into consecutive copies, one child per copy, when no dependence
    is carried backward by the loop.  This turns PolyBench-style imperfect
    nests (e.g. gemm's `for i { C[i][:] *= beta; for k for j C += ... }`)
    into maximal perfect bands that affine tiling and register blocking can
    actually operate on -- the implicit first step a polyhedral scheduler
    (Polly) performs before tiling.
  }];

  let options = [
    Option<"mode", "mode", "std::string", /*default=*/"\"enabler\"",
           "Profitability mode: 'enabler' splits only when fission deepens "
           "a perfect band (feeding the tiler / register blocker); 'always' "
           "splits whenever legal (v1 behavior, kept for A/B runs)">,
    Option<"emitRationale", "emit-rationale", "bool", /*default=*/"false",
           "Emit a remark for every split performed and every legal split "
           "skipped by the profitability mode">,
    Option<"cacheBytes", "cache-bytes", "int64_t", /*default=*/"524288",
           "Cache capacity for the locality guard's evicted-reuse test "
           "(default: half a 1 MiB private L2, matching the tiler's target)">,
  ];

  let dependentDialects = ["mlir::affine::AffineDialect"];
  let constructor = "mlir::createDrAffineLoopDistributePass()";
}

def DrScalarReductionDemotePass
    : Pass<"dr-scalar-reduction-demote", "mlir::func::FuncOp">
{
  let summary = "Lower SSA iter_args reductions to memref-accumulator form";

  let description = [{
    onnx-mlir emits every contraction (matmul, 1x1 conv) as an SSA reduction:

      %r = affine.for %k iter_args(%a = %c0) -> f32 {
             %p = arith.mulf (load A)(load B); %s = arith.addf %a, %p
             affine.yield %s }
      affine.store %r, %C[i, j]

    The affine-register-block matcher instead expects the cgeist -O0 shape: a
    perfect reduction band whose accumulator is carried in MEMORY (load %C
    before, store %C after, body = just the reduction).  This pass bridges the
    two by FISSIONING a single-level add-reduction into

      affine.for ... { affine.store %init, %C[i, j] }          // zero/bias init
      affine.for ... { affine.for %k { %c = load %C; store (%c + A*B), %C } }

    The reduction band is now perfect, so the register-block vector micro-kernel
    fires (1x1 convs vectorize ~9x on resnet50).  An additive epilogue
    (`store (%r + bias[i])`) is folded into the init nest, keeping the band
    perfect.  Only single-level f32/f64 `addf` reductions are handled; nested
    conv reductions (3x3) and non-add (maxpool/softmax) chains are left intact.
  }];

  let options = [
    Option<"emitRationale", "emit-rationale", "bool", /*default=*/"false",
           "Emit a remark for every reduction demoted and every match skipped">,
  ];

  let dependentDialects = ["mlir::affine::AffineDialect"];
  let constructor = "mlir::createDrScalarReductionDemotePass()";
}

def DrScalarReductionPromotePass
    : Pass<"dr-scalar-reduction-promote", "mlir::func::FuncOp">
{
  let summary = "Raise leftover memref accumulators back to iter_args";

  let description = [{
    Inverse of dr-scalar-reduction-demote, run AFTER affine-register-block.
    Bands the demote pass rewrote but register-block did not take (sub-VL
    convs, stride-2 convs, non-vectorizable GEMMs, and the scalar
    borders/tails the conv splitter creates) are left accumulating into the
    real output tensor in memory -- one load/store round-trip per innermost
    iteration, much slower than the original SSA form.  This pass rebuilds
    every such perfect reduction band carrying the accumulator as a scalar
    iter_arg threaded through all levels: one load before the band, one
    store after.
  }];

  let options = [
    Option<"emitRationale", "emit-rationale", "bool", /*default=*/"false",
           "Emit a remark for every accumulator promoted">,
  ];

  let dependentDialects = ["mlir::affine::AffineDialect"];
  let constructor = "mlir::createDrScalarReductionPromotePass()";
}

def DrTestReuseAnalysisPass
    : Pass<"dr-test-reuse-analysis", "mlir::func::FuncOp">
{
  let summary = "Print per-reference loop reuse classification (test only)";

  let description = [{
    Diagnostic printer for drcompiler::reuse::analyzeBandReuse.  For every
    top-level perfect affine band it emits remarks with the per-reference
    reuse kinds (invariant / spatial / streaming, per band loop), the
    per-iteration footprints, and the band-level evicted-reuse verdicts the
    tiler's profitability gate consumes.
  }];

  let options = [
    Option<"cacheBytes", "cache-bytes", "int64_t", /*default=*/"32768",
           "Cache capacity used for the evicted-reuse verdicts">,
  ];

  let dependentDialects = ["mlir::affine::AffineDialect"];
  let constructor = "mlir::createDrTestReuseAnalysisPass()";
}

def DrAffineStencilTimeTilePass
    : Pass<"dr-affine-stencil-time-tile", "mlir::func::FuncOp">
{
  let summary = "Skewed time-tiling for ping-pong stencils (jacobi family)";

  let description = [{
    Matches `for t { B <- f(A); A <- g(B) }` where both space nests are
    perfect, constant-bound, and every load offset is within +/-1 of the
    store point (halo 1).  Assigns each half-step a virtual time
    tau = 2t + phase and skews every space dim by tau; all dependence
    distances become non-negative, so the skewed band is fully permutable
    and is tiled rectangularly (tile-t x tile-s^d), emitted in ORIGINAL
    coordinates as affine max/min iteration windows.  Every (t, point)
    body instance executes whole, exactly once, in an order consistent
    with every flow dependence: bit-identical results.

    Profitability: fires only when one time step's working set (both
    arrays) exceeds l3-size/llc-sharers -- below that the step is already
    cache-resident and skewed windows only add bound overhead (Polly's
    jacobi-1d 0.34x is this mistake).  `force-tile` skips the gate (tests).
  }];

  let options = [
    Option<"tileT", "tile-t", "unsigned", "0",
           "Time-tile depth in t iterations (0 = measured per-rank default: 128 for 2-D, 24 for 3-D, 32 for 1-D)">,
    Option<"tileS", "tile-s", "unsigned", "0",
           "Space-tile width per dimension, in skewed-space points (0 = derive from l3-size so one (Ts+2Tt)^d tile box fits half the effective LLC)">,
    Option<"l3Size", "l3-size", "unsigned", "33554432",
           "Shared last-level cache size in bytes">,
    Option<"llcSharers", "llc-sharers", "unsigned", "1",
           "Co-tenants of the LLC; effective cache = l3-size/this">,
    Option<"forceTile", "force-tile", "bool", "false",
           "Skip the footprint profitability gate (testing)">,
    Option<"cpuCostModelFile", "cpu-cost-model-file", "std::string",
           /*default=*/"",
           "Path to a JSON cost-model file; sources l3-size/llc-sharers from "
           "its `cache` block unless overridden by the CLI options above.">
  ];

  let dependentDialects = [
    "memref::MemRefDialect", "affine::AffineDialect", "arith::ArithDialect"
  ];

  let constructor = "mlir::createDrAffineStencilTimeTilePass()";
}

def DrPinLiveOutPass : Pass<"dr-pin-liveout", "mlir::ModuleOp">
{
  let summary = "Pin live-out memrefs against dead-code elimination";

  let description = [{
    Reads every element of each live-out memref (memref function arguments and
    memref.alloc results) into an opaque external sink (@__dr_observe), inserted
    after the kernel's writes (before the first dealloc, else the return).  This
    keeps the producing stores -- hence the kernel computation -- live when the
    PolyBench DCE guard has been suppressed for the MLIR path and an aggressive
    backend would otherwise delete the unobserved output.

    Intended to run after cgeist has preserved the kernel (cgeist -O0) and after
    the DataRecomputation passes, before lowering.  The sink __dr_observe must be
    linked in as an opaque function, e.g.  void __dr_observe(double){}.
  }];

  let dependentDialects = [
    "affine::AffineDialect", "arith::ArithDialect",
    "memref::MemRefDialect", "func::FuncDialect"
  ];

  let constructor = "mlir::createDrPinLiveOutPass()";
}

def PrintArchHandlerPass : Pass<"print-arch-handler", "mlir::ModuleOp">
{
  let summary = "Print the architecture handler chosen for this module";

  let description = [{
    Diagnostic pass for the unified cost model.  Loads the CPU cost-model
    JSON (if `cpu-cost-model-file` is set) and emits a remark on the module
    with the chosen handler name, vector width, register budgets, and
    weights.  Used to validate end-to-end JSON dispatch in Phase 1 of the
    register-pressure plan.
  }];

  let options = [
    Option<"cpuCostModelFile", "cpu-cost-model-file", "std::string",
           /*default=*/"",
           "Path to a JSON file with cost-model and arch/registers blocks">,
    Option<"handlerOverride", "handler", "std::string", /*default=*/"",
           "Override the JSON handler name (use this handler instead)">
  ];

  let constructor = "mlir::createPrintArchHandlerPass()";
}

def DrAffineLoopFusionPass : Pass<"dr-affine-loop-fusion"> {
  let summary = "Fuse affine loop nests under the unified cost model";

  let description = [{
    Fork of upstream `--affine-loop-fusion` (see
    `lib/Transforms/AffineLoopFusionCostModel/UPSTREAM.md`).  Identical
    behaviour to the upstream pass *except* the profitability decision
    (`isFusionProfitable`) is driven by drcompiler's unified cost model
    (memory + register pressure + ALU combined per ArchHandler).

    See REGISTER_PRESSURE_PLAN.md §7 for the rationale.
  }];

  let options = [
    Option<"computeToleranceThreshold", "compute-tolerance", "double",
           /*default=*/"0.30f",
           "Fractional increase in additional computation tolerated while fusing">,
    Option<"fastMemorySpace", "fast-mem-space", "unsigned", /*default=*/"0",
           "Faster memory space number to promote fusion buffers to">,
    Option<"localBufSizeThreshold", "local-buf-threshold", "uint64_t",
           /*default=*/"0",
           "Threshold size (KiB) for promoting local buffers to fast memory space">,
    Option<"maximalFusion", "maximal", "bool", /*default=*/"false",
           "Enables maximal loop fusion">,
    Option<"affineFusionMode", "mode", "enum FusionMode",
           "mlir::affine::FusionMode::Greedy", "fusion mode to attempt",
           "llvm::cl::values(clEnumValN(mlir::affine::FusionMode::Greedy,"
           " \"greedy\", \"Perform greedy (both producer-consumer and sibling) fusion\"), "
           "clEnumValN( mlir::affine::FusionMode::ProducerConsumer, "
           "\"producer\", \"Perform only producer-consumer fusion\"), "
           "clEnumValN( mlir::affine::FusionMode::Sibling, "
           "\"sibling\", \"Perform only sibling fusion\"))">,
    // Drcompiler-specific knobs (additive over upstream).
    Option<"useUnifiedCostModel", "use-unified-cost-model", "bool",
           /*default=*/"true",
           "Drive isFusionProfitable from the unified cost model (memory + "
           "register-pressure + ALU).  When false, falls back to upstream's "
           "compute-tolerance heuristic.">,
    Option<"sinkSequentialLoops", "sink-sequential-loops", "bool",
           /*default=*/"false",
           "Upstream pre-pass that permutes every fusion-destination nest "
           "(sequential loops inward) to deepen legal fusion depths.  It "
           "mutates nests even when nothing fuses and can destroy spatial "
           "locality (real gemm at O0: k sunk innermost makes B[k][j] a "
           "column walk, 0.18x).  Off by default; enable for upstream "
           "parity experiments.">,
    Option<"cpuCostModelFile", "cpu-cost-model-file", "std::string",
           /*default=*/"",
           "Path to a JSON cost-model file (arch + registers + cache + ops).">,
    Option<"drArchHandler", "arch-handler", "std::string", /*default=*/"",
           "ArchHandler name (generic | x86-64-avx2 | x86-64-avx512 | arm-neon).">,
    Option<"drSpillStrategy", "spill-strategy", "std::string",
           /*default=*/"\"excess-hot\"",
           "Spill aggregation: excess-hot | sum-excess | graph-color">,
    Option<"emitRationale", "emit-rationale", "bool", /*default=*/"false",
           "Emit a remark on the source loop with the unified cost-model "
           "breakdown (mem/reg/alu and unfused baseline) for every fusion "
           "decision.  Default off so production runs are quiet.">
  ];

  let dependentDialects = ["memref::MemRefDialect"];
  let constructor = "mlir::createDrAffineLoopFusionPass()";
}

def DrAffineLoopTilePass
    : Pass<"dr-affine-loop-tile", "mlir::func::FuncOp"> {
  let summary = "Tile affine loop nests under the unified cost model";

  let description = [{
    Fork of upstream `--affine-loop-tile` (see
    `lib/Transforms/AffineLoopTilingCostModel/UPSTREAM.md`).  Identical
    behaviour to upstream *except* the tile-size selection placeholder
    (`getTileSizes`) is driven by drcompiler's unified cost model: a small
    grid search over candidate sizes, scored via ArchHandler::combineCosts
    (memory + register pressure + ALU) and vetoed by RegisterPressure
    over-budget constraints.

    See REGISTER_PRESSURE_PLAN.md §16.
  }];

  let options = [
    Option<"cacheSizeInKiB", "cache-size", "uint64_t", /*default=*/"512",
           "Set size of cache to tile for in KiB (default: 512)">,
    Option<"llcGateInKiB", "llc-gate", "uint64_t", /*default=*/"0",
           "Working-set gate (KiB, 0 = off): tile a band ONLY when its full "
           "memory footprint EXCEEDS this size -- i.e. when the untiled set "
           "spills the last-level cache and the kernel is bandwidth-bound. "
           "Set to the (shared) LLC size for the SPMD tile->shard pipeline so "
           "cache-tiling fires at the L3-spill point and stays out of the way "
           "when the set already fits cache (POLYBENCH_PARALLEL_FINDINGS.md #3). "
           "The relevant cache is the SHARED LLC (owner-computes shards share "
           "the read-only operand), not LLC/sharers.">,
    Option<"separate", "separate", "bool", /*default=*/"false",
           "Separate full and partial tiles (default: false)">,
    Option<"tileSize", "tile-size", "unsigned", /*default=*/"",
           "Use this tile size for all loops">,
    ListOption<"tileSizes", "tile-sizes", "unsigned",
               "List of tile sizes for each perfect nest "
               "(overridden by -tile-size)">,
    Option<"useUnifiedCostModel", "use-unified-cost-model", "bool",
           /*default=*/"true",
           "Drive getTileSizes via the unified cost model when no explicit "
           "tile size is set.  When false, falls back to upstream's "
           "nth_root(footprint/cache) placeholder.">,
    Option<"cpuCostModelFile", "cpu-cost-model-file", "std::string",
           /*default=*/"",
           "Path to a JSON cost-model file (arch + registers + cache + ops).">,
    Option<"drArchHandler", "arch-handler", "std::string", /*default=*/"",
           "ArchHandler name (generic | x86-64-avx2 | x86-64-avx512 | arm-neon).">,
    Option<"drSpillStrategy", "spill-strategy", "std::string",
           /*default=*/"\"excess-hot\"",
           "Spill aggregation: excess-hot | sum-excess | graph-color">,
    Option<"emitRationale", "emit-rationale", "bool", /*default=*/"false",
           "Emit a remark on the outermost loop in the band with the "
           "unified cost-model breakdown (chosen tile size, totals, untiled "
           "baseline) for every tiling decision.">
  ];

  let constructor = "mlir::createDrAffineLoopTilePass()";
}

def PrintRegisterPressurePass
    : Pass<"print-register-pressure", "mlir::ModuleOp">
{
  let summary = "Emit per-function register pressure summary";

  let description = [{
    Diagnostic pass for the RegisterPressureAnalysis.  For every
    func::FuncOp in the module, runs the analysis on the function body and
    emits a single-line remark with per-class peak live counts, per-class
    excess sums, and the total spill-cycle estimate under the chosen
    SpillStrategy.

    Used to validate the analysis against hand-crafted MLIR programs in
    Phase 2 of the register-pressure plan.
  }];

  let options = [
    Option<"cpuCostModelFile", "cpu-cost-model-file", "std::string",
           /*default=*/"",
           "Path to a JSON file with cost-model and arch/registers blocks">,
    Option<"handlerOverride", "handler", "std::string", /*default=*/"",
           "Override the JSON handler name (use this handler instead)">,
    Option<"spillStrategyOpt", "strategy", "std::string",
           /*default=*/"\"excess-hot\"",
           "Spill aggregation strategy: excess-hot | sum-excess | graph-color">,
    Option<"tripCount", "trip-count", "uint64_t", "1",
           "Trip count multiplier for ExcessHot strategy">,
    Option<"showPerOp", "show-per-op", "bool", "false",
           "Emit one remark per op in addition to the per-function summary">
  ];

  let constructor = "mlir::createPrintRegisterPressurePass()";
}

def MemoryFissionPass
    : Pass<"memory-fission", "mlir::ModuleOp">
{
  let summary = "Cache-aware loop fission: materialize redundant computation into buffers";

  let description = [{
    Identifies expensive computation subtrees that are duplicated across
    multiple loops (typically produced by cgeist -O2 loop fusion) and, when
    the cost model indicates it is profitable, hoists the shared computation
    into a separate producer loop backed by a materialization buffer.

    This is the inverse of loop fusion: it introduces memory to avoid
    redundant computation when the buffer fits in cache and the ALU cost
    exceeds the load cost.

    The pass fingerprints computation chains within affine.for loop bodies
    to detect structural duplicates reading from the same source memref.
  }];

  let options = [
    Option<"l1Size", "l1-size", "unsigned", "32768",
           "L1 data cache size in bytes (private)">,
    Option<"l2Size", "l2-size", "unsigned", "1048576",
           "Private (per-core) L2 cache size in bytes. The contention-robust "
           "budget: a materialized buffer whose whole working set fits here is "
           "resident regardless of co-tenants.">,
    Option<"l3Size", "l3-size", "unsigned", "33554432",
           "Shared last-level (L3) cache size in bytes (0 = not modeled)">,
    Option<"llcSharers", "llc-sharers", "unsigned", "1",
           "Co-tenants of the shared LLC. The effective L3 a fission buffer can "
           "count on is l3Size/llcSharers; recompute is preferred unless the "
           "source's re-read reuse distance exceeds it (then fission avoids the "
           "thrash). 1 = isolated/optimistic; set to the LLC core count for a "
           "contention-aware (worst-case) estimate.">,
    Option<"l2OccupancyPct", "l2-occupancy-pct", "unsigned", "50",
           "Percent of private L2 a fission working set may use and still be "
           "treated as guaranteed-resident (margin for code/stack/SMT/prefetch).">,
    Option<"l1Latency", "l1-latency", "unsigned", "4",
           "L1 cache hit latency in cycles">,
    Option<"l2Latency", "l2-latency", "unsigned", "12",
           "L2 cache hit latency in cycles">,
    Option<"l3Latency", "l3-latency", "unsigned", "40",
           "L3 cache hit latency in cycles">,
    Option<"testDiagnostics", "test-diagnostics", "bool", "false",
           "Emit remarks for test diagnostics">,
    Option<"cpuCostModelFile", "cpu-cost-model-file", "std::string", /*default=*/"",
           "Path to a JSON file with per-op cycle costs (see scripts/gen_cpu_cost_model.py)">,
    Option<"debug", "debug", "bool", "false",
           "Enable verbose DRDBG (DRCOMP-FISSION:) trace output">
  ];

  let dependentDialects = [
    "memref::MemRefDialect", "affine::AffineDialect",
    "arith::ArithDialect", "math::MathDialect"
  ];

  let constructor = "mlir::createMemoryFissionPass()";
}

def AffineRegisterBlockPass
    : Pass<"affine-register-block", "mlir::func::FuncOp">
{
  let summary = "Register-block a perfectly-nested affine reduction (GEMM)";

  let description = [{
    Brings DPS-style register-blocking to the memref/affine path (cgeist,
    ONNX-MLIR Krnl->Affine, MARCO).  Given a perfectly-nested affine loop
    band whose innermost loop carries a memory reduction with a loop-invariant
    accumulator address (the canonical `C[i,j] += A[i,k]*B[k,j]` GEMM shape),
    the pass:

      1. unroll-and-jams the two outer (parallel) loops by `mr` x `nr`,
         producing an `mr*nr` block of independent accumulators in the
         reduction-loop body (still backed by memory);
      2. promotes each accumulator to an `affine.for` `iter_args` SSA value
         carried across the reduction loop -- the initial load is hoisted
         before the loop and the final store sunk after it.

    The resulting `iter_args` accumulator block is register-allocated and
    SLP-vectorized by the LLVM backend, closing the ~100x gap between naive
    affine GEMM and a register-blocked micro-kernel (see
    `COSTMODEL_SPIKE_FINDINGS.md`: a fixed 8x16 tile is within 5% of the
    per-arch optimum across register files and vector widths, so the tile is
    a fixed knob rather than a cost-model decision).
  }];

  let options = [
    Option<"mr", "mr", "unsigned", "8",
           "Register-block rows (unroll-and-jam factor for the outer loop)">,
    Option<"nr", "nr", "unsigned", "16",
           "Register-block cols (unroll-and-jam factor for the middle loop)">,
    Option<"cacheTile", "cache-tile", "bool", "false",
           "Cache-block the perfectly-nested GEMM band by mc x nc x kc before "
           "register blocking, to keep the working set resident at large N "
           "(register blocking alone is DRAM-bound once the matrices exceed "
           "the last-level cache). Only applies to perfect nests.">,
    Option<"mc", "mc", "unsigned", "256",
           "Cache-tile rows (i) when cache-tile is set">,
    Option<"nc", "nc", "unsigned", "256",
           "Cache-tile cols (j) when cache-tile is set">,
    Option<"kc", "kc", "unsigned", "256",
           "Cache-tile depth (k) when cache-tile is set (upper bound; the tile is "
           "shrunk to fit the effective cache, see l3-size/llc-sharers)">,
    Option<"l3Size", "l3-size", "unsigned", "33554432",
           "Shared last-level (L3) cache size in bytes. Cache tiling fires only "
           "when the band's working set exceeds the EFFECTIVE cache we can count "
           "on (l3Size/llcSharers) -- otherwise the register-blocked kernel is "
           "already cache-resident and tiling only adds point-bound overhead -- "
           "and tile sizes are clamped so the per-tile working set fits it.">,
    Option<"llcSharers", "llc-sharers", "unsigned", "1",
           "Co-tenants of the shared LLC. The cache a fission/tile can count on "
           "is l3Size/this (a co-tenant can evict the rest); higher => tile "
           "sooner and smaller. 1 = isolated; set to the LLC core count for a "
           "contention-aware estimate.">,
    Option<"familySelect", "family-select", "bool", "true",
           "Auto-select the register-block configuration from operand layout: "
           "broadcast family (C=A*B, a multiplicand stride-1 in the spatial dim) "
           "keeps the wide mr x nr tile with no FP reassociation; dot family "
           "(rank-k, multiplicands stride-1 in the reduction dim) uses a small "
           "square tile and sets fastmath<fast> so LLVM vectorizes the "
           "reduction. When false, always use mr x nr with no reassociation.">,
    Option<"vectorize", "vectorize", "bool", "true",
           "For the broadcast family, emit an explicit vector-dialect "
           "micro-kernel (mr rows x vector<vl> accumulators, contiguous "
           "affine.vector_load of the streamed operand + vector.broadcast of the "
           "broadcast operand) instead of unroll-jamming the inner spatial loop "
           "by nr and relying on LLVM SLP. This survives the >2D accumulator "
           "addressing of tensor contractions (batched matmul) where SLP fails. "
           "Falls back to the scalar+SLP path when a band is not cleanly "
           "vectorizable (non-constant bounds, gather access, etc.).">,
    // PORTABILITY (vl): `vl` is an ELEMENT count.  The static default 8 is the
    // value MachineModel::preferredVectorElems() derives for this dev host (AMD
    // Zen4: 256-bit native FP datapath -> 8 f32 / 8 f64-after-reg-floor).  When
    // a cost-model JSON describes the machine's vector-execution model (arch.
    // vector_bits_native / vector_bits_arch / vec_reg_budget) the pass DERIVES
    // vl from it instead (a native-512 Xeon -> 16 for f32), unless --vl pins a
    // value.  WP-G1 finding: vl=8 vs 16 is throughput-neutral for the GEMM
    // broadcast kernel on Zen4 (-0.008 s); the end-to-end vl=8 win is the
    // small-extent conv interiors clearing `interiorWidth >= VL` (a coverage
    // effect -- see conv-vl / WP-G3), NOT register width.  See MachineModel.h
    // and claude-docs/COSTMODEL_PORTABILITY_FINDINGS.md (WP-G1 + Idun spike).
    Option<"vl", "vl", "unsigned", "8",
           "Vector lane ELEMENT count for the broadcast vector micro-kernel "
           "(8 = a 256-bit ymm for f32, a 512-bit zmm for f64). mr x ceil(nr/vl) "
           "accumulators must fit the vector register file. Default 8 is the "
           "Zen4 value derived by MachineModel::preferredVectorElems; a "
           "cost-model JSON's vector model overrides it per-machine unless this "
           "option is set.">,
    Option<"convVl", "conv-vl", "unsigned", "0",
           "Per-stage VL override for the direct-conv Stage 1d only (0 = use vl). "
           "The conv interiors have a SMALLER vectorizable spatial extent than the "
           "GEMM j-dim (a 14x14 conv interior is 8 wide, below a vl=16 zmm), so "
           "the throughput-optimal conv VL differs from the GEMM VL. Introduced "
           "for the WP-G1 attribution (split the vl16->vl8 win between conv "
           "coverage and GEMM ymm-preference); WP-G3 generalizes this to a "
           "per-band pickVL(extent).">,
    Option<"cpuCostModelFile", "cpu-cost-model-file", "std::string",
           /*default=*/"",
           "Path to a JSON cost-model file; sources l3-size/llc-sharers from "
           "its `cache` block unless overridden by the CLI options above.">,
    Option<"peelKTile", "peel-k-tile", "unsigned", "128",
           "Target k-chunk size for tiling the reduction loop of a peeled "
           "triangular HEAD (0 = off). Applies only when some multiplicand is "
           "k-strided (column access, e.g. covariance's data[k][i]) AND the "
           "streamed working set exceeds l3-size/llc-sharers: the HEAD then "
           "re-streams the full data matrix per row-strip with a >page stride "
           "(20.8 KB row at XL), touching a new TLB page every k step. The "
           "actual chunk is the largest divisor of the k-trip <= this target "
           "with a floor of 64 (targets < 64 disable the chunking), so no "
           "remainder chunk is needed. Measured on covariance XL: a wide "
           "plateau at targets 75-150 (11.0-11.5x vs base; untiled HEAD was "
           "3.7x, Polly 6.25x); 128 sits mid-plateau. Dot-family bands "
           "(stride-1 in k) are excluded by the gate and unaffected.">
  ];

  let dependentDialects = [
    "memref::MemRefDialect", "affine::AffineDialect", "arith::ArithDialect",
    "vector::VectorDialect"
  ];

  let constructor = "mlir::createAffineRegisterBlockPass()";
}

def DrMathStrengthReducePass
    : Pass<"dr-math-strength-reduce", "mlir::func::FuncOp">
{
  let summary = "Strength-reduce math.powf(x, integer) to multiplications";

  let description = [{
    Rewrites `math.powf %x, %c` to repeated `arith.mulf` (exponentiation by
    squaring) when the exponent is a small non-negative integer constant.
    Exact for integer exponents (x^3 = x*x*x), so unconditionally safe
    regardless of fastmath.

    GPT-2's tanh-approximation GELU computes its cubic term as `powf(x, 3.0)`
    -- a full libm transcendental call per element that neither clang -O2 nor
    onnx-mlir --O3 strength-reduces (the `math.powf` carries no fastmath flag,
    so the backend leaves it a `call powf@plt`).  Two multiplies replace it,
    removing ~4.7M libm calls per openai-gpt inference and unblocking
    vectorization of the surrounding pointwise loop.
  }];

  let options = [
    Option<"maxExponent", "max-exponent", "unsigned", /*default=*/"8",
           "Only rewrite integer exponents in [1, max-exponent]">,
    Option<"polyApprox", "poly-approx", "bool", /*default=*/"false",
           "Also lower math.exp/tanh/... to vectorizable polynomial "
           "approximations (upstream populateMathPolynomialApproximationPatterns). "
           "Removes the libm call barrier so the surrounding pointwise loop "
           "auto-vectorizes -- closes the openai-gpt transcendental backend gap "
           "(onnx-mlir --O3 EmitObj does this natively; libmvec cannot vectorize "
           "tanh, so a backend flag is insufficient). Off by default: the "
           "approximations trade ~1e-6 accuracy, gated by the bench norm-rel-err.">,
  ];

  let dependentDialects = [
    "mlir::arith::ArithDialect", "mlir::math::MathDialect",
    "mlir::vector::VectorDialect"
  ];
  let constructor = "mlir::createDrMathStrengthReducePass()";
}

def DrParBubblesPass : Pass<"dr-par-bubbles", "mlir::ModuleOp">
{
  let summary = "Seed per-loop parallel bubbles and classify loop axes (M0)";

  let description = [{
    Front-end of the bubble-widening parallel-codegen track
    (PARALLEL_BUBBLE_SPEC.md).  M0: seed one bubble per loop (affine.for /
    scf.for) and classify each loop's axis via the intra-module
    `ParAliasOracle`:

      PARALLEL     - no loop-carried dependence on this axis (distributable)
      SEQUENTIAL   - carried dependence | reduction (iter_args) | conservative
                     (non-affine / SCF / unprovable aliasing)

    The oracle is tiered: Tier 0 allocation-root provenance disjointness,
    Tier 1 polyhedral affine dependence (checkMemrefAccessDependence), Tier 2
    conservative.  M0 is diagnostic-only (no IR mutation); widening to a fixed
    point, the `par` dialect, peeling/redistribution, and interprocedural call
    consumption land in M1-M4.  Cross-procedure forwarding and SCF dependence
    are deferred (PARALLEL_BUBBLE_SPEC.md §2, §12).
  }];

  let options = [
    Option<"parTestDiagnostics", "par-test-diagnostics", "bool", "false",
           "Emit a per-loop axis classification remark "
           "(PARALLEL / SEQUENTIAL ...)  [M0]">,
    Option<"parTestRegions", "par-test-regions", "bool", "false",
           "Emit a per-maximal-region remark after widening to a fixed point "
           "(perfect-nest climb + conformant sibling fuse): par-region / "
           "par-frontier.  [M1]">,
    Option<"parMaterialize", "par-materialize", "bool", "false",
           "Materialize maximal single-band climb regions into the `par` "
           "dialect (par.region / par.forall; sequential suffix as scf.for; "
           "affine accesses expanded to memref).  First IR-mutating mode.  "
           "[M2]">,
    Option<"parTestSpmd", "par-test-spmd", "bool", "false",
           "Diagnostic-only: pick a whole-function shard axis (S0) and classify "
           "each inter-band edge for barrier elision (S1) -- ELIDE / HALO / "
           "REDISTRIBUTE / BARRIER. Emits par-shard / par-edge / par-spmd "
           "remarks.  See PARALLEL_SPMD_SPEC.md.">,
    Option<"parSpmdPerband", "par-spmd-perband", "bool", "false",
           "Materialize batch-1 within-sample SPMD (S7): ONE par.region; each "
           "top-level band shards its OWN outermost parallel loop into a "
           "par.forall (extents may differ per layer; inner loops incl. the "
           "reduction stay scf.for, within-shard), with a par.barrier between "
           "consecutive bands (the consumer reads the full producer output). "
           "Bails (with a count) if any band's outer loop is not a "
           "materializable parallel axis. See PARALLEL_SPMD_SPEC.md.">,
    Option<"parSpmd", "par-spmd", "bool", "false",
           "Materialize whole-function SPMD (S2): widen every top-level "
           "shard-axis band into ONE par.region, hoist one par.forall over the "
           "shard axis (maximal ELIDE-connected runs share a forall, "
           "owner-computes; inner loops sink as scf.for), and place "
           "par.barrier / par.redistribute only at non-elided edges. Constant "
           "shard extent only; bails (with a remark) on a dynamic axis, an "
           "off-axis / non-materializable / non-contiguous band, or an "
           "inter-band value dependence.  See PARALLEL_SPMD_SPEC.md §7 (S2).">,
  ];

  let dependentDialects = [
    "affine::AffineDialect", "scf::SCFDialect",
    "memref::MemRefDialect", "func::FuncDialect",
    "arith::ArithDialect", "par::ParDialect"
  ];

  let constructor = "mlir::createDrParBubblesPass()";
}

def ConvertParToSCFPass : Pass<"convert-par-to-scf", "mlir::func::FuncOp">
{
  let summary = "Lower the `par` dialect to scf (sequential-equivalent sink)";

  let description = [{
    Lowering target for the `par` dialect (PARALLEL_PAR_DIALECT_SPEC.md §4.3):
    the M2 correctness/test sink.

      par.forall  -> scf.parallel (constant bounds materialized as arith index)
      par.region  -> inlined into the enclosing block
      par.yield   -> erased

    scf.parallel preserves the parallel semantics but executes correctly under
    any schedule, so a materialized region's results can be diffed against the
    original affine program.  par->omp and par->libdrpar are later phases.
  }];

  let dependentDialects = [
    "scf::SCFDialect", "arith::ArithDialect", "memref::MemRefDialect"
  ];

  let constructor = "mlir::createConvertParToSCFPass()";
}

def LowerKrnlGlobalPass : Pass<"lower-krnl-global", "mlir::ModuleOp">
{
  let summary = "Rewrite krnl.global -> memref.global (krnl-free SPMD lowering)";

  let description = [{
    Strips the krnl ops that survive into the affine stage so a whole-kernel
    SPMD (par->omp) module can be lowered to LLVM by HOST mlir-opt (onnx-mlir's
    monolithic convert-krnl-to-llvm rejects omp/scf.parallel):

      krnl.global {name,shape,value} : () -> memref  =>  memref.global +
                                                          memref.get_global
      krnl.entry_point ...                            =>  erased

    krnl ops are unregistered (run dr-opt with -allow-unregistered-dialect);
    matched by op name.  See PARALLEL_SPMD_SPEC.md (parallel back-end).
  }];

  let dependentDialects = ["memref::MemRefDialect", "scf::SCFDialect",
                           "arith::ArithDialect"];
  let constructor = "mlir::createLowerKrnlGlobalPass()";
}

def ConvertParToOMPPass : Pass<"convert-par-to-omp", "mlir::func::FuncOp">
{
  let summary = "Lower the `par` dialect to OpenMP (faithful whole-kernel SPMD)";

  let description = [{
    The faithful SPMD lowering (PARALLEL_SPMD_SPEC.md §7, S3):

      par.region        -> omp.parallel            (ONE team for the kernel)
      par.forall        -> omp.wsloop { omp.loop_nest }   (static worksharing)
        elided edge       -> `nowait` on the wsloop (the explicit boundary
                            omp.barrier provides the sync; no double barrier)
      par.barrier        -> omp.barrier             (only at non-elided edges)
      par.redistribute   -> omp.barrier             (conservative; reblock is
                            Phase B)

    One omp.parallel for the whole region with barriers only where S1 kept
    them -- strictly fewer fork/joins than the per-loop
    `par -> scf.parallel -> --convert-scf-to-openmp` compose.  Lower further
    with --convert-openmp-to-llvm and run with the OpenMP runtime.
  }];

  let dependentDialects = [
    "omp::OpenMPDialect", "arith::ArithDialect", "memref::MemRefDialect"
  ];

  let constructor = "mlir::createConvertParToOMPPass()";
}

#endif
