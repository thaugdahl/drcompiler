//===- MachineModel.h - One machine, one description ----------------------===//
//
// A single resolved description of the target's memory hierarchy, shared by
// every cost-modelled pass (register-block, stencil-time-tile, loop-tile,
// loop-distribute, memory-fission, data-recomputation).  Before v4 each pass
// carried its own l1/l2/l3/latency/sharers CLI options with independent
// defaults; the v2 notes record one drift incident (a stale l2 default 4x too
// small on one path).  MachineModel makes the cost-model JSON the single
// source of truth.
//
// Resolution contract (per COSTMODEL_V4_SPEC §2):
//   * Fields default to the historical built-in values (so a run with NO JSON
//     and NO CLI override is bit-for-bit what it was before v4).
//   * `cpu-cost-model-file` JSON, if present, overrides those defaults.
//   * A pass CLI option, if EXPLICITLY set (hasValue()), wins
//     over both — it is an override, not a competing source of truth.
//
//===----------------------------------------------------------------------===//

#ifndef DRCOMPILER_ANALYSIS_MACHINEMODEL_H
#define DRCOMPILER_ANALYSIS_MACHINEMODEL_H

#include "llvm/ADT/StringRef.h"
#include <algorithm>
#include <cstdint>

namespace drcompiler {

/// A resolved view of the target memory hierarchy.  All sizes are in bytes,
/// all latencies in cycles.  Defaults match the pre-v4 per-pass CLI defaults
/// exactly; see MachineModel.cpp.
struct MachineModel {
  // Cache sizes (bytes).
  int64_t l1Size = 32768;      // dr-l1-size / l1-size default
  int64_t l2Size = 1048576;    // dr-l2-size / l2-size default
  int64_t l3Size = 33554432;   // dr-l3-size / l3-size default
  int64_t cacheLine = 64;      // dr-cache-line-size default

  // Shared-LLC contention: effective LLC = l3Size / llcSharers.
  unsigned llcSharers = 1;     // dr-llc-sharers / llc-sharers default

  // Page / TLB locality (NEW in v4).  Used by the A2 k-chunk target and any
  // page-locality-sensitive tiling.  Defaults are typical Zen4/SKX: 4 KB
  // pages, ~1.5K-entry L2 TLB.
  int64_t pageSize = 4096;
  int64_t l2TlbEntries = 1536;

  // Latencies (cycles).
  unsigned l1Lat = 4;          // dr-l1-latency / l1-latency default
  unsigned l2Lat = 12;         // dr-l2-latency / l2-latency default
  unsigned l3Lat = 40;         // dr-l3-latency / l3-latency default
  unsigned memLat = 200;       // dr-mem-latency / mem-latency default

  // --- Vector-execution model (NEW: WP-G1 portable register-block VL) -------
  // The throughput-optimal vector ELEMENT count for the register-block
  // micro-kernels is the NATIVE FP datapath width in elements, raised if the
  // accumulator tile would not fit the vector register file.  Measured on
  // resnet50 (f32) + the AVX-512 portability spike (f64):
  //   * WIDER than the native datapath buys NO FP throughput -- a wide op is
  //     cracked/double-pumped (on Zen4 a 512-bit FMA is 2 uops over the 2x256
  //     FP pipes, the same FLOP rate as 2 ymm-FMA) -- while halving the
  //     architectural accumulator count (less ILP to hide FMA latency) and, on
  //     Intel, tripping AVX-512 frequency licensing (avx512FreqThrottle).
  //     Confirmed: resnet50 GEMM at vl=16 (zmm) vs vl=8 (ymm) is within noise
  //     (-0.008 s); the vl=8 end-to-end win is purely the small-extent (14x14)
  //     conv interiors clearing `interiorWidth >= VL` -- a coverage effect, not
  //     a datapath one.
  //   * NARROWER than native wastes datapath on a native-wide part.
  //   * BUT the tile (mr*ceil(nr/vl) accumulators) must fit the register file:
  //     a narrow element type at the native width can demand too many regs
  //     (f64 at 256-bit native = 4 elems -> 32 accumulators at mr8/nr16 ->
  //     spill), so vl is raised until it fits.
  // Defaults describe this dev host, AMD Zen4 (Ryzen 7950X3D): the 2x256-bit FP
  // pipes execute AVX-512 double-pumped, so the native datapath is 256-bit even
  // though the ISA encodes 512-bit zmm.  On a native-512 Intel part (e.g. the
  // planned Idun / Xeon Gold spike) a cost-model JSON sets vectorBitsNative=512
  // and f32 derives vl=16.
  int64_t vectorBitsNative = 256; // throughput-effective FP datapath width
  int64_t vectorBitsArch = 512;   // widest ISA-encodable vector (zmm)
  int64_t vecRegBudget = 24;      // vector regs usable for accumulators (of 32
                                  // zmm; ~8 reserved for stream + broadcast)

  // AVX-512 frequency licensing (Intel only): sustained 512-bit FMA drops the
  // core to a lower license frequency (L1/L2 downclock, ~0.85-0.90 of base).
  // DOCUMENTED as a portability knob even though it is INERT on Zen4 (=1.0, no
  // AVX-512 license).  It does not flip the VL DECISION on either Zen4
  // (native 256 < arch 512 -> native wins regardless of throttle) or a
  // native-512 Intel part (wide == native -> throttle scales the whole kernel
  // uniformly, real throttle >= 0.85 >> 0.5 never beats half-width); it feeds
  // cross-machine THROUGHPUT prediction and is the parameter the planned Idun
  // (56-core Xeon Gold) spike measures to confirm the f32 vl=16 choice on a
  // real native-512 part.  See claude-docs/COSTMODEL_PORTABILITY_FINDINGS.md (WP-G1).
  double avx512FreqThrottle = 1.0;

  // True once a cost-model JSON explicitly set any vector-execution field.  The
  // register-block pass derives `vl` from preferredVectorElems() ONLY then, so
  // the default machine keeps the static vl option default (8 == the Zen4
  // value) and every pre-WP-G1 lit test stays byte-identical.
  bool hasExplicitVectorModel = false;

  // --- Thread / parallel-execution model (NEW: CROSSCUTTING.md III.2-III.3) --
  // Every field defaults to a 1-thread, bandwidth-disabled NO-OP: with no
  // explicit thread block the cost is bit-for-bit the single-thread model.
  //   * activeThreads  - parallel workers actually running (roofline divisor).
  //   * smtPerCore     - SMT siblings sharing a physical private L1/L2.
  //   * coresPerLLC    - co-tenants of the shared LLC; an alias of llcSharers
  //                      (the existing field stays the source so old JSON works).
  //   * dram/llcBytesPerCycle - sustained streaming bandwidth in BYTES/CYCLE,
  //                      0 => disabled (latency-only, byte-identical to today).
  //   * lNShared       - cache-sharing topology (default: L1/L2 private, L3
  //                      shared -- correct for Zen4/SKX).
  struct ThreadModel {
    unsigned activeThreads = 1;
    unsigned smtPerCore = 1;
    double dramBytesPerCycle = 0.0;
    double llcBytesPerCycle = 0.0;
    bool l1Shared = false, l2Shared = false, l3Shared = true;
    // Workload deployment mode (the III.4a fork), the per-level bandwidth share:
    //   * INTERSPERSED (default): the kernel is one of `activeThreads` co-equal
    //     tenants sharing the machine -- each gets BW/activeThreads, so a reload
    //     costs bytes * activeThreads / BW (cost RISES with load). Pairs with a
    //     shared, derated LLC (llcSharers). This is what the P1 roofline bench
    //     measured.
    //   * EXCLUSIVE: the kernel OWNS the machine and is parallelized across the
    //     cores -- it gets the full BW (the per-thread WS/N and per-thread BW/N
    //     cancel), so a reload costs bytes / BW. Pairs with an un-derated LLC
    //     (set llc_sharers=1). The single-big-parallel-job deployment.
    bool exclusive = false;
  };
  ThreadModel thread;
  // True once a cost-model JSON explicitly set any thread-model field; the
  // roofline term and per-thread cache split fire ONLY then, so the default
  // machine is byte-identical.
  bool hasExplicitThreadModel = false;

  enum CacheLevel { L1, L2, L3 };

  /// SMT-aware private-cache share: an SMT sibling sharing a physical L1/L2
  /// halves the capacity a thread can count on; `shared=false` (no sibling) or
  /// `smt<=1` reproduces the full size.  Static so consumers can pass their
  /// already-resolved (CLI-overridable) cache size.
  static int64_t effectivePrivateCache(int64_t size, bool shared,
                                       unsigned smt) {
    return shared ? size / static_cast<int64_t>(smt ? smt : 1u) : size;
  }

  /// Per-thread share of cache level `lv` under the current topology: a shared
  /// LLC is divided by co-tenants (llcSharers); a private L1/L2 is divided by
  /// SMT siblings; default smtPerCore=1 / llcSharers=1 reproduces full size.
  int64_t effectiveCache(CacheLevel lv) const {
    switch (lv) {
    case L1:
      return effectivePrivateCache(l1Size, thread.l1Shared, thread.smtPerCore);
    case L2:
      return effectivePrivateCache(l2Size, thread.l2Shared, thread.smtPerCore);
    case L3:
      return thread.l3Shared ? effectiveLLC() : l3Size;
    }
    return l3Size;
  }

  /// Bandwidth-bound cycles to STREAM `bytes` from DRAM (or the shared LLC),
  /// given the per-thread share of that level's bandwidth.  The roofline arm of
  /// the cost: real move time = max(latency-estimate, this).  Returns 0 when the
  /// bandwidth is unmodelled (no thread JSON) so the caller's existing latency
  /// estimate stands unchanged -- the byte-identical gate.
  double streamCycles(int64_t bytes, bool fromDRAM) const {
    double bw = fromDRAM ? thread.dramBytesPerCycle : thread.llcBytesPerCycle;
    if (bw <= 0.0)
      return 0.0;
    // INTERSPERSED: the kernel shares the level's bandwidth with activeThreads
    // co-tenants (effective BW = BW/activeThreads).  EXCLUSIVE: it owns the full
    // bandwidth (the parallelized work's per-thread WS/N and BW/N cancel).
    double effBW =
        thread.exclusive
            ? bw
            : bw / static_cast<double>(thread.activeThreads ? thread.activeThreads
                                                            : 1u);
    return static_cast<double>(bytes) / effBW;
  }

  /// Effective last-level cache after dividing by co-tenant sharers.  A reuse
  /// distance beyond this is priced as a memory access, not an L3 hit.  The
  /// static form is the ONE definition of the contention derate: every consumer
  /// that has already resolved l3Size/llcSharers into local values (the pass
  /// options, possibly CLI-overridden) routes through it instead of re-deriving
  /// `l3 / sharers` inline, so there is a single place to make the share
  /// per-thread (CROSSCUTTING.md P0/ThreadModel).
  static int64_t effectiveLLC(int64_t l3Size, unsigned llcSharers) {
    return l3Size / static_cast<int64_t>(llcSharers ? llcSharers : 1u);
  }
  int64_t effectiveLLC() const { return effectiveLLC(l3Size, llcSharers); }

  /// Bytes addressable without a second-level TLB miss: pageSize * entries.
  /// The natural ceiling for a streaming slab's k-chunk before page-walk cost
  /// dominates.
  int64_t tlbReachBytes() const { return pageSize * l2TlbEntries; }

  /// Throughput-optimal vector ELEMENT count for an `elemBytes`-wide datatype
  /// in a register-block tile of `mr` x `nr` unroll-jam factors.  The native
  /// datapath width in elements, raised (halving the accumulator count) until
  /// the mr*ceil(nr/vl) accumulator tile fits vecRegBudget, never above the
  /// encodable width.  Result is in [1, vectorBitsArch/bits] (a power of two
  /// for the power-of-two bit fields of any real machine).  See the field
  /// comments above for the model and its measured basis.
  ///
  /// Reproduces every known data point:
  ///   Zen4 (native 256): f32 -> 8, f64 -> 4 raised to 8 (fits) == old default.
  ///   Xeon Gold (native 512): f32 -> 16, f64 -> 8.
  int64_t preferredVectorElems(int64_t elemBytes, unsigned mr,
                               unsigned nr) const {
    int64_t bits = 8 * (elemBytes > 0 ? elemBytes : 4);
    int64_t archE = vectorBitsArch / bits;
    if (archE < 1)
      archE = 1;
    int64_t nativeE = vectorBitsNative / bits;
    if (nativeE < 1)
      nativeE = 1;
    // Clamp native to the encodable width: a malformed model with
    // vectorBitsNative > vectorBitsArch must never emit a vector wider than the
    // register file (fromJson also warns + corrects that case).
    int64_t vl = nativeE < archE ? nativeE : archE;
    auto accs = [&](int64_t v) {
      return static_cast<int64_t>(mr) * ((static_cast<int64_t>(nr) + v - 1) / v);
    };
    while (vl < archE && accs(vl) > vecRegBudget)
      vl *= 2;
    return vl;
  }

  // --- GEMM blocking model (NEW: TRANSFORMER_KRNL_SPEC WP-T1) ---------------
  // Register-file budgets bridged from the cost-model JSON `registers` block
  // (CpuRegisterJsonParams: gp/fp/vec/predBudget).  CpuCostModel already parses
  // them but they never reached MachineModel, so the graph-coloring budget and
  // the vector-tile budget (`vecRegBudget`) were two disconnected numbers.
  // Reconciliation (CROSSCUTTING): vecRegBudget (24) = the *accumulator* slice
  // of the full vector file (`registers.vec`, 32 on AVX-512) minus regs reserved
  // for the streaming B-panel + broadcast temporaries.  Register-tile sizing
  // uses vecRegBudget; RegisterPressureAnalysis uses the full registers.vec.
  // Defaults are the x86-64 / AVX-512 architectural counts.
  struct RegisterBudget {
    unsigned gp = 16;   // general-purpose (addresses, loop ivs)
    unsigned fp = 32;   // scalar FP / vector regs
    unsigned vec = 32;  // full vector register file (32 zmm)
    unsigned pred = 8;  // predicate / mask regs (8 AVX-512 k-regs)
  };
  RegisterBudget registers; // bridged in fromJson from the `registers` block

  // True once a cost-model JSON provides a GEMM-model signal (the `registers`
  // block today; arch.fmaUnits in WP-T2; a dedicated `gemm` block later).  The
  // register-block pass consults gemmBlocking() ONLY then, so the default
  // machine keeps the static mr/nr/vl + cache-tile-off path -> byte-identical.
  bool hasExplicitGemmModel = false;

  /// Register-block kernel family selected by gemmBlocking().
  enum class GemmKernel {
    Broadcast,    // deep-K: the mr x nr broadcast micro-kernel (today's path)
    OuterProduct, // tiny-K: jam M & N, fully unroll K in registers (WP-T4)
    Gemv          // M=1 (WP-T7, deferred)
  };

  /// One GEMM's resolved blocking decision (TRANSFORMER_KRNL_SPEC §4.3).
  /// kc/mc/nc == 0 means "untiled / full extent".
  struct GemmTiling {
    unsigned mr = 8, nr = 16, vl = 8;
    int64_t kc = 0, mc = 0, nc = 0;
    GemmKernel kind = GemmKernel::Broadcast;
    bool cacheTile = false;
  };

  /// Does the mr x ceil(nr/vl) accumulator tile fit the accumulator register
  /// budget (vecRegBudget)?  Extracted from preferredVectorElems' fit loop so
  /// the GEMM configurator and the VL deriver share ONE definition.
  bool canFitAccumulators(unsigned mr, unsigned nr, unsigned vl) const {
    if (vl == 0)
      return false;
    int64_t per = (static_cast<int64_t>(nr) + vl - 1) / vl;
    return static_cast<int64_t>(mr) * per <= vecRegBudget;
  }

  /// Largest k-panel such that the A(mr x kc) + B(kc x nr) + C(mr x nr) panels
  /// fit the per-thread effective L1.  `elemBytes` in BYTES.  Returned UNROUNDED;
  /// the caller rounds kc down to a multiple of vl when it needs stride-1
  /// vectorized B-panel loads.  No L1-panel concept existed before WP-T1.
  int64_t maxL1Kc(unsigned mr, unsigned nr, int64_t elemBytes) const {
    int64_t eb = elemBytes > 0 ? elemBytes : 4;
    int64_t budget = effectiveCache(L1) / eb - static_cast<int64_t>(mr) * nr;
    int64_t denom = static_cast<int64_t>(mr) + nr;
    if (denom <= 0)
      return 1;
    int64_t kc = budget / denom;
    return kc < 1 ? 1 : kc;
  }

  /// A macro-tile (mc,nc,kc) and whether the caller should skip tiling.
  struct MacroTile {
    int64_t mc, nc, kc;
    bool skip; // band already fits the budget, or the tile spans the full extent
  };

  /// Shrink the register-block macro-tile (mc,nc,kc), each clamped to its extent
  /// (M,N,K), by halving the largest dim until the per-tile working set
  /// (mc*kc + kc*nc + mc*nc)*elemBytes fits `budgetBytes` (typically the
  /// effective LLC).  `skip` is set when the whole band already fits the budget
  /// (no tiling needed) or the resulting tile spans the full extent (degenerate
  /// -- tiling would only add scalarizing point bounds).  Static + pure: a
  /// byte-identical extraction of the inline cache-tile loop that was in
  /// AffineRegisterBlock.cpp, now the ONE definition shared by the pass and the
  /// GEMM configurator.
  static MacroTile macroTile(int64_t M, int64_t N, int64_t K, int64_t mc,
                             int64_t nc, int64_t kc, int64_t mr, int64_t nr,
                             int64_t vl, int64_t elemBytes, int64_t budgetBytes) {
    int64_t eb = elemBytes > 0 ? elemBytes : 8;
    int64_t ws = (M * K + K * N + M * N) * eb;
    if (budgetBytes <= 0 || ws <= budgetBytes)
      return {0, 0, 0, true};
    int64_t tmc = std::min<int64_t>(mc, M), tnc = std::min<int64_t>(nc, N),
            tkc = std::min<int64_t>(kc, K);
    auto tileWS = [&]() { return (tmc * tkc + tkc * tnc + tmc * tnc) * eb; };
    while (tileWS() > budgetBytes) {
      if (tmc >= tnc && tmc >= tkc && tmc > mr)
        tmc = std::max<int64_t>(mr, tmc / 2);
      else if (tnc >= tkc && tnc > nr)
        tnc = std::max<int64_t>(nr, tnc / 2);
      else if (tkc > vl)
        tkc = std::max<int64_t>(vl, tkc / 2);
      else
        break; // can't shrink further; tile anyway (better than DRAM-bound)
    }
    if (tmc >= M && tnc >= N && tkc >= K)
      return {tmc, tnc, tkc, true};
    return {tmc, tnc, tkc, false};
  }

  /// THE GEMM configurator (TRANSFORMER_KRNL_SPEC §4.3): one query owns the GEMM
  /// tiling decision for a static (M,N,K) contraction.  WP-T1 skeleton -- always
  /// the Broadcast family with today's tiling (mr=8, nr=16, vl from the vector
  /// model, untiled).  WP-T2 adds the roofline kernel-kind dispatch; WP-T3 wires
  /// the pass to call this (only when hasExplicitGemmModel); WP-T5 turns on
  /// cache-tiling for deep-K.  Pure function of the machine + (M,N,K).
  GemmTiling gemmBlocking(int64_t /*M*/, int64_t /*N*/, int64_t /*K*/,
                          int64_t elemBytes) const {
    GemmTiling t;
    t.mr = 8;
    t.nr = 16;
    t.vl = static_cast<unsigned>(preferredVectorElems(elemBytes, t.mr, t.nr));
    t.kind = GemmKernel::Broadcast;
    t.cacheTile = false; // WP-T5 flips this for deep-K
    return t;            // mc=nc=kc=0 => untiled
  }

  /// Built-in defaults (no JSON).
  static MachineModel getDefault() { return MachineModel{}; }

  /// Load from a cost-model JSON file (same file as `cpu-cost-model-file`).
  /// Any field absent from the JSON keeps its built-in default.  On read/parse
  /// failure, logs a warning and returns defaults.  An empty path returns
  /// defaults without touching the filesystem.
  static MachineModel fromJson(llvm::StringRef path);
};

} // namespace drcompiler

#endif // DRCOMPILER_ANALYSIS_MACHINEMODEL_H
