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

  /// Effective last-level cache after dividing by co-tenant sharers.  A reuse
  /// distance beyond this is priced as a memory access, not an L3 hit.
  int64_t effectiveLLC() const {
    return l3Size / static_cast<int64_t>(llcSharers ? llcSharers : 1u);
  }

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
