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

  /// Effective last-level cache after dividing by co-tenant sharers.  A reuse
  /// distance beyond this is priced as a memory access, not an L3 hit.
  int64_t effectiveLLC() const {
    return l3Size / static_cast<int64_t>(llcSharers ? llcSharers : 1u);
  }

  /// Bytes addressable without a second-level TLB miss: pageSize * entries.
  /// The natural ceiling for a streaming slab's k-chunk before page-walk cost
  /// dominates.
  int64_t tlbReachBytes() const { return pageSize * l2TlbEntries; }

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
