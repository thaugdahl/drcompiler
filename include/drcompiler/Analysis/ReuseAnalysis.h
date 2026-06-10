//===- ReuseAnalysis.h - per-reference loop reuse classification ----------===//
//
// Shared profitability primitive for the affine transform suite (tiling,
// distribution, fusion).  For a perfect affine band with constant bounds it
// classifies, per memory reference and per band loop, what kind of reuse the
// loop carries for that reference, and provides analytic working-set /
// reuse-distance estimates parameterized by hypothetical tile sizes.
//
// The model is deliberately constant-coefficient: every subscript must be a
// linear expression of band IVs with constant coefficients (what cgeist -O0 +
// raise-scf-to-affine emit for PolyBench-style code).  Anything else —
// semi-affine subscripts, non-band IVs, symbols, unknown trip counts — makes
// `analyzeBandReuse` return failure, and clients are expected to fall back to
// their conservative default (for the tiler: do not tile).
//
// Footprints are per-reference products of per-dimension extents summed over
// distinct references.  Overlapping references to the same array (stencil
// neighbours A[i][j-1] / A[i][j+1]) are counted separately, so footprints are
// mild over-estimates; that errs toward "evicted" in reuse-distance checks,
// i.e. toward tiling, never toward miscounting reuse as absent.
//
//===----------------------------------------------------------------------===//

#ifndef DRCOMPILER_ANALYSIS_REUSEANALYSIS_H
#define DRCOMPILER_ANALYSIS_REUSEANALYSIS_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace drcompiler {
namespace reuse {

/// What reuse a single band loop carries for a single reference.
enum class ReuseKind {
  /// The loop's IV does not appear in any subscript: successive iterations
  /// touch the SAME elements (temporal reuse carried by this loop).
  Invariant,
  /// The IV appears only in the innermost (fastest-varying) subscript with
  /// |coefficient| == 1: successive iterations walk a cache line.
  Spatial,
  /// The IV strides to a new cache line every iteration.
  Streaming,
};

/// One distinct memory reference (affine.load / affine.store) in the band.
/// References with identical (memref, map, operands) are deduplicated; the
/// classic load+store accumulator pair becomes one entry with multiplicity 2.
struct RefInfo {
  /// Representative op (first occurrence in walk order).
  mlir::Operation *op = nullptr;
  mlir::Value memref;
  /// Number of syntactic accesses folded into this entry.
  unsigned multiplicity = 1;
  /// True if any folded access is a store.
  bool isWrite = false;
  int64_t elemBytes = 0;
  /// coeff[d][l] = coefficient of band loop l's IV in subscript dim d.
  llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4> coeff;
  /// Static memref dim sizes (-1 if dynamic) used to clamp extents.
  llvm::SmallVector<int64_t, 4> dimSizes;
  /// Per band loop classification.
  llvm::SmallVector<ReuseKind, 4> kinds;

  bool invariantIn(unsigned loopIdx) const {
    return kinds[loopIdx] == ReuseKind::Invariant;
  }
};

struct BandReuseInfo {
  llvm::SmallVector<mlir::affine::AffineForOp, 6> band;
  /// Trip count per band loop (exact, or an upper bound when the matching
  /// tripIsExact entry is false — see acceptTripUpperBounds).
  llvm::SmallVector<uint64_t, 6> tripCounts;
  /// False for loops whose trip count is an upper-bound estimate
  /// (triangular bounds: constant-ub minus the constant/zero lb floor).
  llvm::SmallVector<bool, 6> tripIsExact;
  llvm::SmallVector<RefInfo, 8> refs;

  /// Bytes touched by one tile with the given per-loop iteration counts
  /// (tileSizes.size() == band.size(); pass tripCounts for the full band
  /// footprint).  Saturates at INT64_MAX.
  int64_t footprintBytes(llvm::ArrayRef<uint64_t> tileSizes) const;

  /// Footprint of a single reference within one iteration of band loop
  /// `loopIdx` (outer loops and the loop itself fixed, inner loops full).
  int64_t refIterFootprint(unsigned refIdx, unsigned loopIdx) const;

  /// Bytes the given reference touches within one tile of the given shape.
  int64_t refFootprintBytes(unsigned refIdx,
                            llvm::ArrayRef<uint64_t> tileSizes) const;

  /// Bytes touched (all references) between consecutive reuses carried by
  /// band loop `loopIdx` — i.e. one full iteration of that loop.
  int64_t reuseDistanceBytes(unsigned loopIdx) const;

  /// True if `loopIdx` carries non-degenerate temporal reuse (an invariant
  /// reference spanning more than one cache line per iteration of the loop)
  /// that is evicted at a cache of `cacheBytes` (reuse distance exceeds it).
  /// This is the tiler's profitability gate: tiling the inner loops shrinks
  /// the reuse distance below `cacheBytes` and converts misses to hits.
  bool loopCarriesEvictedReuse(unsigned loopIdx, int64_t cacheBytes) const;

  /// True if any loop carries non-degenerate temporal reuse for any ref.
  bool anyTemporalReuse() const;
};

/// Analyze a perfect band of affine loops with constant bounds.  Fails (see
/// file header) whenever the constant-coefficient model does not apply.
/// `walkRoot` restricts which references are collected (default: everything
/// under the band root) — clients evaluating a HYPOTHETICAL band, e.g. loop
/// distribution asking "would this child nest, once isolated under these
/// loops, carry exploitable reuse?", pass the child as walkRoot so sibling
/// units still present in the IR don't poison the analysis.
/// `acceptTripUpperBounds`: instead of failing on a loop without a constant
/// trip count, accept a constant upper bound (constant ub minus the
/// constant lb if present, else 0 — assumes non-negative IVs, which holds
/// for the normalized loops this analysis targets).  Footprints and reuse
/// distances become upper bounds, which errs toward "evicted", i.e. toward
/// transforming — appropriate for the tiler on triangular bands
/// (covariance/syrk j = i..M), not for exactness-sensitive clients.
mlir::FailureOr<BandReuseInfo>
analyzeBandReuse(llvm::ArrayRef<mlir::affine::AffineForOp> band,
                 mlir::Operation *walkRoot = nullptr,
                 bool acceptTripUpperBounds = false);

} // namespace reuse
} // namespace drcompiler

#endif // DRCOMPILER_ANALYSIS_REUSEANALYSIS_H
