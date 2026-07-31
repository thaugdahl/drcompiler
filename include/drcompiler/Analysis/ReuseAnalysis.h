//===- ReuseAnalysis.h - per-reference loop reuse classification ----------===//
//
// Shared profitability primitive for the affine transform suite (tiling,
// distribution, fusion).  For a perfect affine band with constant bounds it
// classifies, per memory reference and per band loop, what kind of reuse the
// loop carries for that reference, and provides analytic working-set /
// reuse-distance estimates parameterized by hypothetical tile sizes.
//
// The model is deliberately constant-coefficient: every subscript must be a
// linear expression with constant coefficients (what cgeist -O0 +
// raise-scf-to-affine emit for PolyBench-style code).  Semi-affine subscripts,
// map symbols, opaque memory ops and underivable trip counts make
// `analyzeBandReuse` return failure — with a `ReuseReject` saying which —
// and clients fall back to their conservative default (for the tiler: do not
// tile).
//
// The band need NOT sit at the top of the nest, and its subscripts need not
// name only band IVs.  A loop ENCLOSING the band is fixed for the band's whole
// execution, so it shifts a reference's base address without widening any
// extent; a loop BELOW the band sweeps its full range once per band iteration,
// so it widens extents by a tile-shape-independent constant.  Admitting both
// (with `collectMaximalPerfectBands` to enumerate bands inside imperfect
// nests) took PolyBench-L from 15 of 30 kernels with no analyzable band at all
// down to zero, and is what lets the tiler see symm's reduction band.
//
// Footprints come in two flavours, and the distinction is load-bearing:
// `footprintBytes` is a CAPACITY figure, so references to the same memref are
// unioned via a per-dimension index hull (summing them over-counted a 5-point
// jacobi-2d 3x and an in-place seidel-2d 9x).  `refFootprintBytes` is a
// TRAFFIC figure and stays per-reference.  Where the hull cannot prove two
// references overlap — different enclosing-IV coefficients — they are summed,
// which errs toward "evicted", i.e. toward tiling, never toward miscounting
// reuse as absent.
//
//===----------------------------------------------------------------------===//

#ifndef DRCOMPILER_ANALYSIS_REUSEANALYSIS_H
#define DRCOMPILER_ANALYSIS_REUSEANALYSIS_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

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
  /// Constant term of each subscript dim (the stencil offset: A[i][j-1] has
  /// constOffset = {0, -1}).
  llvm::SmallVector<int64_t, 4> constOffset;
  /// Static memref dim sizes (-1 if dynamic) used to clamp extents.
  llvm::SmallVector<int64_t, 4> dimSizes;
  /// Extra extent, in elements, that subscript dim `d` spans because of loops
  /// nested BELOW the band.  Their IVs are not band IVs, but unlike an
  /// enclosing IV they do sweep during one band iteration, so they widen the
  /// footprint by `|coeff| * (trip - 1)` each.  This is what puts symm's
  /// `[%i, %j]` band — whose body holds a deeper `%k` loop — in model.
  llvm::SmallVector<int64_t, 4> innerExtent;
  /// Per subscript dim, the coefficients of ENCLOSING (non-band) loop IVs,
  /// sorted by value so two refs are comparable.  Such an IV is fixed for the
  /// whole execution of the band, so it contributes nothing to any extent —
  /// but it does mean two references can sit at unrelated addresses, so the
  /// capacity hull in `footprintBytes` only unions refs whose `outerCoeff`
  /// agrees dim-wise.  Empty for a band with no enclosing affine.for.
  llvm::SmallVector<llvm::SmallVector<std::pair<mlir::Value, int64_t>, 2>, 4>
      outerCoeff;
  /// Per band loop classification.
  llvm::SmallVector<ReuseKind, 4> kinds;

  bool invariantIn(unsigned loopIdx) const {
    return kinds[loopIdx] == ReuseKind::Invariant;
  }
};

/// A GROUP of references to the same memref whose subscripts share one
/// linear coefficient matrix and differ only in the constant offset vector —
/// the stencil-neighbour pattern (A[i][j-1], A[i][j], A[i][j+1], ...).
/// Per-reference analysis sees these as unrelated streams ("no temporal
/// reuse"); the group view exposes the reuse a time-tiling/skewing transform
/// can exploit: members re-touch each other's elements `span` iterations
/// apart along the loops that index the differing dims.
struct RefGroup {
  /// Indices into BandReuseInfo::refs (>= 2 entries).
  llvm::SmallVector<unsigned, 4> members;
  /// span[d] = max - min constant offset over members in subscript dim d
  /// (the halo width of the stencil along that dim).
  llvm::SmallVector<int64_t, 4> span;
  /// carriesReuse[l] = true if band loop l indexes (nonzero coefficient)
  /// some subscript dim whose member offsets differ: one iteration of l
  /// brings a member onto an element another member touched up to span
  /// iterations earlier.
  llvm::SmallVector<bool, 6> carriesReuse;
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
  /// Stencil-neighbour reference groups (only groups with >= 2 members are
  /// recorded).  Purely additive analysis output: no existing footprint /
  /// reuse-distance / eviction verdict consults it.
  llvm::SmallVector<RefGroup, 4> groups;
  /// Cache line size used to round footprints: hardware fetches whole lines,
  /// so a reference whose fastest-varying subscript spans fewer elements than
  /// a line still costs a full line per row.  Without this a column-strided
  /// f64 walk is under-counted by lineBytes/elemBytes (8x), which biases the
  /// tiler toward tiles that do not actually fit.  Clients holding a machine
  /// model should overwrite this from it; 64 reproduces the prior default.
  int64_t cacheLineBytes = 64;

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
  ///
  /// KNOWN TOO PERMISSIVE: it asks whether reuse is evicted, not whether
  /// blocking would fix it, and five of ten PolyBench-L TILE verdicts reduce
  /// zero misses as a result (FALCON_ORACLE_SPIKE.md §8).  Closing the gap
  /// needs a modelled miss count; a cheaper innermost-contiguity test was
  /// measured and does not discriminate (it also rejects 2mm and 3mm, which
  /// gain 9.3x and 9.1x).
  bool loopCarriesEvictedReuse(unsigned loopIdx, int64_t cacheBytes,
                               int64_t cacheLineBytes = 64) const;

  /// True if any loop carries non-degenerate temporal reuse for any ref.
  bool anyTemporalReuse(int64_t cacheLineBytes = 64) const;
};

/// Collect every MAXIMAL perfect affine band under `root`, including bands
/// nested inside an imperfect enclosing loop.
///
/// Clients that enumerate only bands rooted at a top-level `affine.for` see
/// nothing useful in the very common ping-pong shape
///
///     affine.for %t { affine.for %i { affine.for %j { A } }
///                     affine.for %i { affine.for %j { B } } }
///
/// because `getPerfectlyNestedLoops` truncates at `%t` (its body holds two
/// loops), and the analysis then rejects the truncated band with
/// `NonBandOperand` — every subscript mentions the `%i`/`%j` that truncation
/// just excluded.  The two `[%i, %j]` bands are entirely within the
/// constant-coefficient model; nobody was looking at them.  This accounted for
/// ALL 26 rejected bands over PolyBench-L, gating 15 of 30 kernels outright.
///
/// A loop starts a maximal band iff it is not itself the perfectly-nested body
/// of an enclosing `affine.for` (matching `getPerfectlyNestedLoops`: a loop is
/// perfect around its child iff its body is exactly that child plus the
/// terminator).
void collectMaximalPerfectBands(
    mlir::Operation *root,
    llvm::SmallVectorImpl<llvm::SmallVector<mlir::affine::AffineForOp, 6>>
        &bands);

/// Why `analyzeBandReuse` rejected a band.  Every `failure()` return is
/// attributable to exactly one of these, so a client (or the test pass) can
/// report *which* modelling restriction bit rather than an opaque
/// "UNANALYZABLE" — the histogram over a corpus is what tells you which
/// restriction is worth lifting.
enum class ReuseReject {
  None,
  /// Empty band.
  EmptyBand,
  /// Some band loop has no derivable constant trip count / upper bound.
  TripCount,
  /// A side-effecting op that is not an affine load/store (call, memref.load,
  /// vector transfer, ...) sits under the band.
  OpaqueOp,
  /// memref element type is not an int or float (e.g. a nested memref).
  NonScalarElem,
  /// The access map has symbols.
  SymbolicSubscript,
  /// A subscript is semi-affine (mod/floordiv/ceildiv) or has a non-constant
  /// coefficient.
  NonLinearSubscript,
  /// A subscript depends on a value that is neither a band IV nor a constant.
  NonBandOperand,
  /// The band contains no analyzable memory reference at all.
  NoRefs,
};

llvm::StringRef toString(ReuseReject r);

/// Analyze a perfect band of affine loops with constant bounds.  Fails (see
/// file header) whenever the constant-coefficient model does not apply;
/// `reject`, when non-null, receives the reason.
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
                 bool acceptTripUpperBounds = false,
                 ReuseReject *reject = nullptr);

} // namespace reuse
} // namespace drcompiler

#endif // DRCOMPILER_ANALYSIS_REUSEANALYSIS_H
