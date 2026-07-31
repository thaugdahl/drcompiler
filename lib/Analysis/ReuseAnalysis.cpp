//===- ReuseAnalysis.cpp - per-reference loop reuse classification --------===//

#include "drcompiler/Analysis/ReuseAnalysis.h"

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::affine;
using namespace drcompiler::reuse;

namespace {

/// Extract constant coefficients of a linear affine expression over the map's
/// dims.  Returns false for anything non-linear in the dims (mod, floordiv,
/// ceildiv, dim*dim) or with a non-constant coefficient.  Symbols are allowed
/// only with zero coefficient (they never appear once `coeffOk` rejects them).
bool collectLinearCoeffs(AffineExpr expr, unsigned numDims,
                         SmallVectorImpl<int64_t> &dimCoeffs,
                         int64_t &constTerm) {
  dimCoeffs.assign(numDims, 0);
  constTerm = 0;
  // Recursive walk accumulating `scale * expr`.
  std::function<bool(AffineExpr, int64_t)> walk = [&](AffineExpr e,
                                                      int64_t scale) -> bool {
    if (auto c = dyn_cast<AffineConstantExpr>(e)) {
      // Constants shift the base address; irrelevant to extents, but they
      // are the stencil offset that distinguishes group members.
      constTerm += scale * c.getValue();
      return true;
    }
    if (auto d = dyn_cast<AffineDimExpr>(e)) {
      dimCoeffs[d.getPosition()] += scale;
      return true;
    }
    if (isa<AffineSymbolExpr>(e))
      return false; // symbolic subscripts are out of model
    auto bin = dyn_cast<AffineBinaryOpExpr>(e);
    if (!bin)
      return false;
    switch (bin.getKind()) {
    case AffineExprKind::Add:
      return walk(bin.getLHS(), scale) && walk(bin.getRHS(), scale);
    case AffineExprKind::Mul: {
      // Affine guarantees one side is a (symbolic) constant; require literal.
      if (auto rc = dyn_cast<AffineConstantExpr>(bin.getRHS()))
        return walk(bin.getLHS(), scale * rc.getValue());
      if (auto lc = dyn_cast<AffineConstantExpr>(bin.getLHS()))
        return walk(bin.getRHS(), scale * lc.getValue());
      return false;
    }
    default:
      return false; // mod / floordiv / ceildiv: semi-affine, out of model
    }
  };
  return walk(simplifyAffineExpr(expr, numDims, /*numSymbols=*/0), 1);
}

/// Saturating multiply for footprint products.
int64_t satMul(int64_t a, int64_t b) {
  if (a == 0 || b == 0)
    return 0;
  if (a > std::numeric_limits<int64_t>::max() / b)
    return std::numeric_limits<int64_t>::max();
  return a * b;
}

int64_t satAdd(int64_t a, int64_t b) {
  if (a > std::numeric_limits<int64_t>::max() - b)
    return std::numeric_limits<int64_t>::max();
  return a + b;
}

/// Maximum value an affine expression can take, given a per-dim maximum for
/// each of its operands.  Only sound for expressions whose dim coefficients
/// are all non-negative (then the expression is monotone non-decreasing in
/// every operand, so the maximum is attained at the operand maxima).  Returns
/// nullopt for anything else, including symbols and semi-affine operators.
std::optional<int64_t> affineExprMax(AffineExpr expr, unsigned numDims,
                                    ArrayRef<int64_t> dimMax) {
  SmallVector<int64_t, 4> coeffs;
  int64_t constTerm = 0;
  if (!collectLinearCoeffs(expr, numDims, coeffs, constTerm))
    return std::nullopt;
  int64_t acc = constTerm;
  for (unsigned d = 0; d < numDims; ++d) {
    if (coeffs[d] == 0)
      continue;
    if (coeffs[d] < 0)
      return std::nullopt; // not monotone: maximum is not at the operand max
    acc = satAdd(acc, satMul(coeffs[d], dimMax[d]));
  }
  return acc;
}

/// Constant EXCLUSIVE upper bound on `forOp`'s induction variable: a value `m`
/// with `iv < m` on every iteration, or nullopt when none can be derived.
///
/// Handles the triangular shape PolyBench is full of — `affine.for %j = 0 to
/// #map(%i)` with `#map = (d0) -> (d0 + 1)` — by substituting the enclosing
/// loop's own bound for `%i`.  `ivBound` holds the exclusive bound of every
/// enclosing band loop already processed (band order is outer to inner, so
/// they are always available).
///
/// A multi-result (`min`) bound is handled by taking the tightest result.
std::optional<int64_t>
constantIvUpperBound(AffineForOp forOp,
                     const llvm::SmallDenseMap<Value, int64_t, 8> &ivBound) {
  if (forOp.hasConstantUpperBound())
    return forOp.getConstantUpperBound();

  AffineMap map = forOp.getUpperBoundMap();
  if (map.getNumResults() == 0 || map.getNumSymbols() != 0)
    return std::nullopt;
  ValueRange operands = forOp.getUpperBoundOperands();
  if (operands.size() != map.getNumDims())
    return std::nullopt;

  // Largest value each operand can take: an enclosing IV's exclusive bound
  // minus one.
  SmallVector<int64_t, 4> dimMax(map.getNumDims());
  for (unsigned d = 0, e = map.getNumDims(); d < e; ++d) {
    auto it = ivBound.find(operands[d]);
    if (it == ivBound.end())
      return std::nullopt; // not an enclosing band IV with a known range
    if (it->second <= 0)
      return std::nullopt;
    dimMax[d] = it->second - 1;
  }

  std::optional<int64_t> best;
  for (AffineExpr res : map.getResults()) {
    std::optional<int64_t> m = affineExprMax(res, map.getNumDims(), dimMax);
    if (!m)
      return std::nullopt;
    best = best ? std::min(*best, *m) : *m;
  }
  return best;
}

/// Extent, in elements, that one reference spans along subscript dim `d` under
/// the given per-loop iteration counts.  Not clamped to the memref dim.
int64_t refDimExtent(const RefInfo &ref, unsigned d,
                     ArrayRef<uint64_t> tileSizes) {
  int64_t extent = 1;
  for (unsigned l = 0, nl = tileSizes.size(); l < nl; ++l) {
    int64_t span = satMul(std::abs(ref.coeff[d][l]),
                          static_cast<int64_t>(tileSizes[l]) - 1);
    extent = satAdd(extent, span);
  }
  // Loops nested below the band sweep their full range regardless of the
  // hypothetical tile shape, so their contribution is a constant widening.
  if (d < ref.innerExtent.size())
    extent = satAdd(extent, ref.innerExtent[d]);
  return extent;
}

/// Round a contiguous byte run up to whole cache lines.
int64_t roundToLines(int64_t bytes, int64_t lineBytes) {
  if (lineBytes <= 1 || bytes <= 0)
    return bytes;
  int64_t lines = (bytes + lineBytes - 1) / lineBytes;
  return satMul(lines, lineBytes);
}

/// Footprint of one reference under per-loop iteration counts.
///
/// The fastest-varying subscript is the contiguous one (row-major identity
/// layout, the same assumption ReuseKind::Spatial rests on), so its byte run
/// is rounded up to whole lines and the outer dims multiply that.  A column
/// walk therefore costs one full line per row instead of one element.
int64_t refFootprint(const RefInfo &ref, ArrayRef<uint64_t> tileSizes,
                     int64_t lineBytes) {
  unsigned nd = ref.coeff.size();
  if (nd == 0)
    return roundToLines(ref.elemBytes, lineBytes);

  auto clampedExtent = [&](unsigned d) {
    int64_t extent = refDimExtent(ref, d, tileSizes);
    if (ref.dimSizes[d] >= 0)
      extent = std::min(extent, ref.dimSizes[d]);
    return extent;
  };

  int64_t bytes = roundToLines(satMul(ref.elemBytes, clampedExtent(nd - 1)),
                               lineBytes);
  for (unsigned d = 0; d + 1 < nd; ++d)
    bytes = satMul(bytes, clampedExtent(d));
  return bytes;
}

/// Per-memref index-range hull, used for the capacity footprint.
struct MemrefHull {
  void *memref = nullptr;
  /// Enclosing-IV coefficients shared by every member (see RefInfo::outerCoeff).
  /// Two references indexed by different enclosing IVs may sit anywhere
  /// relative to each other, so they get separate hulls and their footprints
  /// are summed — the pre-existing over-estimate, which errs toward "evicted",
  /// i.e. toward tiling.
  llvm::SmallVector<llvm::SmallVector<std::pair<mlir::Value, int64_t>, 2>, 4>
      outerCoeff;
  int64_t elemBytes = 0;
  llvm::SmallVector<int64_t, 4> lo, hi; // half-open [lo, hi) per subscript dim
  llvm::SmallVector<int64_t, 4> dimSizes;

  int64_t bytes(int64_t lineBytes) const {
    unsigned nd = lo.size();
    auto extentAt = [&](unsigned d) {
      int64_t extent = satAdd(hi[d], -lo[d]);
      if (dimSizes[d] >= 0)
        extent = std::min(extent, dimSizes[d]);
      return extent;
    };
    if (nd == 0)
      return roundToLines(elemBytes, lineBytes);
    int64_t total =
        roundToLines(satMul(elemBytes, extentAt(nd - 1)), lineBytes);
    for (unsigned d = 0; d + 1 < nd; ++d)
      total = satMul(total, extentAt(d));
    return total;
  }
};

} // namespace

int64_t BandReuseInfo::footprintBytes(ArrayRef<uint64_t> tileSizes) const {
  assert(tileSizes.size() == band.size() && "tile size per band loop");
  // This is a CAPACITY figure -- bytes that must be resident -- so references
  // to the same memref must be unioned, not summed: they overlap in cache.
  // Summing them over-counted a 5-point jacobi-2d stencil 3x, a 7-point
  // heat-3d 3.9x and an in-place 9-point seidel-2d 9x, because each neighbour
  // was priced as an independent array.  Per dim the union is approximated by
  // the hull of the members' index ranges, which is exact for the stencil case
  // (same coefficient matrix, differing constant offsets) and a sound
  // over-approximation when the coefficient matrices differ.
  //
  // Traffic weighting stays per-reference: see refFootprintBytes.
  llvm::SmallVector<MemrefHull, 4> hulls;
  for (const RefInfo &ref : refs) {
    void *key = ref.memref.getAsOpaquePointer();
    MemrefHull *h = nullptr;
    for (MemrefHull &cand : hulls)
      if (cand.memref == key && cand.outerCoeff == ref.outerCoeff) {
        h = &cand;
        break;
      }
    unsigned nd = ref.coeff.size();
    if (!h) {
      hulls.push_back(MemrefHull{key, ref.outerCoeff, ref.elemBytes, {}, {},
                                 ref.dimSizes});
      h = &hulls.back();
      h->lo.assign(nd, std::numeric_limits<int64_t>::max());
      h->hi.assign(nd, std::numeric_limits<int64_t>::min());
    }
    for (unsigned d = 0; d < nd; ++d) {
      int64_t base = ref.constOffset[d];
      h->lo[d] = std::min(h->lo[d], base);
      h->hi[d] = std::max(h->hi[d],
                          satAdd(base, refDimExtent(ref, d, tileSizes)));
    }
  }
  int64_t total = 0;
  for (const MemrefHull &h : hulls)
    total = satAdd(total, h.bytes(cacheLineBytes));
  return total;
}

int64_t BandReuseInfo::refFootprintBytes(unsigned refIdx,
                                         ArrayRef<uint64_t> tileSizes) const {
  return refFootprint(refs[refIdx], tileSizes, cacheLineBytes);
}

int64_t BandReuseInfo::refIterFootprint(unsigned refIdx,
                                        unsigned loopIdx) const {
  SmallVector<uint64_t, 6> sizes(band.size(), 1);
  for (unsigned l = loopIdx + 1, e = band.size(); l < e; ++l)
    sizes[l] = tripCounts[l];
  return refFootprint(refs[refIdx], sizes, cacheLineBytes);
}

int64_t BandReuseInfo::reuseDistanceBytes(unsigned loopIdx) const {
  SmallVector<uint64_t, 6> sizes(band.size(), 1);
  for (unsigned l = loopIdx + 1, e = band.size(); l < e; ++l)
    sizes[l] = tripCounts[l];
  return footprintBytes(sizes);
}

bool BandReuseInfo::loopCarriesEvictedReuse(unsigned loopIdx, int64_t cacheBytes,
                                            int64_t kCacheLineBytes) const {
  // A reference benefits from tiling around `loopIdx` only if (a) the loop
  // carries temporal reuse for it, (b) the reused window is more than one
  // cache line (a scalar accumulator lives in a register; tiling buys
  // nothing), and (c) the data touched between reuses overflows the cache.
  // NOTE: this gate is known to be too permissive -- five of ten TILE verdicts
  // on PolyBench-L reduce zero cache misses (FALCON_ORACLE_SPIKE.md §8).  An
  // extra condition requiring the reused reference to be contiguous in the
  // innermost loop was tried and REJECTED: it correctly drops all five inert
  // kernels but also drops 2mm and 3mm, which gain 9.3x and 9.1x.  The same
  // array is `stream` in one loop order and `spat` in another while blocking
  // pays off either way, so innermost-contiguity does not discriminate.
  // Distinguishing them needs a modelled miss count, not a footprint sum.
  bool anyNonDegenerateInvariant = false;
  for (unsigned r = 0, e = refs.size(); r < e; ++r) {
    if (!refs[r].invariantIn(loopIdx))
      continue;
    if (refIterFootprint(r, loopIdx) > kCacheLineBytes) {
      anyNonDegenerateInvariant = true;
      break;
    }
  }
  if (!anyNonDegenerateInvariant)
    return false;
  return reuseDistanceBytes(loopIdx) > cacheBytes;
}

bool BandReuseInfo::anyTemporalReuse(int64_t kCacheLineBytes) const {
  for (unsigned l = 0, nl = band.size(); l < nl; ++l)
    for (unsigned r = 0, nr = refs.size(); r < nr; ++r)
      if (refs[r].invariantIn(l) && refIterFootprint(r, l) > kCacheLineBytes)
        return true;
  return false;
}

void drcompiler::reuse::collectMaximalPerfectBands(
    Operation *root,
    llvm::SmallVectorImpl<SmallVector<AffineForOp, 6>> &bands) {
  root->walk([&](AffineForOp forOp) {
    // Skip loops that are the perfectly-nested interior of an enclosing loop:
    // the enclosing loop's own band already covers them, so emitting them
    // separately would report the same nest at every depth.
    if (auto parent = dyn_cast_or_null<AffineForOp>(forOp->getParentOp())) {
      Block *body = parent.getBody();
      // Same perfection test getPerfectlyNestedLoops uses.
      if (body->getOperations().size() == 2 &&
          &body->front() == forOp.getOperation())
        return;
    }
    SmallVector<AffineForOp, 6> band;
    getPerfectlyNestedLoops(band, forOp);
    if (!band.empty())
      bands.push_back(std::move(band));
  });
}

llvm::StringRef drcompiler::reuse::toString(ReuseReject r) {
  switch (r) {
  case ReuseReject::None:
    return "none";
  case ReuseReject::EmptyBand:
    return "empty-band";
  case ReuseReject::TripCount:
    return "trip-count";
  case ReuseReject::OpaqueOp:
    return "opaque-op";
  case ReuseReject::NonScalarElem:
    return "non-scalar-elem";
  case ReuseReject::SymbolicSubscript:
    return "symbolic-subscript";
  case ReuseReject::NonLinearSubscript:
    return "non-linear-subscript";
  case ReuseReject::NonBandOperand:
    return "non-band-operand";
  case ReuseReject::NoRefs:
    return "no-refs";
  }
  llvm_unreachable("covered switch");
}

FailureOr<BandReuseInfo>
drcompiler::reuse::analyzeBandReuse(ArrayRef<AffineForOp> band,
                                    Operation *walkRoot,
                                    bool acceptTripUpperBounds,
                                    ReuseReject *reject) {
  ReuseReject sink = ReuseReject::None;
  ReuseReject &why = reject ? *reject : sink;
  why = ReuseReject::None;
  if (band.empty()) {
    why = ReuseReject::EmptyBand;
    return failure();
  }

  BandReuseInfo info;
  info.band.assign(band.begin(), band.end());

  // Exclusive constant bound on each band IV, so an inner loop with a
  // triangular bound (`0 to #map(%i)`) can be bounded by substituting the
  // enclosing loop's range.  Populated outer to inner as the band is walked.
  llvm::SmallDenseMap<Value, int64_t, 8> ivBound;

  // A band need not start at the top of the nest: `collectMaximalPerfectBands`
  // reports the two `[%i, %j]` bands inside a ping-pong stencil's `%t` loop.
  // Those bands' subscripts and bounds legitimately mention the enclosing IVs,
  // which are FIXED for the whole band execution.  Seed the bound map with
  // them (outermost first, so a chain of triangular bounds resolves) and
  // remember the set so subscripts can admit them with zero extent.
  llvm::SmallDenseSet<Value, 8> outerIvs;
  {
    SmallVector<AffineForOp, 4> enclosing;
    for (Operation *p = band.front()->getParentOp(); p; p = p->getParentOp())
      if (auto forOp = dyn_cast<AffineForOp>(p))
        enclosing.push_back(forOp);
    for (AffineForOp forOp : llvm::reverse(enclosing)) {
      outerIvs.insert(forOp.getInductionVar());
      if (std::optional<int64_t> ub = constantIvUpperBound(forOp, ivBound))
        ivBound[forOp.getInductionVar()] = *ub;
    }
  }

  for (AffineForOp forOp : band) {
    std::optional<uint64_t> tc = getConstantTripCount(forOp);
    if (tc && *tc > 0) {
      info.tripCounts.push_back(*tc);
      info.tripIsExact.push_back(true);
      if (forOp.hasConstantUpperBound())
        ivBound[forOp.getInductionVar()] = forOp.getConstantUpperBound();
      continue;
    }
    if (acceptTripUpperBounds) {
      // `ub` covers both a literal constant bound and a triangular bound
      // resolved against the enclosing loops.
      std::optional<int64_t> ub = constantIvUpperBound(forOp, ivBound);
      int64_t lbFloor =
          forOp.hasConstantLowerBound() ? forOp.getConstantLowerBound() : 0;
      if (ub && *ub > lbFloor) {
        // Unlike the constant path this must divide by the step itself:
        // getConstantTripCount does the ceilDiv, we don't get it for free.
        int64_t step = std::max<int64_t>(forOp.getStepAsInt(), 1);
        int64_t trips = (*ub - lbFloor + step - 1) / step;
        info.tripCounts.push_back(static_cast<uint64_t>(trips));
        info.tripIsExact.push_back(false);
        ivBound[forOp.getInductionVar()] = *ub;
        continue;
      }
    }
    why = ReuseReject::TripCount;
    return failure();
  }

  llvm::SmallDenseMap<Value, unsigned, 8> ivIndex;
  for (unsigned l = 0, e = band.size(); l < e; ++l) {
    AffineForOp forOp = band[l];
    ivIndex[forOp.getInductionVar()] = l;
  }

  // Loops nested strictly BELOW the band (the band is maximal, so these sit
  // under an imperfect body).  Their IVs are not band IVs, but they sweep
  // during a band iteration, so a subscript naming one is still analyzable —
  // it just spans that loop's whole range.  Without this, symm's `[%i, %j]`
  // band is rejected outright because its body holds a `%k` reduction loop.
  // Walked outer to inner (pre-order) so a chain of triangular bounds
  // resolves against the bounds already recorded.
  llvm::SmallDenseMap<Value, int64_t, 8> innerTrip;
  AffineForOp innermost = band.back();
  innermost.getBody()->walk([&](AffineForOp forOp) {
    std::optional<int64_t> trips;
    if (std::optional<uint64_t> tc = getConstantTripCount(forOp)) {
      trips = static_cast<int64_t>(*tc);
      if (forOp.hasConstantUpperBound())
        ivBound[forOp.getInductionVar()] = forOp.getConstantUpperBound();
    } else if (std::optional<int64_t> ub =
                   constantIvUpperBound(forOp, ivBound)) {
      int64_t lbFloor =
          forOp.hasConstantLowerBound() ? forOp.getConstantLowerBound() : 0;
      if (*ub > lbFloor) {
        int64_t step = std::max<int64_t>(forOp.getStepAsInt(), 1);
        trips = (*ub - lbFloor + step - 1) / step;
        ivBound[forOp.getInductionVar()] = *ub;
      }
    }
    if (trips && *trips > 0)
      innerTrip[forOp.getInductionVar()] = *trips;
  });

  // Dedup key: memref + map + operand list.
  using Key = std::tuple<void *, AffineMap, SmallVector<Value, 4>>;
  llvm::SmallVector<std::pair<Key, unsigned>, 8> seen;

  AffineForOp bandRoot = band.front();
  Operation *root = walkRoot ? walkRoot : bandRoot.getOperation();
  WalkResult res = root->walk([&](Operation *op) -> WalkResult {
    Value memref;
    AffineMap map;
    SmallVector<Value, 4> operands;
    bool isWrite = false;
    if (auto load = dyn_cast<AffineLoadOp>(op)) {
      memref = load.getMemRef();
      map = load.getAffineMap();
      operands.assign(load.getMapOperands().begin(),
                      load.getMapOperands().end());
    } else if (auto store = dyn_cast<AffineStoreOp>(op)) {
      memref = store.getMemRef();
      map = store.getAffineMap();
      operands.assign(store.getMapOperands().begin(),
                      store.getMapOperands().end());
      isWrite = true;
    } else {
      // Any other memory access (memref.load, vector transfer, call) is out
      // of model.
      if (!isMemoryEffectFree(op) && !isa<AffineForOp, AffineYieldOp>(op)) {
        why = ReuseReject::OpaqueOp;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }

    // Fold affine.apply chains into the map so subscripts that are not raw
    // IVs (cgeist sometimes materializes them) still expose their
    // coefficients, then canonicalize so dedup sees one spelling.
    fullyComposeAffineMapAndOperands(&map, &operands);
    canonicalizeMapAndOperands(&map, &operands);

    // Dedup syntactically identical accesses.
    Key key{memref.getAsOpaquePointer(), map, operands};
    for (auto &kv : seen) {
      if (kv.first == key) {
        info.refs[kv.second].multiplicity++;
        info.refs[kv.second].isWrite |= isWrite;
        return WalkResult::advance();
      }
    }

    auto memrefTy = dyn_cast<MemRefType>(memref.getType());
    if (!memrefTy || !memrefTy.getElementType().isIntOrFloat()) {
      why = ReuseReject::NonScalarElem;
      return WalkResult::interrupt();
    }

    RefInfo ref;
    ref.op = op;
    ref.memref = memref;
    ref.isWrite = isWrite;
    ref.elemBytes =
        std::max<int64_t>(1, memrefTy.getElementTypeBitWidth() / 8);
    for (int64_t s : memrefTy.getShape())
      ref.dimSizes.push_back(ShapedType::isDynamic(s) ? -1 : s);

    // Per-subscript linear coefficients over the map's dims, then re-keyed to
    // band loops via the operand list.
    unsigned numDims = map.getNumDims();
    if (map.getNumSymbols() != 0) {
      why = ReuseReject::SymbolicSubscript;
      return WalkResult::interrupt();
    }
    for (AffineExpr expr : map.getResults()) {
      SmallVector<int64_t, 4> dimCoeffs;
      int64_t constTerm = 0;
      if (!collectLinearCoeffs(expr, numDims, dimCoeffs, constTerm)) {
        why = ReuseReject::NonLinearSubscript;
        return WalkResult::interrupt();
      }
      ref.constOffset.push_back(constTerm);
      SmallVector<int64_t, 4> loopCoeffs(band.size(), 0);
      SmallVector<std::pair<Value, int64_t>, 2> outer;
      int64_t innerSpan = 0;
      for (unsigned d = 0; d < numDims; ++d) {
        if (dimCoeffs[d] == 0)
          continue;
        auto it = ivIndex.find(operands[d]);
        if (it != ivIndex.end()) {
          loopCoeffs[it->second] += dimCoeffs[d];
          continue;
        }
        // A loop below the band sweeps its full range during every band
        // iteration, so it widens this dim by a tile-shape-independent amount.
        if (auto inner = innerTrip.find(operands[d]); inner != innerTrip.end()) {
          innerSpan = satAdd(innerSpan, satMul(std::abs(dimCoeffs[d]),
                                               inner->second - 1));
          continue;
        }
        // An enclosing loop's IV is a constant for the duration of the band,
        // so it shifts the base address without widening any extent.  The
        // shift itself is unknown, hence it is recorded rather than folded
        // into constOffset.
        if (outerIvs.contains(operands[d])) {
          outer.push_back({operands[d], dimCoeffs[d]});
          continue;
        }
        why = ReuseReject::NonBandOperand;
        return WalkResult::interrupt();
      }
      llvm::sort(outer, [](const std::pair<Value, int64_t> &a,
                           const std::pair<Value, int64_t> &b) {
        return a.first.getAsOpaquePointer() < b.first.getAsOpaquePointer();
      });
      ref.outerCoeff.push_back(std::move(outer));
      ref.innerExtent.push_back(innerSpan);
      ref.coeff.push_back(std::move(loopCoeffs));
    }

    // Classify each band loop for this reference.
    unsigned rank = ref.coeff.size();
    for (unsigned l = 0, e = band.size(); l < e; ++l) {
      bool inOuter = false, inLast = false;
      int64_t lastCoeff = 0;
      for (unsigned d = 0; d < rank; ++d) {
        if (ref.coeff[d][l] == 0)
          continue;
        if (d + 1 == rank) {
          inLast = true;
          lastCoeff = ref.coeff[d][l];
        } else {
          inOuter = true;
        }
      }
      ReuseKind kind;
      if (!inOuter && !inLast)
        kind = ReuseKind::Invariant;
      else if (!inOuter && std::abs(lastCoeff) == 1)
        kind = ReuseKind::Spatial;
      else
        kind = ReuseKind::Streaming;
      ref.kinds.push_back(kind);
    }

    seen.push_back({std::move(key), static_cast<unsigned>(info.refs.size())});
    info.refs.push_back(std::move(ref));
    return WalkResult::advance();
  });

  if (res.wasInterrupted())
    return failure();
  if (info.refs.empty()) {
    why = ReuseReject::NoRefs;
    return failure();
  }

  // Group stencil neighbours: same memref, same per-loop coefficient matrix,
  // differing only in the constant offset vector.
  for (unsigned r = 0, e = info.refs.size(); r < e; ++r) {
    bool grouped = false;
    for (RefGroup &g : info.groups) {
      const RefInfo &rep = info.refs[g.members.front()];
      if (rep.memref == info.refs[r].memref &&
          rep.coeff == info.refs[r].coeff) {
        g.members.push_back(r);
        grouped = true;
        break;
      }
    }
    if (!grouped) {
      RefGroup g;
      g.members.push_back(r);
      info.groups.push_back(std::move(g));
    }
  }
  llvm::erase_if(info.groups,
                 [](const RefGroup &g) { return g.members.size() < 2; });
  for (RefGroup &g : info.groups) {
    const RefInfo &rep = info.refs[g.members.front()];
    unsigned rank = rep.coeff.size();
    SmallVector<bool, 4> dimDiffers(rank, false);
    for (unsigned d = 0; d < rank; ++d) {
      int64_t mn = std::numeric_limits<int64_t>::max();
      int64_t mx = std::numeric_limits<int64_t>::min();
      for (unsigned m : g.members) {
        mn = std::min(mn, info.refs[m].constOffset[d]);
        mx = std::max(mx, info.refs[m].constOffset[d]);
      }
      g.span.push_back(mx - mn);
      dimDiffers[d] = mx != mn;
    }
    for (unsigned l = 0, nl = info.band.size(); l < nl; ++l) {
      bool carries = false;
      for (unsigned d = 0; d < rank; ++d)
        carries |= dimDiffers[d] && rep.coeff[d][l] != 0;
      g.carriesReuse.push_back(carries);
    }
  }
  return info;
}
