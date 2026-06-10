//===- ReuseAnalysis.cpp - per-reference loop reuse classification --------===//

#include "drcompiler/Analysis/ReuseAnalysis.h"

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/AffineExpr.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
using namespace mlir::affine;
using namespace drcompiler::reuse;

namespace {

/// Extract constant coefficients of a linear affine expression over the map's
/// dims.  Returns false for anything non-linear in the dims (mod, floordiv,
/// ceildiv, dim*dim) or with a non-constant coefficient.  Symbols are allowed
/// only with zero coefficient (they never appear once `coeffOk` rejects them).
bool collectLinearCoeffs(AffineExpr expr, unsigned numDims,
                         SmallVectorImpl<int64_t> &dimCoeffs) {
  dimCoeffs.assign(numDims, 0);
  // Recursive walk accumulating `scale * expr`.
  std::function<bool(AffineExpr, int64_t)> walk = [&](AffineExpr e,
                                                      int64_t scale) -> bool {
    if (auto c = dyn_cast<AffineConstantExpr>(e))
      return true; // constants shift the base address; irrelevant to extents
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

/// Footprint of one reference under per-loop iteration counts.
int64_t refFootprint(const RefInfo &ref, ArrayRef<uint64_t> tileSizes) {
  int64_t bytes = ref.elemBytes;
  for (unsigned d = 0, nd = ref.coeff.size(); d < nd; ++d) {
    int64_t extent = 1;
    for (unsigned l = 0, nl = tileSizes.size(); l < nl; ++l) {
      int64_t span = satMul(std::abs(ref.coeff[d][l]),
                            static_cast<int64_t>(tileSizes[l]) - 1);
      extent = satAdd(extent, span);
    }
    if (ref.dimSizes[d] >= 0)
      extent = std::min(extent, ref.dimSizes[d]);
    bytes = satMul(bytes, extent);
  }
  return bytes;
}

} // namespace

int64_t BandReuseInfo::footprintBytes(ArrayRef<uint64_t> tileSizes) const {
  assert(tileSizes.size() == band.size() && "tile size per band loop");
  int64_t total = 0;
  for (const RefInfo &ref : refs)
    total = satAdd(total, refFootprint(ref, tileSizes));
  return total;
}

int64_t BandReuseInfo::refFootprintBytes(unsigned refIdx,
                                         ArrayRef<uint64_t> tileSizes) const {
  return refFootprint(refs[refIdx], tileSizes);
}

int64_t BandReuseInfo::refIterFootprint(unsigned refIdx,
                                        unsigned loopIdx) const {
  SmallVector<uint64_t, 6> sizes(band.size(), 1);
  for (unsigned l = loopIdx + 1, e = band.size(); l < e; ++l)
    sizes[l] = tripCounts[l];
  return refFootprint(refs[refIdx], sizes);
}

int64_t BandReuseInfo::reuseDistanceBytes(unsigned loopIdx) const {
  SmallVector<uint64_t, 6> sizes(band.size(), 1);
  for (unsigned l = loopIdx + 1, e = band.size(); l < e; ++l)
    sizes[l] = tripCounts[l];
  return footprintBytes(sizes);
}

bool BandReuseInfo::loopCarriesEvictedReuse(unsigned loopIdx,
                                            int64_t cacheBytes) const {
  // A reference benefits from tiling around `loopIdx` only if (a) the loop
  // carries temporal reuse for it, (b) the reused window is more than one
  // cache line (a scalar accumulator lives in a register; tiling buys
  // nothing), and (c) the data touched between reuses overflows the cache.
  constexpr int64_t kCacheLineBytes = 64;
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

bool BandReuseInfo::anyTemporalReuse() const {
  constexpr int64_t kCacheLineBytes = 64;
  for (unsigned l = 0, nl = band.size(); l < nl; ++l)
    for (unsigned r = 0, nr = refs.size(); r < nr; ++r)
      if (refs[r].invariantIn(l) && refIterFootprint(r, l) > kCacheLineBytes)
        return true;
  return false;
}

FailureOr<BandReuseInfo>
drcompiler::reuse::analyzeBandReuse(ArrayRef<AffineForOp> band,
                                    Operation *walkRoot,
                                    bool acceptTripUpperBounds) {
  if (band.empty())
    return failure();

  BandReuseInfo info;
  info.band.assign(band.begin(), band.end());

  for (AffineForOp forOp : band) {
    std::optional<uint64_t> tc = getConstantTripCount(forOp);
    if (tc && *tc > 0) {
      info.tripCounts.push_back(*tc);
      info.tripIsExact.push_back(true);
      continue;
    }
    if (acceptTripUpperBounds && forOp.hasConstantUpperBound()) {
      int64_t ub = forOp.getConstantUpperBound();
      int64_t lbFloor =
          forOp.hasConstantLowerBound() ? forOp.getConstantLowerBound() : 0;
      if (ub > lbFloor) {
        info.tripCounts.push_back(static_cast<uint64_t>(ub - lbFloor));
        info.tripIsExact.push_back(false);
        continue;
      }
    }
    return failure();
  }

  llvm::SmallDenseMap<Value, unsigned, 8> ivIndex;
  for (unsigned l = 0, e = band.size(); l < e; ++l) {
    AffineForOp forOp = band[l];
    ivIndex[forOp.getInductionVar()] = l;
  }

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
      if (!isMemoryEffectFree(op) && !isa<AffineForOp, AffineYieldOp>(op))
        return WalkResult::interrupt();
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
    if (!memrefTy || !memrefTy.getElementType().isIntOrFloat())
      return WalkResult::interrupt();

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
    if (map.getNumSymbols() != 0)
      return WalkResult::interrupt();
    for (AffineExpr expr : map.getResults()) {
      SmallVector<int64_t, 4> dimCoeffs;
      if (!collectLinearCoeffs(expr, numDims, dimCoeffs))
        return WalkResult::interrupt();
      SmallVector<int64_t, 4> loopCoeffs(band.size(), 0);
      for (unsigned d = 0; d < numDims; ++d) {
        if (dimCoeffs[d] == 0)
          continue;
        auto it = ivIndex.find(operands[d]);
        if (it == ivIndex.end())
          return WalkResult::interrupt(); // depends on a non-band value
        loopCoeffs[it->second] += dimCoeffs[d];
      }
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
  if (info.refs.empty())
    return failure();
  return info;
}
