//===- Peel.cpp - triangular peels ===//
// WP2: extracted from AffineRegisterBlock.cpp.  See RegisterBlock/Internal.h.
//===----------------------------------------------------------------------===//

#include "Internal.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "affine-register-block"

using namespace mlir;
using affine::AffineForOp;
using affine::AffineLoadOp;
using affine::AffineStoreOp;

namespace drcompiler {
namespace rb {

/// Diagonal-peel a triangular reduction band so its bulk becomes register-
/// blockable.  Given a perfect band whose inner spatial bound depends on the
/// outer IV (e.g. syrk's `j: 0..i+1`):
///     for i = 0 to N { for j = LB to f(i) { red(i,j) } }
/// rewrite it to
///     for ii = 0 to N step mr {
///       for i = ii to ii+mr { for j = LB to ii   { red } }   // HEAD: rectangular
///       for i = ii to ii+mr { for j = ii to f(i) { red } }   // DIAG: mr-wide scalar
///     }
/// The HEAD's inner bound `ii` is invariant in `i`, so unroll-and-jam of `i`
/// becomes legal and Stage 3 register-blocks it; the DIAG keeps the ragged
/// bound and stays scalar (its own unroll-jam fails harmlessly).  Returns true
/// if it rewrote; a no-op (false) on rectangular bands and non-perfect nests.
bool peelTriangularNest(AffineForOp sOut, unsigned mr,
                               unsigned kTileTarget, int64_t effLLC,
                               IRRewriter &rewriter) {
  AffineForOp sIn = onlyChildFor(sOut);
  if (!sIn)
    return false;
  AffineForOp red = onlyChildFor(sIn);
  if (!red || !isInnermost(red) || collectAccumulators(red).empty())
    return false;
  if (!sOut.hasConstantLowerBound() || !sOut.hasConstantUpperBound() ||
      sOut.getStepAsInt() != 1)
    return false;
  int64_t lo = sOut.getConstantLowerBound(), hi = sOut.getConstantUpperBound();
  if (mr == 0)
    return false;
  // Strip-mine only the mr-divisible prefix [lo, stripHi); the < mr remainder
  // rows become a scalar epilogue clone of the ORIGINAL nest (original
  // coordinates -- the normalized form is jam-bait, see the lb-DIAG comment
  // below).  An epilogue of a triangular nest is < mr rows: negligible time.
  int64_t stripHi = lo + ((hi - lo) / (int64_t)mr) * (int64_t)mr;
  if (stripHi == lo)
    return false; // trip < mr: nothing to strip
  Value iv = sOut.getInductionVar();
  // Two triangular orientations:
  //  - upper-triangular-in-bound (syrk): `j = LB .. f(i)` — ub depends on i;
  //  - lower-bound-from-IV (covariance/correlation): `j = g(i) .. UB` — lb
  //    depends on i, ub constant.  Mirrored split: the DIAG covers the
  //    ragged `g(ni) .. ii+mr` corner, the HEAD `ii+mr .. UB` is invariant
  //    in i' and register-blocks.
  bool ubTriangular = llvm::is_contained(sIn.getUpperBoundOperands(), iv) &&
                      !llvm::is_contained(sIn.getLowerBoundOperands(), iv);
  bool lbTriangular = llvm::is_contained(sIn.getLowerBoundOperands(), iv) &&
                      !llvm::is_contained(sIn.getUpperBoundOperands(), iv) &&
                      sIn.hasConstantUpperBound();
  if (!ubTriangular && !lbTriangular)
    return false;
  // The HEAD lower bound ii+mr must stay within the j range: require the
  // strip cover (stripHi) not to exceed j's constant upper bound (the DIAG's
  // j runs up to ii+mr <= stripHi).
  if (lbTriangular && stripHi > sIn.getConstantUpperBound())
    return false;

  // A2: k-strip the HEAD's reduction loop.  When a multiplicand is k-STRIDED
  // (column access, e.g. covariance's data[k][i] / data[k][j] with a row
  // stride > page size) and the streamed working set exceeds the effective
  // LLC, the HEAD re-streams the whole data matrix per row-strip and every
  // k step touches a new TLB page.  Chunking k by Tk keeps the slab's page
  // set TLB/cache-resident across the j sweep.  Per-(i,j) additions stay in
  // ascending k order with exact intermediate store/load roundtrips, so the
  // result is BIT-IDENTICAL.  Tk is the largest divisor of the k-trip <=
  // the target, so no remainder chunk exists.  Row-major (stride-1-in-k,
  // dot-family) bands are excluded: their streams prefetch fine and the
  // extra loop level only costs.
  int64_t kTileSize = 0;
  if (kTileTarget > 0 && red.hasConstantLowerBound() &&
      red.hasConstantUpperBound()) {
    int64_t ktrip =
        red.getConstantUpperBound() - red.getConstantLowerBound();
    Value kIV = red.getInductionVar();
    bool kStrided = false;
    int64_t elemBytes = 8;
    for (Operation &op : red.getBody()->without_terminator()) {
      auto ld = dyn_cast<AffineLoadOp>(&op);
      if (!ld || !llvm::is_contained(ld.getMapOperands(), kIV))
        continue;
      auto mt = cast<MemRefType>(ld.getMemRef().getType());
      if (mt.getElementType().isIntOrFloat())
        elemBytes = std::max<int64_t>(
            1, (int64_t)mt.getElementType().getIntOrFloatBitWidth() / 8);
      if (!innermostStrideOne(ld, kIV))
        kStrided = true;
    }
    int64_t jExt = lbTriangular ? sIn.getConstantUpperBound() : hi;
    if (kStrided && effLLC > 0 && ktrip * jExt * elemBytes > effLLC)
      for (int64_t d = std::min<int64_t>(kTileTarget, ktrip); d >= 64; --d)
        if (ktrip % d == 0) {
          kTileSize = d;
          break;
        }
  }
  AffineMap addTkMap =
      kTileSize ? AffineMap::get(1, 0, getAffineDimExpr(0, sOut.getContext()) +
                                           kTileSize)
                : AffineMap();

  MLIRContext *ctx = sOut.getContext();
  Location loc = sOut.getLoc();
  AffineExpr d0 = getAffineDimExpr(0, ctx), d1 = getAffineDimExpr(1, ctx);
  // `ii` (the strip IV) is a loop induction variable, hence a valid affine
  // *dimension* (not a symbol).
  AffineMap idMap = AffineMap::get(1, 0, d0);       // (d0)      -> d0
  AffineMap addMap = AffineMap::get(2, 0, d0 + d1); // (d0, d1)  -> d0 + d1
  AffineMap sInLbMap = sIn.getLowerBoundMap();
  SmallVector<Value> sInLbOps(sIn.getLowerBoundOperands());
  AffineMap sInUbMap = sIn.getUpperBoundMap();
  SmallVector<Value> sInUbOps(sIn.getUpperBoundOperands());

  // Build one half as a *constant-bound* i-loop `for i' = 0 to mr` (so its
  // unroll-and-jam matches the rectangular path), with the real row index
  // recovered as `ni = ii + i'`.  For the HEAD the row offset is computed
  // INSIDE the j-loop, keeping head-i' a perfect single-child nest (unroll-jam
  // cannot thread a non-loop op sitting between the jammed loops); its j runs
  // `LB..ii` (invariant in i').  For the DIAG the offset must precede the j-loop
  // (it feeds the ragged `ii..f(ni)` bound), but the DIAG is left scalar.
  AffineMap addMrMap =
      AffineMap::get(1, 0, d0 + (int64_t)mr); // (d0) -> d0 + mr

  auto buildHalf = [&](Value ii, bool diag, Value kkIV) -> AffineForOp {
    if (diag && lbTriangular) {
      // lb-triangular DIAG: emit in ORIGINAL coordinates — a real
      // `for i2 = ii .. ii+mr { for j = g(i2) .. ii+mr }` whose ragged
      // bound hangs directly off i2's IV.  The normalized 0..mr + apply
      // form used below reads as jam-able to Stage 3 (the bound dependence
      // hides behind the affine.apply), and unroll-jamming it collapses
      // the DIAG's j range to empty — measured on covariance as a lost
      // diagonal (checksum p+40 -> p+64).  This shape is exactly the
      // original triangular nest Stage 3 provably skips.
      auto i2 = rewriter.create<AffineForOp>(loc, ValueRange{ii}, idMap,
                                             ValueRange{ii}, addMrMap, 1);
      rewriter.setInsertionPointToStart(i2.getBody());
      Value ni2 = i2.getInductionVar();
      SmallVector<Value> jLbOps(sInLbOps);
      for (Value &v : jLbOps)
        if (v == iv)
          v = ni2;
      auto jL = rewriter.create<AffineForOp>(loc, ValueRange(jLbOps),
                                             sInLbMap, ValueRange{ii},
                                             addMrMap, 1);
      rewriter.setInsertionPointToStart(jL.getBody());
      IRMapping m;
      m.map(iv, ni2);
      m.map(sIn.getInductionVar(), jL.getInductionVar());
      rewriter.clone(*red.getOperation(), m);
      return i2;
    }
    auto iL = rewriter.create<AffineForOp>(loc, 0, (int64_t)mr, 1);
    rewriter.setInsertionPointToStart(iL.getBody());
    Value iLocal = iL.getInductionVar();
    auto emitApply = [&]() {
      return rewriter.create<affine::AffineApplyOp>(loc, addMap,
                                                    ValueRange{iLocal, ii});
    };
    AffineForOp jL;
    Value ni;
    if (!diag) {
      if (ubTriangular) {
        jL = rewriter.create<AffineForOp>(loc, ValueRange(sInLbOps), sInLbMap,
                                          ValueRange{ii}, idMap, 1); // 0 .. ii
      } else {
        // lb-triangular HEAD: j = ii+mr .. UB, invariant in i'.
        jL = rewriter.create<AffineForOp>(loc, ValueRange{ii}, addMrMap,
                                          ValueRange(sInUbOps), sInUbMap, 1);
      }
      rewriter.setInsertionPointToStart(jL.getBody());
      ni = emitApply(); // row offset computed inside the (perfect) j-loop
    } else {
      ni = emitApply(); // offset precedes j (feeds the ragged bound)
      if (ubTriangular) {
        SmallVector<Value> jUbOps(sInUbOps);
        for (Value &v : jUbOps)
          if (v == iv)
            v = ni;
        jL = rewriter.create<AffineForOp>(loc, ValueRange{ii}, idMap,
                                          ValueRange(jUbOps), sInUbMap,
                                          1); // ii .. f(ni)
      } else {
        // lb-triangular DIAG: j = g(ni) .. ii+mr, ragged mr-wide corner.
        SmallVector<Value> jLbOps(sInLbOps);
        for (Value &v : jLbOps)
          if (v == iv)
            v = ni;
        jL = rewriter.create<AffineForOp>(loc, ValueRange(jLbOps), sInLbMap,
                                          ValueRange{ii}, addMrMap, 1);
      }
      rewriter.setInsertionPointToStart(jL.getBody());
    }
    IRMapping m;
    m.map(iv, ni);
    m.map(sIn.getInductionVar(), jL.getInductionVar());
    auto redClone = cast<AffineForOp>(rewriter.clone(*red.getOperation(), m));
    if (!diag && kkIV) {
      redClone.setLowerBound(ValueRange{kkIV}, idMap);
      redClone.setUpperBound(ValueRange{kkIV}, addTkMap);
    }
    return iL;
  };

  // Emit one half; the HEAD gets the optional kk chunk loop wrapped around
  // its i' nest (kk sits ABOVE i' so Stage 3 still sees the perfect
  // i'-j-k band: red's parent must remain the j loop).
  auto emitHalf = [&](Value ii, bool diag) -> AffineForOp {
    if (!diag && kTileSize) {
      auto kkL = rewriter.create<AffineForOp>(
          loc, red.getConstantLowerBound(), red.getConstantUpperBound(),
          kTileSize);
      rewriter.setInsertionPointToStart(kkL.getBody());
      buildHalf(ii, /*diag=*/false, kkL.getInductionVar());
      return kkL;
    }
    return buildHalf(ii, diag, Value());
  };

  rewriter.setInsertionPoint(sOut);
  auto strip = rewriter.create<AffineForOp>(loc, lo, stripHi, (int64_t)mr);
  if (stripHi != hi) {
    // Remainder epilogue: the original triangular nest over [stripHi, hi).
    // Cloned verbatim (original IVs, ragged bound on a real IV) so Stage 3
    // provably skips it, exactly like the pre-peel nest.
    rewriter.setInsertionPointAfter(strip);
    auto epi = cast<AffineForOp>(rewriter.clone(*sOut.getOperation()));
    epi.setConstantLowerBound(stripHi);
  }
  rewriter.setInsertionPointToStart(strip.getBody());
  Value ii = strip.getInductionVar();
  // Per-row j order stays ascending: ub-triangular runs HEAD (LB..ii) before
  // DIAG (ii..f(ni)); lb-triangular runs DIAG (g(ni)..ii+mr) before HEAD
  // (ii+mr..UB).  Reductions are insensitive either way; keep it tidy.
  AffineForOp first = emitHalf(ii, /*diag=*/!ubTriangular);
  rewriter.setInsertionPointAfter(first);
  emitHalf(ii, /*diag=*/ubTriangular);
  rewriter.eraseOp(sOut);
  return true;
}

/// Diagonal-peel a triangular REDUCTION band: a perfect band `for i { for k =
/// g(i)..N { for j { red }}}` whose reduction loop's lower bound depends on the
/// outer spatial IV `i` (e.g. trmm's `k = i..N`).  The mr rows of an i-strip
/// have different k-ranges, so unroll-and-jam can't fuse them.  Split into:
///   MAIN:   for i'=0..mr { for j { for k = ii+mr-1 .. N  { red(ii+i', j, k) }}}
///           -- k range uniform across the strip, pushed innermost, so Stage 3
///              register-blocks it (no interchange needed).
///   CORNER: for i'=0..mr { for j { for k = (ii+i') .. ii+mr-1 { red }}}
///           -- ragged per row, left scalar (Stage 3 skips it).
/// Together they cover k in [ii+i', N) for each row.  Returns true if rewritten.
bool peelTriangularReduction(AffineForOp sOut, unsigned mr,
                                    IRRewriter &rewriter) {
  if (mr < 2)
    return false;
  AffineForOp redMid = onlyChildFor(sOut); // k
  if (!redMid)
    return false;
  AffineForOp inner = onlyChildFor(redMid); // j
  if (!inner || !isInnermost(inner))
    return false;
  AffineStoreOp store;
  AffineLoadOp load;
  if (!findAccPair(inner, store, load))
    return false;
  // The accumulator must vary in the inner spatial loop and be invariant in the
  // reduction (so `redMid` is the reduction and `inner` the spatial dim).
  if (!addrDependsOnIV(store, inner.getInductionVar()) ||
      addrDependsOnIV(store, redMid.getInductionVar()))
    return false;
  Value iv = sOut.getInductionVar();
  // Triangular reduction: k's lower bound depends on i, upper bound is constant.
  if (!llvm::is_contained(redMid.getLowerBoundOperands(), iv) ||
      !redMid.hasConstantUpperBound())
    return false;
  if (!sOut.hasConstantLowerBound() || !sOut.hasConstantUpperBound() ||
      sOut.getStepAsInt() != 1)
    return false;
  int64_t lo = sOut.getConstantLowerBound(), hi = sOut.getConstantUpperBound();
  // Strip-mine the mr-divisible prefix; the < mr remainder rows become a
  // scalar epilogue clone of the original nest (see peelTriangularNest).
  int64_t stripHi = lo + ((hi - lo) / (int64_t)mr) * (int64_t)mr;
  if (stripHi == lo)
    return false; // trip < mr
  if (!inner.hasConstantLowerBound() || !inner.hasConstantUpperBound())
    return false;

  MLIRContext *ctx = sOut.getContext();
  Location loc = sOut.getLoc();
  AffineExpr d0 = getAffineDimExpr(0, ctx), d1 = getAffineDimExpr(1, ctx);
  AffineMap idMap = AffineMap::get(1, 0, d0);          // (d0)     -> d0
  AffineMap addMap = AffineMap::get(2, 0, d0 + d1);    // (d0, d1) -> d0 + d1
  AffineMap offMap =                                   // (d0)     -> d0 + mr-1
      AffineMap::get(1, 0, d0 + (int64_t)(mr - 1));
  AffineMap kUbMap = redMid.getUpperBoundMap();
  SmallVector<Value> kUbOps(redMid.getUpperBoundOperands());
  int64_t jlo = inner.getConstantLowerBound(), jhi = inner.getConstantUpperBound();
  int64_t jstep = inner.getStepAsInt(), kstep = redMid.getStepAsInt();

  auto buildHalf = [&](Value ii, bool corner) -> AffineForOp {
    auto iL = rewriter.create<AffineForOp>(loc, 0, (int64_t)mr, 1); // i'
    rewriter.setInsertionPointToStart(iL.getBody());
    Value iLocal = iL.getInductionVar();
    auto jL = rewriter.create<AffineForOp>(loc, jlo, jhi, jstep);
    rewriter.setInsertionPointToStart(jL.getBody());
    AffineForOp kL;
    Value ni;
    if (!corner) {
      // MAIN: k = ii+mr-1 .. N (uniform); row offset computed INSIDE k-body so
      // i'/j/k stay perfect single-child nests (Stage 3 can unroll-jam them).
      kL = rewriter.create<AffineForOp>(loc, ValueRange{ii}, offMap,
                                        ValueRange(kUbOps), kUbMap, kstep);
      rewriter.setInsertionPointToStart(kL.getBody());
      ni = rewriter.create<affine::AffineApplyOp>(loc, addMap,
                                                  ValueRange{iLocal, ii});
    } else {
      // CORNER: k = (ii+i') .. ii+mr-1 (ragged); offset precedes k (feeds its
      // lower bound).  Left scalar.
      ni = rewriter.create<affine::AffineApplyOp>(loc, addMap,
                                                  ValueRange{iLocal, ii});
      kL = rewriter.create<AffineForOp>(loc, ValueRange{ni}, idMap,
                                        ValueRange{ii}, offMap, kstep);
      rewriter.setInsertionPointToStart(kL.getBody());
    }
    IRMapping m;
    m.map(iv, ni);
    m.map(redMid.getInductionVar(), kL.getInductionVar());
    m.map(inner.getInductionVar(), jL.getInductionVar());
    for (Operation &op : inner.getBody()->without_terminator())
      rewriter.clone(op, m);
    return iL;
  };

  rewriter.setInsertionPoint(sOut);
  auto strip = rewriter.create<AffineForOp>(loc, lo, stripHi, (int64_t)mr);
  if (stripHi != hi) {
    rewriter.setInsertionPointAfter(strip);
    auto epi = cast<AffineForOp>(rewriter.clone(*sOut.getOperation()));
    epi.setConstantLowerBound(stripHi);
  }
  rewriter.setInsertionPointToStart(strip.getBody());
  Value ii = strip.getInductionVar();
  AffineForOp mainI = buildHalf(ii, /*corner=*/false);
  rewriter.setInsertionPointAfter(mainI);
  buildHalf(ii, /*corner=*/true);
  rewriter.eraseOp(sOut);
  return true;
}

/// In-place triangular reduction band with the reduction INNERMOST (the
/// PolyBench trmm shape once dr-affine-loop-distribute has fissioned the
/// trailing alpha-scale into its own nest):
///     for i { for j { for k = i+c .. N { B[i][j] += A[k][i] * B[k][j] } } }
/// The accumulator's own memref is read at row k, so accumulatorAliasesInput
/// rightly refuses to register-block it as-is (rows of one strip read each
/// other).  Split each mr-strip of i by the k range instead:
///   CORNER (emitted FIRST, in ORIGINAL coordinates -- ragged bounds on real
///           IVs, and its i2 loop carries a real dependence so Stage 3's
///           parallel check provably skips it):
///     for i2 = ii..ii+mr { for j { for k = i2+c .. ii+mr-1+c { red } } }
///   MAIN  (k >= ii+mr-1+c > every strip row, so its B[k][j] reads are
///          row-disjoint from the strip's accumulators; certified with
///          dr.acc_no_alias so Stage 3 register-blocks it):
///     for i' = 0..mr { for j { for k = ii+mr-1+c .. N { red(ii+i') } } }
/// Soundness/bit-identity: per (i,j) the additions stay in ascending k order
/// (CORNER covers the low k's first), and a row is only ever READ by rows
/// above it, which run earlier in both schedules, so every read sees exactly
/// the value the original schedule saw.
bool peelInPlaceTriangularInnermost(AffineForOp sOut, unsigned mr,
                                           IRRewriter &rewriter) {
  if (mr < 2)
    return false;
  AffineForOp sIn = onlyChildFor(sOut);
  if (!sIn)
    return false;
  AffineForOp red = onlyChildFor(sIn);
  if (!red || !isInnermost(red))
    return false;
  if (!sOut.hasConstantLowerBound() || !sOut.hasConstantUpperBound() ||
      sOut.getStepAsInt() != 1 || sIn.getStepAsInt() != 1 ||
      red.getStepAsInt() != 1)
    return false;
  if (!sIn.hasConstantLowerBound() || !sIn.hasConstantUpperBound())
    return false;
  if (!red.hasConstantUpperBound())
    return false;
  Value iv = sOut.getInductionVar(), jIV = sIn.getInductionVar(),
        kIV = red.getInductionVar();
  // Reduction lower bound must be exactly `i + c` with c >= 1 (c == 0 would
  // make k == i read the accumulator row itself).
  AffineMap lbMap = red.getLowerBoundMap();
  if (lbMap.getNumResults() != 1 || red.getLowerBoundOperands().size() != 1 ||
      red.getLowerBoundOperands()[0] != iv)
    return false;
  int64_t c = -1;
  if (auto bin = dyn_cast<AffineBinaryOpExpr>(lbMap.getResult(0))) {
    auto cst = dyn_cast<AffineConstantExpr>(bin.getRHS());
    if (bin.getKind() != AffineExprKind::Add || !cst ||
        !isa<AffineDimExpr>(bin.getLHS()))
      return false;
    c = cst.getValue();
  }
  if (c < 1)
    return false;

  // Body shape: exactly one store (the accumulator, addressed [i, j]), a
  // matching load it depends on, optional reads of OTHER memrefs, and any
  // read of the accumulator's memref addressed EXACTLY [k, j] (the in-place
  // multiplicand; at least one must exist or the plain peels apply).
  Block *body = red.getBody();
  MLIRContext *ctx = sOut.getContext();
  AffineMap id2 = AffineMap::getMultiDimIdentityMap(2, ctx);
  AffineStoreOp accSt;
  for (Operation &op : body->without_terminator()) {
    if (auto st = dyn_cast<AffineStoreOp>(&op)) {
      if (accSt)
        return false;
      accSt = st;
    } else if (!isMemoryEffectFree(&op) && !isa<AffineLoadOp>(&op)) {
      return false;
    }
  }
  if (!accSt || accSt.getAffineMap() != id2)
    return false;
  auto stOps = accSt.getMapOperands();
  if (stOps.size() != 2 || stOps[0] != iv || stOps[1] != jIV)
    return false;
  AffineLoadOp accLd;
  bool sawInPlaceRead = false;
  for (Operation &op : body->without_terminator()) {
    auto ld = dyn_cast<AffineLoadOp>(&op);
    if (!ld || ld.getMemRef() != accSt.getMemRef())
      continue;
    auto ldOps = ld.getMapOperands();
    if (ld.getAffineMap() == id2 && ldOps.size() == 2 && ldOps[0] == iv &&
        ldOps[1] == jIV) {
      accLd = ld;
      continue;
    }
    if (ld.getAffineMap() == id2 && ldOps.size() == 2 && ldOps[0] == kIV &&
        ldOps[1] == jIV) {
      sawInPlaceRead = true;
      continue;
    }
    return false; // any other access shape to the accumulator memref
  }
  if (!accLd || !sawInPlaceRead ||
      !dependsOn(accSt.getValueToStore(), accLd.getResult(), body))
    return false;

  int64_t lo = sOut.getConstantLowerBound(), hi = sOut.getConstantUpperBound();
  int64_t stripHi = lo + ((hi - lo) / (int64_t)mr) * (int64_t)mr;
  if (stripHi == lo)
    return false;
  int64_t jlo = sIn.getConstantLowerBound(), jhi = sIn.getConstantUpperBound();

  Location loc = sOut.getLoc();
  AffineExpr d0 = getAffineDimExpr(0, ctx), d1 = getAffineDimExpr(1, ctx);
  AffineMap idMap = AffineMap::get(1, 0, d0);
  AffineMap addMap = AffineMap::get(2, 0, d0 + d1);
  AffineMap addMrMap = AffineMap::get(1, 0, d0 + (int64_t)mr);
  AffineMap mainLbMap = AffineMap::get(1, 0, d0 + (int64_t)(mr - 1) + c);
  AffineMap kUbMap = red.getUpperBoundMap();
  SmallVector<Value> kUbOps(red.getUpperBoundOperands());

  rewriter.setInsertionPoint(sOut);
  auto strip = rewriter.create<AffineForOp>(loc, lo, stripHi, (int64_t)mr);
  if (stripHi != hi) {
    rewriter.setInsertionPointAfter(strip);
    auto epi = cast<AffineForOp>(rewriter.clone(*sOut.getOperation()));
    epi.setConstantLowerBound(stripHi);
  }
  rewriter.setInsertionPointToStart(strip.getBody());
  Value ii = strip.getInductionVar();

  // CORNER: original coordinates, k = i2+c .. ii+mr-1+c.
  auto i2L = rewriter.create<AffineForOp>(loc, ValueRange{ii}, idMap,
                                          ValueRange{ii}, addMrMap, 1);
  rewriter.setInsertionPointToStart(i2L.getBody());
  auto jC = rewriter.create<AffineForOp>(loc, jlo, jhi, 1);
  rewriter.setInsertionPointToStart(jC.getBody());
  auto kC = rewriter.create<AffineForOp>(loc,
                                         ValueRange{i2L.getInductionVar()},
                                         lbMap, ValueRange{ii}, mainLbMap, 1);
  rewriter.setInsertionPointToStart(kC.getBody());
  {
    IRMapping m;
    m.map(iv, i2L.getInductionVar());
    m.map(jIV, jC.getInductionVar());
    m.map(kIV, kC.getInductionVar());
    for (Operation &op : body->without_terminator())
      rewriter.clone(op, m);
  }

  // MAIN: uniform k range, row offset inside the k body (jam-proof shape).
  rewriter.setInsertionPointAfter(i2L);
  auto iM = rewriter.create<AffineForOp>(loc, 0, (int64_t)mr, 1);
  rewriter.setInsertionPointToStart(iM.getBody());
  auto jM = rewriter.create<AffineForOp>(loc, jlo, jhi, 1);
  rewriter.setInsertionPointToStart(jM.getBody());
  auto kM = rewriter.create<AffineForOp>(loc, ValueRange{ii}, mainLbMap,
                                         ValueRange(kUbOps), kUbMap, 1);
  kM->setAttr(kAccNoAliasAttr, rewriter.getUnitAttr());
  rewriter.setInsertionPointToStart(kM.getBody());
  {
    Value ni = rewriter.create<affine::AffineApplyOp>(
        loc, addMap, ValueRange{iM.getInductionVar(), ii});
    IRMapping m;
    m.map(iv, ni);
    m.map(jIV, jM.getInductionVar());
    m.map(kIV, kM.getInductionVar());
    for (Operation &op : body->without_terminator())
      rewriter.clone(op, m);
  }

  rewriter.eraseOp(sOut);
  return true;
}

} // namespace rb
} // namespace drcompiler
