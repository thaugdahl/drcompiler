//===- Vectorize.cpp - explicit vector micro-kernels ===//
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
#include <functional>

#define DEBUG_TYPE "affine-register-block"

using namespace mlir;
using affine::AffineForOp;
using affine::AffineLoadOp;
using affine::AffineStoreOp;

namespace drcompiler {
namespace rb {

/// True if `load`'s innermost (fastest-varying) memref dimension is indexed
/// exactly by induction variable `iv` with coefficient 1 -- i.e. consecutive
/// `iv` values give stride-1 (contiguous) addresses.
bool innermostStrideOne(AffineLoadOp load, Value iv) {
  AffineMap m = load.getAffineMap();
  if (m.getNumResults() == 0)
    return false;
  AffineExpr last = m.getResult(m.getNumResults() - 1);
  // Find which map dim is `iv`.
  auto operands = load.getMapOperands();
  unsigned pos = m.getNumDims();
  for (unsigned d = 0; d < m.getNumDims(); ++d)
    if (d < operands.size() && operands[d] == iv) {
      pos = d;
      break;
    }
  if (pos == m.getNumDims())
    return false; // `iv` is not a dim of this map
  // Stride-1 iff `iv` appears in the innermost result with coefficient exactly 1
  // and nothing else multiplies it: i.e. `last == dim(pos) + g` where g is
  // independent of dim(pos).  Covers a pure `B[k][j]` (g = 0) AND a conv
  // `in[ic][oh+kh][ow+kw]` (last = ow + kw, g = kw): incrementing `iv` by 1
  // moves the address by exactly one element either way.
  AffineExpr dimP = getAffineDimExpr(pos, m.getContext());
  // `last - dimP` does not auto-cancel like terms; simplify before testing so a
  // genuine `dimP + g` (coeff 1) reports stride-1.
  AffineExpr diff =
      simplifyAffineExpr(last - dimP, m.getNumDims(), m.getNumSymbols());
  return !diff.isFunctionOfDim(pos);
}

/// Classify a reduction band by operand layout.  `red` is the innermost
/// reduction loop (k); `sIn` is the inner spatial loop (j, the one that will be
/// unroll-jammed by nr).  A *multiplicand* is a load whose address involves k
/// (the accumulator is k-invariant and excluded).
///   - Broadcast (C=A*B): some multiplicand is stride-1 in j (e.g. B[k][j]).
///     LLVM SLP-vectorizes the independent j-lanes; wants a WIDE tile and NO
///     reassociation (each lane is a distinct output, no cross-lane sum).
///   - Dot (rank-k, C=A*A^T): multiplicands stride-1 in k, none in j (e.g.
///     A[i][k], A[j][k]).  LLVM must reduction-vectorize over k; wants a SMALL
///     square tile (each accumulator is a full vector) and REQUIRES reassoc.
/// `nMul` returns the multiplicand count (a register-pressure proxy: more input
/// streams -> a tighter tile).
RBFamily detectFamily(AffineForOp red, AffineForOp sIn, int &nMul) {
  Value k = red.getInductionVar(), j = sIn.getInductionVar();
  bool strideJ = false, strideK = false;
  nMul = 0;
  for (Operation &op : red.getBody()->without_terminator()) {
    auto load = dyn_cast<AffineLoadOp>(&op);
    if (!load)
      continue;
    if (!llvm::is_contained(load.getMapOperands(), k))
      continue; // k-invariant => accumulator, not a multiplicand
    ++nMul;
    strideJ |= innermostStrideOne(load, j);
    strideK |= innermostStrideOne(load, k);
  }
  if (strideJ)
    return RBFamily::Broadcast;
  if (strideK)
    return RBFamily::Dot;
  return RBFamily::Broadcast;
}

/// Can the scalar reduction DAG rooted at `v` be vectorized along `jIV` (the
/// inner spatial dim)?  Supported: the accumulator load (-> iter_arg), affine
/// loads that are either stride-1 in j (-> vector load) or invariant in j (->
/// broadcast), FP mul/add/sub/div/neg, and j-invariant scalars (-> broadcast).
/// A load that varies in j but is NOT innermost-stride-1 would need a gather ->
/// unsupported (caller falls back to the scalar+SLP path).
static bool canVectorizeDAG(Value v, const llvm::SmallPtrSetImpl<Operation *> &accLoads,
                            Value jIV, Block *redBody) {
  Operation *def = v.getDefiningOp();
  if (!def || def->getBlock() != redBody)
    return true; // loop-invariant scalar -> broadcast
  if (accLoads.contains(def))
    return true; // accumulator load -> iter_arg
  if (auto ld = dyn_cast<AffineLoadOp>(def)) {
    if (innermostStrideOne(ld, jIV))
      return true;
    return !llvm::is_contained(ld.getMapOperands(), jIV); // invariant in j
  }
  if (isa<arith::MulFOp, arith::AddFOp, arith::SubFOp, arith::DivFOp>(def))
    return canVectorizeDAG(def->getOperand(0), accLoads, jIV, redBody) &&
           canVectorizeDAG(def->getOperand(1), accLoads, jIV, redBody);
  if (isa<arith::NegFOp>(def))
    return canVectorizeDAG(def->getOperand(0), accLoads, jIV, redBody);
  return false;
}

/// Emit, at `b`'s insertion point, the vector<VL> equivalent of the scalar value
/// `v` from the (still-live) original reduction body `oldRedBody`.  `accToIter`
/// maps each accumulator load to its new vector iter_arg; `remap` rewrites old
/// operands to values that dominate the new loop (the new reduction IV and the
/// hoisted index `affine.apply`s).  See canVectorizeDAG for supported shapes.
static Value vectorizeReductionValue(Value v, DenseMap<Value, Value> &accToIter,
                                     IRMapping &remap, Value jIV,
                                     VectorType vecTy, OpBuilder &b,
                                     Block *oldRedBody) {
  if (Value it = accToIter.lookup(v))
    return it; // accumulator load -> carried vector iter_arg
  Operation *def = v.getDefiningOp();
  if (!def || def->getBlock() != oldRedBody) // loop-invariant scalar
    return b.create<vector::BroadcastOp>(v.getLoc(), vecTy,
                                         remap.lookupOrDefault(v));
  if (auto ld = dyn_cast<AffineLoadOp>(def)) {
    SmallVector<Value> ops;
    for (Value o : ld.getMapOperands())
      ops.push_back(remap.lookupOrDefault(o));
    if (innermostStrideOne(ld, jIV))
      return b.create<affine::AffineVectorLoadOp>(
          ld.getLoc(), vecTy, ld.getMemRef(), ld.getAffineMap(), ops);
    Value scalar = b.create<AffineLoadOp>(ld.getLoc(), ld.getMemRef(),
                                          ld.getAffineMap(), ops);
    return b.create<vector::BroadcastOp>(ld.getLoc(), vecTy, scalar);
  }
  Location loc = def->getLoc();
  Value l = vectorizeReductionValue(def->getOperand(0), accToIter, remap, jIV,
                                    vecTy, b, oldRedBody);
  if (isa<arith::NegFOp>(def))
    return b.create<arith::NegFOp>(loc, l);
  Value r = vectorizeReductionValue(def->getOperand(1), accToIter, remap, jIV,
                                    vecTy, b, oldRedBody);
  if (isa<arith::MulFOp>(def))
    return b.create<arith::MulFOp>(loc, l, r);
  if (isa<arith::AddFOp>(def))
    return b.create<arith::AddFOp>(loc, l, r);
  if (isa<arith::SubFOp>(def))
    return b.create<arith::SubFOp>(loc, l, r);
  return b.create<arith::DivFOp>(loc, l, r);
}

/// Vectorize a broadcast reduction band along the inner spatial loop `sIn` with
/// vector width `VL`, instead of unroll-jamming it by nr and relying on LLVM
/// SLP.  The (already mr-unroll-jammed) accumulators become mr `vector<VL>`
/// values carried over the reduction: the streamed operand is a contiguous
/// `affine.vector_load`, the broadcast operand a scalar load + `vector.broadcast`,
/// each FMA a vector op.  This makes vectorization explicit so it survives the
/// >2D accumulator addressing of tensor contractions (batched matmul), where SLP
/// fails.  Returns success if it vectorized.
LogicalResult vectorizeBroadcastBand(AffineForOp red, AffineForOp sIn,
                                            unsigned VL, IRRewriter &rewriter) {
  if (VL < 2)
    return failure();
  Value jIV = sIn.getInductionVar();
  if (sIn.getStepAsInt() != 1)
    return failure();
  // The inner spatial extent must be a constant multiple of VL.  This holds for
  // an untiled loop (trip = N) and for a cache-tiled point loop whose extent is
  // exactly the tile size (`tc .. tc+tile`, symbolic bound but constant trip) --
  // so the vector micro-kernel composes with cache tiling.  A non-constant
  // trip (a symbolic bound, e.g. j = k+1..M) is handled below by an AFFINE
  // vl-split: main loop up to lb + ((ub-lb) floordiv VL)*VL, scalar tail.
  std::optional<uint64_t> trip = affine::getConstantTripCount(sIn);
  // The j-body must be PERFECT: re-stepping j by VL turns every remaining
  // scalar statement into a once-per-VL-lanes operation.  Measured on
  // covariance's mean nest `for j { mean[j]=0; for i acc; mean[j]/=n }`:
  // only every 16th element was initialized and divided — a silent
  // miscompile, not a missed optimization.  Anything besides the reduction
  // loop and the terminator (modulo pure ops feeding only the bound/body)
  // bails to the scalar path.
  for (Operation &op : sIn.getBody()->without_terminator())
    if (&op != red.getOperation() && !isMemoryEffectFree(&op))
      return failure();
  // (The vl-remainder is peeled below, AFTER the band is confirmed vectorizable.)
  SmallVector<Acc> accs = collectAccumulators(red);
  if (accs.empty())
    return failure();
  // The explicit vector micro-kernel handles ALL ranks (2D gemm through tensor
  // contractions).  Vectorization is emitted in the vector dialect rather than
  // betting on LLVM-SLP firing downstream, so the codegen -- and the cost model
  // that reasons about it -- is predictable; we no longer have to guess whether
  // SLP will vectorize a 2D tile (it loses on awkward N / strided / tail cases).
  // The full mr x ceil(nr/vl) register tile is completed by the caller, which
  // unroll-jams this vl-stepped loop into nrVec vector columns.  SLP remains only
  // as a guarded fallback when this returns failure (a non-vectorizable band).
  llvm::SmallPtrSet<Operation *, 8> accLoads;
  for (Acc &a : accs) {
    if (!innermostStrideOne(a.load, jIV)) // store/load must be vectorizable in j
      return failure();
    accLoads.insert(a.load);
  }
  Block *redBody = red.getBody();
  for (Acc &a : accs)
    if (!canVectorizeDAG(a.storedVal, accLoads, jIV, redBody))
      return failure();
  auto elemTy = cast<MemRefType>(accs[0].memref.getType()).getElementType();
  auto vecTy = VectorType::get({(int64_t)VL}, elemTy);
  Value oldKIV = red.getInductionVar();
  Location loc = red.getLoc();

  // Peel the vl-remainder so the explicit kernel fires on ANY N: split `sIn` into
  // a vl-divisible main loop (vectorized below) + a scalar tail clone.  Bailing
  // instead drops the band to the SLP fallback, which LOSES to clang on
  // non-vl-divisible N (clang masks the tail; we don't).  The tail is a small
  // (<VL) scalar copy left memory-backed; LLVM handles it.  Done here -- after the
  // rank/stride/DAG checks -- so only bands we will actually vectorize are split
  // (a rank<3 or non-vectorizable band has already returned failure above).
  if (trip && *trip % VL != 0) {
    if (!sIn.hasConstantLowerBound() || !sIn.hasConstantUpperBound())
      return failure();
    int64_t lb = sIn.getConstantLowerBound();
    int64_t mainUb = lb + (int64_t)((*trip / VL) * VL);
    if (mainUb == lb) // trip < VL: nothing to vectorize, leave to SLP
      return failure();
    rewriter.setInsertionPointAfter(sIn);
    rewriter.clone(*sIn); // scalar tail [mainUb, ub), still memory-backed
    auto tail = cast<AffineForOp>(sIn->getNextNode());
    tail.setConstantLowerBound(mainUb);
    sIn.setConstantUpperBound(mainUb);
  } else if (!trip) {
    // SYMBOLIC trip (gramschmidt's projection sweep j = k+1..M under the
    // sequential k): same split with affine bounds.  mainUb =
    // lb + ((ub - lb) floordiv VL) * VL, expressed over the concatenated
    // lb/ub operands; (mainUb - lb) is a VL multiple by construction, so
    // the VL-stepped main loop ends exactly at mainUb.  A runtime trip
    // < VL makes the main loop zero-trip and the tail cover everything.
    AffineMap lbM = sIn.getLowerBoundMap(), ubM = sIn.getUpperBoundMap();
    if (lbM.getNumResults() != 1 || ubM.getNumResults() != 1 ||
        lbM.getNumSymbols() != 0 || ubM.getNumSymbols() != 0)
      return failure();
    unsigned nlb = lbM.getNumDims(), nub = ubM.getNumDims();
    SmallVector<Value> ops(sIn.getLowerBoundOperands());
    ops.append(sIn.getUpperBoundOperands().begin(),
               sIn.getUpperBoundOperands().end());
    AffineExpr lbE = lbM.getResult(0);
    AffineExpr ubE = ubM.getResult(0).shiftDims(nub, nlb);
    AffineExpr mainUbE =
        lbE + (ubE - lbE).floorDiv((int64_t)VL) * (int64_t)VL;
    AffineMap mainUbMap = AffineMap::get(nlb + nub, 0, mainUbE);
    rewriter.setInsertionPointAfter(sIn);
    rewriter.clone(*sIn);
    auto tail = cast<AffineForOp>(sIn->getNextNode());
    tail.setLowerBound(ops, mainUbMap);
    sIn.setUpperBound(ops, mainUbMap);
  }
  // The inner spatial loop now strides by VL (one VL-lane along j per iteration).
  sIn.setStep(VL);

  // Hoist the accumulators' index operands (the i-offset affine.applys that
  // unroll-and-jam left inside the reduction body) to in front of the loop, so
  // the pre-loop vector inits and the post-loop vector stores can reference them.
  rewriter.setInsertionPoint(red);
  IRMapping hoistMap;
  SmallVector<SmallVector<Value>> hoistedOps(accs.size());
  for (auto [i, a] : llvm::enumerate(accs)) {
    SmallVector<Value> ops;
    for (Value o : a.operands)
      ops.push_back(hoistOperand(o, red, rewriter, hoistMap));
    hoistedOps[i] = ops;
  }
  SmallVector<Value> initVals;
  for (auto [i, a] : llvm::enumerate(accs))
    initVals.push_back(rewriter.create<affine::AffineVectorLoadOp>(
        a.loc, vecTy, a.memref, a.map, hoistedOps[i]));

  // Build the vector reduction loop from scratch (reading the still-live scalar
  // body), then erase the original.
  Block *oldRedBody = red.getBody();
  auto newK = rewriter.create<AffineForOp>(
      loc, red.getLowerBoundOperands(), red.getLowerBoundMap(),
      red.getUpperBoundOperands(), red.getUpperBoundMap(), red.getStepAsInt(),
      initVals, [&](OpBuilder &b, Location bloc, Value iv, ValueRange args) {
        IRMapping remap = hoistMap;   // old i-applys -> hoisted clones
        remap.map(oldKIV, iv);        // old reduction IV -> new reduction IV
        DenseMap<Value, Value> accToIter;
        for (auto [i, a] : llvm::enumerate(accs))
          accToIter[a.load.getResult()] = args[i];
        SmallVector<Value> yields;
        for (Acc &a : accs)
          yields.push_back(vectorizeReductionValue(
              a.storedVal, accToIter, remap, jIV, vecTy, b, oldRedBody));
        b.create<affine::AffineYieldOp>(bloc, yields);
      });

  // Sink the final vector stores after the loop, then drop the scalar loop.
  rewriter.setInsertionPointAfter(newK);
  for (auto [i, a] : llvm::enumerate(accs))
    rewriter.create<affine::AffineVectorStoreOp>(
        a.loc, newK.getResult(i), a.memref, a.map, hoistedOps[i]);
  rewriter.eraseOp(red);
  return success();
}

/// C1 (WP-O2 part 3b): fold producer `affine.apply`s into every affine
/// load/store map under `sp`, so an onnx-mlir precomputed index (`%b =
/// apply(kw, ow); load in[.., %b]`) exposes `ow` as a real dim of the op's own
/// map and the stride-1 test becomes exact: `in[.., ow+kw-1]` composes to a
/// last result `d_ow + d_kw - 1` (stride-1, vector load), while a stride-2
/// stem composes to `d_ow*2 + d_kw - 3` (correctly NOT stride-1, bails).
/// Scoped to the one conv band being vectorized -- canonicalizing the whole
/// function could perturb the 1x1 GEMM path that already works.
static void composeBandMemOps(AffineForOp sp, IRRewriter &rewriter) {
  SmallVector<Operation *> memOps;
  sp.walk([&](Operation *op) {
    if (isa<AffineLoadOp, AffineStoreOp>(op))
      memOps.push_back(op);
  });
  for (Operation *op : memOps) {
    if (auto ld = dyn_cast<AffineLoadOp>(op)) {
      AffineMap map = ld.getAffineMap();
      SmallVector<Value> ops(ld.getMapOperands());
      affine::fullyComposeAffineMapAndOperands(&map, &ops);
      affine::canonicalizeMapAndOperands(&map, &ops);
      if (map == ld.getAffineMap() && ValueRange(ops) == ld.getMapOperands())
        continue;
      rewriter.setInsertionPoint(ld);
      rewriter.replaceOpWithNewOp<AffineLoadOp>(ld, ld.getMemRef(), map, ops);
    } else {
      auto st = cast<AffineStoreOp>(op);
      AffineMap map = st.getAffineMap();
      SmallVector<Value> ops(st.getMapOperands());
      affine::fullyComposeAffineMapAndOperands(&map, &ops);
      affine::canonicalizeMapAndOperands(&map, &ops);
      if (map == st.getAffineMap() && ValueRange(ops) == st.getMapOperands())
        continue;
      rewriter.setInsertionPoint(st);
      rewriter.replaceOpWithNewOp<AffineStoreOp>(st, st.getValueToStore(),
                                                 st.getMemRef(), map, ops);
    }
  }
}

/// C2 (WP-O2 part 3b): solve one clamped bound map of a padded conv's inner
/// reduction loop for the spatial interior.  `m` is a max (lower, isLower) or
/// min (upper) bound map whose dim `owPos` is the spatial IV; supported (v1)
/// shape: exactly two results, one constant `cst` and one `a*ow + b` with
/// a in {+1,-1} and no other dims/symbols (all stride-1 onnx convs).  The
/// clamp is inactive -- the constant result dominates -- where `e(ow) <= cst`
/// (max) resp. `e(ow) >= cst` (min); tighten the interior `[lo, hi)` by that
/// half-line and return `cst`.  Anything else fails (band stays scalar).
static LogicalResult solveClampMap(AffineMap m, unsigned owPos, bool isLower,
                                   int64_t &lo, int64_t &hi, int64_t &cst) {
  if (m.getNumResults() != 2)
    return failure();
  std::optional<int64_t> c;
  std::optional<std::pair<int64_t, int64_t>> lin; // (a, b)
  for (AffineExpr e : m.getResults()) {
    if (auto ce = dyn_cast<AffineConstantExpr>(e)) {
      if (c)
        return failure();
      c = ce.getValue();
      continue;
    }
    for (unsigned d = 0; d < m.getNumDims(); ++d)
      if (d != owPos && e.isFunctionOfDim(d))
        return failure();
    for (unsigned s = 0; s < m.getNumSymbols(); ++s)
      if (e.isFunctionOfSymbol(s))
        return failure();
    AffineExpr dOw = getAffineDimExpr(owPos, m.getContext());
    AffineExpr bP = simplifyAffineExpr(e - dOw, m.getNumDims(), m.getNumSymbols());
    AffineExpr bN = simplifyAffineExpr(e + dOw, m.getNumDims(), m.getNumSymbols());
    if (lin)
      return failure();
    if (auto bc = dyn_cast<AffineConstantExpr>(bP))
      lin = {{1, bc.getValue()}};
    else if (auto bc = dyn_cast<AffineConstantExpr>(bN))
      lin = {{-1, bc.getValue()}};
    else
      return failure();
  }
  if (!c || !lin)
    return failure();
  auto [a, b] = *lin;
  if (isLower) { // max(a*ow+b, c): inactive where a*ow+b <= c
    if (a < 0)
      lo = std::max(lo, b - *c); // ow >= b - c
    else
      hi = std::min(hi, *c - b + 1); // ow <= c - b
  } else { // min(a*ow+b, c): inactive where a*ow+b >= c
    if (a < 0)
      hi = std::min(hi, b - *c + 1); // ow <= b - c
    else
      lo = std::max(lo, *c - b); // ow >= c - b
  }
  cst = *c;
  return success();
}

/// Vectorize a direct-conv reduction BAND along the spatial loop `sp` (ow).
/// `bandLoops` is the reduction band outer->inner (e.g. ic, kh, kw); the
/// innermost loop holds a single memref accumulator Y[.., ow] that is stride-1
/// in ow and invariant across the whole band.  Re-steps `sp` by VL and rebuilds
/// the band carrying a `vector<VL>` accumulator through every level: the weight
/// load (ow-invariant) becomes a broadcast, the input load (stride-1 in ow, e.g.
/// in[ic][oh+kh][ow+kw]) a contiguous vector load, the accumulator a carried
/// vector iter_arg loaded once before the band and stored once after.  This is
/// the GEMM broadcast micro-kernel generalized from a single reduction loop to a
/// multi-loop contraction band -- the WP-O2 direct-conv kernel.  vl-divisible
/// ow only (caller leaves a non-divisible remainder to the scalar tail).
LogicalResult vectorizeConvBand(AffineForOp sp, ArrayRef<AffineForOp> bandLoops,
                                unsigned VL, IRRewriter &rewriter) {
  if (VL < 2 || sp.getStepAsInt() != 1 || bandLoops.empty())
    return failure();
  // Fold precomputed-index applies into the band's own load/store maps FIRST,
  // so every check below (stride-1, DAG, the reaches-ow guard) sees the real
  // ow dependence instead of bailing on (or worse, mis-broadcasting) an
  // apply-hidden index.  Composing is semantics-preserving, so a band that
  // still bails afterwards is left composed-but-scalar -- harmless.
  composeBandMemOps(sp, rewriter);
  Value owIV = sp.getInductionVar();
  AffineForOp inner = bandLoops.back();
  SmallVector<Acc> accs = collectAccumulators(inner);
  if (accs.size() != 1)
    return failure();
  Acc a = accs[0];
  if (!innermostStrideOne(a.load, owIV))
    return failure();
  llvm::SmallPtrSet<Operation *, 4> accLoads;
  accLoads.insert(a.load.getOperation());
  if (!canVectorizeDAG(a.storedVal, accLoads, owIV, inner.getBody()))
    return failure();
  // SAFETY: canVectorizeDAG treats a load whose direct operands omit `owIV` as
  // ow-invariant (-> broadcast).  But onnx-mlir precomputes indices via
  // affine.apply, so `in[.., apply(kw, ow)]` hides ow behind the apply and a
  // stride-2 stem (in idx = ow*2+kw-3) would be mis-broadcast -> invalid/ wrong
  // code.  Bail if ANY band load reaches owIV transitively yet is not provably
  // stride-1 in ow.  (Composing the applies to vectorize such convs is WP-O2
  // part 3b; until then this keeps the pass correct.)
  std::function<bool(Value)> reachesOw = [&](Value v) -> bool {
    if (v == owIV)
      return true;
    Operation *d = v.getDefiningOp();
    if (!d || !isa<affine::AffineApplyOp>(d))
      return false;
    for (Value o : d->getOperands())
      if (reachesOw(o))
        return true;
    return false;
  };
  for (Operation &op : inner.getBody()->without_terminator())
    if (auto ld = dyn_cast<AffineLoadOp>(&op))
      if (!innermostStrideOne(ld, owIV))
        for (Value o : ld.getMapOperands())
          if (reachesOw(o))
            return failure();
  // C2: a padded conv clamps the inner reduction bounds by ow (kw in
  // [max(eLb(ow),cLb), min(eUb(ow),cUb))), which is un-vectorizable: the VL
  // ow-lanes would each need a different kw trip count.  Solve the INTERIOR
  // [owLo, owHi) where both clamps are provably inactive, split ow into
  // [scalar left border | interior | scalar right border], and rewrite the
  // interior's clamped bounds to their constants.  The borders keep the
  // original band verbatim (a pure index-set split, no FP reorder).  Loops
  // clamped by some OTHER IV (kh by oh) are identical across ow-lanes and are
  // left alone.
  if (!sp.hasConstantLowerBound() || !sp.hasConstantUpperBound())
    return failure();
  int64_t spLb = sp.getConstantLowerBound(), spUb = sp.getConstantUpperBound();
  int64_t owLo = spLb, owHi = spUb;
  struct Clamp {
    AffineForOp loop;
    int64_t cLb, cUb;
  };
  SmallVector<Clamp> clamps;
  for (AffineForOp L : bandLoops) {
    auto owDim = [&](Operation::operand_range ops, AffineMap m) -> int {
      for (unsigned d = 0; d < m.getNumDims(); ++d)
        if (ops[d] == owIV)
          return (int)d;
      return -1;
    };
    AffineMap lbM = L.getLowerBoundMap(), ubM = L.getUpperBoundMap();
    int lbPos = owDim(L.getLowerBoundOperands(), lbM);
    int ubPos = owDim(L.getUpperBoundOperands(), ubM);
    bool lbDep = lbPos >= 0 && llvm::any_of(lbM.getResults(), [&](AffineExpr e) {
                   return e.isFunctionOfDim(lbPos);
                 });
    bool ubDep = ubPos >= 0 && llvm::any_of(ubM.getResults(), [&](AffineExpr e) {
                   return e.isFunctionOfDim(ubPos);
                 });
    if (!lbDep && !ubDep)
      continue;
    int64_t cLb, cUb;
    if (lbDep) {
      if (failed(solveClampMap(lbM, lbPos, /*isLower=*/true, owLo, owHi, cLb)))
        return failure();
    } else if (L.hasConstantLowerBound()) {
      cLb = L.getConstantLowerBound();
    } else {
      return failure();
    }
    if (ubDep) {
      if (failed(solveClampMap(ubM, ubPos, /*isLower=*/false, owLo, owHi, cUb)))
        return failure();
    } else if (L.hasConstantUpperBound()) {
      cUb = L.getConstantUpperBound();
    } else {
      return failure();
    }
    clamps.push_back({L, cLb, cUb});
  }
  // Sub-VL interior (the 14x14 / 7x7 layers at VL=16): nothing to vectorize;
  // bail BEFORE mutating so the band stays a single scalar loop.
  if (owHi - owLo < (int64_t)VL)
    return failure();
  if (!clamps.empty()) {
    if (owLo > spLb) {
      rewriter.setInsertionPoint(sp);
      auto left = cast<AffineForOp>(rewriter.clone(*sp.getOperation()));
      left.setConstantUpperBound(owLo);
    }
    if (owHi < spUb) {
      rewriter.setInsertionPointAfter(sp);
      auto right = cast<AffineForOp>(rewriter.clone(*sp.getOperation()));
      right.setConstantLowerBound(owHi);
    }
    sp.setConstantLowerBound(owLo);
    sp.setConstantUpperBound(owHi);
    for (Clamp &cl : clamps) {
      cl.loop.setConstantLowerBound(cl.cLb);
      cl.loop.setConstantUpperBound(cl.cUb);
    }
  }
  std::optional<uint64_t> trip = affine::getConstantTripCount(sp);
  if (!trip || *trip % VL != 0)
    return failure();

  Location loc = sp.getLoc();
  auto elemTy = cast<MemRefType>(a.memref.getType()).getElementType();
  auto vecTy = VectorType::get({(int64_t)VL}, elemTy);

  // Seed: vector_load the accumulator slab once, before the band.
  rewriter.setInsertionPoint(bandLoops.front());
  Value seed = rewriter.create<affine::AffineVectorLoadOp>(loc, vecTy, a.memref,
                                                           a.map, a.operands);

  Block *oldInnerBody = inner.getBody();
  // Rebuild the band: one vector iter_arg threaded through every level; the
  // innermost vectorizes the scalar stored-value DAG along ow.
  std::function<Value(unsigned, Value, IRMapping &, OpBuilder &)> build =
      [&](unsigned lvl, Value sd, IRMapping &remap, OpBuilder &bld) -> Value {
    AffineForOp old = bandLoops[lvl];
    auto nf = bld.create<AffineForOp>(
        loc, old.getLowerBoundOperands(), old.getLowerBoundMap(),
        old.getUpperBoundOperands(), old.getUpperBoundMap(), old.getStepAsInt(),
        ValueRange{sd}, [&](OpBuilder &b2, Location l2, Value iv, ValueRange args) {
          IRMapping rm = remap;
          rm.map(old.getInductionVar(), iv);
          Value res;
          if (lvl + 1 < bandLoops.size())
            res = build(lvl + 1, args[0], rm, b2);
          else {
            DenseMap<Value, Value> accToIter;
            accToIter[a.load.getResult()] = args[0];
            res = vectorizeReductionValue(a.storedVal, accToIter, rm, owIV, vecTy,
                                          b2, oldInnerBody);
          }
          b2.create<affine::AffineYieldOp>(l2, res);
        });
    return nf.getResult(0);
  };
  IRMapping remap;
  Value result = build(0, seed, remap, rewriter);

  // Store the final vector slab once, after the band; then drop the scalar band
  // and re-step ow by VL.
  rewriter.create<affine::AffineVectorStoreOp>(loc, result, a.memref, a.map,
                                               a.operands);
  rewriter.eraseOp(bandLoops.front());
  sp.setStep(VL);
  return success();
}

/// Explicit reduction-vectorization for the DOT (rank-k) family: vectorize the
/// reduction loop `red` (k) itself by VL, carrying one `vector<VL>` partial sum
/// per accumulator across k-chunks, then horizontal-reduce, add the original C,
/// and scalar-store.  Multiplicands are stride-1 in k (contiguous vector loads
/// over k); the accumulator is k-invariant.  This makes the dot kernel
/// EXPLICITLY vectorized instead of relying on LLVM to reduction-vectorize the
/// fastmath'd scalar loop -- the cost model then reasons about real vector ops,
/// not a bet on LLVM.  Reuses `vectorizeReductionValue` with jIV=k (a load
/// stride-1 in k -> vector load; the k-invariant acc load -> the carried
/// iter_arg).  Requires trip(red) % VL == 0; otherwise the caller falls back to
/// scalar + reassoc + LLVM reduction-vec (still correct).
LogicalResult vectorizeDotBand(AffineForOp red, unsigned VL,
                                      IRRewriter &rewriter) {
  if (VL < 2 || red.getStepAsInt() != 1)
    return failure();
  std::optional<uint64_t> trip = affine::getConstantTripCount(red);
  if (!trip)
    return failure();
  SmallVector<Acc> accs = collectAccumulators(red);
  if (accs.empty())
    return failure();
  Value kIV = red.getInductionVar();
  Block *redBody = red.getBody();
  // The vectorized loop is rebuilt from the accumulators' def-use DAGs
  // alone; any other effectful op in the reduction body (a store to a
  // different array, a call) would be silently DROPPED.  Bail instead.
  {
    llvm::SmallPtrSet<Operation *, 8> accStores;
    for (Acc &a : accs)
      accStores.insert(a.store);
    for (Operation &op : redBody->without_terminator()) {
      if (isMemoryEffectFree(&op) || isa<AffineLoadOp>(op))
        continue;
      if (isa<AffineStoreOp>(op) && accStores.contains(&op))
        continue;
      return failure();
    }
  }
  llvm::SmallPtrSet<Operation *, 8> accLoads;
  for (Acc &a : accs)
    accLoads.insert(a.load);
  for (Acc &a : accs) {
    // Reduction shape `acc +/- prod`, with `prod` vectorizable along k.
    if (!a.storedVal.getDefiningOp<arith::AddFOp>() &&
        !a.storedVal.getDefiningOp<arith::SubFOp>())
      return failure();
    if (!canVectorizeDAG(a.storedVal, accLoads, kIV, redBody))
      return failure();
  }
  // Peel the k-remainder: split the reduction into a vl-divisible vector MAIN
  // (vectorized below) + a scalar TAIL that accumulates the leftover k into the
  // SAME C.  Both are partial sums of one reduction, so -- unlike the broadcast
  // peel (disjoint output columns) -- the tail must run AFTER the main stores and
  // read the partial result.  The tail is the original scalar body over
  // [mainUb, K); cloning it before the main rewrite preserves it, and placing it
  // after `red` means it follows the main + its stores once `red` is erased.
  // Bail (-> scalar+reassoc+SLP) when k < VL (no vectorizable main).
  if (*trip % VL != 0) {
    if (!red.hasConstantLowerBound() || !red.hasConstantUpperBound())
      return failure();
    int64_t lb = red.getConstantLowerBound();
    int64_t mainUb = lb + (int64_t)((*trip / VL) * VL);
    if (mainUb == lb)
      return failure();
    rewriter.setInsertionPointAfter(red);
    rewriter.clone(*red); // scalar tail (full original body)
    auto tail = cast<AffineForOp>(red->getNextNode());
    tail.setConstantLowerBound(mainUb); // tail = [mainUb, K), accumulates into C
    red.setConstantUpperBound(mainUb);  // main = [lb, mainUb), vectorized below
  }
  auto elemTy = cast<MemRefType>(accs[0].memref.getType()).getElementType();
  auto vecTy = VectorType::get({(int64_t)VL}, elemTy);
  Location loc = red.getLoc();

  rewriter.setInsertionPoint(red);
  IRMapping hoistMap;
  SmallVector<SmallVector<Value>> hoistedOps(accs.size());
  for (auto [i, a] : llvm::enumerate(accs)) {
    SmallVector<Value> ops;
    for (Value o : a.operands)
      ops.push_back(hoistOperand(o, red, rewriter, hoistMap));
    hoistedOps[i] = ops;
  }
  // Partial sums start at zero (the original C is added back after the
  // horizontal reduction); save the original C scalars first.
  Value zeroElem =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elemTy));
  Value zeroVec = rewriter.create<vector::BroadcastOp>(loc, vecTy, zeroElem);
  SmallVector<Value> cOrig, initVals;
  for (auto [i, a] : llvm::enumerate(accs)) {
    cOrig.push_back(
        rewriter.create<AffineLoadOp>(a.loc, a.memref, a.map, hoistedOps[i]));
    initVals.push_back(zeroVec);
  }

  Block *oldRedBody = red.getBody();
  auto newK = rewriter.create<AffineForOp>(
      loc, red.getLowerBoundOperands(), red.getLowerBoundMap(),
      red.getUpperBoundOperands(), red.getUpperBoundMap(), (int64_t)VL, initVals,
      [&](OpBuilder &b, Location bloc, Value iv, ValueRange args) {
        IRMapping remap = hoistMap;
        remap.map(kIV, iv); // old k -> new (vl-stepped) k; vector load at [.,k]
        DenseMap<Value, Value> accToIter;
        for (auto [i, a] : llvm::enumerate(accs))
          accToIter[a.load.getResult()] = args[i];
        SmallVector<Value> yields;
        for (Acc &a : accs)
          yields.push_back(vectorizeReductionValue(a.storedVal, accToIter, remap,
                                                   kIV, vecTy, b, oldRedBody));
        b.create<affine::AffineYieldOp>(bloc, yields);
      });

  rewriter.setInsertionPointAfter(newK);
  for (auto [i, a] : llvm::enumerate(accs)) {
    Value hsum = rewriter.create<vector::ReductionOp>(
        a.loc, vector::CombiningKind::ADD, newK.getResult(i), /*acc=*/Value(),
        arith::FastMathFlags::fast);
    Value cfin = rewriter.create<arith::AddFOp>(a.loc, cOrig[i], hsum);
    rewriter.create<AffineStoreOp>(a.loc, cfin, a.memref, a.map, hoistedOps[i]);
  }
  rewriter.eraseOp(red);
  return success();
}


} // namespace rb
} // namespace drcompiler
