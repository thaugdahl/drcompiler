//===- AffineRegisterBlock.cpp - Register-block affine GEMM ---------------===//
//
// Brings DPS-style register-blocking to the memref/affine path.  A perfectly
// nested affine band of the canonical GEMM shape
//
//     affine.for %i { affine.for %j { affine.for %k {
//       C[i,j] = C[i,j] + A[i,k] * B[k,j]   // memory reduction over k
//     }}}
//
// is rewritten by (1) unroll-and-jamming the two outer loops by mr x nr, then
// (2) promoting each of the mr*nr accumulators from a load/store pair into an
// affine.for iter_args SSA value carried across the reduction loop.  The
// resulting register block is register-allocated + SLP-vectorized by LLVM.
//
// The (mr, nr) tile is a fixed knob, not a cost-model decision: the
// cost-model validation spike (COSTMODEL_SPIKE_FINDINGS.md) showed a fixed
// 8x16 is within 5% of the per-arch optimum across register files and vector
// widths.
//
//===----------------------------------------------------------------------===//

#include "drcompiler/Transforms/AffineRegisterBlock.h"
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
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
#define GEN_PASS_DEF_AFFINEREGISTERBLOCKPASS
#include "drcompiler/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using affine::AffineForOp;
using affine::AffineLoadOp;
using affine::AffineStoreOp;

namespace {

/// If `forOp`'s body is exactly one nested affine.for (plus the terminator),
/// return it; otherwise return null.  This identifies a perfect nest level.
static AffineForOp onlyChildFor(AffineForOp forOp) {
  Block *body = forOp.getBody();
  AffineForOp child;
  for (Operation &op : body->without_terminator()) {
    if (auto f = dyn_cast<AffineForOp>(&op)) {
      if (child) // more than one loop -> not a perfect single-child nest
        return nullptr;
      child = f;
    } else if (isa<AffineForOp>(op)) {
      return nullptr;
    } else {
      // Any non-loop op at this level means the nested loop is not the sole
      // occupant -> not a perfect nest for our purposes.
      return nullptr;
    }
  }
  return child;
}

/// One promotable accumulator: a load/store pair to the same memref location
/// whose address is invariant in the reduction loop, where the stored value
/// is (transitively) computed from the loaded value.
struct Acc {
  AffineLoadOp load;
  AffineStoreOp store;
  Value memref;
  AffineMap map;
  SmallVector<Value> operands;
  Value storedVal;
  Location loc;
};

/// Produce a value equivalent to `v` that is usable *before* `loop`.  Index
/// operands of an accumulator access (e.g. the `affine.apply #map(%i)` that
/// unroll-and-jam inserts inside the loop body) are pure functions of the
/// outer induction variables; clone that computation in front of the loop so
/// the hoisted init-load and the sunk final-store can reference it.
static Value hoistOperand(Value v, AffineForOp loop, IRRewriter &rewriter,
                          IRMapping &map) {
  if (Value m = map.lookupOrNull(v))
    return m;
  Operation *def = v.getDefiningOp();
  if (!def || !loop->isAncestor(def))
    return v; // defined outside the loop -> already dominates
  for (Value o : def->getOperands())
    hoistOperand(o, loop, rewriter, map);
  rewriter.clone(*def, map);
  return map.lookup(v);
}

/// True if `def` is reachable backwards from `root` through op operands,
/// staying within `loopBody` block.
static bool dependsOn(Value root, Value def, Block *loopBody) {
  SmallVector<Value> work{root};
  llvm::SmallPtrSet<Value, 16> seen;
  while (!work.empty()) {
    Value v = work.pop_back_val();
    if (v == def)
      return true;
    if (!seen.insert(v).second)
      continue;
    Operation *op = v.getDefiningOp();
    if (!op || op->getBlock() != loopBody)
      continue;
    for (Value o : op->getOperands())
      work.push_back(o);
  }
  return false;
}

/// Same affine access (map + operand list) on the same memref?
static bool sameAccess(Value memA, AffineMap mapA, ValueRange opsA, Value memB,
                       AffineMap mapB, ValueRange opsB) {
  if (memA != memB || mapA != mapB || opsA.size() != opsB.size())
    return false;
  for (auto [a, b] : llvm::zip(opsA, opsB))
    if (a != b)
      return false;
  return true;
}

/// True if `body` reads the accumulator's memref at an address that is NOT
/// itself stored back in `body` -- i.e. a read-only cross-element access to the
/// accumulator array, so the "reduction" is not independent and reblocking is
/// illegal.  Catches in-place factorizations like LU's `A[i][j] -=
/// A[i][k]*A[k][j]`: `A[i][k]`/`A[k][j]` read the `A` array but are never stored
/// here, so they overlap the `A[i][j]` accumulator across iterations.
///
/// Crucially, a load from the accumulator memref that DOES have a matching store
/// (same address) is a *sibling accumulator*, not an alias -- this is exactly
/// what unroll-and-jam produces (C[i][j], C[i+1][j], ... all in memref C), and
/// must be allowed.  A matmul's A/B are distinct memrefs from C, so they are
/// never even considered; the standard BLAS no-alias assumption stands for
/// distinct memref SSA values.
static bool accumulatorAliasesInput(Block *body, Value accMemref) {
  SmallVector<AffineStoreOp> stores;
  for (Operation &op : body->without_terminator())
    if (auto st = dyn_cast<AffineStoreOp>(&op))
      if (st.getMemRef() == accMemref)
        stores.push_back(st);
  for (Operation &op : body->without_terminator()) {
    auto ld = dyn_cast<AffineLoadOp>(&op);
    if (!ld || ld.getMemRef() != accMemref)
      continue;
    SmallVector<Value> ldOps(ld.getMapOperands().begin(),
                             ld.getMapOperands().end());
    bool stored = false;
    for (AffineStoreOp st : stores) {
      SmallVector<Value> stOps(st.getMapOperands().begin(),
                               st.getMapOperands().end());
      if (sameAccess(ld.getMemRef(), ld.getAffineMap(), ldOps, st.getMemRef(),
                     st.getAffineMap(), stOps)) {
        stored = true;
        break;
      }
    }
    if (!stored)
      return true; // read of the accumulator array with no matching store
  }
  return false;
}

/// Collect accumulator load/store pairs in the innermost loop `kLoop`.
static SmallVector<Acc> collectAccumulators(AffineForOp kLoop) {
  Value kIV = kLoop.getInductionVar();
  Block *body = kLoop.getBody();
  SmallVector<Acc> accs;

  for (Operation &op : body->without_terminator()) {
    auto store = dyn_cast<AffineStoreOp>(&op);
    if (!store)
      continue;
    // Address must be invariant in k: the reduction IV must not appear in the
    // store's index operands.
    SmallVector<Value> stOps(store.getMapOperands().begin(),
                             store.getMapOperands().end());
    if (llvm::is_contained(stOps, kIV))
      continue;
    // Find a matching load (same memref + access) in the loop body.
    AffineLoadOp matchLoad;
    for (Operation &op2 : body->without_terminator()) {
      auto load = dyn_cast<AffineLoadOp>(&op2);
      if (!load)
        continue;
      SmallVector<Value> ldOps(load.getMapOperands().begin(),
                               load.getMapOperands().end());
      if (sameAccess(load.getMemRef(), load.getAffineMap(), ldOps,
                     store.getMemRef(), store.getAffineMap(), stOps)) {
        matchLoad = load;
        break;
      }
    }
    if (!matchLoad)
      continue;
    // The stored value must be computed from the loaded value (a reduction).
    if (!dependsOn(store.getValueToStore(), matchLoad.getResult(), body))
      continue;
    // The accumulator must not alias a multiplicand (else the reduction is not
    // independent and reblocking is illegal -- e.g. LU).
    if (accumulatorAliasesInput(body, store.getMemRef()))
      continue;
    accs.push_back({matchLoad, store, store.getMemRef(), store.getAffineMap(),
                    stOps, store.getValueToStore(), store.getLoc()});
  }
  return accs;
}

/// Promote every accumulator in `kLoop` to an iter_args SSA value.  Returns
/// success if at least one accumulator was promoted.
static LogicalResult promoteReductions(AffineForOp kLoop, IRRewriter &rewriter) {
  SmallVector<Acc> accs = collectAccumulators(kLoop);
  if (accs.empty())
    return failure();

  // Hoist the initial accumulator loads before the loop -> iter_args inits.
  // Index operands defined inside the loop (affine.apply offsets from
  // unroll-and-jam) are cloned in front of the loop first.
  rewriter.setInsertionPoint(kLoop);
  IRMapping hoistMap;
  SmallVector<SmallVector<Value>> hoistedOperands(accs.size());
  SmallVector<Value> initVals;
  for (auto [i, a] : llvm::enumerate(accs)) {
    SmallVector<Value> ops;
    for (Value o : a.operands)
      ops.push_back(hoistOperand(o, kLoop, rewriter, hoistMap));
    hoistedOperands[i] = ops;
    auto init = rewriter.create<AffineLoadOp>(a.loc, a.memref, a.map, ops);
    initVals.push_back(init.getResult());
  }

  // Build the new yields: each new iter_arg replaces the in-loop load, and the
  // stored value becomes the yielded next-iteration accumulator.
  auto yieldFn = [&](OpBuilder &, Location,
                     ArrayRef<BlockArgument> newArgs) -> SmallVector<Value> {
    SmallVector<Value> yields;
    for (auto [i, a] : llvm::enumerate(accs)) {
      a.load.getResult().replaceAllUsesWith(newArgs[i]);
      yields.push_back(a.storedVal);
    }
    // The in-loop load is now dead; the in-loop store is superseded by the
    // post-loop store.  Erase both.
    for (Acc &a : accs)
      rewriter.eraseOp(a.store);
    for (Acc &a : accs)
      rewriter.eraseOp(a.load);
    return yields;
  };

  FailureOr<LoopLikeOpInterface> res = kLoop.replaceWithAdditionalYields(
      rewriter, initVals, /*replaceInitOperandUsesInLoop=*/false, yieldFn);
  if (failed(res))
    return failure();
  auto newLoop = cast<AffineForOp>(res->getOperation());

  // Sink the final stores after the loop, reusing the hoisted index operands.
  rewriter.setInsertionPointAfter(newLoop);
  for (auto [i, a] : llvm::enumerate(accs))
    rewriter.create<AffineStoreOp>(a.loc, newLoop.getResult(i), a.memref, a.map,
                                   hoistedOperands[i]);
  return success();
}

/// True if `iv` is in the backward def chain of `v` (crossing regions).
static bool valueDependsOnIV(Value v, Value iv) {
  SmallVector<Value> work{v};
  llvm::SmallPtrSet<Value, 16> seen;
  while (!work.empty()) {
    Value x = work.pop_back_val();
    if (x == iv)
      return true;
    if (!seen.insert(x).second)
      continue;
    if (Operation *d = x.getDefiningOp())
      for (Value o : d->getOperands())
        work.push_back(o);
  }
  return false;
}

/// Does the store's address depend on induction variable `iv`?
static bool addrDependsOnIV(AffineStoreOp store, Value iv) {
  for (Value o : store.getMapOperands())
    if (valueDependsOnIV(o, iv))
      return true;
  return false;
}

/// Find the first accumulator load/store pair directly in `loop`'s body (a
/// store with a matching same-address load whose value the store depends on).
/// No invariance filter -- used for canonicalization where the reduction may
/// be carried by an enclosing loop.
static bool findAccPair(AffineForOp loop, AffineStoreOp &outStore,
                        AffineLoadOp &outLoad) {
  Block *body = loop.getBody();
  for (Operation &op : body->without_terminator()) {
    auto store = dyn_cast<AffineStoreOp>(&op);
    if (!store)
      continue;
    SmallVector<Value> stOps(store.getMapOperands().begin(),
                             store.getMapOperands().end());
    AffineLoadOp matchLoad;
    for (Operation &op2 : body->without_terminator()) {
      auto load = dyn_cast<AffineLoadOp>(&op2);
      if (!load)
        continue;
      SmallVector<Value> ldOps(load.getMapOperands().begin(),
                               load.getMapOperands().end());
      if (sameAccess(load.getMemRef(), load.getAffineMap(), ldOps,
                     store.getMemRef(), store.getAffineMap(), stOps)) {
        matchLoad = load;
        break;
      }
    }
    if (!matchLoad)
      continue;
    if (!dependsOn(store.getValueToStore(), matchLoad.getResult(), body))
      continue;
    if (accumulatorAliasesInput(body, store.getMemRef()))
      continue; // aliasing reduction (e.g. LU) -> not register-blockable
    outStore = store;
    outLoad = matchLoad;
    return true;
  }
  return false;
}

/// Is `loop` innermost (no directly-nested affine.for)?
static bool isInnermost(AffineForOp loop) {
  for (Operation &o : *loop.getBody())
    if (isa<AffineForOp>(o))
      return false;
  return true;
}

/// The innermost reduction loop under `root`: an innermost loop with a
/// k-invariant accumulator (collectAccumulators non-empty).
static AffineForOp findReductionLoopUnder(Operation *root) {
  AffineForOp found;
  root->walk([&](AffineForOp loop) {
    if (found || !isInnermost(loop))
      return;
    if (!collectAccumulators(loop).empty())
      found = loop;
  });
  return found;
}

/// Nearest enclosing affine.for of `from` whose IV indexes the accumulator.
static AffineForOp enclosingSpatial(AffineForOp from, AffineStoreOp store) {
  AffineForOp p = from->getParentOfType<AffineForOp>();
  while (p) {
    if (addrDependsOnIV(store, p.getInductionVar()))
      return p;
    p = p->getParentOfType<AffineForOp>();
  }
  return nullptr;
}

/// Distribute (fission) an imperfectly-nested loop whose body is a sequence of
/// affine.for loops into one loop per child, preserving order.  Used to peel a
/// matmul nest away from a sibling (e.g. a PolyBench `C *= beta` loop) so the
/// matmul becomes a perfect band that `tilePerfectlyNested` can cache-block.
/// Legal here because the children's only shared memref is C and the original
/// per-iteration order (all of child0 before child1 for a given i) is preserved
/// row-by-row -> distributing to all-child0 then all-child1 keeps every C
/// dependence.  Returns the new loops, or {} if the body isn't a clean sequence
/// of loops (a stray op would need replication) or bounds aren't constant.
static SmallVector<AffineForOp> distributeLoop(AffineForOp iLoop,
                                               IRRewriter &rewriter) {
  SmallVector<AffineForOp> children;
  for (Operation &op : iLoop.getBody()->without_terminator()) {
    if (auto f = dyn_cast<AffineForOp>(&op))
      children.push_back(f);
    else if (!isMemoryEffectFree(&op))
      return {}; // side-effecting stray op -> unsafe to replicate/reorder
  }
  if (children.size() < 2)
    return {};
  if (!iLoop.hasConstantLowerBound() || !iLoop.hasConstantUpperBound())
    return {};
  int64_t lb = iLoop.getConstantLowerBound();
  int64_t ub = iLoop.getConstantUpperBound();
  int64_t step = iLoop.getStepAsInt();

  // True if `user` lies inside `child`'s region (walk the parent chain).
  auto insideChild = [](Operation *user, AffineForOp child) {
    for (Operation *p = user; p; p = p->getParentOp())
      if (p == child.getOperation())
        return true;
    return false;
  };

  SmallVector<AffineForOp> newLoops;
  for (AffineForOp child : children) {
    // Which pure stray ops (e.g. a hoisted `arith.index_cast %i`) does this
    // child transitively depend on?  Iterate body ops in reverse so a stray op
    // feeding another needed stray op is itself marked needed.
    llvm::SmallPtrSet<Operation *, 8> needed;
    Block *body = iLoop.getBody();
    for (Operation &opR :
         llvm::reverse(llvm::make_range(body->begin(), std::prev(body->end())))) {
      Operation *o = &opR;
      if (isa<AffineForOp>(o))
        continue;
      bool used = false;
      for (Value r : o->getResults()) {
        for (Operation *user : r.getUsers())
          if (insideChild(user, child) || needed.count(user)) {
            used = true;
            break;
          }
        if (used)
          break;
      }
      if (used)
        needed.insert(o);
    }
    // Fresh sibling before the original (preserving child order); reset the
    // insertion point every iteration so loop N+1 is NOT nested in loop N.
    rewriter.setInsertionPoint(iLoop);
    auto ni = rewriter.create<AffineForOp>(iLoop.getLoc(), lb, ub, step);
    IRMapping map;
    map.map(iLoop.getInductionVar(), ni.getInductionVar());
    rewriter.setInsertionPointToStart(ni.getBody());
    // Clone, in original body order, the needed stray ops then this child.
    for (Operation &op : body->without_terminator()) {
      if (&op == child.getOperation() || needed.count(&op))
        rewriter.clone(op, map);
    }
    newLoops.push_back(ni);
  }
  rewriter.eraseOp(iLoop);
  return newLoops;
}

/// One canonicalization step: if an innermost loop holds an accumulator whose
/// address varies in the innermost loop but is invariant in an enclosing loop
/// (the reduction loop, e.g. the `k` of a PolyBench i-k-j GEMM), interchange
/// so the reduction loop becomes innermost.  Returns true if it changed the IR.
static bool canonicalizeOnce(func::FuncOp func) {
  SmallVector<AffineForOp> inners;
  func.walk([&](AffineForOp loop) {
    if (isInnermost(loop))
      inners.push_back(loop);
  });
  for (AffineForOp inner : inners) {
    AffineStoreOp store;
    AffineLoadOp load;
    if (!findAccPair(inner, store, load))
      continue;
    if (!addrDependsOnIV(store, inner.getInductionVar()))
      continue; // already canonical: reduction is the innermost loop
    // Walk up to the reduction loop (IV not in the accumulator address).
    AffineForOp red = inner->getParentOfType<AffineForOp>();
    while (red && addrDependsOnIV(store, red.getInductionVar()))
      red = red->getParentOfType<AffineForOp>();
    if (!red || onlyChildFor(red) != inner)
      continue; // need a single spatial loop between red and inner
    // Only interchange a reduction we can actually register-block afterwards.
    // If the reduction trip count depends on an outer IV (a triangular
    // reduction, e.g. trmm's `k = i..N`), unroll-jamming the spatial loops would
    // give the mr rows different reduction ranges -- we have no reduction-peel
    // for that.  Interchanging without blocking would just leave a cache-hostile
    // order, so leave such nests untouched (clang vectorizes the original).
    if (!red.hasConstantLowerBound() || !red.hasConstantUpperBound())
      continue;
    SmallVector<AffineForOp, 2> band{red, inner};
    if (!affine::isValidLoopInterchangePermutation(band, {1, 0}))
      continue;
    affine::interchangeLoops(red, inner);
    return true;
  }
  return false;
}

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
static bool peelTriangularNest(AffineForOp sOut, unsigned mr,
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
  if (mr == 0 || (hi - lo) % (int64_t)mr != 0)
    return false;
  Value iv = sOut.getInductionVar();
  // Triangular iff the inner spatial upper bound depends on i but its lower
  // bound does not (the common `0..i+c` rank-k shape).
  if (!llvm::is_contained(sIn.getUpperBoundOperands(), iv))
    return false;
  if (llvm::is_contained(sIn.getLowerBoundOperands(), iv))
    return false;

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
  auto buildHalf = [&](Value ii, bool diag) -> AffineForOp {
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
      jL = rewriter.create<AffineForOp>(loc, ValueRange(sInLbOps), sInLbMap,
                                        ValueRange{ii}, idMap, 1); // 0 .. ii
      rewriter.setInsertionPointToStart(jL.getBody());
      ni = emitApply(); // row offset computed inside the (perfect) j-loop
    } else {
      ni = emitApply(); // offset precedes j (feeds the ragged bound)
      SmallVector<Value> jUbOps(sInUbOps);
      for (Value &v : jUbOps)
        if (v == iv)
          v = ni;
      jL = rewriter.create<AffineForOp>(loc, ValueRange{ii}, idMap,
                                        ValueRange(jUbOps), sInUbMap,
                                        1); // ii .. f(ni)
      rewriter.setInsertionPointToStart(jL.getBody());
    }
    IRMapping m;
    m.map(iv, ni);
    m.map(sIn.getInductionVar(), jL.getInductionVar());
    rewriter.clone(*red.getOperation(), m);
    return iL;
  };

  rewriter.setInsertionPoint(sOut);
  auto strip = rewriter.create<AffineForOp>(loc, lo, hi, (int64_t)mr);
  rewriter.setInsertionPointToStart(strip.getBody());
  Value ii = strip.getInductionVar();
  AffineForOp headI = buildHalf(ii, /*diag=*/false);
  rewriter.setInsertionPointAfter(headI);
  buildHalf(ii, /*diag=*/true);
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
static bool peelTriangularReduction(AffineForOp sOut, unsigned mr,
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
  if ((hi - lo) % (int64_t)mr != 0)
    return false;
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
  auto strip = rewriter.create<AffineForOp>(loc, lo, hi, (int64_t)mr);
  rewriter.setInsertionPointToStart(strip.getBody());
  Value ii = strip.getInductionVar();
  AffineForOp mainI = buildHalf(ii, /*corner=*/false);
  rewriter.setInsertionPointAfter(mainI);
  buildHalf(ii, /*corner=*/true);
  rewriter.eraseOp(sOut);
  return true;
}

/// Register-block family, selected from operand layout (OPERAND_PACKING_FINDINGS.md).
enum class RBFamily { Broadcast, Dot };

/// True if `load`'s innermost (fastest-varying) memref dimension is indexed
/// exactly by induction variable `iv` with coefficient 1 -- i.e. consecutive
/// `iv` values give stride-1 (contiguous) addresses.
static bool innermostStrideOne(AffineLoadOp load, Value iv) {
  AffineMap m = load.getAffineMap();
  if (m.getNumResults() == 0)
    return false;
  auto dim = dyn_cast<AffineDimExpr>(m.getResult(m.getNumResults() - 1));
  if (!dim)
    return false;
  unsigned pos = dim.getPosition();
  auto operands = load.getMapOperands();
  return pos < operands.size() && operands[pos] == iv;
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
static RBFamily detectFamily(AffineForOp red, AffineForOp sIn, int &nMul) {
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
static LogicalResult vectorizeBroadcastBand(AffineForOp red, AffineForOp sIn,
                                            unsigned VL, IRRewriter &rewriter) {
  if (VL < 2)
    return failure();
  Value jIV = sIn.getInductionVar();
  if (sIn.getStepAsInt() != 1)
    return failure();
  // The inner spatial extent must be a constant multiple of VL.  This holds for
  // an untiled loop (trip = N) and for a cache-tiled point loop whose extent is
  // exactly the tile size (`tc .. tc+tile`, symbolic bound but constant trip) --
  // so the vector micro-kernel composes with cache tiling.  A *partial* last
  // tile (`tc .. min(tc+tile, N)`) has a non-constant trip -> we bail here and
  // fall back to scalar+SLP, which is correct (no out-of-bounds vector access).
  std::optional<uint64_t> trip = affine::getConstantTripCount(sIn);
  if (!trip)
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
  if (*trip % VL != 0) {
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
static LogicalResult vectorizeDotBand(AffineForOp red, unsigned VL,
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

class AffineRegisterBlockPass final
    : public impl::AffineRegisterBlockPassBase<AffineRegisterBlockPass> {
public:
  using AffineRegisterBlockPassBase<
      AffineRegisterBlockPass>::AffineRegisterBlockPassBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    IRRewriter rewriter(&getContext());

    // Stage 1: canonicalize reduction nests so the reduction loop is innermost
    // (handles the PolyBench i-k-j order via k<->j interchange).
    while (canonicalizeOnce(func))
      ;

    // Stage 1a: family selection.  The transform is identical for both BLAS-3
    // families, but the operand layout dictates which LLVM vectorization
    // strategy wins -- and therefore the tile shape and whether the FP
    // reduction may be reassociated.  See OPERAND_PACKING_FINDINGS.md:
    //   broadcast (C=A*B): wide tile (mr x nr), no reassoc -> SLP over j-lanes.
    //   dot (rank-k):      small square tile, reassoc       -> reduction over k.
    // A fixed wide tile + reassoc would spill the dot grid AND flip gemm to the
    // wrong (horizontal-sum) strategy; hence per-layout selection.
    unsigned mrEff = mr, nrEff = nr;
    bool reassoc = false;
    if (familySelect) {
      bool anyDot = false;
      int maxMul = 0;
      func.walk([&](AffineForOp red) {
        if (!isInnermost(red) || collectAccumulators(red).empty())
          return;
        AffineForOp sIn = red->getParentOfType<AffineForOp>();
        if (!sIn)
          return;
        int nMul = 0;
        if (detectFamily(red, sIn, nMul) == RBFamily::Dot) {
          anyDot = true;
          maxMul = std::max(maxMul, nMul);
        }
      });
      if (anyDot) {
        reassoc = true;
        // Shrink the square tile as the input-stream count rises: syrk (2
        // multiplicands) fits 4x4 (16 acc + 8 loads <= 32 zmm); syr2k (4
        // multiplicands) needs 2x2.
        mrEff = nrEff = (maxMul > 2) ? 2u : 4u;
      }
    }

    // Stage 1b: optional cache blocking.  Register blocking alone is DRAM-bound
    // once the matrices exceed the last-level cache (the full B is re-streamed
    // per i-block).  Tile each perfectly-nested GEMM band by mc x nc x kc so
    // the working set stays resident; the register-block micro-kernel then runs
    // on the cache-resident point loops.
    if (cacheTile) {
      // Distribute any imperfect matmul outer loop (e.g. PolyBench's i-loop
      // carrying a beta-scaling sibling) so the matmul becomes a perfect band.
      bool changed = true;
      while (changed) {
        changed = false;
        SmallVector<AffineForOp> reds;
        func.walk([&](AffineForOp r) {
          if (isInnermost(r) && !collectAccumulators(r).empty())
            reds.push_back(r);
        });
        for (AffineForOp r : reds) {
          AffineStoreOp st;
          AffineLoadOp ld;
          if (!findAccPair(r, st, ld))
            continue;
          AffineForOp sIn = r->getParentOfType<AffineForOp>();
          AffineForOp sOut = sIn ? enclosingSpatial(sIn, st) : AffineForOp();
          if (!sOut)
            continue;
          // Already a perfect i-j-k band?
          if (onlyChildFor(sOut) == sIn && onlyChildFor(sIn) == r)
            continue;
          if (!distributeLoop(sOut, rewriter).empty()) {
            changed = true;
            break;
          }
        }
      }

      // Diagonal-peel triangular reduction bands (e.g. syrk/syr2k `j:0..i+1`)
      // so their rectangular bulk becomes register-blockable.
      bool peeled = true;
      while (peeled) {
        peeled = false;
        SmallVector<AffineForOp> cands;
        func.walk([&](AffineForOp s) {
          AffineForOp sIn = onlyChildFor(s);
          if (!sIn)
            return;
          AffineForOp r = onlyChildFor(sIn);
          if (r && isInnermost(r))
            cands.push_back(s);
        });
        for (AffineForOp s : cands)
          if (peelTriangularNest(s, mrEff, rewriter)) {
            peeled = true;
            break;
          }
      }

      // Diagonal-peel triangular *reduction* bands (e.g. trmm's `k = i..N`,
      // reduction lower bound depends on the outer spatial IV).  First distribute
      // any sibling (e.g. trmm's `C[i][j] = 0` init loop) so the band is perfect,
      // then split into a register-blockable MAIN + scalar CORNER.
      bool rpeeled = true;
      while (rpeeled) {
        rpeeled = false;
        SmallVector<AffineForOp> cands;
        func.walk([&](AffineForOp s) {
          for (Operation &op : s.getBody()->without_terminator()) {
            auto k = dyn_cast<AffineForOp>(&op);
            if (!k)
              continue;
            AffineForOp j = onlyChildFor(k);
            if (!j || !isInnermost(j))
              continue;
            AffineStoreOp st;
            AffineLoadOp ld;
            if (!findAccPair(j, st, ld))
              continue;
            if (addrDependsOnIV(st, k.getInductionVar()) ||
                !addrDependsOnIV(st, j.getInductionVar()))
              continue; // k must be the reduction, j the spatial dim
            if (!llvm::is_contained(k.getLowerBoundOperands(),
                                    s.getInductionVar()))
              continue; // reduction lower bound must depend on the outer IV
            cands.push_back(s);
            break;
          }
        });
        for (AffineForOp s : cands) {
          if (!onlyChildFor(s)) { // a sibling (init loop) is present -> fission
            if (!distributeLoop(s, rewriter).empty()) {
              rpeeled = true;
              break;
            }
            continue;
          }
          if (peelTriangularReduction(s, mrEff, rewriter)) {
            rpeeled = true;
            break;
          }
        }
      }

      // Tile each perfect GEMM band by mc x nc x kc.  Only fully-rectangular
      // (constant-bound) bands are tiled; a peeled triangular head has a
      // parametric j-bound and is register-blocked untiled.
      SmallVector<SmallVector<AffineForOp, 3>> bands;
      func.walk([&](AffineForOp iLoop) {
        AffineForOp jLoop = onlyChildFor(iLoop);
        if (!jLoop)
          return;
        AffineForOp kLoop = onlyChildFor(jLoop);
        if (!kLoop || !isInnermost(kLoop) ||
            collectAccumulators(kLoop).empty())
          return;
        if (!iLoop.hasConstantUpperBound() || !jLoop.hasConstantUpperBound() ||
            !kLoop.hasConstantUpperBound())
          return;
        bands.push_back({iLoop, jLoop, kLoop});
      });
      for (auto &band : bands) {
        SmallVector<AffineForOp, 3> in(band.begin(), band.end());
        // Skip a degenerate tiling: if every loop already fits its tile, tiling
        // emits `step >extent` loops with min/max (#map) point bounds that the
        // LLVM vectorizer can't analyze -> the register-block micro-kernel goes
        // scalar (observed: small-N tensors collapse to ~0.1x).  Only tile when
        // at least one dim genuinely exceeds its tile.
        bool worthTiling =
            in[0].getConstantUpperBound() - in[0].getConstantLowerBound() >
                (int64_t)mc ||
            in[1].getConstantUpperBound() - in[1].getConstantLowerBound() >
                (int64_t)nc ||
            in[2].getConstantUpperBound() - in[2].getConstantLowerBound() >
                (int64_t)kc;
        if (!worthTiling)
          continue;
        SmallVector<unsigned, 3> sizes{mc, nc, kc};
        (void)affine::tilePerfectlyNested(in, sizes);
      }
    }

    // Stage 2: collect the distinct outer spatial loops of each reduction.
    SmallVector<AffineForOp> sOuts;
    func.walk([&](AffineForOp red) {
      if (!isInnermost(red) || collectAccumulators(red).empty())
        return;
      AffineStoreOp store;
      AffineLoadOp load;
      if (!findAccPair(red, store, load))
        return;
      AffineForOp sIn = red->getParentOfType<AffineForOp>();
      if (!sIn || !addrDependsOnIV(store, sIn.getInductionVar()))
        return;
      AffineForOp sOut = enclosingSpatial(sIn, store);
      if (!sOut)
        return;
      // The two loops we unroll-and-jam must be parallel: jamming a loop that
      // carries a dependence reorders dependent iterations and is illegal.
      // gemm/syrk spatial loops are parallel; a factorization's carried sweep
      // (e.g. gramschmidt's outer k, which updates A in place) is not -> skip,
      // leaving it untouched.  (The alias guard already rejects LU earlier; this
      // is the general safety net.)
      if (!affine::isLoopParallel(sOut) || !affine::isLoopParallel(sIn))
        return;
      if (!llvm::is_contained(sOuts, sOut))
        sOuts.push_back(sOut);
    });

    // Stage 3: per reduction, unroll-and-jam the two spatial loops by mr x nr
    // (the outer one may be imperfectly nested, e.g. a PolyBench beta-scaling
    // sibling loop) and promote the mr*nr accumulators to iter_args.
    //
    // After unroll-jam, `sOut` may be a dangling handle: when its trip count
    // equals mr (a peeled triangular head), loopUnrollJamByFactor promotes it
    // to a single iteration and erases it.  So the reduction is re-found by
    // walking the stable `func` -- findReductionLoopUnder skips already-promoted
    // reductions (their accumulator load/store are gone), so it keeps advancing
    // to the next unprocessed one across multi-matmul kernels.
    for (AffineForOp sOut : sOuts) {
      // Skip a diagonal remainder: its inner spatial bound depends on sOut's IV
      // (the ragged tail), so unroll-and-jam cannot fuse the inner loops.
      if (AffineForOp sIn0 = onlyChildFor(sOut)) {
        bool tri = false;
        for (Value o : sIn0.getUpperBoundOperands())
          tri |= valueDependsOnIV(o, sOut.getInductionVar());
        if (tri)
          continue;
      }
      if (mrEff > 1 && failed(affine::loopUnrollJamByFactor(sOut, mrEff)))
        continue;
      AffineForOp red = findReductionLoopUnder(func);
      if (!red)
        continue;
      AffineForOp sIn = red->getParentOfType<AffineForOp>();
      if (!sIn)
        continue;
      // Broadcast family: emit an EXPLICIT vector micro-kernel along the inner
      // spatial loop (vector dialect), for all ranks -- no reliance on LLVM-SLP.
      // Tile width (vector columns): a >=3D accumulator (tensor contraction) is
      // best as mr mr-only vectors (measured: a wider tile over-subscribes and
      // regresses bmm/ttm), so nrVec=1; a 2D accumulator uses the nr-wide tile.
      SmallVector<Acc> accsForRank = collectAccumulators(red);
      unsigned accRank =
          accsForRank.empty()
              ? 2u
              : cast<MemRefType>(accsForRank[0].memref.getType()).getRank();
      unsigned nrVec = accRank >= 3 ? 1u : (nrEff + vl - 1) / vl;
      if (vectorize && !reassoc &&
          succeeded(vectorizeBroadcastBand(red, sIn, vl, rewriter))) {
        // Complete the mr x nrVec register tile explicitly: unroll-jam the (now
        // vl-stepped) inner spatial loop into nrVec vector columns, so the tile
        // width matches the nr-wide tile WITHOUT betting on SLP to widen it.
        if (nrVec > 1)
          (void)affine::loopUnrollJamByFactor(sIn, nrVec);
        continue;
      }
      // Guarded fallback: the band was not explicitly vectorizable (non-constant
      // bounds, gather, unsupported DAG).  Fall back to scalar promotion + LLVM
      // SLP and record it -- SLP is never the silent default.
      LLVM_DEBUG(llvm::dbgs() << "affine-register-block: explicit vectorization "
                                 "declined; SLP fallback for band at "
                              << sIn.getLoc() << "\n");
      if (nrEff > 1 && failed(affine::loopUnrollJamByFactor(sIn, nrEff)))
        continue;
      red = findReductionLoopUnder(func);
      if (!red)
        continue;
      // Dot family: EXPLICIT reduction-vectorization over k (vector dialect, no
      // reliance on LLVM reduction-vec).  Falls back to scalar promotion +
      // reassoc + LLVM when the k-trip isn't vl-divisible or the band isn't
      // cleanly vectorizable along k.
      if (vectorize && reassoc &&
          succeeded(vectorizeDotBand(red, vl, rewriter)))
        continue;
      (void)promoteReductions(red, rewriter);
    }

    // Stage 4: set fast-math on the kernel's FP ops.  MLIR lowering emits
    // flagless FP ops and `clang -ffast-math` does NOT retroactively flag a .ll,
    // so without this the backend never forms FMAs (it emits separate mulpd +
    // addpd, ~half FP throughput) -- measured: vfmadd=0 on the broadcast kernel.
    //   - Broadcast family: `contract` only.  This lets the backend fuse
    //     mul+add into FMA *without* reassociating, so the accumulation order
    //     (and the SLP/vector strategy) is unchanged -- pure throughput win.
    //   - Dot family: `fast` (contract + reassoc).  The k-reduction can only be
    //     vectorized by LLVM with reassociation (the rank-k "loss" was this);
    //     reassoc would flip gemm to the wrong horizontal-sum strategy, hence it
    //     is reserved for the dot family.
    {
      auto flags = reassoc ? arith::FastMathFlags::fast
                           : arith::FastMathFlags::contract;
      auto fma = arith::FastMathFlagsAttr::get(&getContext(), flags);
      func.walk([&](Operation *op) {
        if (isa<arith::MulFOp, arith::AddFOp, arith::SubFOp, arith::DivFOp,
                arith::NegFOp>(op))
          op->setAttr("fastmath", fma);
      });
    }
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createAffineRegisterBlockPass() {
  return std::make_unique<AffineRegisterBlockPass>();
}
} // namespace mlir
