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

#include "drcompiler/Analysis/MachineModel.h"
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

#define DEBUG_TYPE "affine-register-block"

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

/// Unit attribute the in-place triangular peel sets on the MAIN reduction
/// loop it emits: the peel has PROVEN (from the loop bounds it constructed)
/// that every read of the accumulator's memref in that loop is row-disjoint
/// from the strip's accumulators, so accumulatorAliasesInput may be skipped.
/// Never set this by hand.
static const char kAccNoAliasAttr[] = "dr.acc_no_alias";

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
    // independent and reblocking is illegal -- e.g. LU).  A loop the in-place
    // peel certified (dr.acc_no_alias) has row-disjointness proven by
    // construction; skip the syntactic guard there.
    if (!kLoop->hasAttr(kAccNoAliasAttr) &&
        accumulatorAliasesInput(body, store.getMemRef()))
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
    if (!loop->hasAttr(kAccNoAliasAttr) &&
        accumulatorAliasesInput(body, store.getMemRef()))
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
    SmallVector<Acc> accs = collectAccumulators(loop);
    if (accs.empty())
      return;
    // Skip purely rank-0 (scalar) reductions.  They are never register-block
    // targets -- Stage 2 excludes them (addrDependsOnIV is false on a rank-0
    // store) -- and must not SHADOW a real >=1D reduction during the post-jam
    // re-find: symm emits a scatter band (rank-2 acc, branch B) next to a
    // temp2 scalar reduction (rank-0, branch A); walk order hits temp2 first,
    // so without this guard Stage 3 would re-find temp2 and leave the scatter
    // band unblocked (the rank-0 slice of the v3 Stage-3 cross-talk bug).
    if (llvm::all_of(accs, [](const Acc &a) {
          return cast<MemRefType>(a.memref.getType()).getRank() == 0;
        }))
      return;
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

/// WP4 (COSTMODEL_V4_SPEC §5): raise the PolyBench symm scatter into a
/// register-blockable triangular reduction.  Matches the j-body
///   for i { for j {
///     store 0 -> t (rank-0 acc)                  // temp2 = 0
///     for k = 0 .. i {                           // triangular ub = i
///       M[k][j] = M[k][j] + s(i,j,k)             // SCATTER (addr varies in k,
///       t       = t       + r(i,j,k)             //          invariant in i)
///     }                                          // t: rank-0 reduction (scalar)
///     <epilogue ops, incl. M[i][j] = f(t, M[i][j], ...)>
///   } }
/// splitting it IN PLACE into:
///   (A) the original nest with the scatter store + its exclusive feeders
///       removed -- temp2 and the epilogue stay, unchanged and scalar; and
///   (B) a fresh, already-i-innermost scatter nest inserted AFTER (A):
///         for k = 0..N { for j { for i = k+1..N { M[k][j] += s(i,j,k) } } }
///       which the existing in-place triangular register-blocker (Stage 1c) +
///       broadcast vectorizer then crush (verified: the hand-interchanged nest
///       register-blocks to an mr x vl vector micro-kernel).
/// Bit-identical: a scatter into row r accumulates contributions only from
/// i>r, which in the original outer-i order all run AFTER row r's epilogue (the
/// epilogue at i=r reads M[r][j] before any i>r scatter touches row r); so
/// emitting branch A (all epilogues) before branch B (all scatters) preserves
/// every M dependence.  temp2 reads only inputs (B,A), so fissioning it from the
/// scatter is trivially legal.
static bool raiseSymmScatter(func::FuncOp func, IRRewriter &rewriter) {
  // Match candidates without mutating during the walk.
  SmallVector<AffineForOp> iCands;
  func.walk([&](AffineForOp iLoop) {
    AffineForOp jLoop = onlyChildFor(iLoop);
    if (!jLoop)
      return; // bounds may be parametric (cgeist -O0): only need affine maps
    // A single inner k-loop, triangular (ub depends on i).
    AffineForOp kLoop;
    for (Operation &op : jLoop.getBody()->without_terminator())
      if (auto f = dyn_cast<AffineForOp>(&op)) {
        if (kLoop)
          return; // >1 inner loop -> not the symm shape
        kLoop = f;
      }
    if (!kLoop || !isInnermost(kLoop))
      return;
    bool tri = false;
    for (Value o : kLoop.getUpperBoundOperands())
      tri |= valueDependsOnIV(o, iLoop.getInductionVar());
    if (tri)
      iCands.push_back(iLoop);
  });

  bool changed = false;
  for (AffineForOp iLoop : iCands) {
    AffineForOp jLoop = onlyChildFor(iLoop);
    AffineForOp kLoop;
    for (Operation &op : jLoop.getBody()->without_terminator())
      if (auto f = dyn_cast<AffineForOp>(&op))
        kLoop = f;
    Value iIV = iLoop.getInductionVar(), jIV = jLoop.getInductionVar(),
          kIV = kLoop.getInductionVar();
    Block *kBody = kLoop.getBody();

    // Find THE scatter store (a >=2D store whose address varies in k but not in
    // i) with a matching same-address load it reduces into, AND require a
    // separate rank-0 reduction (temp2) so this fires only on the symm shape.
    AffineStoreOp scatter;
    bool hasRank0Reduction = false, twoScatters = false;
    for (Operation &op : kBody->without_terminator()) {
      auto st = dyn_cast<AffineStoreOp>(&op);
      if (!st)
        continue;
      auto mrTy = cast<MemRefType>(st.getMemRef().getType());
      if (mrTy.getRank() == 0) {
        hasRank0Reduction = true;
        continue;
      }
      if (!addrDependsOnIV(st, kIV) || !addrDependsOnIV(st, jIV) ||
          addrDependsOnIV(st, iIV))
        continue;
      SmallVector<Value> stOps(st.getMapOperands().begin(),
                               st.getMapOperands().end());
      AffineLoadOp matchLoad;
      for (Operation &op2 : kBody->without_terminator())
        if (auto ld = dyn_cast<AffineLoadOp>(&op2)) {
          SmallVector<Value> ldOps(ld.getMapOperands().begin(),
                                   ld.getMapOperands().end());
          if (sameAccess(ld.getMemRef(), ld.getAffineMap(), ldOps,
                         st.getMemRef(), st.getAffineMap(), stOps)) {
            matchLoad = ld;
            break;
          }
        }
      if (!matchLoad || !dependsOn(st.getValueToStore(), matchLoad.getResult(),
                                   kBody))
        continue;
      if (scatter)
        twoScatters = true;
      scatter = st;
    }
    if (!scatter || twoScatters || !hasRank0Reduction)
      continue;

    // --- Build branch B: for k2=lb_i..ub_i { for j2 { for i2=k2+1..ub_i {
    // scatter } } }.  Bounds may be parametric (cgeist -O0 keeps n/m symbolic):
    // copy the i- and j-loops' affine bound maps + operands verbatim.  i's
    // bound operands are defined before the i-loop, so they dominate the new
    // nest inserted right after it.
    MLIRContext *ctx = func.getContext();
    Location loc = scatter.getLoc();
    rewriter.setInsertionPointAfter(iLoop);
    auto kB = rewriter.create<AffineForOp>(
        loc, iLoop.getLowerBoundOperands(), iLoop.getLowerBoundMap(),
        iLoop.getUpperBoundOperands(), iLoop.getUpperBoundMap(),
        iLoop.getStepAsInt());
    rewriter.setInsertionPoint(kB.getBody()->getTerminator());
    auto jB = rewriter.create<AffineForOp>(
        loc, jLoop.getLowerBoundOperands(), jLoop.getLowerBoundMap(),
        jLoop.getUpperBoundOperands(), jLoop.getUpperBoundMap(),
        jLoop.getStepAsInt());
    rewriter.setInsertionPoint(jB.getBody()->getTerminator());
    // i2 lower bound = k2 + 1 (triangular); upper bound = i's upper bound.
    AffineMap lbMap = AffineMap::get(1, 0, getAffineDimExpr(0, ctx) + 1);
    auto iB = rewriter.create<AffineForOp>(
        loc, ValueRange{kB.getInductionVar()}, lbMap,
        iLoop.getUpperBoundOperands(), iLoop.getUpperBoundMap(),
        kLoop.getStepAsInt());

    // Backward cone of the scatter within the k-body (value + address feeders).
    llvm::SmallPtrSet<Operation *, 16> cone;
    SmallVector<Value> work{scatter.getValueToStore()};
    work.append(scatter.getMapOperands().begin(),
                scatter.getMapOperands().end());
    while (!work.empty()) {
      Value v = work.pop_back_val();
      Operation *d = v.getDefiningOp();
      if (!d || d->getBlock() != kBody || !cone.insert(d).second)
        continue;
      for (Value o : d->getOperands())
        work.push_back(o);
    }
    // Clone the cone (in body order) + the scatter store into i2's body, with
    // the original (i,j,k) IVs remapped to the new (i2,j2,k2).
    IRMapping map;
    map.map(iIV, iB.getInductionVar());
    map.map(jIV, jB.getInductionVar());
    map.map(kIV, kB.getInductionVar());
    rewriter.setInsertionPoint(iB.getBody()->getTerminator());
    for (Operation &op : kBody->without_terminator())
      if (cone.count(&op) || &op == scatter.getOperation())
        rewriter.clone(op, map);

    // --- Prune branch A: erase the scatter store, then its now-dead feeders.
    rewriter.eraseOp(scatter);
    bool erased = true;
    while (erased) {
      erased = false;
      for (Operation &op : llvm::make_early_inc_range(
               kBody->without_terminator()))
        if (cone.count(&op) && op.use_empty()) {
          rewriter.eraseOp(&op);
          erased = true;
        }
    }
    changed = true;
  }
  return changed;
}

/// WP5 (COSTMODEL_V4_SPEC §6 — the spec's block-interleave mechanism, replaced
/// after a spike showed it only buys 1.38x vs 8.4x for this): interchange a
/// BLAS-2 column-major streaming nest to row-major.  A perfect 2-loop nest
///   for outer { for inner: ... M[inner][outer] ... }
/// reads the 2-D array M with the INNER loop in the ROW position -> stride =
/// row length (column-major; 8 useful bytes per 64-byte line).  When the nest
/// sits under a SEQUENTIAL outer sweep it is NOT a register-blockable BLAS-3
/// tile (GEMM/syrk/covariance have a PARALLEL spatial sweep there and must keep
/// the reduction innermost for register reuse), so interchanging to
///   for inner { for outer: ... M[inner][outer] ... }
/// makes M stride-1 (row-major).  Measured: gramschmidt's dot + A-update go
/// column->row-major for an 8.4x inner-kernel speedup, BIT-identical (the
/// reduction's accumulation order over the now-outer loop is preserved).  LLVM
/// then vectorizes the stride-1 inner loop; the register-block stages leave
/// these nests alone (after interchange the innermost loop carries no
/// loop-invariant accumulator).
static bool interchangeBlas2RowMajor(func::FuncOp func) {
  SmallVector<std::pair<AffineForOp, AffineForOp>> work; // (outer, inner)
  func.walk([&](AffineForOp inner) {
    if (!isInnermost(inner))
      return;
    AffineForOp outer = inner->getParentOfType<AffineForOp>();
    if (!outer || onlyChildFor(outer) != inner)
      return; // need a perfect 2-loop nest
    AffineForOp sweep = outer->getParentOfType<AffineForOp>();
    if (!sweep || affine::isLoopParallel(sweep))
      return; // a parallel enclosing sweep => BLAS-3 tile; leave for reg-block
    // The inner (streamed) loop must be rectangular.  A TRIANGULAR inner loop
    // (bound depends on an outer IV, e.g. trmm/lu's `k = i+1..N`) is an in-place
    // triangular reduction the peel register-blocks -- interchanging it would
    // break that 16x path.  gramschmidt's streamed loop is a plain `0..M`.
    if (!inner.hasConstantLowerBound() || !inner.hasConstantUpperBound())
      return;
    Value iIV = inner.getInductionVar(), oIV = outer.getInductionVar();
    // Look for a 2-D access M[inner][outer]: inner IV in the row dim, outer IV
    // in the column dim -> the inner loop strides M by a full row.
    bool colMajor = false;
    auto check = [&](AffineMap m, ValueRange ops) {
      if (m.getNumResults() != 2)
        return;
      auto r0 = dyn_cast<AffineDimExpr>(m.getResult(0));
      auto r1 = dyn_cast<AffineDimExpr>(m.getResult(1));
      if (r0 && r1 && ops[r0.getPosition()] == iIV &&
          ops[r1.getPosition()] == oIV)
        colMajor = true;
    };
    inner.getBody()->walk([&](Operation *op) {
      if (auto ld = dyn_cast<AffineLoadOp>(op))
        check(ld.getAffineMap(), ld.getMapOperands());
      else if (auto st = dyn_cast<AffineStoreOp>(op))
        check(st.getAffineMap(), st.getMapOperands());
    });
    if (!colMajor)
      return;
    SmallVector<AffineForOp, 2> band{outer, inner};
    if (!affine::isValidLoopInterchangePermutation(band, {1, 0}))
      return;
    work.push_back({outer, inner});
  });
  for (auto [outer, inner] : work)
    affine::interchangeLoops(outer, inner);
  return !work.empty();
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
static bool innermostStrideOne(AffineLoadOp load, Value iv);

static bool peelTriangularNest(AffineForOp sOut, unsigned mr,
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
static bool peelInPlaceTriangularInnermost(AffineForOp sOut, unsigned mr,
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

class AffineRegisterBlockPass final
    : public impl::AffineRegisterBlockPassBase<AffineRegisterBlockPass> {
public:
  using AffineRegisterBlockPassBase<
      AffineRegisterBlockPass>::AffineRegisterBlockPassBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    IRRewriter rewriter(&getContext());

    // Cache hierarchy from the single source of truth (MachineModel,
    // COSTMODEL_V4_SPEC §2): JSON file unless a CLI option was set explicitly.
    {
      drcompiler::MachineModel mm =
          drcompiler::MachineModel::fromJson(cpuCostModelFile);
      if (!l3Size.hasValue())
        l3Size = static_cast<unsigned>(mm.l3Size);
      if (!llcSharers.hasValue())
        llcSharers = mm.llcSharers;
    }

    // Stage 0 (WP4): raise the symm scatter into a register-blockable
    // triangular reduction (fission temp2/epilogue from the scatter + emit the
    // scatter already interchanged to i-innermost).  No-op on every other
    // kernel (gated on a rank-0 reduction co-resident with a k-scattered store).
    raiseSymmScatter(func, rewriter);

    // Stage 1: canonicalize reduction nests so the reduction loop is innermost
    // (handles the PolyBench i-k-j order via k<->j interchange).
    while (canonicalizeOnce(func))
      ;

    // Stage 1.5 (WP5): interchange BLAS-2 column-major streaming nests under a
    // SEQUENTIAL outer sweep to row-major (gramschmidt's dot + A-update).  Runs
    // AFTER canonicalizeOnce (which leaves these already-inner-reduction nests
    // untouched) so nothing reverts it; the result is inert to the BLAS-3
    // register-block stages below.
    interchangeBlas2RowMajor(func);

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

    // Stage 1c (run unconditionally; historically nested under cache-tile,
    // which default-off meant triangular peeling NEVER ran in the default
    // configs): diagonal-peel triangular bands so their rectangular bulk
    // becomes register-blockable.  Handles both orientations: ub-from-IV
    // (syrk `j: 0..i+1`) and lb-from-IV (covariance/correlation `j: i..M`).
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
      int64_t effLLC = (int64_t)l3Size / (int64_t)(llcSharers ? llcSharers : 1u);
      for (AffineForOp s : cands)
        if (peelTriangularNest(s, mrEff, peelKTile, effLLC, rewriter)) {
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

    // In-place triangular reduction bands with the reduction innermost
    // (PolyBench trmm after distribute fissions its alpha-scale): split each
    // i-strip into a sequential CORNER (intra-strip k's) and a certified
    // row-disjoint MAIN that Stage 3 register-blocks.
    bool ipeeled = true;
    while (ipeeled) {
      ipeeled = false;
      SmallVector<AffineForOp> cands;
      func.walk([&](AffineForOp s) {
        AffineForOp sIn = onlyChildFor(s);
        if (!sIn)
          return;
        AffineForOp r = onlyChildFor(sIn);
        if (r && isInnermost(r) &&
            llvm::is_contained(r.getLowerBoundOperands(), s.getInductionVar()))
          cands.push_back(s);
      });
      for (AffineForOp s : cands)
        if (peelInPlaceTriangularInnermost(s, mrEff, rewriter)) {
          ipeeled = true;
          break;
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
        int64_t ie = in[0].getConstantUpperBound() - in[0].getConstantLowerBound();
        int64_t je = in[1].getConstantUpperBound() - in[1].getConstantLowerBound();
        int64_t ke = in[2].getConstantUpperBound() - in[2].getConstantLowerBound();
        // Element bytes from the accumulator memref.
        int64_t eb = 8;
        if (SmallVector<Acc> a = collectAccumulators(in[2]); !a.empty()) {
          Type et = cast<MemRefType>(a[0].memref.getType()).getElementType();
          if (et.isIntOrFloat())
            eb = std::max<int64_t>(1, (int64_t)et.getIntOrFloatBitWidth() / 8);
        }
        // The cache we can COUNT ON under contention: a co-tenant can evict the
        // shared L3, so only l3Size/llcSharers is guaranteed (private caches are
        // not derated).  Tile ONLY when the band's working set (A + B + C) does
        // not fit it -- otherwise the register-blocked micro-kernel already runs
        // cache-resident and tiling just adds min/max point-bound overhead that
        // scalarizes the kernel (observed: small-N collapse to ~0.1x).  Higher
        // llcSharers => tile sooner and smaller (can't rely on the shared L3).
        unsigned sharers = llcSharers ? llcSharers : 1u;
        int64_t effLLC = (int64_t)l3Size / (int64_t)sharers;
        int64_t ws = (ie * ke + ke * je + ie * je) * eb;
        if (effLLC <= 0 || ws <= effLLC)
          continue;
        // Clamp each tile to its extent, then halve the largest until the
        // per-tile working set (mc*kc + kc*nc + mc*nc)*eb fits the effective
        // cache.  Halving 256 keeps mr/nr/vl-friendly multiples.
        int64_t tmc = std::min<int64_t>(mc, ie), tnc = std::min<int64_t>(nc, je),
                tkc = std::min<int64_t>(kc, ke);
        auto tileWS = [&]() {
          return (tmc * tkc + tkc * tnc + tmc * tnc) * eb;
        };
        while (tileWS() > effLLC) {
          if (tmc >= tnc && tmc >= tkc && tmc > (int64_t)mr)
            tmc = std::max<int64_t>(mr, tmc / 2);
          else if (tnc >= tkc && tnc > (int64_t)nr)
            tnc = std::max<int64_t>(nr, tnc / 2);
          else if (tkc > (int64_t)vl)
            tkc = std::max<int64_t>(vl, tkc / 2);
          else
            break; // can't shrink further; tile anyway (better than DRAM-bound)
        }
        // Degenerate: a tile spanning the full extent gives no blocking benefit
        // and yields scalarizing point bounds -- leave it register-blocked untiled.
        if (tmc >= ie && tnc >= je && tkc >= ke)
          continue;
        SmallVector<unsigned, 3> sizes{(unsigned)tmc, (unsigned)tnc,
                                       (unsigned)tkc};
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

    // Stage 3b: explicit broadcast vectorization for reductions Stage 3
    // cannot reach through a parallel (sOut, sIn) PAIR -- e.g. a projection
    // sweep under a SEQUENTIAL outer loop (gramschmidt's
    // R[k][j] += Q[i][k]*A[i][j] under k; measured 0.94x vs base because
    // distribute splits the sweep and nothing downstream captured it).
    // vectorizeBroadcastBand only re-steps sIn by vl (lanes must be
    // independent => sIn parallel is required) and rebuilds the reduction
    // loop; the sequential ancestor is untouched, so no unroll-jam legality
    // is involved.  Already-promoted/vectorized nests have no scalar
    // load/store accumulator pair left and are skipped naturally.
    if (vectorize && !reassoc) {
      SmallVector<AffineForOp> reds;
      func.walk([&](AffineForOp r) {
        if (isInnermost(r) && !collectAccumulators(r).empty())
          reds.push_back(r);
      });
      for (AffineForOp red : reds) {
        AffineForOp sIn = red->getParentOfType<AffineForOp>();
        if (!sIn || onlyChildFor(sIn) != red)
          continue;
        AffineStoreOp store;
        AffineLoadOp load;
        if (!findAccPair(red, store, load))
          continue;
        if (!addrDependsOnIV(store, sIn.getInductionVar()))
          continue;
        if (!affine::isLoopParallel(sIn))
          continue;
        (void)vectorizeBroadcastBand(red, sIn, vl, rewriter);
      }
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
