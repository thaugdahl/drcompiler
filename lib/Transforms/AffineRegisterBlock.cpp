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
#include "RegisterBlock/Internal.h"
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

namespace drcompiler {
namespace rb {

/// If `forOp`'s body is exactly one nested affine.for (plus the terminator),
/// return it; otherwise return null.  This identifies a perfect nest level.
AffineForOp onlyChildFor(AffineForOp forOp) {
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

/// Produce a value equivalent to `v` that is usable *before* `loop`.  Index
/// operands of an accumulator access (e.g. the `affine.apply #map(%i)` that
/// unroll-and-jam inserts inside the loop body) are pure functions of the
/// outer induction variables; clone that computation in front of the loop so
/// the hoisted init-load and the sunk final-store can reference it.
Value hoistOperand(Value v, AffineForOp loop, IRRewriter &rewriter,
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
bool dependsOn(Value root, Value def, Block *loopBody) {
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
bool accumulatorAliasesInput(Block *body, Value accMemref) {
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
const char kAccNoAliasAttr[] = "dr.acc_no_alias";

/// Collect accumulator load/store pairs in the innermost loop `kLoop`.
SmallVector<Acc> collectAccumulators(AffineForOp kLoop) {
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
bool addrDependsOnIV(AffineStoreOp store, Value iv) {
  for (Value o : store.getMapOperands())
    if (valueDependsOnIV(o, iv))
      return true;
  return false;
}

/// Find the first accumulator load/store pair directly in `loop`'s body (a
/// store with a matching same-address load whose value the store depends on).
/// No invariance filter -- used for canonicalization where the reduction may
/// be carried by an enclosing loop.
bool findAccPair(AffineForOp loop, AffineStoreOp &outStore,
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
bool isInnermost(AffineForOp loop) {
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

} // namespace rb
} // namespace drcompiler

namespace {
using namespace drcompiler::rb;

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

    // Stage 1d (WP-O2): direct-conv reduction BANDS.  A de-promoted 3x3 conv is
    // `for oc,oh,ow { for ic,kh,kw { Y[oc,oh,ow] += in[ic,oh+kh,ow+kw]*w[...] }}`
    // -- a MULTI-loop reduction band (ic/kh/kw) under the spatial loop ow.  The
    // GEMM stages below need the spatial loop to DIRECTLY enclose a single
    // reduction loop, so they never fire on conv.  Here we vectorize ow (stride-1
    // in both Y and `in`) directly, carrying a vector accumulator through the
    // whole band.  Detected before Stage 2 so the GEMM matcher never sees these.
    if (vectorize) {
      SmallVector<std::pair<AffineForOp, SmallVector<AffineForOp>>> convBands;
      func.walk([&](AffineForOp inner) {
        if (!isInnermost(inner))
          return;
        AffineStoreOp store;
        AffineLoadOp load;
        if (!findAccPair(inner, store, load))
          return;
        // Grow the band upward through enclosing REDUCTION loops (IVs absent
        // from the accumulator address), each perfectly enclosing the band.
        SmallVector<AffineForOp> band{inner};
        AffineForOp p = inner->getParentOfType<AffineForOp>();
        while (p && !addrDependsOnIV(store, p.getInductionVar())) {
          if (onlyChildFor(p) != band.front())
            break;
          band.insert(band.begin(), p);
          p = p->getParentOfType<AffineForOp>();
        }
        // A single-loop reduction is a GEMM (Stage 2/3 handles it); only a true
        // band (>=2 reduction loops) is a conv.
        if (band.size() < 2 || !p)
          return;
        AffineForOp sp = p; // the spatial loop indexing the accumulator (ow)
        if (!addrDependsOnIV(store, sp.getInductionVar()) ||
            onlyChildFor(sp) != band.front() || !affine::isLoopParallel(sp))
          return;
        convBands.push_back({sp, std::move(band)});
      });
      for (auto &cb : convBands)
        (void)vectorizeConvBand(cb.first, cb.second, vl, rewriter);
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
      // PER-BAND family selection.  The global familySelect above sets a single
      // function-wide mode (one PolyBench kernel = one family).  onnx-mlir emits
      // MANY contractions per function -- a 1x1 conv GEMM (broadcast) and a
      // genuine rank-k (dot) can coexist, and the global `anyDot` flag would
      // force every band into dot mode, disabling the broadcast vector kernel
      // for ALL of them (measured: resnet50's 33 demoted GEMMs all fell to the
      // scalar 4x4 dot tile -> 0 vector ops).  Detect the family of THIS band's
      // reduction (before jamming, since the jam factor depends on it).  For a
      // single-kernel function this is identical to the global decision.
      unsigned mrB = mrEff, nrB = nrEff;
      bool reassocB = reassoc;
      if (familySelect) {
        if (AffineForOp predRed = findReductionLoopUnder(sOut)) {
          if (AffineForOp predSin = predRed->getParentOfType<AffineForOp>()) {
            int nMul = 0;
            if (detectFamily(predRed, predSin, nMul) == RBFamily::Dot) {
              reassocB = true;
              mrB = nrB = (nMul > 2) ? 2u : 4u;
            } else {
              reassocB = false;
              mrB = mr;
              nrB = nr;
            }
          }
        }
      }
      if (mrB > 1 && failed(affine::loopUnrollJamByFactor(sOut, mrB)))
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
      unsigned nrVec = accRank >= 3 ? 1u : (nrB + vl - 1) / vl;
      if (vectorize && !reassocB &&
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
      if (nrB > 1 && failed(affine::loopUnrollJamByFactor(sIn, nrB)))
        continue;
      red = findReductionLoopUnder(func);
      if (!red)
        continue;
      // Dot family: EXPLICIT reduction-vectorization over k (vector dialect, no
      // reliance on LLVM reduction-vec).  Falls back to scalar promotion +
      // reassoc + LLVM when the k-trip isn't vl-divisible or the band isn't
      // cleanly vectorizable along k.
      if (vectorize && reassocB &&
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
