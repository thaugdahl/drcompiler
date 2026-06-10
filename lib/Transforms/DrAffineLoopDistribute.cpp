//===- DrAffineLoopDistribute.cpp - Fission imperfect affine loops -------===//
//
// PolyBench kernels are frequently written as imperfect nests:
//
//   for (i) { C[i][j] *= beta;  for (k) for (j) C[i][j] += ... }
//
// Both upstream affine-loop-tile and our dr-affine-loop-tile (and
// affine-register-block) only operate on maximal PERFECT bands rooted at
// outermost loops, so the imperfect i-loop reduces "tiling gemm" to
// strip-mining the i-loop — no cache reuse is ever captured.  Polyhedral
// schedulers (Polly) implicitly distribute such loops before tiling.
//
// This pass performs classical loop fission: a loop L whose body contains
// two or more top-level affine.for children (and otherwise only pure scalar
// ops) is split into consecutive copies of L, one child nest per copy, when
// no dependence is carried backward by L (i.e. no dependence from a later
// child to an earlier child carried exactly at L's depth).  All-forward
// dependences (e.g. "init C row i" -> "update C row i") are preserved by
// fission: every iteration of copy 1 still executes before any dependent
// iteration of copy 2.
//
// Pure scalar ops in L's body are kept in every copy (they are cheap and
// dead copies fold away later); loops whose body contains top-level memory
// accesses or calls are left untouched.
//
//===----------------------------------------------------------------------===//

#include "drcompiler/Transforms/DrAffineLoopDistribute.h"
#include "drcompiler/Analysis/ReuseAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "dr-affine-loop-distribute"

namespace mlir {
#define GEN_PASS_DEF_DRAFFINELOOPDISTRIBUTEPASS
#include "drcompiler/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;

namespace {

struct DrAffineLoopDistributePass
    : public impl::DrAffineLoopDistributePassBase<DrAffineLoopDistributePass> {

  void runOnOperation() override {
    func::FuncOp fn = getOperation();
    if (fn.isExternal())
      return;
    // Fixpoint: re-walk after every successful split.  A split erases loops
    // (sibling units in each copy) and can make new loops distributable
    // (e.g. 2mm's `for i { for j { tmp=0; for k } }`: splitting j leaves i
    // with two perfectly nestable j-children), so a pointer worklist would
    // dangle; fresh pre-order walks are cheap at kernel scale and always
    // see a consistent IR.
    bool changed = true;
    while (changed) {
      changed = false;
      fn.walk<WalkOrder::PreOrder>([&](AffineForOp loop) -> WalkResult {
        SmallVector<AffineForOp, 4> copies;
        if (tryDistribute(loop, copies)) {
          changed = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
    }
  }

private:
  static void collectAccesses(AffineForOp root,
                              SmallVectorImpl<Operation *> &ops) {
    root->walk([&](Operation *op) {
      if (isa<AffineReadOpInterface, AffineWriteOpInterface>(op))
        ops.push_back(op);
    });
  }

  /// Returns true (and fills `copies`, original loop first) if `loop` was
  /// distributed into one copy per distribution unit.
  ///
  /// A unit is either a top-level child affine.for nest or a maximal
  /// contiguous run of top-level memory-effectful non-loop ops (a statement
  /// block, e.g. the `tmp[i][j] = 0` matmul init that makes 2mm/3mm nests
  /// imperfect).  Memory-effect-free scalar ops are replicated into every
  /// copy; dead replicas fold away later.
  bool tryDistribute(AffineForOp loop, SmallVectorImpl<AffineForOp> &copies) {
    Block *body = loop.getBody();
    if (loop->getNumResults() > 0)
      return false;

    // Partition body ops (by index) into units.
    struct Unit {
      bool isLoop = false;
      SmallVector<unsigned, 4> opIdx; // indices into body op order
    };
    SmallVector<Unit, 4> units;
    SmallVector<Operation *, 16> bodyOps;
    unsigned numLoops = 0;
    // First sweep: collect ops, loop units, and maximal contiguous runs of
    // non-loop ops that contain at least one memory-effectful op.
    SmallVector<SmallVector<unsigned, 8>, 4> runs;
    bool runOpen = false, runHasEffect = false;
    auto closeRun = [&]() {
      if (runOpen && !runHasEffect)
        runs.pop_back(); // pure-only run: everything stays replicated
      runOpen = runHasEffect = false;
    };
    for (Operation &op : body->without_terminator()) {
      unsigned idx = bodyOps.size();
      bodyOps.push_back(&op);
      if (auto childFor = dyn_cast<AffineForOp>(op)) {
        if (childFor->getNumResults() > 0)
          return false; // iter_args: leave untouched
        closeRun();
        units.push_back(Unit{/*isLoop=*/true, {idx}});
        ++numLoops;
        continue;
      }
      if (op.getNumRegions() > 0)
        return false;
      if (!runOpen) {
        runs.emplace_back();
        runOpen = true;
      }
      runs.back().push_back(idx);
      runHasEffect |= !isMemoryEffectFree(&op);
    }
    closeRun();

    // Each effectful run becomes a statement unit owning its effectful ops
    // plus every pure run-op that is dead once the unit is erased (its users
    // all land inside the unit, computed to fixpoint).  Remaining pure ops
    // are replicated into every copy.  SSA safety: a unit op whose result is
    // used outside the unit (a load feeding a later nest) blocks fission —
    // erasing it would orphan the user, and replicating a read across the
    // split is unsound when an intervening copy writes that location.
    for (const SmallVector<unsigned, 8> &run : runs) {
      llvm::SmallPtrSet<Operation *, 8> inUnit;
      for (unsigned idx : run)
        if (!isMemoryEffectFree(bodyOps[idx]))
          inUnit.insert(bodyOps[idx]);
      bool changed = true;
      while (changed) {
        changed = false;
        for (unsigned idx : llvm::reverse(run)) {
          Operation *op = bodyOps[idx];
          if (inUnit.contains(op) || !isMemoryEffectFree(op))
            continue;
          bool allUsersInUnit = !op->use_empty();
          for (Operation *user : op->getUsers())
            allUsersInUnit &= inUnit.contains(user);
          if (allUsersInUnit) {
            inUnit.insert(op);
            changed = true;
          }
        }
      }
      for (Operation *op : inUnit)
        for (Operation *user : op->getUsers())
          if (!inUnit.contains(user))
            return false;
      Unit u;
      u.isLoop = false;
      for (unsigned idx : run)
        if (inUnit.contains(bodyOps[idx]))
          u.opIdx.push_back(idx);
      // Splice the statement unit into program-order position among units.
      units.push_back(std::move(u));
      for (unsigned i = units.size() - 1; i > 0; --i)
        if (units[i - 1].opIdx.front() > units[i].opIdx.front())
          std::swap(units[i - 1], units[i]);
    }
    if (units.size() < 2 || numLoops == 0)
      return false;

    // Legality: no dependence between distinct units carried exactly by
    // `loop`.  Dependences carried by enclosing loops are unaffected by
    // fission (the copies stay together inside them); loop-independent
    // dependences are always forward in program order and remain satisfied
    // (all iterations of an earlier copy run before any of a later copy).
    unsigned loopDepth = getNestingDepth(loop) + 1;
    SmallVector<SmallVector<Operation *, 8>, 4> accesses(units.size());
    for (unsigned i = 0; i < units.size(); ++i) {
      for (unsigned idx : units[i].opIdx) {
        if (units[i].isLoop)
          collectAccesses(cast<AffineForOp>(bodyOps[idx]), accesses[i]);
        else if (isa<AffineReadOpInterface, AffineWriteOpInterface>(
                     bodyOps[idx]))
          accesses[i].push_back(bodyOps[idx]);
        else if (!isMemoryEffectFree(bodyOps[idx]))
          return false; // effectful op we cannot reason about (call, memref.*)
        // else: pure op absorbed into the unit (dead once the unit is
        // erased); no memory access to model.
      }
    }

    for (unsigned j = 1; j < units.size(); ++j) {
      for (unsigned i = 0; i < j; ++i) {
        for (Operation *a : accesses[j]) {
          for (Operation *b : accesses[i]) {
            bool aWrites = isa<AffineWriteOpInterface>(a);
            bool bWrites = isa<AffineWriteOpInterface>(b);
            if (!aWrites && !bWrites)
              continue;
            MemRefAccess srcAccess(a), dstAccess(b);
            if (srcAccess.memref != dstAccess.memref)
              continue;
            DependenceResult result = checkMemrefAccessDependence(
                srcAccess, dstAccess, loopDepth,
                /*dependenceConstraints=*/nullptr,
                /*dependenceComponents=*/nullptr);
            if (hasDependence(result)) {
              LLVM_DEBUG(llvm::dbgs()
                         << "fission illegal: dep carried by loop between "
                            "units\n");
              return false;
            }
          }
        }
      }
    }

    llvm::SmallPtrSet<Operation *, 16> unitMember;
    for (const Unit &u : units)
      for (unsigned idx : u.opIdx)
        unitMember.insert(bodyOps[idx]);

    // Profitability ('enabler' mode): fission must DEEPEN some perfect band,
    // i.e. some loop-unit copy must be a perfect nest rooted at `loop` after
    // sibling units are erased and dead pure replicas are cleaned.  A copy is
    // spoiled by any replicated pure op that stays live (feeds the kept
    // child): it sits at body level and breaks perfect nesting, so the tiler
    // could not analyze the band anyway and splitting buys nothing.
    if (mode == "enabler") {
      bool deepens = false;
      for (const Unit &u : units) {
        if (!u.isLoop)
          continue;
        Operation *child = bodyOps[u.opIdx.front()];
        bool clean = true;
        for (unsigned idx = 0; idx < bodyOps.size() && clean; ++idx) {
          Operation *op = bodyOps[idx];
          if (op == child || unitMember.contains(op))
            continue; // unit members are erased or kept wholesale
          // Live in this copy iff some transitive user sits inside `child`.
          SmallVector<Operation *, 8> work(op->getUsers().begin(),
                                           op->getUsers().end());
          llvm::SmallPtrSet<Operation *, 8> seen;
          while (!work.empty()) {
            Operation *user = work.pop_back_val();
            if (!seen.insert(user).second)
              continue;
            if (child->isAncestor(user)) {
              clean = false;
              break;
            }
            for (Operation *uu : user->getUsers())
              work.push_back(uu);
          }
        }
        if (clean) {
          deepens = true;
          break;
        }
      }
      if (!deepens) {
        if (emitRationale)
          loop->emitRemark(
              "distribute-rationale: SKIP legal split (no band deepened)");
        return false;
      }

      // Locality guard: splitting separates units that today execute back to
      // back per iteration.  If they SHARE data (atax's two j-loops both
      // read row A[i][:]), the fused form gets that reuse for free from the
      // cache; the split form refetches it a whole sweep later.  That cost
      // is only worth paying when the split feeds a downstream win — some
      // post-split band carrying temporal reuse the tiler can capture
      // (gemm/2mm: the matmul band).  No such band + shared data => skip.
      bool reuseBenefit = false;
      {
        // Perfect ancestors of `loop` (the post-split copies stay nested in
        // them, so they are part of every hypothetical band).
        SmallVector<AffineForOp, 4> ancestors;
        Operation *cur = loop;
        while (auto parent = cur->getParentOfType<AffineForOp>()) {
          Block *pb = parent.getBody();
          if (std::next(pb->begin()) != std::prev(pb->end()))
            break; // imperfect above: band stops here
          ancestors.push_back(parent);
          cur = parent;
        }
        std::reverse(ancestors.begin(), ancestors.end());
        for (const Unit &u : units) {
          if (!u.isLoop)
            continue;
          SmallVector<AffineForOp, 6> band(ancestors.begin(), ancestors.end());
          band.push_back(loop);
          SmallVector<AffineForOp, 6> chain;
          getPerfectlyNestedLoops(chain,
                                  cast<AffineForOp>(bodyOps[u.opIdx.front()]));
          band.append(chain.begin(), chain.end());
          // Restrict the walk to this unit's nest: the sibling units are
          // still present under `loop` and would otherwise poison the band.
          auto infoOr = drcompiler::reuse::analyzeBandReuse(
              band, bodyOps[u.opIdx.front()]);
          if (failed(infoOr)) {
            // Out of the constant-coefficient model (e.g. a triangular band
            // like covariance's j = i..M around its k-reduction).  Unknown
            // is not "no benefit": the harmful shared-sweep cases this guard
            // exists for (atax/bicg BLAS-2 bodies) analyze cleanly, so stay
            // optimistic here and let the split stand.
            reuseBenefit = true;
            break;
          }
          for (unsigned l = 0, e = band.size(); l < e && !reuseBenefit; ++l)
            reuseBenefit = infoOr->loopCarriesEvictedReuse(l, cacheBytes);
          if (reuseBenefit)
            break;
        }
      }
      if (!reuseBenefit) {
        // No tiling fuel anywhere: only split if the units are disjoint.
        llvm::SmallDenseMap<void *, unsigned, 8> firstUnit;
        bool shared = false;
        for (unsigned i = 0; i < units.size() && !shared; ++i) {
          llvm::SmallPtrSet<void *, 8> mine;
          for (Operation *a : accesses[i])
            mine.insert(isa<AffineWriteOpInterface>(a)
                            ? cast<AffineWriteOpInterface>(a)
                                  .getMemRef()
                                  .getAsOpaquePointer()
                            : cast<AffineReadOpInterface>(a)
                                  .getMemRef()
                                  .getAsOpaquePointer());
          for (void *m : mine) {
            auto it = firstUnit.find(m);
            if (it != firstUnit.end() && it->second != i) {
              shared = true;
              break;
            }
            firstUnit[m] = i;
          }
        }
        if (shared) {
          if (emitRationale)
            loop->emitRemark("distribute-rationale: SKIP legal split "
                             "(shared data, no reuse benefit)");
          return false;
        }
      }
    }

    // Transform: clone `loop` (units.size() - 1) times after the original;
    // copy k keeps only unit k's ops (plus replicated pure ops).
    OpBuilder b(loop->getBlock(), std::next(Block::iterator(loop)));
    SmallVector<AffineForOp, 4> all;
    all.push_back(loop);
    for (unsigned k = 1; k < units.size(); ++k) {
      auto clone = cast<AffineForOp>(b.clone(*loop.getOperation()));
      all.push_back(clone);
      b.setInsertionPointAfter(clone);
    }
    for (unsigned k = 0; k < all.size(); ++k) {
      SmallVector<Operation *, 16> copyOps;
      for (Operation &op : all[k].getBody()->without_terminator())
        copyOps.push_back(&op);
      assert(copyOps.size() == bodyOps.size() && "clone mismatch");
      for (unsigned u = 0; u < units.size(); ++u) {
        if (u == k)
          continue;
        for (unsigned idx : llvm::reverse(units[u].opIdx))
          copyOps[idx]->erase();
      }
      // Sweep dead pure replicas so loop-unit copies become PERFECT nests
      // (a live-but-unused body-level op would hide the band from the tiler
      // and from getPerfectlyNestedLoops-based clients).
      bool swept = true;
      while (swept) {
        swept = false;
        for (Operation &op :
             llvm::make_early_inc_range(llvm::reverse(*all[k].getBody()))) {
          if (op.hasTrait<OpTrait::IsTerminator>())
            continue;
          if (isMemoryEffectFree(&op) && op.use_empty() &&
              op.getNumRegions() == 0) {
            op.erase();
            swept = true;
          }
        }
      }
    }
    if (emitRationale)
      loop->emitRemark("distribute-rationale: SPLIT into " +
                       std::to_string(all.size()) + " loops");
    // Report only the loop-unit copies for re-examination.
    for (unsigned k = 0; k < all.size(); ++k)
      if (units[k].isLoop)
        copies.push_back(all[k]);
    LLVM_DEBUG(llvm::dbgs() << "distributed loop into " << all.size()
                            << " copies\n");
    return true;
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createDrAffineLoopDistributePass() {
  return std::make_unique<DrAffineLoopDistributePass>();
}
