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
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
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
    // Process loops outermost-first; after a successful split, re-examine
    // the copies (a split can expose further distributable levels).
    SmallVector<AffineForOp> worklist;
    for (Block &b : fn.getBody())
      for (auto loop : b.getOps<AffineForOp>())
        worklist.push_back(loop);
    while (!worklist.empty()) {
      AffineForOp loop = worklist.pop_back_val();
      SmallVector<AffineForOp, 4> copies;
      if (tryDistribute(loop, copies)) {
        // Examine each copy again (and their children via the else-branch).
        for (AffineForOp c : copies)
          worklist.push_back(c);
      } else {
        for (auto child : loop.getBody()->getOps<AffineForOp>())
          worklist.push_back(child);
      }
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
  /// distributed into one copy per top-level child nest.
  bool tryDistribute(AffineForOp loop, SmallVectorImpl<AffineForOp> &copies) {
    Block *body = loop.getBody();

    SmallVector<AffineForOp, 4> children;
    for (Operation &op : body->without_terminator()) {
      if (auto childFor = dyn_cast<AffineForOp>(op)) {
        if (childFor->getNumResults() > 0)
          return false; // iter_args: leave untouched
        children.push_back(childFor);
        continue;
      }
      // Everything else must be pure scalar computation we can replicate.
      if (!isMemoryEffectFree(&op) || op.getNumRegions() > 0)
        return false;
    }
    if (children.size() < 2)
      return false;
    if (loop->getNumResults() > 0)
      return false;

    // Legality: no dependence from a later child to an earlier child carried
    // exactly by `loop`.  Dependences carried by loops enclosing `loop` are
    // unaffected by fission (the copies stay together inside them), and
    // loop-independent backward dependences cannot exist (program order).
    unsigned loopDepth = getNestingDepth(loop) + 1;
    SmallVector<SmallVector<Operation *, 8>, 4> accesses(children.size());
    for (unsigned i = 0; i < children.size(); ++i)
      collectAccesses(children[i], accesses[i]);

    for (unsigned j = 1; j < children.size(); ++j) {
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
                         << "fission illegal: backward dep carried by loop\n");
              return false;
            }
          }
        }
      }
    }

    // Transform: clone `loop` (children.size() - 1) times after the
    // original; copy k keeps only child k.  The original keeps child 0.
    OpBuilder b(loop->getBlock(), std::next(Block::iterator(loop)));
    SmallVector<AffineForOp, 4> all;
    all.push_back(loop);
    for (unsigned k = 1; k < children.size(); ++k) {
      auto clone = cast<AffineForOp>(b.clone(*loop.getOperation()));
      all.push_back(clone);
      b.setInsertionPointAfter(clone);
    }
    for (unsigned k = 0; k < all.size(); ++k) {
      SmallVector<AffineForOp, 4> kids;
      for (auto child : all[k].getBody()->getOps<AffineForOp>())
        kids.push_back(child);
      assert(kids.size() == children.size() && "clone mismatch");
      for (unsigned c = 0; c < kids.size(); ++c)
        if (c != k)
          kids[c].erase();
    }
    copies.append(all.begin(), all.end());
    LLVM_DEBUG(llvm::dbgs() << "distributed loop into " << all.size()
                            << " copies\n");
    return true;
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createDrAffineLoopDistributePass() {
  return std::make_unique<DrAffineLoopDistributePass>();
}
