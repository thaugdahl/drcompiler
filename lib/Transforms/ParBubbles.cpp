//===- ParBubbles.cpp - Bubble-widening front-end (M0 + M1) -------------===//
//
// Bubble-widening parallel-codegen front-end (PARALLEL_BUBBLE_SPEC.md).
//
//   M0 (par-test-diagnostics): seed one bubble per loop and classify each
//      loop's axis via the intra-module ParAliasOracle.
//   M1 (par-test-regions): widen to a fixed point and report MAXIMAL regions.
//      Two moves, Clean/Hard only (no peel/redistribute yet, that is M3):
//        * climb  — collapse a perfect affine band into one region, per-level
//                   par/seq classification (GEMM -> par={i,j} seq={k}).
//        * fuse   — merge adjacent conformant sibling bands when they touch
//                   provably-disjoint memory (Tier 0).  A rejected fuse is a
//                   frozen frontier (reason: outer-sequential | non-conformant
//                   | shared-write).
//
// Both modes are diagnostic-only — no IR is mutated.  Materialization to the
// `par` dialect is M2; same-buffer aligned fuse + offset peel is M3; cross-
// procedure call consumption is M4.
//
//===----------------------------------------------------------------------===//

#include "drcompiler/Transforms/ParBubbles.h"
#include "drcompiler/Analysis/ParAliasOracle.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "dr-par-bubbles"

namespace mlir {
#define GEN_PASS_DEF_DRPARBUBBLESPASS
#include "drcompiler/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using drcompiler::par::ConflictKind;
using drcompiler::par::ParAliasOracle;

namespace {

//===----------------------------------------------------------------------===//
// M0 — per-loop axis classification
//===----------------------------------------------------------------------===//

enum class AxisKind { Parallel, Carried, Reduction, Conservative };

static StringRef describe(AxisKind k) {
  switch (k) {
  case AxisKind::Parallel:
    return "par-bubble axis: PARALLEL";
  case AxisKind::Carried:
    return "par-bubble axis: SEQUENTIAL (carried dependence)";
  case AxisKind::Reduction:
    return "par-bubble axis: SEQUENTIAL (reduction)";
  case AxisKind::Conservative:
    return "par-bubble axis: SEQUENTIAL (conservative)";
  }
  return "par-bubble axis: SEQUENTIAL (conservative)";
}

/// Classify the loop carried by `loopOp` (an affine.for or scf.for).
static AxisKind classifyLoop(Operation *loopOp, const ParAliasOracle &oracle) {
  // iter_args / results thread an accumulator across iterations: a reduction.
  if (loopOp->getNumResults() > 0)
    return AxisKind::Reduction;
  // M0/M1 model dependence only on affine.for; SCF is conservative for now.
  if (isa<scf::ForOp>(loopOp))
    return AxisKind::Conservative;
  switch (oracle.axisConflict(loopOp)) {
  case ConflictKind::None:
  case ConflictKind::LoopIndependent:
    return AxisKind::Parallel;
  case ConflictKind::Carried:
    return AxisKind::Carried;
  case ConflictKind::Unknown:
    return AxisKind::Conservative;
  }
  return AxisKind::Conservative;
}

//===----------------------------------------------------------------------===//
// M1 — region formation (climb + conformant fuse)
//===----------------------------------------------------------------------===//

static bool isAllocLike(Value v) {
  Operation *def = v.getDefiningOp();
  return def && isa<memref::AllocOp, memref::AllocaOp>(def);
}

static Value memrefOf(Operation *op) {
  if (auto w = dyn_cast<affine::AffineWriteOpInterface>(op))
    return w.getMemRef();
  if (auto r = dyn_cast<affine::AffineReadOpInterface>(op))
    return r.getMemRef();
  return nullptr;
}

static void collectAccesses(Operation *root, SmallVectorImpl<Operation *> &out) {
  root->walk([&](Operation *op) {
    if (isa<affine::AffineReadOpInterface, affine::AffineWriteOpInterface>(op))
      out.push_back(op);
  });
}

/// A loop is a *band root* unless it is the sole perfect child of an enclosing
/// affine.for (in which case getPerfectlyNestedLoops folds it into the parent's
/// band).
static bool isBandRoot(affine::AffineForOp loop) {
  auto parent = dyn_cast<affine::AffineForOp>(loop->getParentOp());
  if (!parent)
    return true;
  Block *pb = parent.getBody();
  Block::iterator begin = pb->begin();
  Block::iterator term = std::prev(pb->end());
  // Perfect nesting: the body holds exactly this loop + terminator.
  if (std::next(begin) == term && &*begin == loop.getOperation())
    return false;
  return true;
}

/// Two sibling bands are conformant when their outermost loops share constant
/// bounds and step (so they can be distributed over one iteration space).
static bool conformant(affine::AffineForOp a, affine::AffineForOp b) {
  if (!a.hasConstantLowerBound() || !b.hasConstantLowerBound())
    return false;
  if (!a.hasConstantUpperBound() || !b.hasConstantUpperBound())
    return false;
  return a.getConstantLowerBound() == b.getConstantLowerBound() &&
         a.getConstantUpperBound() == b.getConstantUpperBound() &&
         a.getStepAsInt() == b.getStepAsInt();
}

/// Clean (M1, Tier-0 only): every cross access pair that involves a write must
/// land on provably-distinct allocations.  Same-buffer aligned fusion needs the
/// slice/dependence machinery and is deferred to M3.
static bool crossClean(affine::AffineForOp a, affine::AffineForOp b) {
  SmallVector<Operation *, 16> aa, bb;
  collectAccesses(a, aa);
  collectAccesses(b, bb);
  for (Operation *x : aa) {
    for (Operation *y : bb) {
      bool writes = isa<affine::AffineWriteOpInterface>(x) ||
                    isa<affine::AffineWriteOpInterface>(y);
      if (!writes)
        continue; // read/read never conflicts
      Value mx = memrefOf(x), my = memrefOf(y);
      if (mx == my)
        return false;
      Value rx = ParAliasOracle::allocationRoot(mx);
      Value ry = ParAliasOracle::allocationRoot(my);
      if (rx == ry)
        return false;
      if (isAllocLike(rx) && isAllocLike(ry))
        continue; // distinct buffers: no interaction
      return false; // unprovable (args / globals): conservative
    }
  }
  return true;
}

struct FuseCheck {
  bool ok;
  StringRef reason;
};

static FuseCheck canFuseSibling(affine::AffineForOp a, affine::AffineForOp b,
                                const ParAliasOracle &oracle) {
  if (classifyLoop(a.getOperation(), oracle) != AxisKind::Parallel ||
      classifyLoop(b.getOperation(), oracle) != AxisKind::Parallel)
    return {false, "outer-sequential"};
  if (!conformant(a, b))
    return {false, "non-conformant"};
  if (!crossClean(a, b))
    return {false, "shared-write"};
  return {true, ""};
}

static void classifyBand(ArrayRef<affine::AffineForOp> band,
                         const ParAliasOracle &oracle,
                         SmallVectorImpl<AxisKind> &levels) {
  for (affine::AffineForOp l : band)
    levels.push_back(classifyLoop(l.getOperation(), oracle));
}

static std::string axisList(ArrayRef<AxisKind> levels, bool wantParallel) {
  std::string s = "[";
  bool first = true;
  for (unsigned i = 0; i < levels.size(); ++i) {
    bool isPar = levels[i] == AxisKind::Parallel;
    if (isPar != wantParallel)
      continue;
    if (!first)
      s += ",";
    s += std::to_string(i);
    first = false;
  }
  s += "]";
  return s;
}

static void emitRegion(affine::AffineForOp rep, unsigned bands,
                       ArrayRef<AxisKind> levels) {
  rep->emitRemark("par-region: bands=" + std::to_string(bands) + " par=" +
                  axisList(levels, /*wantParallel=*/true) + " seq=" +
                  axisList(levels, /*wantParallel=*/false));
}

/// Form maximal regions from the band roots of one block (program order) and
/// emit one remark per region, plus a frozen-frontier remark at each rejected
/// fuse boundary.
static void processBlock(ArrayRef<affine::AffineForOp> roots,
                         const ParAliasOracle &oracle) {
  unsigned n = roots.size(), i = 0;
  while (i < n) {
    SmallVector<affine::AffineForOp, 4> band;
    affine::getPerfectlyNestedLoops(band, roots[i]);
    SmallVector<AxisKind, 4> levels;
    classifyBand(band, oracle, levels);

    unsigned j = i + 1, bands = 1;
    StringRef rejectReason;
    bool rejected = false;
    while (j < n) {
      FuseCheck fc = canFuseSibling(roots[j - 1], roots[j], oracle);
      if (!fc.ok) {
        rejectReason = fc.reason;
        rejected = true;
        break;
      }
      ++bands;
      ++j;
    }

    emitRegion(roots[i], bands, levels);
    if (rejected)
      roots[j]->emitRemark(
          (Twine("par-frontier: frozen (") + rejectReason + ")").str());
    i = j;
  }
}

//===----------------------------------------------------------------------===//

struct DrParBubblesPass
    : public impl::DrParBubblesPassBase<DrParBubblesPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    ParAliasOracle oracle;

    // M0 — per-loop axis classification.
    if (parTestDiagnostics) {
      module.walk([&](func::FuncOp fn) {
        if (fn.isExternal())
          return;
        fn.walk([&](Operation *op) {
          if (!isa<affine::AffineForOp, scf::ForOp>(op))
            return;
          AxisKind k = classifyLoop(op, oracle);
          LLVM_DEBUG(llvm::dbgs() << "seed bubble @" << op << " -> "
                                  << describe(k) << "\n");
          op->emitRemark(describe(k));
        });
      });
    }

    // M1 — maximal-region formation (climb + conformant fuse).
    if (parTestRegions) {
      module.walk([&](func::FuncOp fn) {
        if (fn.isExternal())
          return;
        llvm::DenseMap<Block *, SmallVector<affine::AffineForOp, 4>> byBlock;
        SmallVector<Block *, 8> order;
        fn.walk([&](affine::AffineForOp loop) {
          if (!isBandRoot(loop))
            return;
          auto &v = byBlock[loop->getBlock()];
          if (v.empty())
            order.push_back(loop->getBlock());
          v.push_back(loop);
        });
        for (Block *b : order)
          processBlock(byBlock[b], oracle);
      });
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createDrParBubblesPass() {
  return std::make_unique<DrParBubblesPass>();
}
