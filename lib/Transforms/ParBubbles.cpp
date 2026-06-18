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
#include "drcompiler/Dialect/Par/IR/ParOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
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
// M2 — materialize a single climb region into the `par` dialect
//===----------------------------------------------------------------------===//

static unsigned leadingParallel(ArrayRef<AxisKind> levels) {
  unsigned p = 0;
  for (AxisKind k : levels) {
    if (k != AxisKind::Parallel)
      break;
    ++p;
  }
  return p;
}

/// A band is materializable when every loop has constant bounds and no
/// iter_args, and the innermost body holds only affine load/store ops and
/// region-free, side-effect-free ops (arith/math).  Anything else (calls,
/// non-affine memref ops, nested regions) bails — left as affine.for.
static bool bandMaterializable(ArrayRef<affine::AffineForOp> band) {
  for (affine::AffineForOp l : band) {
    if (!l.hasConstantLowerBound() || !l.hasConstantUpperBound())
      return false;
    if (l.getNumResults() != 0)
      return false;
  }
  affine::AffineForOp inner = band.back();
  for (Operation &op : inner.getBody()->without_terminator()) {
    if (isa<affine::AffineLoadOp, affine::AffineStoreOp>(op))
      continue;
    if (op.getNumRegions() == 0 && isMemoryEffectFree(&op))
      continue;
    return false;
  }
  return true;
}

/// Clone one body op, expanding affine accesses to memref (affine validity does
/// not hold under par.forall / scf.for, whose IVs are not affine dims).
static void cloneBodyOp(OpBuilder &b, Operation *op, IRMapping &map) {
  Location loc = op->getLoc();
  if (auto ld = dyn_cast<affine::AffineLoadOp>(op)) {
    SmallVector<Value> operands;
    for (Value v : ld.getMapOperands())
      operands.push_back(map.lookupOrDefault(v));
    auto idx = affine::expandAffineMap(b, loc, ld.getAffineMap(), operands);
    auto nl = b.create<memref::LoadOp>(loc, map.lookupOrDefault(ld.getMemRef()),
                                       *idx);
    map.map(ld.getResult(), nl.getResult());
    return;
  }
  if (auto st = dyn_cast<affine::AffineStoreOp>(op)) {
    Value val = map.lookupOrDefault(st.getValueToStore());
    SmallVector<Value> operands;
    for (Value v : st.getMapOperands())
      operands.push_back(map.lookupOrDefault(v));
    auto idx = affine::expandAffineMap(b, loc, st.getAffineMap(), operands);
    b.create<memref::StoreOp>(loc, val, map.lookupOrDefault(st.getMemRef()),
                              *idx);
    return;
  }
  b.clone(*op, map);
}

/// Build the sequential suffix loops band[idx..] as scf.for, then clone the
/// innermost body.
static void buildSeq(OpBuilder &b, ArrayRef<affine::AffineForOp> band,
                     unsigned idx, IRMapping &map) {
  if (idx == band.size()) {
    affine::AffineForOp inner = band.back();
    for (Operation &op : inner.getBody()->without_terminator())
      cloneBodyOp(b, &op, map);
    return;
  }
  affine::AffineForOp l = band[idx];
  Location loc = l.getLoc();
  Value lb = b.create<arith::ConstantIndexOp>(loc, l.getConstantLowerBound());
  Value ub = b.create<arith::ConstantIndexOp>(loc, l.getConstantUpperBound());
  Value st = b.create<arith::ConstantIndexOp>(loc, l.getStepAsInt());
  auto forOp = b.create<scf::ForOp>(loc, lb, ub, st);
  map.map(l.getInductionVar(), forOp.getInductionVar());
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToStart(forOp.getBody());
  buildSeq(b, band, idx + 1, map);
}

/// Wrap the leading parallel prefix of `band` in par.region { par.forall ... }.
/// Returns false (no mutation) when nothing is materializable.
static bool materializeBand(ArrayRef<affine::AffineForOp> band,
                            ArrayRef<AxisKind> levels) {
  unsigned p = leadingParallel(levels);
  if (p == 0 || !bandMaterializable(band))
    return false;

  affine::AffineForOp root = band.front();
  OpBuilder b(root);
  Location loc = root.getLoc();

  auto regionOp = b.create<par::RegionOp>(loc);
  b.createBlock(&regionOp.getRegion());

  SmallVector<int64_t> lbs, ubs, steps;
  for (unsigned d = 0; d < p; ++d) {
    affine::AffineForOp bd = band[d];
    lbs.push_back(bd.getConstantLowerBound());
    ubs.push_back(bd.getConstantUpperBound());
    steps.push_back(bd.getStepAsInt());
  }
  auto forall = b.create<par::ForallOp>(
      loc, b.getDenseI64ArrayAttr(lbs), b.getDenseI64ArrayAttr(ubs),
      b.getDenseI64ArrayAttr(steps));
  Block *fblk = b.createBlock(&forall.getRegion());
  fblk->addArguments(SmallVector<Type>(p, b.getIndexType()),
                     SmallVector<Location>(p, loc));

  IRMapping map;
  for (unsigned d = 0; d < p; ++d) {
    affine::AffineForOp bd = band[d];
    map.map(bd.getInductionVar(), fblk->getArgument(d));
  }
  b.setInsertionPointToStart(fblk);
  buildSeq(b, band, p, map);
  b.create<par::YieldOp>(loc, ValueRange{});

  b.setInsertionPointToEnd(regionOp.getBody());
  b.create<par::YieldOp>(loc, ValueRange{});

  root.erase();
  return true;
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

    // M2 — materialize single climb regions into the `par` dialect.
    if (parMaterialize) {
      module.walk([&](func::FuncOp fn) {
        if (fn.isExternal())
          return;
        // Re-walk to a fixed point: materializeBand erases the affine nest and
        // emits par/scf, so a fresh walk never re-sees a materialized region.
        bool changed = true;
        while (changed) {
          changed = false;
          fn.walk([&](affine::AffineForOp loop) -> WalkResult {
            if (!isBandRoot(loop))
              return WalkResult::advance();
            SmallVector<affine::AffineForOp, 4> band;
            affine::getPerfectlyNestedLoops(band, loop);
            SmallVector<AxisKind, 4> levels;
            classifyBand(band, oracle, levels);
            if (materializeBand(band, levels)) {
              changed = true;
              return WalkResult::interrupt();
            }
            return WalkResult::advance();
          });
        }
      });
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createDrParBubblesPass() {
  return std::make_unique<DrParBubblesPass>();
}
