//===- ParBubbles.cpp - Bubble-widening front-end (M0) ------------------===//
//
// M0 of the bubble-widening parallel-codegen track (PARALLEL_BUBBLE_SPEC.md):
// seed one bubble per loop and classify each loop axis as parallel or
// sequential via the intra-module ParAliasOracle.  Diagnostic-only — no IR is
// mutated.  Widening to fixed point (M1), the `par` dialect + materialization
// (M2), peeling/redistribution (M3), and interprocedural call consumption (M4)
// build on this seed.
//
//===----------------------------------------------------------------------===//

#include "drcompiler/Transforms/ParBubbles.h"
#include "drcompiler/Analysis/ParAliasOracle.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

/// How a seeded bubble's own loop axis is classified (PARALLEL_BUBBLE_SPEC.md §3).
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
  // A loop carrying iter_args / results threads an accumulator across
  // iterations — a (possibly associative) reduction.  v1 keeps it sequential;
  // cross-thread reduction-split is the M5 stretch goal.
  if (loopOp->getNumResults() > 0)
    return AxisKind::Reduction;

  // M0 models dependence only on affine.for; SCF is conservative for now.
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

struct DrParBubblesPass
    : public impl::DrParBubblesPassBase<DrParBubblesPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    ParAliasOracle oracle;

    module.walk([&](func::FuncOp fn) {
      if (fn.isExternal())
        return;
      // Seed: one bubble per loop; classify its axis.
      fn.walk([&](Operation *op) {
        if (!isa<affine::AffineForOp, scf::ForOp>(op))
          return;
        AxisKind k = classifyLoop(op, oracle);
        LLVM_DEBUG(llvm::dbgs() << "seed bubble @" << op << " -> "
                                << describe(k) << "\n");
        if (parTestDiagnostics)
          op->emitRemark(describe(k));
      });
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createDrParBubblesPass() {
  return std::make_unique<DrParBubblesPass>();
}
