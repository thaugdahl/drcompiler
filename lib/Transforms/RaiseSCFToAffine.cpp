//===- RaiseSCFToAffine.cpp - Raise structured loops back to affine ------===//
//
// Front-end normalisation, the same role `raise-malloc-to-memref` plays for
// Polygeist output: put a front end's loops into the form the affine analyses
// need, without changing what the program computes.
//
// MARCO lowers every Modelica equation to an `scf.for` nest whose bounds are
// either constants or `index` block arguments of the equation's own function,
// with unit step and no iteration-carried values:
//
//   func.func @equation_16(%lb0: index, %ub0: index, ...) {
//     scf.for %i = %lb0 to %ub0 step %c1 {
//       scf.for %j = %lb1 to %ub1 step %c1 { ... }
//     }
//   }
//
// Every DRCC locality transform (fusion, tiling, distribution, register
// blocking) is an affine transform, so none of it can see any of this — the
// measured affine coverage of a lowered MARCO model is ~2 %.  The gap is not a
// missing analysis: the loops are already rectangular, unit-step and
// constant/symbol-bounded, i.e. syntactically affine.  They are just spelled in
// `scf`.
//
// This pass rewrites the loops.  It deliberately does nothing about memory
// accesses, because upstream already handles them once the enclosing loops are
// affine (an `scf.for` induction variable is not a valid affine dimension,
// which is why the upstream passes must run *after* this one):
//
//   --fold-memref-alias-ops     folds MARCO's rank-0 `memref.subview` +
//                               `memref.store` back into an indexed store
//   --affine-raise-from-memref  folds `arith.addi %iv, %c` index arithmetic
//                               into the map of an `affine.load`/`affine.store`
//
// so the full bridge is
//
//   inline, canonicalize, symbol-dce, fold-memref-alias-ops,
//   func.func(dr-raise-scf-to-affine, affine-raise-from-memref), canonicalize
//
// Legality is syntactic and checked per loop; anything not provably convertible
// is left as `scf.for` (and reported under `emit-rationale`).  Converting only
// part of a nest is safe: `affine.for` nested inside `scf.for` is valid IR, and
// the affine analyses simply treat the `scf` parent conservatively.
//
//===----------------------------------------------------------------------===//

#include "drcompiler/Transforms/RaiseSCFToAffine.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "dr-raise-scf-to-affine"

namespace mlir {
#define GEN_PASS_DEF_DRRAISESCFTOAFFINEPASS
#include "drcompiler/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;

namespace {

/// An `affine.for` bound: a map plus the SSA values it is applied to.
struct Bound {
  AffineMap map;
  SmallVector<Value, 1> operands;
};

/// Express `v` as an affine.for bound, or fail.
///
/// Two forms are accepted, which between them cover every loop MARCO emits:
///   * a constant index                -> `affine_map<() -> (c)>`
///   * a valid affine symbol           -> `affine_map<()[s0] -> (s0)>`
/// A function's `index` block arguments are valid symbols, so equation
/// functions raise without being inlined first.
///
/// Bounds that are affine functions of enclosing induction variables (i.e.
/// triangular nests) are not handled; MARCO's equation nests are rectangular.
static std::optional<Bound> toBound(Value v, Operation *forOp) {
  MLIRContext *ctx = forOp->getContext();

  if (std::optional<int64_t> c = getConstantIntValue(v))
    return Bound{AffineMap::getConstantMap(*c, ctx), {}};

  Region *scope = getAffineScope(forOp);
  if (scope && isValidSymbol(v, scope))
    return Bound{AffineMap::get(/*dimCount=*/0, /*symbolCount=*/1,
                                getAffineSymbolExpr(0, ctx)),
                 {v}};

  return std::nullopt;
}

/// Why a loop could not be raised — reported under `emit-rationale`.
static const char *checkLegality(scf::ForOp forOp,
                                 std::optional<Bound> &lb,
                                 std::optional<Bound> &ub,
                                 int64_t &step) {
  // affine.for does support iter_args, but raising a reduction also means
  // proving the yield feeds the region argument in the obvious way.  MARCO's
  // equation loops carry nothing, so this is left out rather than guessed at.
  if (forOp.getNumResults() != 0)
    return "loop carries iter_args";

  std::optional<int64_t> stepCst = getConstantIntValue(forOp.getStep());
  if (!stepCst)
    return "step is not a constant";
  if (*stepCst <= 0)
    return "step is not positive";
  step = *stepCst;

  lb = toBound(forOp.getLowerBound(), forOp);
  if (!lb)
    return "lower bound is neither a constant nor a valid affine symbol";

  ub = toBound(forOp.getUpperBound(), forOp);
  if (!ub)
    return "upper bound is neither a constant nor a valid affine symbol";

  return nullptr;
}

/// Replace `forOp` with an equivalent `affine.for`, moving the body across.
static void raiseOne(scf::ForOp forOp, const Bound &lb, const Bound &ub,
                     int64_t step) {
  OpBuilder builder(forOp);
  auto newFor = builder.create<AffineForOp>(forOp.getLoc(), lb.operands, lb.map,
                                            ub.operands, ub.map, step);

  Block *oldBody = forOp.getBody();
  Block *newBody = newFor.getBody();

  // scf.yield carries no operands here (no iter_args), so it just goes away;
  // the affine.yield the builder already created terminates the new body.
  oldBody->getTerminator()->erase();
  newBody->getOperations().splice(newBody->getTerminator()->getIterator(),
                                  oldBody->getOperations());
  oldBody->getArgument(0).replaceAllUsesWith(newFor.getInductionVar());
  forOp.erase();
}

struct DrRaiseSCFToAffinePass
    : public impl::DrRaiseSCFToAffinePassBase<DrRaiseSCFToAffinePass> {

  void runOnOperation() override {
    func::FuncOp fn = getOperation();
    if (fn.isExternal())
      return;

    // Post-order: innermost loops first.  Raising an inner loop before its
    // parent is safe (bounds never reference an enclosing induction variable
    // here) and keeps the parent's body splice a single move.
    SmallVector<scf::ForOp> worklist;
    fn.walk([&](scf::ForOp forOp) { worklist.push_back(forOp); });

    unsigned raised = 0, skipped = 0;
    for (scf::ForOp forOp : worklist) {
      std::optional<Bound> lb, ub;
      int64_t step = 1;

      if (const char *why = checkLegality(forOp, lb, ub, step)) {
        ++skipped;
        LLVM_DEBUG(llvm::dbgs() << "dr-raise-scf-to-affine: skip -- " << why << "\n");
        if (emitRationale)
          forOp.emitRemark() << "dr-raise-scf-to-affine: SKIP -- " << why;
        continue;
      }

      raiseOne(forOp, *lb, *ub, step);
      ++raised;
    }

    if (emitRationale && (raised || skipped))
      fn.emitRemark() << "dr-raise-scf-to-affine: raised " << raised
                      << " scf.for, skipped " << skipped;

    if (emitRationale)
      warnOnDuplicateGlobals(fn);
  }

  /// Raising the loops is not on its own enough to make the affine transforms
  /// safe on this IR.  `checkMemrefAccessDependence` bails with `NoDependence`
  /// whenever the two accesses' memref *SSA values* differ
  /// (AffineAnalysis.cpp), and MARCO re-fetches every array with its own
  /// `memref.get_global` inside every equation -- 609 of them for a 127-
  /// equation model, for 16 distinct arrays.  Two nests that read and write the
  /// same global through different `get_global` results are therefore reported
  /// as independent, and a transform that trusts that answer may reorder them.
  ///
  /// The fix is upstream and cheap, but it has to run *after* raising (LICM
  /// needs affine.for to hoist out of):
  ///
  ///   func.func(affine-loop-invariant-code-motion, cse)
  ///
  /// which on the same model takes 609 `get_global` down to 349 and is what
  /// makes fusion fire at all.  Flag the situation so it cannot pass silently.
  static void warnOnDuplicateGlobals(func::FuncOp fn) {
    llvm::DenseMap<StringAttr, unsigned> perSymbol;
    fn.walk([&](memref::GetGlobalOp op) { ++perSymbol[op.getNameAttr().getAttr()]; });

    unsigned dupSymbols = 0, dupOps = 0;
    for (auto &[sym, n] : perSymbol)
      if (n > 1) {
        ++dupSymbols;
        dupOps += n;
      }

    if (dupSymbols)
      fn.emitRemark()
          << "dr-raise-scf-to-affine: " << dupOps << " memref.get_global ops "
          << "for " << dupSymbols << " distinct globals are duplicated. Affine "
          << "dependence analysis compares memref SSA values, so these read as "
          << "separate buffers -- run "
          << "func.func(affine-loop-invariant-code-motion,cse) before any "
          << "affine transform";
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createDrRaiseSCFToAffinePass() {
  return std::make_unique<DrRaiseSCFToAffinePass>();
}
