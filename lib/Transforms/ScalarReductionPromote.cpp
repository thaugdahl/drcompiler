//===- ScalarReductionPromote.cpp - memref accumulator -> iter_args ------===//
//
// Inverse of dr-scalar-reduction-demote, run AFTER affine-register-block.
//
// The demote pass rewrites every onnx-mlir iter_args reduction into the
// memref-accumulator form register-block matches.  Register-block then
// re-promotes the bands it takes (GEMM Stage 3 jams into iter_args; the conv
// Stage 1d carries a vector iter_arg) -- but every band it does NOT take is
// left accumulating into the REAL output tensor in memory, one round-trip per
// innermost iteration.  That is much slower than the original iter_args form
// (measured 3.2s vs 2.1s whole-resnet50: sub-VL 14x14/7x7 convs, stride-2
// convs, dot-misclassified GEMMs, and the scalar borders/tails the conv
// splitter creates).  The WP-O1 assumption that backend mem2reg cleans these
// up does not hold: the accumulator is a real heap buffer, not an alloca.
//
// This pass walks innermost loops with a memory accumulator (same-address
// load/store pair, address invariant across the enclosing perfect reduction
// band), and rebuilds the band carrying the accumulator as a scalar iter_arg
// threaded through every level: one load before the band, one store after.
//
//   for kw { %c = load Y[oh,ow]; store %c + in*w, Y[oh,ow] }
//     ==>
//   %i = load Y[oh,ow]
//   %r = for kw iter_args(%a = %i) { yield %a + in*w }
//   store %r, Y[oh,ow]
//
// Generic over the reduction op (works for any stored value computed from the
// load -- addf chains, maxnumf, fused mul-adds); the alias guard in
// collectAccumulators (no other access to the accumulator memref in the body)
// makes deferring the store safe.
//
//===----------------------------------------------------------------------===//

#include "drcompiler/Transforms/ScalarReductionPromote.h"
#include "RegisterBlock/Internal.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "dr-scalar-reduction-promote"

namespace mlir {
#define GEN_PASS_DEF_DRSCALARREDUCTIONPROMOTEPASS
#include "drcompiler/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;
using drcompiler::rb::Acc;

namespace {

/// A promotable band: perfectly nested reduction loops outer->inner whose
/// innermost body carries exactly one memory accumulator with an address
/// invariant to every level and available above the band.
struct Match {
  SmallVector<AffineForOp> band; // outer -> inner
  Acc acc;
  Match(SmallVector<AffineForOp> b, Acc a)
      : band(std::move(b)), acc(std::move(a)) {}
};

/// True if `v` is defined outside `loop` (i.e. usable before it).
static bool availableAbove(Value v, AffineForOp loop) {
  Operation *def = v.getDefiningOp();
  if (!def)
    return !loop->isAncestor(cast<BlockArgument>(v).getOwner()->getParentOp());
  return !loop->isAncestor(def);
}

static std::optional<Match> matchBand(AffineForOp inner) {
  if (!drcompiler::rb::isInnermost(inner))
    return std::nullopt;
  // A loop carrying results is already (partly) an SSA reduction -- skip.
  if (inner.getNumResults() != 0)
    return std::nullopt;
  SmallVector<Acc> accs = drcompiler::rb::collectAccumulators(inner);
  if (accs.size() != 1)
    return std::nullopt;
  Acc &a = accs[0];
  // Every other effectful op in the body must be a load (cloned verbatim);
  // a second store or a call would be reordered illegally by the rebuild.
  for (Operation &op : inner.getBody()->without_terminator()) {
    if (&op == a.load.getOperation() || &op == a.store.getOperation())
      continue;
    if (isMemoryEffectFree(&op) || isa<AffineLoadOp>(op))
      continue;
    return std::nullopt;
  }
  // Grow the band upward through perfectly-nesting loops whose IV the
  // accumulator address does not depend on (the reduction band demote
  // created, or the conv splitter's scalar border/tail clones).
  SmallVector<AffineForOp> band{inner};
  AffineForOp p = inner->getParentOfType<AffineForOp>();
  while (p && p.getNumResults() == 0 &&
         drcompiler::rb::onlyChildFor(p) == band.front() &&
         !drcompiler::rb::addrDependsOnIV(a.store, p.getInductionVar())) {
    band.insert(band.begin(), p);
    p = p->getParentOfType<AffineForOp>();
  }
  // The accumulator address must be computable BEFORE the band.
  for (Value o : a.operands)
    if (!availableAbove(o, band.front()))
      return std::nullopt;
  return Match(std::move(band), std::move(a));
}

/// Rebuild `m.band` carrying the accumulator as a scalar iter_arg through
/// every level; load once before, store once after, erase the old band.
static void promoteBand(Match &m, OpBuilder &b) {
  AffineForOp outer = m.band.front();
  Acc &a = m.acc;
  Location loc = outer.getLoc();

  b.setInsertionPoint(outer);
  Value init = b.create<AffineLoadOp>(a.loc, a.memref, a.map, a.operands);

  std::function<Value(unsigned, Value, IRMapping &, OpBuilder &)> build =
      [&](unsigned lvl, Value seed, IRMapping &remap, OpBuilder &bld) -> Value {
    AffineForOp old = m.band[lvl];
    SmallVector<Value> lbOps = llvm::to_vector(old.getLowerBoundOperands());
    SmallVector<Value> ubOps = llvm::to_vector(old.getUpperBoundOperands());
    for (Value &v : lbOps)
      v = remap.lookupOrDefault(v);
    for (Value &v : ubOps)
      v = remap.lookupOrDefault(v);
    auto nf = bld.create<AffineForOp>(
        loc, lbOps, old.getLowerBoundMap(), ubOps, old.getUpperBoundMap(),
        old.getStepAsInt(), ValueRange{seed},
        [&](OpBuilder &b2, Location l2, Value iv, ValueRange args) {
          IRMapping rm = remap;
          rm.map(old.getInductionVar(), iv);
          Value res;
          if (lvl + 1 < m.band.size())
            res = build(lvl + 1, args[0], rm, b2);
          else {
            // Innermost: clone the body, the accumulator load becoming the
            // iter_arg and its store becoming the yield.
            rm.map(a.load.getResult(), args[0]);
            for (Operation &op : old.getBody()->without_terminator())
              if (&op != a.load.getOperation() && &op != a.store.getOperation())
                b2.clone(op, rm);
            res = rm.lookupOrDefault(a.storedVal);
          }
          b2.create<AffineYieldOp>(l2, res);
        });
    return nf.getResult(0);
  };
  IRMapping remap;
  Value result = build(0, init, remap, b);
  b.create<AffineStoreOp>(a.loc, result, a.memref, a.map, a.operands);
  outer.erase();
}

struct DrScalarReductionPromotePass
    : public impl::DrScalarReductionPromotePassBase<
          DrScalarReductionPromotePass> {

  void runOnOperation() override {
    func::FuncOp fn = getOperation();
    if (fn.isExternal())
      return;
    // Collect first: the rewrite erases loops mid-walk.
    SmallVector<Match> matches;
    fn.walk([&](AffineForOp inner) {
      if (std::optional<Match> m = matchBand(inner))
        matches.push_back(std::move(*m));
    });
    OpBuilder b(fn.getContext());
    for (Match &m : matches) {
      if (emitRationale)
        m.band.front().emitRemark("promoting ")
            << m.band.size() << "-level memref accumulator to iter_args";
      promoteBand(m, b);
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createDrScalarReductionPromotePass() {
  return std::make_unique<DrScalarReductionPromotePass>();
}
