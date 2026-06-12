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
// This pass walks innermost loops with one or more memory accumulators
// (same-address load/store pairs, addresses invariant across the enclosing
// perfect reduction band), and rebuilds the band carrying each accumulator as a
// scalar iter_arg threaded through every level: one load before the band, one
// store after.  N>1 (WP-G2 safety net) is the register-block mr-jam leftover --
// a band that slipped vectorization keeps `mr` same-shape accumulators in the
// innermost body; promoting all of them is what keeps a missed vectorization in
// registers instead of DRAM.
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
/// innermost body carries N independent memory accumulators, each with an
/// address invariant to every band level and available above the band.  N>1 is
/// the register-block mr-jam leftover (the WP-G2 safety net): a band that slips
/// vectorization keeps `mr` same-shape accumulators in the innermost body; each
/// becomes its own iter_arg so the band degrades to registers, not DRAM.
struct Match {
  SmallVector<AffineForOp> band; // outer -> inner
  SmallVector<Acc> accs;
  Match(SmallVector<AffineForOp> b, SmallVector<Acc> a)
      : band(std::move(b)), accs(std::move(a)) {}
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
  if (accs.empty())
    return std::nullopt;
  // Every other effectful op in the body must be a load (cloned verbatim);
  // a store or call NOT belonging to one of the accumulators would be reordered
  // illegally by the rebuild.
  llvm::SmallPtrSet<Operation *, 16> accOps;
  for (Acc &a : accs) {
    accOps.insert(a.load.getOperation());
    accOps.insert(a.store.getOperation());
  }
  for (Operation &op : inner.getBody()->without_terminator()) {
    if (accOps.contains(&op))
      continue;
    if (isMemoryEffectFree(&op) || isa<AffineLoadOp>(op))
      continue;
    return std::nullopt;
  }
  // Grow the band upward through perfectly-nesting loops whose IV NO
  // accumulator address depends on (the reduction band demote created, the conv
  // splitter's scalar border/tail clones, or an mr-jam's reduction loop).
  SmallVector<AffineForOp> band{inner};
  AffineForOp p = inner->getParentOfType<AffineForOp>();
  while (p && p.getNumResults() == 0 &&
         drcompiler::rb::onlyChildFor(p) == band.front() &&
         llvm::none_of(accs, [&](const Acc &a) {
           return drcompiler::rb::addrDependsOnIV(a.store, p.getInductionVar());
         })) {
    band.insert(band.begin(), p);
    p = p->getParentOfType<AffineForOp>();
  }
  // Every accumulator address must be computable BEFORE the band.
  for (Acc &a : accs)
    for (Value o : a.operands)
      if (!availableAbove(o, band.front()))
        return std::nullopt;
  return Match(std::move(band), std::move(accs));
}

/// Rebuild `m.band` carrying the N accumulators as scalar iter_args through
/// every level; load each once before, store each once after, erase the old
/// band.
static void promoteBand(Match &m, OpBuilder &b) {
  AffineForOp outer = m.band.front();
  Location loc = outer.getLoc();
  llvm::SmallPtrSet<Operation *, 16> accOps;
  for (Acc &a : m.accs) {
    accOps.insert(a.load.getOperation());
    accOps.insert(a.store.getOperation());
  }

  b.setInsertionPoint(outer);
  SmallVector<Value> inits;
  for (Acc &a : m.accs)
    inits.push_back(b.create<AffineLoadOp>(a.loc, a.memref, a.map, a.operands));

  std::function<SmallVector<Value>(unsigned, ValueRange, IRMapping &,
                                   OpBuilder &)>
      build = [&](unsigned lvl, ValueRange seeds, IRMapping &remap,
                  OpBuilder &bld) -> SmallVector<Value> {
    AffineForOp old = m.band[lvl];
    SmallVector<Value> lbOps = llvm::to_vector(old.getLowerBoundOperands());
    SmallVector<Value> ubOps = llvm::to_vector(old.getUpperBoundOperands());
    for (Value &v : lbOps)
      v = remap.lookupOrDefault(v);
    for (Value &v : ubOps)
      v = remap.lookupOrDefault(v);
    auto nf = bld.create<AffineForOp>(
        loc, lbOps, old.getLowerBoundMap(), ubOps, old.getUpperBoundMap(),
        old.getStepAsInt(), seeds,
        [&](OpBuilder &b2, Location l2, Value iv, ValueRange args) {
          IRMapping rm = remap;
          rm.map(old.getInductionVar(), iv);
          SmallVector<Value> res;
          if (lvl + 1 < m.band.size())
            res = build(lvl + 1, args, rm, b2);
          else {
            // Innermost: each accumulator load becomes its iter_arg; clone the
            // shared body; each accumulator's stored value becomes its yield.
            for (auto [i, a] : llvm::enumerate(m.accs))
              rm.map(a.load.getResult(), args[i]);
            for (Operation &op : old.getBody()->without_terminator())
              if (!accOps.contains(&op))
                b2.clone(op, rm);
            for (Acc &a : m.accs)
              res.push_back(rm.lookupOrDefault(a.storedVal));
          }
          b2.create<AffineYieldOp>(l2, res);
        });
    return SmallVector<Value>(nf.getResults().begin(), nf.getResults().end());
  };
  IRMapping remap;
  SmallVector<Value> results = build(0, inits, remap, b);
  for (auto [i, a] : llvm::enumerate(m.accs))
    b.create<AffineStoreOp>(a.loc, results[i], a.memref, a.map, a.operands);
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
