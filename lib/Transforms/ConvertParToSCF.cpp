//===- ConvertParToSCF.cpp - Lower `par` to scf (M2 test sink) ----------===//
//
// par.forall -> scf.parallel (constant bounds -> arith index constants)
// par.region -> inlined into the enclosing block
// par.yield  -> erased
//
// Sequential-equivalent reference lowering (PARALLEL_PAR_DIALECT_SPEC.md §4.3).
//
//===----------------------------------------------------------------------===//

#include "drcompiler/Transforms/ConvertParToSCF.h"
#include "drcompiler/Dialect/Par/IR/ParOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTPARTOSCFPASS
#include "drcompiler/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

/// Associative combinator for a par.reduce kind (see ParOps.td).
static Value combine(OpBuilder &b, Location loc, int64_t kind, Value l,
                     Value r) {
  switch (kind) {
  case 0: return b.create<arith::AddFOp>(loc, l, r);
  case 1: return b.create<arith::MulFOp>(loc, l, r);
  case 2: return b.create<arith::MaxNumFOp>(loc, l, r);
  case 3: return b.create<arith::MinNumFOp>(loc, l, r);
  case 4: return b.create<arith::AddIOp>(loc, l, r);
  case 5: return b.create<arith::MulIOp>(loc, l, r);
  case 6: return b.create<arith::MaxSIOp>(loc, l, r);
  default: return b.create<arith::MinSIOp>(loc, l, r);
  }
}

/// A reducing par.forall (1-D, carries results) -> a SEQUENTIAL scf.for with
/// iter_args: the spec's sequential reference (par->scf runs sequentially), so
/// the accumulators are threaded and combined per kind.  Exact, no FP reassoc.
static void lowerReduceForall(par::ForallOp forall) {
  OpBuilder b(forall);
  Location loc = forall.getLoc();
  Value lb = b.create<arith::ConstantIndexOp>(loc, forall.getLowerBounds()[0]);
  Value ub = forall.isDynamicUpperBound(0)
                 ? forall.getDynamicUpperBound(0)
                 : b.create<arith::ConstantIndexOp>(loc,
                                                    forall.getUpperBounds()[0])
                       .getResult();
  Value st = b.create<arith::ConstantIndexOp>(loc, forall.getSteps()[0]);
  auto red = cast<par::ReduceOp>(forall.getBody()->getTerminator());
  SmallVector<Value> inits(forall.getInitVals());
  SmallVector<int64_t> kinds(red.getKinds());

  // scf.for with iter_args via the bodyBuilder idiom (it creates the body +
  // terminator; we clone the forall body and combine into the accumulators).
  auto forOp = b.create<scf::ForOp>(
      loc, lb, ub, st, inits,
      [&](OpBuilder &nb, Location nloc, Value iv, ValueRange iterArgs) {
        IRMapping map;
        map.map(forall.getInductionVars()[0], iv);
        for (Operation &op : forall.getBody()->without_terminator())
          nb.clone(op, map);
        SmallVector<Value> yields;
        for (unsigned i = 0, e = red.getContributions().size(); i < e; ++i)
          yields.push_back(combine(nb, nloc, kinds[i], iterArgs[i],
                                   map.lookupOrDefault(red.getContributions()[i])));
        nb.create<scf::YieldOp>(nloc, yields);
      });
  forall.replaceAllUsesWith(forOp.getResults());
  forall.erase();
}

/// par.forall -> scf.parallel; the body is cloned in (only the induction
/// variables are remapped — every other operand dominates the new op).
static void lowerForall(par::ForallOp forall) {
  if (!forall.getResults().empty()) {
    lowerReduceForall(forall);
    return;
  }
  OpBuilder b(forall);
  Location loc = forall.getLoc();
  ArrayRef<int64_t> lo = forall.getLowerBounds();
  ArrayRef<int64_t> hi = forall.getUpperBounds();
  ArrayRef<int64_t> st = forall.getSteps();

  SmallVector<Value> lbs, ubs, steps;
  for (size_t d = 0, e = lo.size(); d < e; ++d) {
    lbs.push_back(b.create<arith::ConstantIndexOp>(loc, lo[d]));
    ubs.push_back(forall.isDynamicUpperBound(d)
                      ? forall.getDynamicUpperBound(d)
                      : b.create<arith::ConstantIndexOp>(loc, hi[d]).getResult());
    steps.push_back(b.create<arith::ConstantIndexOp>(loc, st[d]));
  }

  auto parOp = b.create<scf::ParallelOp>(loc, lbs, ubs, steps);
  Block *pb = parOp.getBody();
  IRMapping map;
  for (size_t d = 0, e = lo.size(); d < e; ++d)
    map.map(forall.getInductionVars()[d], pb->getArgument(d));

  b.setInsertionPoint(pb->getTerminator());
  for (Operation &op : forall.getBody()->without_terminator())
    b.clone(op, map);
  forall.erase();
}

/// Splice a single-block op's body (minus terminator) into the enclosing block
/// and erase the op.  Used for par.region and par.critical (sequential sink:
/// one worker in order == the original program order).
static void inlineSingleBlock(Operation *op, Block *body) {
  Operation *term = body->getTerminator();
  Block *parent = op->getBlock();
  parent->getOperations().splice(Block::iterator(op), body->getOperations(),
                                 body->begin(), Block::iterator(term));
  op->erase();
}

struct ConvertParToSCFPass
    : public impl::ConvertParToSCFPassBase<ConvertParToSCFPass> {
  void runOnOperation() override {
    func::FuncOp fn = getOperation();

    // Inline single-worker critical slabs (sequential sink == in-order).
    SmallVector<par::CriticalOp> crits;
    fn.walk([&](par::CriticalOp c) { crits.push_back(c); });
    for (par::CriticalOp c : crits)
      inlineSingleBlock(c, c.getBody());

    // Barriers and redistributes are no-ops under sequential execution.
    SmallVector<Operation *> syncs;
    fn.walk([&](Operation *op) {
      if (isa<par::BarrierOp, par::RedistributeOp>(op))
        syncs.push_back(op);
    });
    for (Operation *op : syncs)
      op->erase();

    SmallVector<par::ForallOp> foralls;
    fn.walk([&](par::ForallOp f) { foralls.push_back(f); });
    for (par::ForallOp f : foralls)
      lowerForall(f);

    SmallVector<par::RegionOp> regions;
    fn.walk([&](par::RegionOp r) { regions.push_back(r); });
    for (par::RegionOp r : regions)
      inlineSingleBlock(r, r.getBody());
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createConvertParToSCFPass() {
  return std::make_unique<ConvertParToSCFPass>();
}
