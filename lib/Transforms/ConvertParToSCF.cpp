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

/// par.forall -> scf.parallel; the body is cloned in (only the induction
/// variables are remapped — every other operand dominates the new op).
static void lowerForall(par::ForallOp forall) {
  OpBuilder b(forall);
  Location loc = forall.getLoc();
  ArrayRef<int64_t> lo = forall.getLowerBounds();
  ArrayRef<int64_t> hi = forall.getUpperBounds();
  ArrayRef<int64_t> st = forall.getSteps();

  SmallVector<Value> lbs, ubs, steps;
  for (size_t d = 0, e = lo.size(); d < e; ++d) {
    lbs.push_back(b.create<arith::ConstantIndexOp>(loc, lo[d]));
    ubs.push_back(b.create<arith::ConstantIndexOp>(loc, hi[d]));
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

/// par.region -> inline its body ops into the enclosing block.
static void inlineRegion(par::RegionOp region) {
  Block *body = region.getBody();
  Operation *term = body->getTerminator();
  Block *parent = region->getBlock();
  parent->getOperations().splice(Block::iterator(region),
                                 body->getOperations(), body->begin(),
                                 Block::iterator(term));
  region.erase();
}

struct ConvertParToSCFPass
    : public impl::ConvertParToSCFPassBase<ConvertParToSCFPass> {
  void runOnOperation() override {
    func::FuncOp fn = getOperation();

    SmallVector<par::ForallOp> foralls;
    fn.walk([&](par::ForallOp f) { foralls.push_back(f); });
    for (par::ForallOp f : foralls)
      lowerForall(f);

    SmallVector<par::RegionOp> regions;
    fn.walk([&](par::RegionOp r) { regions.push_back(r); });
    for (par::RegionOp r : regions)
      inlineRegion(r);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createConvertParToSCFPass() {
  return std::make_unique<ConvertParToSCFPass>();
}
