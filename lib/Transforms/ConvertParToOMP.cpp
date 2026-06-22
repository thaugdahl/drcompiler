//===- ConvertParToOMP.cpp - Faithful `par` -> OpenMP lowering (S3) -----===//
//
// The faithful whole-kernel SPMD lowering (PARALLEL_SPMD_SPEC.md §7, S3):
//
//   par.region        -> omp.parallel           (ONE worker team for the kernel)
//   par.forall        -> omp.wsloop { omp.loop_nest }  (static worksharing)
//     elided edge      -> `nowait` on the wsloop (no implicit barrier; the
//                         explicit boundary omp.barrier provides the sync)
//   par.barrier        -> omp.barrier            (only at non-elided edges)
//   par.redistribute   -> omp.barrier            (conservative: a reblock is at
//                         least a full barrier; data movement is Phase B)
//
// This is strictly fewer fork/joins than the `par -> scf.parallel ->
// --convert-scf-to-openmp` compose (which forks per loop): one omp.parallel for
// the whole region, barriers only where S1 kept them.  In an S2 region each
// par.forall is one maximal ELIDE-run, so consecutive foralls are always split
// by an explicit barrier/redistribute -> every wsloop but the last is `nowait`.
//
//===----------------------------------------------------------------------===//

#include "drcompiler/Transforms/ConvertParToOMP.h"
#include "drcompiler/Dialect/Par/IR/ParOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTPARTOOMPPASS
#include "drcompiler/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

/// par.forall -> omp.wsloop [nowait] { omp.loop_nest (ivs) { <body> omp.yield } }
/// Constant bounds are materialized as arith index constants in the enclosing
/// (omp.parallel) block; only the induction variables are remapped.
static void lowerForall(OpBuilder &b, par::ForallOp forall, bool nowait,
                        IRMapping &m) {
  // Reducing foralls need an omp reduction clause (S6, not yet wired); until
  // then convert-par-to-scf is the reduce reference.  Fail loudly rather than
  // silently dropping the par.reduce.
  if (!forall.getResults().empty()) {
    forall.emitError("convert-par-to-omp: reducing par.forall not yet supported "
                     "(lower via convert-par-to-scf; omp reduction is S6)");
    return;
  }
  Location loc = forall.getLoc();
  ArrayRef<int64_t> lo = forall.getLowerBounds();
  ArrayRef<int64_t> hi = forall.getUpperBounds();
  ArrayRef<int64_t> st = forall.getSteps();

  SmallVector<Value> lbs, ubs, steps;
  for (size_t d = 0, e = lo.size(); d < e; ++d) {
    lbs.push_back(b.create<arith::ConstantIndexOp>(loc, lo[d]));
    ubs.push_back(forall.isDynamicUpperBound(d)
                      ? m.lookupOrDefault(forall.getDynamicUpperBound(d))
                      : b.create<arith::ConstantIndexOp>(loc, hi[d]).getResult());
    steps.push_back(b.create<arith::ConstantIndexOp>(loc, st[d]));
  }

  auto ws = b.create<omp::WsloopOp>(loc);
  if (nowait)
    ws.setNowait(true);
  // A band tagged load-imbalanced (triangular / IV-dependent inner extent) gets
  // a dynamic schedule so workers pull rows on demand instead of a static block.
  if (forall->hasAttr("par.dynamic"))
    ws.setScheduleKind(omp::ClauseScheduleKind::Dynamic);
  Block *wsBlk = b.createBlock(&ws.getRegion());
  b.setInsertionPointToStart(wsBlk);

  auto ln = b.create<omp::LoopNestOp>(
      loc, /*collapse_num_loops=*/(uint64_t)lo.size(), lbs, ubs, steps,
      /*loop_inclusive=*/false, /*tile_sizes=*/DenseI64ArrayAttr{});
  Block *lnBlk = b.createBlock(&ln.getRegion());
  SmallVector<Type> ivTypes(lo.size(), b.getIndexType());
  SmallVector<Location> ivLocs(lo.size(), loc);
  lnBlk->addArguments(ivTypes, ivLocs);

  // Clone the body under the region map (so it resolves replicated glue) plus
  // this loop's IVs; IVs are loop-local so use a copy, not the shared map.
  IRMapping bodyMap = m;
  for (size_t d = 0, e = lo.size(); d < e; ++d)
    bodyMap.map(forall.getInductionVars()[d], lnBlk->getArgument(d));
  b.setInsertionPointToStart(lnBlk);
  for (Operation &op : forall.getBody()->without_terminator())
    b.clone(op, bodyMap);
  b.create<omp::YieldOp>(loc);
}

/// par.region -> omp.parallel { <forall/barrier/single/replicated sequence>
/// omp.terminator }.  One IRMapping threads the whole region so replicated glue
/// (cloned at region level) is visible to later foralls/singles.
static void lowerRegion(par::RegionOp region) {
  OpBuilder b(region);
  Location loc = region.getLoc();
  auto parallel = b.create<omp::ParallelOp>(loc);
  Block *pblk = b.createBlock(&parallel.getRegion());

  SmallVector<Operation *> ops;
  for (Operation &op : region.getBody()->without_terminator())
    ops.push_back(&op);

  IRMapping m;
  for (size_t i = 0, n = ops.size(); i < n; ++i) {
    b.setInsertionPointToEnd(pblk);
    Operation *op = ops[i];
    if (auto forall = dyn_cast<par::ForallOp>(op)) {
      // Suppress the implicit end-of-wsloop barrier when an explicit boundary
      // barrier/redistribute follows (an elided edge inside an S2 region).
      bool nowait = (i + 1 < n) &&
                    isa<par::BarrierOp, par::RedistributeOp>(ops[i + 1]);
      lowerForall(b, forall, nowait, m);
    } else if (isa<par::BarrierOp, par::RedistributeOp>(op)) {
      b.create<omp::BarrierOp>(loc);
    } else if (auto crit = dyn_cast<par::CriticalOp>(op)) {
      // Single worker runs the slab in order (implicit end barrier).
      auto single = b.create<omp::SingleOp>(
          loc, ValueRange{}, ValueRange{}, ValueRange{}, ArrayAttr{},
          UnitAttr{}, ValueRange{}, ArrayAttr{}, UnitAttr{});
      Block *sblk = b.createBlock(&single.getRegion());
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(sblk);
      IRMapping cmap = m;
      for (Operation &cop : crit.getBody()->without_terminator())
        b.clone(cop, cmap);
      b.create<omp::TerminatorOp>(loc);
    } else {
      Operation *cl = b.clone(*op, m); // replicated glue / control flow
      // A cloned control-flow op (e.g. a sequential scf.for time loop wrapping
      // spatial par.forall bands -- the stencil SEQWRAP shape) may hold nested
      // par.forall ops the direct-children walk above never saw.  Lower each in
      // place to an omp.wsloop WITHOUT nowait, so its implicit end barrier
      // synchronizes consecutive bands within the loop body (par.barrier cannot
      // nest under scf.for, so the implicit barrier is the only sync available).
      SmallVector<par::ForallOp> nested;
      cl->walk([&](par::ForallOp f) { nested.push_back(f); });
      for (par::ForallOp f : nested) {
        OpBuilder fb(f);
        lowerForall(fb, f, /*nowait=*/false, m);
        f.erase();
      }
    }
  }

  b.setInsertionPointToEnd(pblk);
  b.create<omp::TerminatorOp>(loc);
  region.erase();
}

struct ConvertParToOMPPass
    : public impl::ConvertParToOMPPassBase<ConvertParToOMPPass> {
  void runOnOperation() override {
    func::FuncOp fn = getOperation();
    SmallVector<par::RegionOp> regions;
    fn.walk([&](par::RegionOp r) { regions.push_back(r); });
    for (par::RegionOp r : regions)
      lowerRegion(r);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createConvertParToOMPPass() {
  return std::make_unique<ConvertParToOMPPass>();
}
