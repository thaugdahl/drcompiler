//===- AffineStencilTimeTile.cpp - skewed time-tiling for stencils --------===//
//
// Time-tiles ping-pong stencils (PolyBench jacobi-1d/2d, heat-3d at -O0):
//
//     for t { nest1: B <- f(A);  nest2: A <- g(B) }
//
// where both nests are perfect constant-bound bands over the same space and
// every load offset is within +/-1 of the store point (halo 1).  One time
// step streams both whole arrays; once they exceed the cache, every step
// re-misses everything.  Tiling t alone is illegal (the halo couples
// neighbouring points across steps) -- the classical fix is skewing.
//
// Schedule derivation (the legality argument, kept here on purpose):
// give each half-step a virtual time tau = 2t + phase (phase 0 = nest1,
// 1 = nest2) and skew every space dim by tau: i'_k = i_k + tau.  Every
// dependence (an update at tau reads points at tau-1 within halo 1) then
// has distance vector (Delta tau, Delta i'_k) = (1, 1 + delta_k) with
// delta_k in {-1,0,1}, i.e. ALL components >= 0: the (tau, i'_1..i'_d)
// band is fully permutable, so rectangular tiling of the skewed space with
// lexicographic tile order is legal (Irigoin/Triolet, Wolf/Lam).
//
// The tile body is emitted in ORIGINAL coordinates -- the skew only shapes
// the per-(tile, t) iteration windows, which become affine max/min bounds:
//
//   for tt step Tt:                       // time tile
//     for ii_k step Ts (skewed range):    // space tiles, per dim
//       for t = tt .. min(tt+Tt, T):
//         nest1, dim k in [max(lo_k, ii_k - 2t),   min(hi_k, ii_k+Ts - 2t))
//         nest2, dim k in [max(lo_k, ii_k - 2t-1), min(hi_k, ii_k+Ts - 2t-1))
//
// (windows = the tile's slice of skewed space translated back by tau, so
// for fixed tau the ii_k windows partition the space exactly: every (t, i)
// body instance runs exactly once, whole -- no FP op is reordered within a
// point update, and per-point update order is preserved, so the result is
// BIT-IDENTICAL.)  Bounds hang off real IVs via maps (no affine.apply
// between loops): the shape is unroll-jam-proof by construction.
//
//===----------------------------------------------------------------------===//

#include "drcompiler/Analysis/MachineModel.h"
#include "drcompiler/Transforms/AffineStencilTimeTile.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
#define GEN_PASS_DEF_DRAFFINESTENCILTIMETILEPASS
#include "drcompiler/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "dr-affine-stencil-time-tile"

using namespace mlir;
using affine::AffineForOp;
using affine::AffineLoadOp;
using affine::AffineStoreOp;

namespace {

/// A matched ping-pong stencil step: one perfect constant-bound space nest.
struct SpaceNest {
  AffineForOp root;
  SmallVector<AffineForOp, 3> loops; // outer..inner
  Value readArr, writeArr;
};

/// If `forOp` roots a perfect nest of constant-bound unit-step loops whose
/// innermost body is straight-line loads/stores, fill `nest` and return true.
static bool matchSpaceNest(AffineForOp forOp, SpaceNest &nest) {
  nest.root = forOp;
  nest.loops.clear();
  AffineForOp cur = forOp;
  while (true) {
    if (!cur.hasConstantLowerBound() || !cur.hasConstantUpperBound() ||
        cur.getStepAsInt() != 1)
      return false;
    nest.loops.push_back(cur);
    if (nest.loops.size() > 3)
      return false;
    AffineForOp child;
    bool sawNonLoop = false;
    for (Operation &op : cur.getBody()->without_terminator()) {
      if (auto f = dyn_cast<AffineForOp>(&op)) {
        if (child)
          return false; // two sibling loops
        child = f;
      } else {
        sawNonLoop = true;
      }
    }
    if (!child) {
      // Innermost: the non-loop ops are the stencil body, validated by
      // classifyBody.
      return true;
    }
    if (sawNonLoop)
      return false; // imperfect level
    cur = child;
  }
}

/// Check the innermost body of `nest`: every subscript of every access must
/// be `IV_k + c` with |c| <= 1 (c == 0 for stores), loads all from one
/// memref, stores all to one DIFFERENT memref, nothing else effectful.
/// Records the two arrays.
static bool classifyBody(SpaceNest &nest) {
  Block *body = nest.loops.back().getBody();
  Value readArr, writeArr;
  auto subscriptsOk = [&](AffineMap map, ValueRange ops, bool isStore) {
    if (map.getNumResults() != nest.loops.size() ||
        map.getNumSymbols() != 0 || ops.size() != map.getNumDims())
      return false;
    for (unsigned d = 0, e = map.getNumResults(); d < e; ++d) {
      AffineExpr expr = map.getResult(d);
      int64_t c = 0;
      AffineExpr dimExpr = expr;
      if (auto bin = dyn_cast<AffineBinaryOpExpr>(expr)) {
        if (bin.getKind() != AffineExprKind::Add)
          return false;
        auto cst = dyn_cast<AffineConstantExpr>(bin.getRHS());
        if (!cst)
          return false;
        c = cst.getValue();
        dimExpr = bin.getLHS();
      }
      auto dim = dyn_cast<AffineDimExpr>(dimExpr);
      if (!dim)
        return false;
      // Subscript d must use space IV d (same order as the nest).
      if (ops[dim.getPosition()] != nest.loops[d].getInductionVar())
        return false;
      if (isStore ? c != 0 : (c < -1 || c > 1))
        return false;
    }
    return true;
  };
  for (Operation &op : body->without_terminator()) {
    if (auto ld = dyn_cast<AffineLoadOp>(&op)) {
      if (!subscriptsOk(ld.getAffineMap(), ld.getMapOperands(), false))
        return false;
      if (readArr && ld.getMemRef() != readArr)
        return false;
      readArr = ld.getMemRef();
    } else if (auto st = dyn_cast<AffineStoreOp>(&op)) {
      if (!subscriptsOk(st.getAffineMap(), st.getMapOperands(), true))
        return false;
      if (writeArr && st.getMemRef() != writeArr)
        return false;
      writeArr = st.getMemRef();
    } else if (!isMemoryEffectFree(&op)) {
      return false;
    }
  }
  if (!readArr || !writeArr || readArr == writeArr)
    return false;
  nest.readArr = readArr;
  nest.writeArr = writeArr;
  return true;
}

class DrAffineStencilTimeTilePass final
    : public impl::DrAffineStencilTimeTilePassBase<
          DrAffineStencilTimeTilePass> {
public:
  using DrAffineStencilTimeTilePassBase<
      DrAffineStencilTimeTilePass>::DrAffineStencilTimeTilePassBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Cache hierarchy from the single source of truth (MachineModel,
    // COSTMODEL_V4_SPEC §2): JSON file unless a CLI option was set explicitly.
    {
      drcompiler::MachineModel mm =
          drcompiler::MachineModel::fromJson(cpuCostModelFile);
      if (!l3Size.hasValue())
        l3Size = static_cast<unsigned>(mm.l3Size);
      if (!llcSharers.hasValue())
        llcSharers = mm.llcSharers;
    }

    SmallVector<AffineForOp> tLoops, fdtdLoops;
    func.walk([&](AffineForOp t) {
      if (t->getParentOfType<AffineForOp>())
        return; // only top-level time loops
      SmallVector<AffineForOp, 4> nests;
      for (Operation &op : t.getBody()->without_terminator()) {
        if (auto f = dyn_cast<AffineForOp>(&op))
          nests.push_back(f);
        else if (!isMemoryEffectFree(&op))
          return;
      }
      if (nests.size() == 2)
        tLoops.push_back(t);
      else if (nests.size() == 4)
        fdtdLoops.push_back(t); // 4-phase stencil (fdtd-2d)
    });

    for (AffineForOp t : tLoops)
      (void)timeTile(t);
    for (AffineForOp t : fdtdLoops)
      (void)fdtdTimeTile(t);
  }

private:
  LogicalResult timeTile(AffineForOp t) {
    if (!t.hasConstantLowerBound() || !t.hasConstantUpperBound() ||
        t.getStepAsInt() != 1)
      return failure();
    // The classifier above only guarantees exactly 2 AffineForOp children
    // among possibly-other memory-effect-free ops in t's body (e.g. a stray
    // effect-free scalar op) -- re-filter here instead of blindly casting
    // every op, which crashed on non-stencil kernels (e.g. symm) that
    // happen to match the "2 nests" shape by coincidence.
    SmallVector<AffineForOp, 2> nestRoots;
    for (Operation &op : t.getBody()->without_terminator())
      if (auto f = dyn_cast<AffineForOp>(&op))
        nestRoots.push_back(f);
    if (nestRoots.size() != 2)
      return failure();

    SpaceNest n1, n2;
    if (!matchSpaceNest(nestRoots[0], n1) || !classifyBody(n1) ||
        !matchSpaceNest(nestRoots[1], n2) || !classifyBody(n2))
      return failure();
    // Ping-pong: nest2 consumes what nest1 produced and refreshes nest1's
    // input.  Identical space bands (same depth + bounds).
    if (n1.writeArr != n2.readArr || n1.readArr != n2.writeArr)
      return failure();
    unsigned d = n1.loops.size();
    if (n2.loops.size() != d)
      return failure();
    SmallVector<int64_t, 3> slo(d), shi(d);
    for (unsigned k = 0; k < d; ++k) {
      slo[k] = n1.loops[k].getConstantLowerBound();
      shi[k] = n1.loops[k].getConstantUpperBound();
      if (n2.loops[k].getConstantLowerBound() != slo[k] ||
          n2.loops[k].getConstantUpperBound() != shi[k])
        return failure();
      if (shi[k] <= slo[k])
        return failure();
    }

    // Profitability: time-tiling pays only when one step's working set
    // (both arrays) overflows the cache share we can count on; below that
    // the t loop already runs cache-resident and the skewed windows only
    // add min/max overhead (Polly's jacobi-1d 0.34x is this mistake).
    if (!forceTile) {
      int64_t elemBytes = 8;
      if (auto mt = dyn_cast<MemRefType>(n1.readArr.getType()))
        if (mt.getElementType().isIntOrFloat())
          elemBytes = std::max<int64_t>(
              1, (int64_t)mt.getElementType().getIntOrFloatBitWidth() / 8);
      int64_t pts = 1;
      for (unsigned k = 0; k < d; ++k)
        pts *= shi[k] - slo[k];
      int64_t stepBytes = 2 * pts * elemBytes;
      int64_t effLLC =
          drcompiler::MachineModel::effectiveLLC(l3Size, llcSharers);
      if (stepBytes <= effLLC)
        return failure();
    }

    int64_t tlo = t.getConstantLowerBound(), thi = t.getConstantUpperBound();
    // tile-t = 0: measured XL optima (plateaus): d=2 -> 128 (jacobi-2d
    // 2.9x, flat 64..128), d=3 -> 24 (heat-3d 2.08x; 32 already degrades
    // because the skew slack 2*Tt eats the whole cache-derived strip).
    int64_t Tt = tileT ? std::max<int64_t>(1, tileT)
                       : (d == 3 ? 24 : (d == 2 ? 128 : 32));
    // tile-s = 0: derive the strip width from the cache so one tile's
    // working set (both arrays over a (Ts + 2*Tt)^d skew-extended box)
    // stays within half the effective LLC.
    int64_t Ts = tileS;
    if (Ts == 0) {
      int64_t effLLC =
          drcompiler::MachineModel::effectiveLLC(l3Size, llcSharers);
      double box = (double)effLLC / 2.0 / 16.0; // 2 arrays x 8 B
      double side = std::pow(box, 1.0 / (double)d);
      Ts = (int64_t)side - 2 * Tt;
    }
    Ts = std::max<int64_t>(8, Ts);

    MLIRContext *ctx = t.getContext();
    Location loc = t.getLoc();
    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(t);

    AffineExpr d0 = getAffineDimExpr(0, ctx), d1 = getAffineDimExpr(1, ctx);

    // for tt = tlo .. thi step Tt
    auto ttL = rewriter.create<AffineForOp>(loc, tlo, thi, Tt);
    rewriter.setInsertionPointToStart(ttL.getBody());
    // Space tile loops over the BAND-LOCAL skewed ranges
    // [slo_k + 2*tt, shi_k + 2*(tt + Tt)): within one tt band the skew
    // offset only spans 2*Tt, so this covers every window the band can
    // reach while avoiding the (whole-2T-extension)^d grid of empty tiles
    // that otherwise dominates at d >= 3.
    SmallVector<AffineForOp, 3> iiLs;
    for (unsigned k = 0; k < d; ++k) {
      AffineMap iiLb =
          AffineMap::get(1, 0, getAffineConstantExpr(slo[k], ctx) + 2 * d0);
      AffineMap iiUb = AffineMap::get(
          1, 0, getAffineConstantExpr(shi[k], ctx) + 2 * d0 + 2 * Tt);
      auto iiL = rewriter.create<AffineForOp>(
          loc, ValueRange{ttL.getInductionVar()}, iiLb,
          ValueRange{ttL.getInductionVar()}, iiUb, Ts);
      rewriter.setInsertionPointToStart(iiL.getBody());
      iiLs.push_back(iiL);
    }
    // for t' = tt .. min(tt+Tt, thi)
    AffineMap tLb = AffineMap::get(1, 0, d0);
    AffineMap tUb = AffineMap::get(1, 0, {d0 + Tt, getAffineConstantExpr(thi, ctx)}, ctx);
    auto tL = rewriter.create<AffineForOp>(
        loc, ValueRange{ttL.getInductionVar()}, tLb,
        ValueRange{ttL.getInductionVar()}, tUb, 1);
    rewriter.setInsertionPointToStart(tL.getBody());

    // Clone one phase nest with windowed bounds; phase 0 = nest1 (tau = 2t),
    // phase 1 = nest2 (tau = 2t + 1).
    auto emitPhase = [&](SpaceNest &n, int64_t phase) {
      auto cloned = cast<AffineForOp>(rewriter.clone(*n.root.getOperation()));
      // Rebind each level's bounds:
      //   lb = max(slo_k, ii_k - 2t - phase)
      //   ub = min(shi_k, ii_k + Ts - 2t - phase)
      AffineForOp cur = cloned;
      for (unsigned k = 0; k < d; ++k) {
        AffineMap lb = AffineMap::get(
            2, 0, {getAffineConstantExpr(slo[k], ctx), d0 - 2 * d1 - phase},
            ctx);
        AffineMap ub = AffineMap::get(
            2, 0,
            {getAffineConstantExpr(shi[k], ctx), d0 + Ts - 2 * d1 - phase},
            ctx);
        SmallVector<Value, 2> ops{iiLs[k].getInductionVar(),
                                  tL.getInductionVar()};
        cur.setLowerBound(ops, lb);
        cur.setUpperBound(ops, ub);
        if (k + 1 < d) {
          for (Operation &op : cur.getBody()->without_terminator())
            if (auto f = dyn_cast<AffineForOp>(&op)) {
              cur = f;
              break;
            }
        }
      }
      // Re-map any use of the original t IV inside the clone.
      cloned->walk([&](Operation *op) {
        for (OpOperand &o : op->getOpOperands())
          if (o.get() == t.getInductionVar())
            o.set(tL.getInductionVar());
      });
      rewriter.setInsertionPointAfter(cloned);
    };
    emitPhase(n1, 0);
    emitPhase(n2, 1);

    rewriter.eraseOp(t);
    return success();
  }

  // fdtd-2d (COSTMODEL_V4_SPEC §4.3): a time loop with FOUR phases over three
  // 2-D arrays + a 1-D source.  All inter-phase dependences have virtual-time
  // distance >= 1 at halo <= 1, so the tau-ONLY skew (i'=i+tau, j'=j+tau with
  // tau = P*t + phase, P=4, c=0) makes every distance non-negative -- no new
  // schedule math vs jacobi, only (a) per-phase space bands and (b) the 1-D
  // border phase modelled as a degenerate 2-D phase with i-band [row, row+1).
  // fdtd is memory-bound (arithmetic intensity ~0.15 flop/byte) with no
  // intra-phase recurrence, so the inner loops vectorize and time-tiling cuts
  // real byte traffic.  Emitted in ORIGINAL coordinates; windows partition the
  // skewed space exactly -> BIT-identical.
  LogicalResult fdtdTimeTile(AffineForOp t) {
    if (!t.hasConstantLowerBound() || !t.hasConstantUpperBound() ||
        t.getStepAsInt() != 1)
      return failure();
    SmallVector<AffineForOp, 4> roots;
    for (Operation &op : t.getBody()->without_terminator())
      if (auto f = dyn_cast<AffineForOp>(&op))
        roots.push_back(f);
    if (roots.size() != 4)
      return failure();

    // A subscript is `IV + c`; record (which IV, c).  Returns false on any
    // non-affine / symbol / multi-dim subscript.
    auto decode = [&](AffineExpr e, ValueRange ops, Value &iv,
                      int64_t &c) -> bool {
      c = 0;
      AffineExpr dim = e;
      if (auto bin = dyn_cast<AffineBinaryOpExpr>(e)) {
        if (bin.getKind() != AffineExprKind::Add)
          return false;
        auto cst = dyn_cast<AffineConstantExpr>(bin.getRHS());
        if (!cst)
          return false;
        c = cst.getValue();
        dim = bin.getLHS();
      }
      if (auto cst = dyn_cast<AffineConstantExpr>(dim)) {
        iv = nullptr;
        c = cst.getValue();
        return true;
      }
      auto dd = dyn_cast<AffineDimExpr>(dim);
      if (!dd)
        return false;
      iv = ops[dd.getPosition()];
      return true;
    };

    struct Phase {
      bool border;
      AffineForOp iLoop, jLoop; // iLoop null for border
      int64_t slo[2], shi[2];   // per-dim space band
      Block *body;
    };
    SmallVector<Phase, 4> ph;
    int nBorder = 0;
    for (AffineForOp r : roots) {
      SpaceNest n;
      if (!matchSpaceNest(r, n))
        return failure();
      Phase x;
      if (n.loops.size() == 2) {
        // 2-D phase: store at (i,j) offset 0, every i/j-indexed load halo <= 1.
        x.border = false;
        x.iLoop = n.loops[0];
        x.jLoop = n.loops[1];
        x.body = x.jLoop.getBody();
        Value iIV = x.iLoop.getInductionVar(), jIV = x.jLoop.getInductionVar();
        AffineStoreOp store;
        for (Operation &op : x.body->without_terminator()) {
          if (auto ld = dyn_cast<AffineLoadOp>(&op)) {
            if (ld.getAffineMap().getNumResults() != 2)
              return failure();
            for (unsigned d = 0; d < 2; ++d) {
              Value iv;
              int64_t c;
              if (!decode(ld.getAffineMap().getResult(d), ld.getMapOperands(),
                          iv, c))
                return failure();
              if ((iv == iIV || iv == jIV) && (c < -1 || c > 1))
                return failure(); // halo > 1 -> tau-only skew illegal
            }
          } else if (auto st = dyn_cast<AffineStoreOp>(&op)) {
            if (store || st.getAffineMap().getNumResults() != 2)
              return failure();
            Value iv0, iv1;
            int64_t c0, c1;
            if (!decode(st.getAffineMap().getResult(0), st.getMapOperands(),
                        iv0, c0) ||
                !decode(st.getAffineMap().getResult(1), st.getMapOperands(),
                        iv1, c1))
              return failure();
            if (iv0 != iIV || iv1 != jIV || c0 != 0 || c1 != 0)
              return failure();
            store = st;
          } else if (!isMemoryEffectFree(&op)) {
            return failure();
          }
        }
        if (!store)
          return failure();
        x.slo[0] = x.iLoop.getConstantLowerBound();
        x.shi[0] = x.iLoop.getConstantUpperBound();
        x.slo[1] = x.jLoop.getConstantLowerBound();
        x.shi[1] = x.jLoop.getConstantUpperBound();
      } else if (n.loops.size() == 1) {
        // 1-D border: for j { A[row][j] = f(...) } -- row is a constant.
        x.border = true;
        ++nBorder;
        x.iLoop = nullptr;
        x.jLoop = n.loops[0];
        x.body = x.jLoop.getBody();
        Value jIV = x.jLoop.getInductionVar();
        AffineStoreOp store;
        for (Operation &op : x.body->without_terminator())
          if (auto st = dyn_cast<AffineStoreOp>(&op)) {
            if (store || st.getAffineMap().getNumResults() != 2)
              return failure();
            Value iv0, iv1;
            int64_t row, c1;
            if (!decode(st.getAffineMap().getResult(0), st.getMapOperands(),
                        iv0, row) ||
                !decode(st.getAffineMap().getResult(1), st.getMapOperands(),
                        iv1, c1))
              return failure();
            if (iv0 != nullptr || iv1 != jIV || c1 != 0)
              return failure(); // first index must be a constant row
            store = st;
            x.slo[0] = row;
            x.shi[0] = row + 1;
          }
        if (!store)
          return failure();
        x.slo[1] = x.jLoop.getConstantLowerBound();
        x.shi[1] = x.jLoop.getConstantUpperBound();
      } else {
        return failure();
      }
      ph.push_back(x);
    }
    if (nBorder != 1)
      return failure();

    const int64_t P = 4;
    int64_t minSlo[2] = {INT64_MAX, INT64_MAX}, maxShi[2] = {INT64_MIN,
                                                             INT64_MIN};
    for (auto &x : ph)
      for (int k = 0; k < 2; ++k) {
        minSlo[k] = std::min(minSlo[k], x.slo[k]);
        maxShi[k] = std::max(maxShi[k], x.shi[k]);
      }

    // Profitability: time-tile only when one full sweep (all distinct arrays)
    // overflows the effective LLC.  Estimate arrays touched = 3 (ex/ey/hz).
    if (!forceTile) {
      int64_t pts = (maxShi[0] - minSlo[0]) * (maxShi[1] - minSlo[1]);
      int64_t sweepBytes = 3 * pts * 8;
      int64_t effLLC =
          drcompiler::MachineModel::effectiveLLC(l3Size, llcSharers);
      if (sweepBytes <= effLLC)
        return failure();
    }

    int64_t tlo = t.getConstantLowerBound(), thi = t.getConstantUpperBound();
    // Measured XL optima (fdtd-2d 2000x2600x1000): Tt=16, Ts=64 -> 2.51x.
    // The skew slack is P*Tt = 4*Tt per dim (twice jacobi's), so the tile must
    // stay L2-resident across the band -- a SMALL Ts (~64) beats the L3-derived
    // strip (Ts~780 gave only 1.86x: the per-tile working set then fits only the
    // shared L3, not the private L2, so there is far less reuse).  Hardcoded as
    // the jacobi emitter does its own measured optima; override with tile-t/-s.
    int64_t Tt = tileT ? std::max<int64_t>(1, tileT) : 16;
    int64_t Ts = tileS ? tileS : 64;
    Ts = std::max<int64_t>(8, Ts);

    MLIRContext *ctx = t.getContext();
    Location loc = t.getLoc();
    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(t);
    AffineExpr d0 = getAffineDimExpr(0, ctx), d1 = getAffineDimExpr(1, ctx);
    auto C = [&](int64_t v) { return getAffineConstantExpr(v, ctx); };

    // tt = tlo .. thi step Tt
    auto ttL = rewriter.create<AffineForOp>(loc, tlo, thi, Tt);
    rewriter.setInsertionPointToStart(ttL.getBody());
    Value tt = ttL.getInductionVar();
    // ii = minSloI + P*tt .. maxShiI + P*tt + P*Tt + P step Ts (union band)
    auto iiL = rewriter.create<AffineForOp>(
        loc, ValueRange{tt}, AffineMap::get(1, 0, C(minSlo[0]) + P * d0),
        ValueRange{tt}, AffineMap::get(1, 0, C(maxShi[0] + P * Tt + P) + P * d0),
        Ts);
    rewriter.setInsertionPointToStart(iiL.getBody());
    Value ii = iiL.getInductionVar();
    auto jjL = rewriter.create<AffineForOp>(
        loc, ValueRange{tt}, AffineMap::get(1, 0, C(minSlo[1]) + P * d0),
        ValueRange{tt}, AffineMap::get(1, 0, C(maxShi[1] + P * Tt + P) + P * d0),
        Ts);
    rewriter.setInsertionPointToStart(jjL.getBody());
    Value jj = jjL.getInductionVar();
    // t' = tt .. min(tt+Tt, thi)
    auto tL = rewriter.create<AffineForOp>(
        loc, ValueRange{tt}, AffineMap::get(1, 0, d0), ValueRange{tt},
        AffineMap::get(1, 0, {d0 + Tt, C(thi)}, ctx), 1);
    rewriter.setInsertionPointToStart(tL.getBody());
    Value tnew = tL.getInductionVar();
    Value tileIV[2] = {ii, jj};

    // Emit each phase at its virtual time tau = P*t' + p, windowed per dim:
    //   lb = max(slo_pk, tileIV_k - P*t' - p), ub = min(shi_pk, +Ts ...).
    for (auto [p, x] : llvm::enumerate(ph)) {
      OpBuilder::InsertionGuard g(rewriter);
      Value cur[2];
      for (int k = 0; k < 2; ++k) {
        AffineMap lb = AffineMap::get(
            2, 0, {C(x.slo[k]), d0 - P * d1 - (int64_t)p}, ctx);
        AffineMap ub = AffineMap::get(
            2, 0, {C(x.shi[k]), d0 + Ts - P * d1 - (int64_t)p}, ctx);
        auto L = rewriter.create<AffineForOp>(
            loc, ValueRange{tileIV[k], tnew}, lb, ValueRange{tileIV[k], tnew},
            ub, 1);
        cur[k] = L.getInductionVar();
        rewriter.setInsertionPointToStart(L.getBody());
      }
      // Clone the phase body, remapping (t, i?, j) to (tnew, cur[0], cur[1]).
      IRMapping map;
      map.map(t.getInductionVar(), tnew);
      if (!x.border)
        map.map(x.iLoop.getInductionVar(), cur[0]);
      map.map(x.jLoop.getInductionVar(), cur[1]);
      for (Operation &op : x.body->without_terminator())
        rewriter.clone(op, map);
    }

    rewriter.eraseOp(t);
    return success();
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createDrAffineStencilTimeTilePass() {
  return std::make_unique<DrAffineStencilTimeTilePass>();
}
} // namespace mlir
