//===- ScalarReductionDemote.cpp - iter_args -> memref accumulator -------===//
//
// onnx-mlir lowers every contraction (matmul, 1x1 conv) to an SSA reduction
// carried in iter_args:
//
//   %r = affine.for %k iter_args(%a = %c0) -> f32 {
//          %p = arith.mulf (load A)(load B)
//          %s = arith.addf %a, %p
//          affine.yield %s }
//   affine.store %r, %C[i, j]
//
// The affine-register-block vector micro-kernel instead matches the cgeist -O0
// shape: a PERFECT reduction band whose accumulator lives in memory (load %C
// before the reduction, store after, body = just the reduction).  An iter_args
// loop never touches %C inside the band, so the matcher finds no accumulator
// and bails -- which is exactly why the v4 codegen was a no-op on onnx-mlir
// output.
//
// This pass fissions a single-level add-reduction into the two-nest memref form
// the matcher wants:
//
//   affine.for <spatial> { affine.store %init, %C[i, j] }              // init
//   affine.for <spatial> { affine.for %k {                            // reduce
//       %c = affine.load %C[i, j]
//       %p = arith.mulf ...; %s = arith.addf %c, %p
//       affine.store %s, %C[i, j] } }
//
// The reduction band is now perfect, so register-block vectorizes it (measured
// ~9x on resnet50's 1x1 convs).  The init store sits in its OWN spatial nest so
// the reduction band stays perfect (a leading store in the band body fails the
// vectorizer's perfect-body check -- the reason a naive in-place de-promotion
// only reaches the slower scalar/SLP path).
//
// v1 scope: single-level f32/f64 `addf` reductions whose result is stored
// directly.  Nested conv reductions (3x3: the inner addf feeds an outer loop's
// yield, not a store) and non-add chains (maxpool/softmax `maxnumf`) fail the
// match and are left intact -- they are WP-O2 / out of scope.
//
//===----------------------------------------------------------------------===//

#include "drcompiler/Transforms/ScalarReductionDemote.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "dr-scalar-reduction-demote"

namespace mlir {
#define GEN_PASS_DEF_DRSCALARREDUCTIONDEMOTEPASS
#include "drcompiler/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;

namespace {

/// A matched add-reduction (single-level GEMM, or a nested ic/kh/kw conv band)
/// ready to demote.
struct Match {
  AffineForOp red;        // the OUTERMOST iter_args reduction loop (result stored)
  AffineStoreOp store;    // the store consuming the band's result (the accumulator)
  Value init;             // the value the init nest stores (seed, or bias for Case B)
  arith::AddFOp addf;     // the INNERMOST reduction add
  arith::AddFOp epilogue; // Case B: the `band + bias` add to fold away (else null)
  Operation *epiBiasDef;  // Case B: the bias def to clone into the init nest (or null)
  SmallVector<AffineForOp> redBand; // reduction loops outer->inner (1 = GEMM, 3 = 3x3 conv)
  SmallVector<AffineForOp> spatial; // enclosing perfect spatial loops, outer->inner
};

/// True if `v` is defined outside `loop`'s region (loop-invariant to it).
static bool definedOutside(Value v, AffineForOp loop) {
  Operation *def = v.getDefiningOp();
  if (!def)
    return !loop->isAncestor(cast<BlockArgument>(v).getOwner()->getParentOp());
  return !loop->isAncestor(def);
}

/// True if `v` is a constant +0.0 (the additive identity -- a band seeded with 0
/// can fold an additive `+ bias` epilogue into a bias-valued init).
static bool isZeroConst(Value v) {
  if (auto c = v.getDefiningOp<arith::ConstantOp>())
    if (auto f = dyn_cast<FloatAttr>(c.getValue()))
      return f.getValue().isZero();
  return false;
}

/// Match a demotable reduction rooted at `red`.  Returns false (no change) on
/// anything outside the v1 contract.
static bool matchReduction(AffineForOp red, Match &m) {
  if (red.getNumResults() != 1 || red.getInits().size() != 1)
    return false;
  Type et = red.getResult(0).getType();
  if (!et.isF32() && !et.isF64())
    return false;

  // Descend the (possibly nested) reduction band: each level is a single-iter_arg
  // loop that threads the accumulator; the innermost yields addf(acc, product).
  // GEMM/1x1-conv = one level; a 3x3 conv threads ic -> kh -> kw.
  SmallVector<AffineForOp> band;
  arith::AddFOp addf;
  for (AffineForOp cur = red;;) {
    if (cur.getNumResults() != 1 || cur.getInits().size() != 1)
      return false;
    band.push_back(cur);
    auto yield = cast<AffineYieldOp>(cur.getBody()->getTerminator());
    Value yv = yield.getOperand(0);
    if (auto a = yv.getDefiningOp<arith::AddFOp>()) {
      Value ia = cur.getRegionIterArgs()[0];
      Value other;
      if (a.getLhs() == ia)
        other = a.getRhs();
      else if (a.getRhs() == ia)
        other = a.getLhs();
      else
        return false;
      if (other == ia) // product must not reuse the accumulator
        return false;
      addf = a;
      break;
    }
    // An interior band level: its body is exactly the next reduction loop (whose
    // init is this level's iter arg) plus the yield of that loop's result.
    auto inner = yv.getDefiningOp<AffineForOp>();
    if (!inner || inner.getInits()[0] != cur.getRegionIterArgs()[0])
      return false;
    if (cur.getBody()->getOperations().size() != 2)
      return false;
    cur = inner;
    if (band.size() > 4) // bound the descent (ic/kh/kw is 3)
      return false;
  }

  // The band's result is consumed either by a store DIRECTLY (Case A) or through
  // a single additive epilogue `band + bias` whose result is stored (Case B --
  // onnx-mlir's conv+bias).  A nested reduction's inner result feeds an outer
  // yield, not a store -> fails here.
  if (!red.getResult(0).hasOneUse())
    return false;
  Operation *user = *red.getResult(0).getUsers().begin();
  AffineStoreOp store;
  arith::AddFOp epilogue;
  Operation *epiBiasDef = nullptr;
  Value initVal = red.getInits()[0];
  if (auto st = dyn_cast<AffineStoreOp>(user)) {
    if (st.getValueToStore() != red.getResult(0))
      return false;
    store = st;
    if (!definedOutside(initVal, red))
      return false;
  } else if (auto ep = dyn_cast<arith::AddFOp>(user)) {
    // Case B: fold `band + bias` by seeding the accumulator with bias and
    // dropping the add.  Correct only when the band seed is the additive
    // identity (0): then result = 0 + sum, and bias + sum = bias-seeded sum.
    if (!ep.getResult().hasOneUse() || !isZeroConst(initVal))
      return false;
    auto st = dyn_cast<AffineStoreOp>(*ep.getResult().getUsers().begin());
    if (!st || st.getValueToStore() != ep.getResult())
      return false;
    Value E = (ep.getLhs() == red.getResult(0)) ? ep.getRhs() : ep.getLhs();
    if (E == red.getResult(0))
      return false;
    // The bias must be clonable into the init nest: either it already dominates
    // the band, or it is one op (e.g. a bias load) in red's block whose own
    // operands dominate the band.
    if (Operation *Edef = E.getDefiningOp()) {
      if (Edef->getBlock() == red->getBlock()) {
        for (Value o : Edef->getOperands())
          if (!definedOutside(o, red))
            return false;
        epiBiasDef = Edef;
      } else if (!definedOutside(E, red))
        return false;
    }
    store = st;
    epilogue = ep;
    initVal = E;
  } else
    return false;
  if (store->getBlock() != red->getBlock())
    return false;

  // The accumulator address must be independent of EVERY reduction-band IV (a
  // true reduction, not a scatter).
  for (AffineForOp rl : band)
    for (Value o : store.getMapOperands())
      if (o == rl.getInductionVar())
        return false;

  // Collect the enclosing perfect spatial band: each enclosing affine.for must
  // hold exactly its single inner loop (+ terminator).  red's own block must
  // hold ONLY the band, the store, and (Case B) the epilogue add + bias def --
  // anything else would be duplicated wrongly by the init nest.
  SmallVector<AffineForOp> spatial;
  Operation *innerBody = red->getBlock()->getParentOp();
  {
    llvm::SmallPtrSet<Operation *, 4> allowed{red.getOperation(),
                                              store.getOperation()};
    if (epilogue)
      allowed.insert(epilogue.getOperation());
    if (epiBiasDef)
      allowed.insert(epiBiasDef);
    for (Operation &op : red->getBlock()->without_terminator())
      if (!allowed.count(&op))
        return false;
  }
  for (Operation *cur = red->getParentOp(); auto f = dyn_cast<AffineForOp>(cur);
       cur = cur->getParentOp()) {
    // Stop (don't bail) at the first imperfect enclosing loop -- e.g. onnx-mlir
    // puts a trip-1 group `affine.apply` between oc and oh.  The collected inner
    // loops are still a clean perfect band; the uncollected outer ones simply
    // enclose both the init nest and the reduction nest, which is fine.
    if (!spatial.empty() && f.getBody()->getOperations().size() != 2)
      break;
    spatial.push_back(f);
  }
  (void)innerBody;
  if (spatial.empty())
    return false;
  std::reverse(spatial.begin(), spatial.end()); // outer -> inner

  m.red = red;
  m.store = store;
  m.init = initVal;
  m.addf = addf;
  m.epilogue = epilogue;
  m.epiBiasDef = epiBiasDef;
  m.redBand = std::move(band);
  m.spatial = std::move(spatial);
  return true;
}

/// Emit a fresh spatial nest mirroring `m.spatial`, whose innermost body stores
/// `m.init` to the accumulator memref at the original store's subscript.
/// Inserted immediately before the outermost spatial loop.
static void emitInitNest(Match &m, OpBuilder &b) {
  AffineForOp outer = m.spatial.front();
  b.setInsertionPoint(outer);
  Location loc = outer.getLoc();
  IRMapping map; // old spatial IV -> new spatial IV

  AffineForOp innermost;
  for (AffineForOp s : m.spatial) {
    SmallVector<Value> lbOps = llvm::to_vector(s.getLowerBoundOperands());
    SmallVector<Value> ubOps = llvm::to_vector(s.getUpperBoundOperands());
    for (Value &v : lbOps)
      if (Value mv = map.lookupOrNull(v))
        v = mv;
    for (Value &v : ubOps)
      if (Value mv = map.lookupOrNull(v))
        v = mv;
    auto nf = b.create<AffineForOp>(loc, lbOps, s.getLowerBoundMap(), ubOps,
                                    s.getUpperBoundMap(), s.getStepAsInt());
    map.map(s.getInductionVar(), nf.getInductionVar());
    b.setInsertionPointToStart(nf.getBody());
    innermost = nf;
  }
  // innermost body: store the seed at the (remapped) accumulator subscript.
  // Case B folds the bias epilogue here: the seed is the bias value, whose def
  // (e.g. a `load bias[oc]`) lives after the band, so it must be CLONED into the
  // init body (its operands dominate this point); plain Case A uses the seed
  // value directly (it already dominates).
  Value seed = m.init;
  if (m.epiBiasDef) {
    // Remap the bias def's operands to the init nest's IVs (e.g. a `load
    // bias[oc]` must use the init nest's oc, not the original loop's).
    Operation *cl = b.clone(*m.epiBiasDef, map);
    seed = cl->getResult(0);
  }
  SmallVector<Value> idx = llvm::to_vector(m.store.getMapOperands());
  for (Value &v : idx)
    if (Value mv = map.lookupOrNull(v))
      v = mv;
  b.create<AffineStoreOp>(loc, seed, m.store.getMemRef(), m.store.getAffineMap(),
                          idx);
}

/// Rewrite the reduction band in place to accumulate into the memref: rebuild
/// the band loops with NO iter_args, and in the innermost body load / add /
/// store the accumulator.  For a 3x3 conv the whole ic/kh/kw nest is rebuilt.
static void demoteReduction(Match &m, OpBuilder &b) {
  AffineForOp outer = m.redBand.front();
  AffineForOp innerOld = m.redBand.back();
  Location loc = outer.getLoc();
  Value mem = m.store.getMemRef();
  AffineMap idxMap = m.store.getAffineMap();
  SmallVector<Value> idx = llvm::to_vector(m.store.getMapOperands());

  // Rebuild each band level as a plain loop, nesting inward; map old IV -> new.
  b.setInsertionPoint(outer);
  IRMapping map;
  for (AffineForOp rl : m.redBand) {
    auto nf = b.create<AffineForOp>(
        loc, rl.getLowerBoundOperands(), rl.getLowerBoundMap(),
        rl.getUpperBoundOperands(), rl.getUpperBoundMap(), rl.getStepAsInt());
    map.map(rl.getInductionVar(), nf.getInductionVar());
    b.setInsertionPointToStart(nf.getBody());
  }
  // Innermost body: load the accumulator (subscripts are the live spatial IVs),
  // clone the product DAG (acc -> loaded), store the sum back.
  auto c = b.create<AffineLoadOp>(loc, mem, idxMap, idx);
  map.map(innerOld.getRegionIterArgs()[0], c.getResult());
  auto innerYield = cast<AffineYieldOp>(innerOld.getBody()->getTerminator());
  for (Operation &op : innerOld.getBody()->without_terminator())
    b.clone(op, map);
  Value acc = map.lookupOrDefault(innerYield.getOperand(0));
  b.create<AffineStoreOp>(loc, acc, mem, idxMap, idx);

  // The original store of the band's result is now redundant; the accumulator
  // already holds the final value after the band.  For Case B the store fed off
  // the bias-add epilogue, which is folded into the init nest -- erase the store,
  // the dead epilogue add, and its now-dead bias def.
  m.store.erase();
  if (m.epilogue) {
    m.epilogue.erase();
    if (m.epiBiasDef && m.epiBiasDef->use_empty())
      m.epiBiasDef->erase();
  }
  outer.erase();
}

struct DrScalarReductionDemotePass
    : public impl::DrScalarReductionDemotePassBase<DrScalarReductionDemotePass> {

  void runOnOperation() override {
    func::FuncOp fn = getOperation();
    if (fn.isExternal())
      return;

    // Collect matches first; the rewrite erases/creates loops, so mutating
    // during the walk is unsafe.
    SmallVector<Match> matches;
    fn.walk([&](AffineForOp red) {
      Match m;
      if (matchReduction(red, m))
        matches.push_back(std::move(m));
    });

    OpBuilder b(fn.getContext());
    for (Match &m : matches) {
      if (emitRationale)
        m.red.emitRemark("demoting ")
            << m.redBand.size() << "-level add-reduction to memref accumulator";
      emitInitNest(m, b);
      demoteReduction(m, b);
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createDrScalarReductionDemotePass() {
  return std::make_unique<DrScalarReductionDemotePass>();
}
