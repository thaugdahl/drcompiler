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

/// A matched single-level add-reduction ready to demote.
struct Match {
  AffineForOp red;        // the iter_args reduction loop
  AffineStoreOp store;    // the store consuming red's result (the accumulator)
  Value init;             // the iter_args init value (accumulator seed)
  arith::AddFOp addf;     // the reduction add inside the body
  SmallVector<AffineForOp> spatial; // enclosing perfect spatial loops, outer->inner
};

/// True if `v` is defined outside `loop`'s region (loop-invariant to it).
static bool definedOutside(Value v, AffineForOp loop) {
  Operation *def = v.getDefiningOp();
  if (!def)
    return !loop->isAncestor(cast<BlockArgument>(v).getOwner()->getParentOp());
  return !loop->isAncestor(def);
}

/// Match a demotable reduction rooted at `red`.  Returns false (no change) on
/// anything outside the v1 contract.
static bool matchReduction(AffineForOp red, Match &m) {
  if (red.getNumResults() != 1 || red.getInits().size() != 1)
    return false;
  Type et = red.getResult(0).getType();
  if (!et.isF32() && !et.isF64())
    return false;

  // Body terminator must yield an addf(acc, product) where acc is the region
  // iter arg and product is independent of it.
  auto yield = cast<AffineYieldOp>(red.getBody()->getTerminator());
  auto addf = yield.getOperand(0).getDefiningOp<arith::AddFOp>();
  if (!addf)
    return false;
  Value iterArg = red.getRegionIterArgs()[0];
  Value other;
  if (addf.getLhs() == iterArg)
    other = addf.getRhs();
  else if (addf.getRhs() == iterArg)
    other = addf.getLhs();
  else
    return false;
  // The product must not (transitively) reuse the accumulator: a genuine
  // reduction.  A defining op inside the body that is the iterArg is the only
  // way to reuse it; cheap conservative check on the immediate addf operand.
  if (other == iterArg)
    return false;

  // Single use: the result is stored directly (Case A).  A nested conv
  // reduction's result feeds an outer yield, not a store -> fails here.
  if (!red.getResult(0).hasOneUse())
    return false;
  auto store = dyn_cast<AffineStoreOp>(*red.getResult(0).getUsers().begin());
  if (!store || store.getValueToStore() != red.getResult(0))
    return false;
  if (store->getBlock() != red->getBlock())
    return false;

  // The accumulator address must be independent of the reduction IV (a true
  // reduction, not a scatter).
  Value kIV = red.getInductionVar();
  for (Value o : store.getMapOperands())
    if (o == kIV)
      return false;

  // The init must be usable in the init nest (dominates the whole band): a
  // value defined outside the reduction loop.  Constants and block args qualify.
  if (!definedOutside(red.getInits()[0], red))
    return false;

  // Collect the enclosing perfect spatial band: each enclosing affine.for must
  // hold exactly its single inner loop (+ terminator), except the innermost,
  // whose body is {red, store, terminator}.  Imperfect enclosers are not
  // cloned (conservative).
  SmallVector<AffineForOp> spatial;
  Operation *innerBody = red->getBlock()->getParentOp();
  // innermost spatial loop body must be exactly red + store + terminator
  if (red->getBlock()->getOperations().size() != 3)
    return false;
  for (Operation *cur = red->getParentOp(); auto f = dyn_cast<AffineForOp>(cur);
       cur = cur->getParentOp()) {
    if (!spatial.empty()) {
      // f must perfectly enclose the previous spatial loop.
      if (f.getBody()->getOperations().size() != 2)
        return false;
    }
    spatial.push_back(f);
  }
  (void)innerBody;
  if (spatial.empty())
    return false;
  std::reverse(spatial.begin(), spatial.end()); // outer -> inner

  m.red = red;
  m.store = store;
  m.init = red.getInits()[0];
  m.addf = addf;
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
  // innermost body: store init at the (remapped) accumulator subscript.
  SmallVector<Value> idx = llvm::to_vector(m.store.getMapOperands());
  for (Value &v : idx)
    if (Value mv = map.lookupOrNull(v))
      v = mv;
  b.create<AffineStoreOp>(loc, m.init, m.store.getMemRef(), m.store.getAffineMap(),
                          idx);
}

/// Rewrite the reduction loop in place to accumulate into the memref: replace
/// the iter_args carry with a load-before / store-after of the accumulator.
static void demoteReduction(Match &m, OpBuilder &b) {
  AffineForOp red = m.red;
  Location loc = red.getLoc();
  Value mem = m.store.getMemRef();
  AffineMap idxMap = m.store.getAffineMap();
  SmallVector<Value> idx = llvm::to_vector(m.store.getMapOperands());

  // New reduction loop with NO iter args, same bounds.
  b.setInsertionPoint(red);
  auto nf = b.create<AffineForOp>(
      loc, red.getLowerBoundOperands(), red.getLowerBoundMap(),
      red.getUpperBoundOperands(), red.getUpperBoundMap(), red.getStepAsInt());

  b.setInsertionPointToStart(nf.getBody());
  // Load the accumulator seed from memory (subscripts are the live spatial IVs).
  auto c = b.create<AffineLoadOp>(loc, mem, idxMap, idx);

  // Clone the old body (minus the yield) mapping kIV->new kIV and acc->loaded.
  IRMapping map;
  map.map(red.getInductionVar(), nf.getInductionVar());
  map.map(red.getRegionIterArgs()[0], c.getResult());
  auto yield = cast<AffineYieldOp>(red.getBody()->getTerminator());
  for (Operation &op : red.getBody()->without_terminator())
    b.clone(op, map);
  Value acc = map.lookupOrDefault(yield.getOperand(0));

  // Store the updated accumulator back.
  b.create<AffineStoreOp>(loc, acc, mem, idxMap, idx);

  // The original direct store of red's result is now redundant; the accumulator
  // already holds the final value after the loop.
  m.store.erase();
  red.erase();
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
        m.red.emitRemark("demoting single-level add-reduction to memref "
                         "accumulator");
      emitInitNest(m, b);
      demoteReduction(m, b);
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createDrScalarReductionDemotePass() {
  return std::make_unique<DrScalarReductionDemotePass>();
}
