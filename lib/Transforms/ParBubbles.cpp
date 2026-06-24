//===- ParBubbles.cpp - Bubble-widening front-end (M0 + M1) -------------===//
//
// Bubble-widening parallel-codegen front-end (PARALLEL_BUBBLE_SPEC.md).
//
//   M0 (par-test-diagnostics): seed one bubble per loop and classify each
//      loop's axis via the intra-module ParAliasOracle.
//   M1 (par-test-regions): widen to a fixed point and report MAXIMAL regions.
//      Two moves, Clean/Hard only (no peel/redistribute yet, that is M3):
//        * climb  — collapse a perfect affine band into one region, per-level
//                   par/seq classification (GEMM -> par={i,j} seq={k}).
//        * fuse   — merge adjacent conformant sibling bands when they touch
//                   provably-disjoint memory (Tier 0).  A rejected fuse is a
//                   frozen frontier (reason: outer-sequential | non-conformant
//                   | shared-write).
//
// Both modes are diagnostic-only — no IR is mutated.  Materialization to the
// `par` dialect is M2; same-buffer aligned fuse + offset peel is M3; cross-
// procedure call consumption is M4.
//
//===----------------------------------------------------------------------===//

#include "drcompiler/Transforms/ParBubbles.h"
#include "drcompiler/Analysis/ParAliasOracle.h"
#include "drcompiler/Dialect/Par/IR/ParOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Twine.h"
#include <map>
#include <set>
#include <string>
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "dr-par-bubbles"

namespace mlir {
#define GEN_PASS_DEF_DRPARBUBBLESPASS
#include "drcompiler/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using drcompiler::par::ConflictKind;
using drcompiler::par::ParAliasOracle;

namespace {

//===----------------------------------------------------------------------===//
// M0 — per-loop axis classification
//===----------------------------------------------------------------------===//

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

/// Compact one-glyph tag for a per-level AxisKind dump.
static StringRef axisTag(AxisKind k) {
  switch (k) {
  case AxisKind::Parallel:     return "P";
  case AxisKind::Carried:      return "Cd";
  case AxisKind::Reduction:    return "Rd";
  case AxisKind::Conservative: return "Cv";
  }
  return "Cv";
}

/// Classify the loop carried by `loopOp` (an affine.for or scf.for).
static AxisKind classifyLoop(Operation *loopOp, const ParAliasOracle &oracle) {
  // iter_args / results thread an accumulator across iterations: a reduction.
  if (loopOp->getNumResults() > 0)
    return AxisKind::Reduction;
  // M0/M1 model dependence only on affine.for; SCF is conservative for now.
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

//===----------------------------------------------------------------------===//
// M1 — region formation (climb + conformant fuse)
//===----------------------------------------------------------------------===//

static Value memrefOf(Operation *op) {
  if (auto w = dyn_cast<affine::AffineWriteOpInterface>(op))
    return w.getMemRef();
  if (auto r = dyn_cast<affine::AffineReadOpInterface>(op))
    return r.getMemRef();
  return nullptr;
}

static void collectAccesses(Operation *root, SmallVectorImpl<Operation *> &out) {
  root->walk([&](Operation *op) {
    if (isa<affine::AffineReadOpInterface, affine::AffineWriteOpInterface>(op))
      out.push_back(op);
  });
}

/// A loop is a *band root* unless it is the sole perfect child of an enclosing
/// affine.for (in which case getPerfectlyNestedLoops folds it into the parent's
/// band).
static bool isBandRoot(affine::AffineForOp loop) {
  auto parent = dyn_cast<affine::AffineForOp>(loop->getParentOp());
  if (!parent)
    return true;
  Block *pb = parent.getBody();
  Block::iterator begin = pb->begin();
  Block::iterator term = std::prev(pb->end());
  // Perfect nesting: the body holds exactly this loop + terminator.
  if (std::next(begin) == term && &*begin == loop.getOperation())
    return false;
  return true;
}

/// Two sibling bands are conformant when their outermost loops share constant
/// bounds and step (so they can be distributed over one iteration space).
static bool conformant(affine::AffineForOp a, affine::AffineForOp b) {
  if (!a.hasConstantLowerBound() || !b.hasConstantLowerBound())
    return false;
  if (!a.hasConstantUpperBound() || !b.hasConstantUpperBound())
    return false;
  return a.getConstantLowerBound() == b.getConstantLowerBound() &&
         a.getConstantUpperBound() == b.getConstantUpperBound() &&
         a.getStepAsInt() == b.getStepAsInt();
}

/// Clean (M1, Tier-0 only): every cross access pair that involves a write must
/// land on provably-distinct allocations.  Same-buffer aligned fusion needs the
/// slice/dependence machinery and is deferred to M3.
static bool crossClean(affine::AffineForOp a, affine::AffineForOp b) {
  SmallVector<Operation *, 16> aa, bb;
  collectAccesses(a, aa);
  collectAccesses(b, bb);
  for (Operation *x : aa) {
    for (Operation *y : bb) {
      bool writes = isa<affine::AffineWriteOpInterface>(x) ||
                    isa<affine::AffineWriteOpInterface>(y);
      if (!writes)
        continue; // read/read never conflicts
      Value mx = memrefOf(x), my = memrefOf(y);
      if (mx == my)
        return false;
      Value rx = ParAliasOracle::allocationRoot(mx);
      Value ry = ParAliasOracle::allocationRoot(my);
      if (rx != ry)
        continue;     // distinct allocation roots: distinct memory (affine model)
      return false;   // same underlying buffer (views) + write: conservative
    }
  }
  return true;
}

struct FuseCheck {
  bool ok;
  StringRef reason;
};

static FuseCheck canFuseSibling(affine::AffineForOp a, affine::AffineForOp b,
                                const ParAliasOracle &oracle) {
  if (classifyLoop(a.getOperation(), oracle) != AxisKind::Parallel ||
      classifyLoop(b.getOperation(), oracle) != AxisKind::Parallel)
    return {false, "outer-sequential"};
  if (!conformant(a, b))
    return {false, "non-conformant"};
  if (!crossClean(a, b))
    return {false, "shared-write"};
  return {true, ""};
}

static void classifyBand(ArrayRef<affine::AffineForOp> band,
                         const ParAliasOracle &oracle,
                         SmallVectorImpl<AxisKind> &levels) {
  for (affine::AffineForOp l : band)
    levels.push_back(classifyLoop(l.getOperation(), oracle));
}

static std::string axisList(ArrayRef<AxisKind> levels, bool wantParallel) {
  std::string s = "[";
  bool first = true;
  for (unsigned i = 0; i < levels.size(); ++i) {
    bool isPar = levels[i] == AxisKind::Parallel;
    if (isPar != wantParallel)
      continue;
    if (!first)
      s += ",";
    s += std::to_string(i);
    first = false;
  }
  s += "]";
  return s;
}

static void emitRegion(affine::AffineForOp rep, unsigned bands,
                       ArrayRef<AxisKind> levels) {
  rep->emitRemark("par-region: bands=" + std::to_string(bands) + " par=" +
                  axisList(levels, /*wantParallel=*/true) + " seq=" +
                  axisList(levels, /*wantParallel=*/false));
}

/// Form maximal regions from the band roots of one block (program order) and
/// emit one remark per region, plus a frozen-frontier remark at each rejected
/// fuse boundary.
static void processBlock(ArrayRef<affine::AffineForOp> roots,
                         const ParAliasOracle &oracle) {
  unsigned n = roots.size(), i = 0;
  while (i < n) {
    SmallVector<affine::AffineForOp, 4> band;
    affine::getPerfectlyNestedLoops(band, roots[i]);
    SmallVector<AxisKind, 4> levels;
    classifyBand(band, oracle, levels);

    unsigned j = i + 1, bands = 1;
    StringRef rejectReason;
    bool rejected = false;
    while (j < n) {
      FuseCheck fc = canFuseSibling(roots[j - 1], roots[j], oracle);
      if (!fc.ok) {
        rejectReason = fc.reason;
        rejected = true;
        break;
      }
      ++bands;
      ++j;
    }

    emitRegion(roots[i], bands, levels);
    if (rejected)
      roots[j]->emitRemark(
          (Twine("par-frontier: frozen (") + rejectReason + ")").str());
    i = j;
  }
}

//===----------------------------------------------------------------------===//
// M2 — materialize a single climb region into the `par` dialect
//===----------------------------------------------------------------------===//

static unsigned leadingParallel(ArrayRef<AxisKind> levels) {
  unsigned p = 0;
  for (AxisKind k : levels) {
    if (k != AxisKind::Parallel)
      break;
    ++p;
  }
  return p;
}

/// A band is materializable when every loop has constant bounds and no
/// iter_args, and the innermost body holds only affine load/store ops and
/// region-free, side-effect-free ops (arith/math).  Anything else (calls,
/// non-affine memref ops, nested regions) bails — left as affine.for.
static bool bandMaterializable(ArrayRef<affine::AffineForOp> band,
                               bool dynOuterUb = false) {
  for (unsigned d = 0, e = band.size(); d < e; ++d) {
    affine::AffineForOp l = band[d];
    if (!l.hasConstantLowerBound())
      return false;
    // The outer (shard) loop may carry a dynamic upper bound under S2; every
    // inner loop must be constant-bound (it becomes an scf.for with constants).
    if (!l.hasConstantUpperBound() && !(dynOuterUb && d == 0))
      return false;
    if (l.getNumResults() != 0)
      return false;
  }
  affine::AffineForOp inner = band.back();
  for (Operation &op : inner.getBody()->without_terminator()) {
    if (isa<affine::AffineLoadOp, affine::AffineStoreOp>(op))
      continue;
    if (op.getNumRegions() == 0 &&
        (isMemoryEffectFree(&op) || ParAliasOracle::isPureCall(&op)))
      continue;
    return false;
  }
  return true;
}

/// Clone one body op, expanding affine accesses to memref (affine validity does
/// not hold under par.forall / scf.for, whose IVs are not affine dims).
static void cloneBodyOp(OpBuilder &b, Operation *op, IRMapping &map) {
  Location loc = op->getLoc();
  if (auto ld = dyn_cast<affine::AffineLoadOp>(op)) {
    SmallVector<Value> operands;
    for (Value v : ld.getMapOperands())
      operands.push_back(map.lookupOrDefault(v));
    auto idx = affine::expandAffineMap(b, loc, ld.getAffineMap(), operands);
    auto nl = b.create<memref::LoadOp>(loc, map.lookupOrDefault(ld.getMemRef()),
                                       *idx);
    map.map(ld.getResult(), nl.getResult());
    return;
  }
  if (auto st = dyn_cast<affine::AffineStoreOp>(op)) {
    Value val = map.lookupOrDefault(st.getValueToStore());
    SmallVector<Value> operands;
    for (Value v : st.getMapOperands())
      operands.push_back(map.lookupOrDefault(v));
    auto idx = affine::expandAffineMap(b, loc, st.getAffineMap(), operands);
    b.create<memref::StoreOp>(loc, val, map.lookupOrDefault(st.getMemRef()),
                              *idx);
    return;
  }
  // affine.vector_load/store (emitted by affine-register-block) -> vector.load/
  // store: same affine-map expansion, vector element type.  Lets a REGISTER-
  // BLOCKED band (the per-thread vectorized GEMM kernel) shard under par.forall,
  // composing codegen x parallelism (PARALLEL_SPMD_SPEC.md §11.16).
  if (auto vld = dyn_cast<affine::AffineVectorLoadOp>(op)) {
    SmallVector<Value> operands;
    for (Value v : vld.getMapOperands())
      operands.push_back(map.lookupOrDefault(v));
    auto idx = affine::expandAffineMap(b, loc, vld.getAffineMap(), operands);
    auto nl = b.create<vector::LoadOp>(loc, vld.getVectorType(),
                                       map.lookupOrDefault(vld.getMemRef()), *idx);
    map.map(vld.getResult(), nl.getResult());
    return;
  }
  if (auto vst = dyn_cast<affine::AffineVectorStoreOp>(op)) {
    Value val = map.lookupOrDefault(vst.getValueToStore());
    SmallVector<Value> operands;
    for (Value v : vst.getMapOperands())
      operands.push_back(map.lookupOrDefault(v));
    auto idx = affine::expandAffineMap(b, loc, vst.getAffineMap(), operands);
    b.create<vector::StoreOp>(loc, val, map.lookupOrDefault(vst.getMemRef()),
                              *idx);
    return;
  }
  b.clone(*op, map);
}

/// Expand an affine.for bound (a max-affine for the lower, min-affine for the
/// upper) to a single index Value, remapping operands.  Used to de-affine an
/// inner loop whose bounds may depend on the (now non-affine) shard IV.
static Value expandForBound(OpBuilder &b, Location loc, AffineMap m,
                            ValueRange operands, bool isUpper, IRMapping &map) {
  SmallVector<Value> mapped;
  for (Value v : operands)
    mapped.push_back(map.lookupOrDefault(v));
  auto vals = affine::expandAffineMap(b, loc, m, mapped);
  Value acc = (*vals)[0];
  for (unsigned i = 1, e = vals->size(); i < e; ++i)
    acc = isUpper ? b.create<arith::MinSIOp>(loc, acc, (*vals)[i]).getResult()
                  : b.create<arith::MaxSIOp>(loc, acc, (*vals)[i]).getResult();
  return acc;
}

/// Recursively clone an op subtree into the par.forall body, fully DE-AFFINING:
/// affine.for -> scf.for (bounds expanded), affine.load/store -> memref, affine
/// .apply -> expanded SSA.  Lets a parallel shard loop with an IMPERFECT body
/// (sibling sub-loops, an inner memref reduction) materialize as a real forall
/// instead of falling back to par.critical.  Soundness comes from the shard
/// loop being oracle-PARALLEL (owner-computes: each shard writes disjoint
/// memory); the body just runs sequentially per shard iteration.  Callers must
/// gate on deAffinable() first so every op here is handled.
static void cloneDeAffine(OpBuilder &b, Operation *op, IRMapping &map) {
  Location loc = op->getLoc();
  if (auto f = dyn_cast<affine::AffineForOp>(op)) {
    Value lb = expandForBound(b, loc, f.getLowerBoundMap(),
                              f.getLowerBoundOperands(), /*isUpper=*/false, map);
    Value ub = expandForBound(b, loc, f.getUpperBoundMap(),
                              f.getUpperBoundOperands(), /*isUpper=*/true, map);
    Value st = b.create<arith::ConstantIndexOp>(loc, f.getStepAsInt());
    // Carry iter_args (a promoted register-accumulator reduction) through to the
    // scf.for: scf supports them natively.  Init operands map to scf init args,
    // region iter args to scf's, and the affine.yield to scf.yield.  Soundness:
    // the enclosing shard loop is oracle-PARALLEL (owner-computes, disjoint
    // output), and the accumulator is loop-local (an iter_arg, not shared
    // memory), so a within-shard sequential reduction is correct.
    SmallVector<Value> inits;
    for (Value v : f.getInits())
      inits.push_back(map.lookupOrDefault(v));
    auto nf = b.create<scf::ForOp>(
        loc, lb, ub, st, inits,
        [&](OpBuilder &nb, Location nloc, Value iv, ValueRange iterArgs) {
          map.map(f.getInductionVar(), iv);
          for (unsigned i = 0, e = iterArgs.size(); i < e; ++i)
            map.map(f.getRegionIterArgs()[i], iterArgs[i]);
          for (Operation &inner : f.getBody()->without_terminator())
            cloneDeAffine(nb, &inner, map);
          SmallVector<Value> yields;
          if (auto y =
                  dyn_cast<affine::AffineYieldOp>(f.getBody()->getTerminator()))
            for (Value v : y.getOperands())
              yields.push_back(map.lookupOrDefault(v));
          nb.create<scf::YieldOp>(nloc, yields);
        });
    for (unsigned i = 0, e = nf.getNumResults(); i < e; ++i)
      map.map(f.getResult(i), nf.getResult(i));
    return;
  }
  if (auto ap = dyn_cast<affine::AffineApplyOp>(op)) {
    SmallVector<Value> operands;
    for (Value v : ap.getMapOperands())
      operands.push_back(map.lookupOrDefault(v));
    auto r = affine::expandAffineMap(b, loc, ap.getAffineMap(), operands);
    map.map(ap.getResult(), (*r)[0]);
    return;
  }
  cloneBodyOp(b, op, map); // affine.load/store -> memref; else verbatim clone
}

/// Can the shard loop's body be fully de-affined by cloneDeAffine()?  Accepts
/// nested affine.for (no iter_args), affine.load/store/apply, and region-less
/// pure / pure-call ops.  Rejects affine.if, iter_arg reductions, and any other
/// region op (would clone to invalid IR under the non-affine forall IV).
static bool deAffinable(affine::AffineForOp shard) {
  bool ok = true;
  shard.getBody()->walk([&](Operation *op) {
    if (op == shard.getOperation())
      return;
    if (isa<affine::AffineForOp>(op))
      return; // nested affine.for (incl. iter_arg reductions): cloneDeAffine
              // lowers it to scf.for, carrying any iter_args through.
    if (isa<affine::AffineLoadOp, affine::AffineStoreOp, affine::AffineApplyOp,
            affine::AffineYieldOp, affine::AffineVectorLoadOp,
            affine::AffineVectorStoreOp>(op))
      return; // affine.vector_load/store (register-block output) -> cloneBodyOp
              // lowers to vector.load/store; sharding the register-blocked band
              // composes codegen x parallelism.
    if (isa<memref::AllocaOp>(op))
      return; // thread-private stack scratch (e.g. a demote/promote scalar
              // accumulator left over from a GEMM+bias band).  Replicating it
              // per shard is sound: the alloca address never escapes the band
              // (only its loaded value is stored to the owner-computes output),
              // so each forall iteration gets its own scratch.  Without this the
              // alloca's allocation effect bails the whole band to par.critical
              // -- which serialized openai-gpt's QKV/FC GEMMs (PARALLEL_SPMD §11.13).
    if (op->getNumRegions() != 0) {
      ok = false; // affine.if / scf / unknown region op
      return;
    }
    if (!isMemoryEffectFree(op) && !ParAliasOracle::isPureCall(op))
      ok = false; // region-less op with effects we can't safely replicate
  });
  return ok;
}

/// A band is load-IMBALANCED across the shard axis when an inner loop's trip
/// count depends on the shard induction variable (a triangular nest: syrk's
/// `j:0..i`, covariance's gram triangle, trmm).  A static (block) OpenMP
/// schedule then starves the low-IV workers; such a forall is tagged for a
/// `dynamic` schedule by convert-par-to-omp.
static bool bandImbalanced(affine::AffineForOp shard) {
  Value iv = shard.getInductionVar();
  bool found = false;
  shard.getBody()->walk([&](affine::AffineForOp inner) {
    for (Value o : inner.getLowerBoundOperands())
      if (o == iv)
        found = true;
    for (Value o : inner.getUpperBoundOperands())
      if (o == iv)
        found = true;
  });
  return found;
}

/// Clone the (imperfect) shard loop's body into the forall, de-affined.  The
/// shard IV must already be mapped to the forall arg.
static void buildShardedBody(OpBuilder &b, affine::AffineForOp shard,
                             IRMapping &map) {
  for (Operation &op : shard.getBody()->without_terminator())
    cloneDeAffine(b, &op, map);
}

/// Build the sequential suffix loops band[idx..] as scf.for, then clone the
/// innermost body.
static void buildSeq(OpBuilder &b, ArrayRef<affine::AffineForOp> band,
                     unsigned idx, IRMapping &map) {
  if (idx == band.size()) {
    affine::AffineForOp inner = band.back();
    for (Operation &op : inner.getBody()->without_terminator())
      cloneBodyOp(b, &op, map);
    return;
  }
  affine::AffineForOp l = band[idx];
  Location loc = l.getLoc();
  Value lb = b.create<arith::ConstantIndexOp>(loc, l.getConstantLowerBound());
  Value ub = b.create<arith::ConstantIndexOp>(loc, l.getConstantUpperBound());
  Value st = b.create<arith::ConstantIndexOp>(loc, l.getStepAsInt());
  auto forOp = b.create<scf::ForOp>(loc, lb, ub, st);
  map.map(l.getInductionVar(), forOp.getInductionVar());
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToStart(forOp.getBody());
  buildSeq(b, band, idx + 1, map);
}

/// Wrap the leading parallel prefix of `band` in par.region { par.forall ... }.
/// Returns false (no mutation) when nothing is materializable.
static bool materializeBand(ArrayRef<affine::AffineForOp> band,
                            ArrayRef<AxisKind> levels) {
  unsigned p = leadingParallel(levels);
  if (p == 0 || !bandMaterializable(band))
    return false;

  affine::AffineForOp root = band.front();
  OpBuilder b(root);
  Location loc = root.getLoc();

  auto regionOp = b.create<par::RegionOp>(loc);
  b.createBlock(&regionOp.getRegion());

  SmallVector<int64_t> lbs, ubs, steps;
  for (unsigned d = 0; d < p; ++d) {
    affine::AffineForOp bd = band[d];
    lbs.push_back(bd.getConstantLowerBound());
    ubs.push_back(bd.getConstantUpperBound());
    steps.push_back(bd.getStepAsInt());
  }
  auto forall = b.create<par::ForallOp>(
      loc, TypeRange{}, b.getDenseI64ArrayAttr(lbs), b.getDenseI64ArrayAttr(ubs),
      b.getDenseI64ArrayAttr(steps), ValueRange{}, ValueRange{});
  Block *fblk = b.createBlock(&forall.getRegion());
  fblk->addArguments(SmallVector<Type>(p, b.getIndexType()),
                     SmallVector<Location>(p, loc));

  IRMapping map;
  for (unsigned d = 0; d < p; ++d) {
    affine::AffineForOp bd = band[d];
    map.map(bd.getInductionVar(), fblk->getArgument(d));
  }
  b.setInsertionPointToStart(fblk);
  buildSeq(b, band, p, map);
  b.create<par::YieldOp>(loc, ValueRange{});

  b.setInsertionPointToEnd(regionOp.getBody());
  b.create<par::YieldOp>(loc, ValueRange{});

  root.erase();
  return true;
}

//===----------------------------------------------------------------------===//
// M3 — multi-band fuse / barrier materialization (depth-1 parallel bands)
//===----------------------------------------------------------------------===//

static AffineMap accMap(Operation *op) {
  if (auto l = dyn_cast<affine::AffineLoadOp>(op))
    return l.getAffineMap();
  return cast<affine::AffineStoreOp>(op).getAffineMap();
}
static SmallVector<Value> accOperands(Operation *op) {
  if (auto l = dyn_cast<affine::AffineLoadOp>(op))
    return llvm::to_vector(l.getMapOperands());
  return llvm::to_vector(cast<affine::AffineStoreOp>(op).getMapOperands());
}

/// Two same-buffer accesses are element-aligned when their access maps are
/// structurally equal and every operand matches, with each loop's own IV in the
/// same slot (so element i of one only meets element i of the other).
static bool alignedAccess(Operation *a, Operation *b, Value ivA, Value ivB) {
  if (accMap(a) != accMap(b))
    return false;
  SmallVector<Value> oa = accOperands(a), ob = accOperands(b);
  if (oa.size() != ob.size())
    return false;
  for (size_t k = 0, e = oa.size(); k < e; ++k) {
    if (oa[k] == ivA && ob[k] == ivB)
      continue;
    if (oa[k] == ob[k])
      continue;
    return false;
  }
  return true;
}

enum class CrossKind { Fusable, Barrier };

/// Relationship between two conformant depth-1 sibling bands: Fusable (every
/// cross write-pair is on disjoint allocations or element-aligned) or Barrier
/// (an offset / unprovable dependence — keep both parallel, sync between).
static CrossKind classifyCross(affine::AffineForOp a, affine::AffineForOp b) {
  Value ivA = a.getInductionVar(), ivB = b.getInductionVar();
  SmallVector<Operation *, 8> aa, bb;
  collectAccesses(a, aa);
  collectAccesses(b, bb);
  for (Operation *x : aa) {
    for (Operation *y : bb) {
      bool writes = isa<affine::AffineWriteOpInterface>(x) ||
                    isa<affine::AffineWriteOpInterface>(y);
      if (!writes)
        continue;
      Value mx = memrefOf(x), my = memrefOf(y);
      if (mx != my) {
        Value rx = ParAliasOracle::allocationRoot(mx);
        Value ry = ParAliasOracle::allocationRoot(my);
        if (rx != ry)
          continue; // distinct allocation roots: distinct memory (affine model)
        return CrossKind::Barrier; // same buffer via views: conservative
      }
      if (alignedAccess(x, y, ivA, ivB))
        continue;
      return CrossKind::Barrier;
    }
  }
  return CrossKind::Fusable;
}

/// A reshuffle boundary: two bands share a buffer, but the producer wrote it
/// under a different worker→data mapping than the consumer reads it — not a
/// plain constant offset.  Materializes to par.redistribute (vs par.barrier).
struct RedistDesc {
  Value buf;
  std::string from, to;
};

static std::string mapStr(AffineMap m) {
  std::string s;
  llvm::raw_string_ostream os(s);
  m.print(os);
  return s;
}

/// Find a same-buffer cross pair between bands of `A` and `B` whose 1-D access
/// maps differ by more than a constant offset (different IV coefficient — a
/// reverse/scale/permute). Returns the buffer + the two maps, else nullopt.
static std::optional<RedistDesc> findReshuffle(ArrayRef<affine::AffineForOp> A,
                                               ArrayRef<affine::AffineForOp> B) {
  for (affine::AffineForOp ba : A) {
    SmallVector<Operation *, 8> aa;
    collectAccesses(ba, aa);
    for (affine::AffineForOp bb : B) {
      SmallVector<Operation *, 8> bbacc;
      collectAccesses(bb, bbacc);
      for (Operation *x : aa) {
        for (Operation *y : bbacc) {
          bool writes = isa<affine::AffineWriteOpInterface>(x) ||
                        isa<affine::AffineWriteOpInterface>(y);
          if (!writes || memrefOf(x) != memrefOf(y))
            continue;
          AffineMap mx = accMap(x), my = accMap(y);
          if (mx.getNumDims() != 1 || mx.getNumResults() != 1 ||
              mx.getNumSymbols() != 0 || my.getNumDims() != 1 ||
              my.getNumResults() != 1 || my.getNumSymbols() != 0)
            continue; // not a clean 1-D access pair: leave as a barrier
          AffineExpr diff = simplifyAffineExpr(
              mx.getResult(0) - my.getResult(0), /*numDims=*/1, /*numSymbols=*/0);
          if (isa<AffineConstantExpr>(diff))
            continue; // constant offset: a plain shift, barrier suffices
          return RedistDesc{memrefOf(x), mapStr(mx), mapStr(my)};
        }
      }
    }
  }
  return std::nullopt;
}

/// A depth-1 affine.for whose single axis is parallel and whose body is
/// materializable.  These are the bands M3 fuses across; everything else falls
/// back to the M2 single-band path.
static bool isSimpleParallel(affine::AffineForOp loop,
                             const ParAliasOracle &oracle) {
  SmallVector<affine::AffineForOp, 4> band;
  affine::getPerfectlyNestedLoops(band, loop);
  if (band.size() != 1)
    return false;
  if (classifyLoop(loop.getOperation(), oracle) != AxisKind::Parallel)
    return false;
  return bandMaterializable(band);
}

/// True when every value `band`'s body reads from outside the band is defined
/// before `anchor` (so it still dominates a region inserted there).  Guards
/// against fusing a band that uses a value defined between the run's bands.
static bool bodyOperandsDominate(affine::AffineForOp band, Operation *anchor) {
  affine::AffineForOp b = band;
  for (Operation &op : b.getBody()->without_terminator()) {
    for (Value v : op.getOperands()) {
      if (isa<BlockArgument>(v))
        continue; // IV (remapped) / enclosing-scope arg (dominates)
      Operation *def = v.getDefiningOp();
      if (!def || b->isAncestor(def))
        continue; // defined inside the band body
      if (def->getBlock() == anchor->getBlock() &&
          !def->isBeforeInBlock(anchor))
        return false;
    }
  }
  return true;
}

/// Materialize a run of conformant depth-1 simple-parallel sibling bands into a
/// single par.region: maximal fusable subgroups share one par.forall (bodies
/// sequenced); an offset boundary between subgroups becomes a par.barrier.
static void materializeRun(ArrayRef<affine::AffineForOp> run) {
  affine::AffineForOp first = run.front();
  OpBuilder b(first);
  Location loc = first.getLoc();
  auto regionOp = b.create<par::RegionOp>(loc);
  Block *rblk = b.createBlock(&regionOp.getRegion());

  unsigned idx = 0;
  bool firstSub = true;
  SmallVector<affine::AffineForOp, 4> prevSub;
  while (idx < run.size()) {
    SmallVector<affine::AffineForOp, 4> sub;
    sub.push_back(run[idx]);
    unsigned k = idx + 1;
    while (k < run.size()) {
      affine::AffineForOp cand = run[k];
      bool fusable = true;
      for (affine::AffineForOp s : sub)
        if (classifyCross(s, cand) == CrossKind::Barrier) {
          fusable = false;
          break;
        }
      if (!fusable)
        break;
      sub.push_back(cand);
      ++k;
    }

    // Boundary between the previous subgroup and this one: a worker↔data
    // remap of a shared buffer becomes par.redistribute; otherwise a barrier.
    b.setInsertionPointToEnd(rblk);
    if (!firstSub) {
      if (auto rd = findReshuffle(prevSub, sub))
        b.create<par::RedistributeOp>(loc, rd->buf, b.getStringAttr(rd->from),
                                      b.getStringAttr(rd->to));
      else
        b.create<par::BarrierOp>(loc);
    }
    firstSub = false;

    b.setInsertionPointToEnd(rblk);
    affine::AffineForOp f0 = sub.front();
    SmallVector<int64_t> lb{f0.getConstantLowerBound()};
    SmallVector<int64_t> ub{f0.getConstantUpperBound()};
    SmallVector<int64_t> st{f0.getStepAsInt()};
    auto forall = b.create<par::ForallOp>(
        loc, TypeRange{}, b.getDenseI64ArrayAttr(lb), b.getDenseI64ArrayAttr(ub),
        b.getDenseI64ArrayAttr(st), ValueRange{}, ValueRange{});
    Block *fblk = b.createBlock(&forall.getRegion());
    fblk->addArgument(b.getIndexType(), loc);
    Value iv = fblk->getArgument(0);
    b.setInsertionPointToStart(fblk);
    for (affine::AffineForOp band : sub) {
      IRMapping map;
      map.map(band.getInductionVar(), iv);
      for (Operation &op : band.getBody()->without_terminator())
        cloneBodyOp(b, &op, map);
    }
    b.create<par::YieldOp>(loc, ValueRange{});
    prevSub = sub;
    idx = k;
  }

  b.setInsertionPointToEnd(rblk);
  b.create<par::YieldOp>(loc, ValueRange{});

  for (affine::AffineForOp band : run)
    band.erase();
}

/// Materialize all band roots of one block: maximal runs of conformant
/// dominating simple-parallel bands go through materializeRun (fuse/barrier);
/// every other band falls back to the M2 single-band path.
static void materializeBlock(ArrayRef<affine::AffineForOp> roots,
                             const ParAliasOracle &oracle) {
  unsigned n = roots.size(), i = 0;
  while (i < n) {
    affine::AffineForOp head = roots[i];
    if (isSimpleParallel(head, oracle)) {
      SmallVector<affine::AffineForOp, 4> run;
      run.push_back(head);
      unsigned j = i + 1;
      while (j < n && isSimpleParallel(roots[j], oracle) &&
             conformant(head, roots[j]) &&
             bodyOperandsDominate(roots[j], head.getOperation())) {
        run.push_back(roots[j]);
        ++j;
      }
      materializeRun(run);
      i = j;
    } else {
      SmallVector<affine::AffineForOp, 4> band;
      affine::getPerfectlyNestedLoops(band, roots[i]);
      SmallVector<AxisKind, 4> levels;
      classifyBand(band, oracle, levels);
      materializeBand(band, levels); // no-op if not materializable
      ++i;
    }
  }
}

//===----------------------------------------------------------------------===//
// S0/S1 — whole-function shard-axis selection + barrier-elision analysis
// (diagnostic-only spike; see PARALLEL_SPMD_SPEC.md §3-§4)
//===----------------------------------------------------------------------===//

enum class EdgeKind { Elide = 0, Halo = 1, Redistribute = 2, Barrier = 3 };

static EdgeKind worstEdge(EdgeKind a, EdgeKind b) {
  return static_cast<int>(a) >= static_cast<int>(b) ? a : b;
}
static StringRef edgeName(EdgeKind k) {
  switch (k) {
  case EdgeKind::Elide:        return "ELIDE";
  case EdgeKind::Halo:         return "HALO";
  case EdgeKind::Redistribute: return "REDISTRIBUTE";
  case EdgeKind::Barrier:      return "BARRIER";
  }
  return "BARRIER";
}

/// Identity of a band's shard axis (its outermost loop) if that loop is
/// parallel: the printed lb/ub maps, OPERAND-AGNOSTIC so the same axis matches
/// across layers (e.g. every conv's `0 to #map(%dim)` batch loop, where the
/// %dim SSA value differs per layer).  Multi-dim bands welcome (we shard the
/// outer loop; inner loops stay sequential within the shard).
static std::optional<std::string> shardAxisId(affine::AffineForOp band,
                                               const ParAliasOracle &oracle) {
  if (classifyLoop(band.getOperation(), oracle) != AxisKind::Parallel)
    return std::nullopt;
  std::string s;
  llvm::raw_string_ostream os(s);
  os << "lb=";
  band.getLowerBoundMap().print(os);
  os << " ub=";
  band.getUpperBoundMap().print(os);
  return s;
}

/// A short human label for a shard axis (constant extent, else "dyn").
static std::string shardLabel(affine::AffineForOp band) {
  if (band.hasConstantLowerBound() && band.hasConstantUpperBound())
    return "extent=" +
           std::to_string(band.getConstantUpperBound() -
                          band.getConstantLowerBound());
  return "dyn";
}

/// Owner-aligned on the shard axis: same access map, and the shard IV occupies
/// the same operand slots in both (the inner-loop operands are free — they
/// range within the shard).  Sound projection onto the shard axis
/// (PARALLEL_SPMD_SPEC.md §4): shard t reads exactly what shard t wrote.
static bool alignedOnShard(Operation *x, Operation *y, Value sX, Value sY) {
  if (accMap(x) != accMap(y))
    return false;
  SmallVector<Value> ox = accOperands(x), oy = accOperands(y);
  if (ox.size() != oy.size())
    return false;
  for (size_t k = 0, e = ox.size(); k < e; ++k)
    if ((ox[k] == sX) != (oy[k] == sY))
      return false; // shard IV must sit in the same slot(s)
  return true;
}

/// Classify the barrier-elision verdict for the inter-band edge A -> B, sharded
/// on the given shard IVs `sA`/`sB` (the outermost real-extent parallel loop of
/// each, which may sit below degenerate outer loops -- so the caller passes the
/// actual shard IV rather than A/B's outermost induction var).  Sound: any
/// non-provably-owner-aligned dependence keeps its sync.
static EdgeKind classifyEdgeShard(Operation *A, Value sA, Operation *B,
                                  Value sB) {
  SmallVector<Operation *, 8> aa, bb;
  collectAccesses(A, aa);
  collectAccesses(B, bb);
  EdgeKind worst = EdgeKind::Elide;
  for (Operation *x : aa) {
    for (Operation *y : bb) {
      bool writes = isa<affine::AffineWriteOpInterface>(x) ||
                    isa<affine::AffineWriteOpInterface>(y);
      if (!writes)
        continue;
      Value mx = memrefOf(x), my = memrefOf(y);
      if (mx != my) {
        if (ParAliasOracle::allocationRoot(mx) !=
            ParAliasOracle::allocationRoot(my))
          continue; // distinct buffers -> no cross-shard interaction
        worst = worstEdge(worst, EdgeKind::Barrier); // same root, views
        continue;
      }
      if (alignedOnShard(x, y, sA, sB))
        continue; // owner-aligned on the shard axis
      AffineMap mxm = accMap(x), mym = accMap(y);
      if (mxm.getNumDims() == 1 && mxm.getNumResults() == 1 &&
          mxm.getNumSymbols() == 0 && mym.getNumDims() == 1 &&
          mym.getNumResults() == 1 && mym.getNumSymbols() == 0) {
        AffineExpr diff = simplifyAffineExpr(
            mxm.getResult(0) - mym.getResult(0), 1, 0);
        worst = worstEdge(worst, isa<AffineConstantExpr>(diff)
                                     ? EdgeKind::Halo        // 1-D shift -> halo
                                     : EdgeKind::Redistribute); // 1-D remap
      } else {
        worst = worstEdge(worst, EdgeKind::Barrier); // multi-dim non-aligned
      }
    }
  }
  return worst;
}

/// Diagnostic S0/S1 convenience: classify A -> B sharded on each band's
/// outermost induction var.
static EdgeKind classifyEdge(affine::AffineForOp A, affine::AffineForOp B) {
  return classifyEdgeShard(A.getOperation(), A.getInductionVar(),
                           B.getOperation(), B.getInductionVar());
}

//===----------------------------------------------------------------------===//
// S2 — whole-function widening + materialization
// (PARALLEL_SPMD_SPEC.md §7).  Widen every top-level shard-axis band into ONE
// par.region; hoist one par.forall over the shard axis per maximal
// ELIDE-connected run (owner-computes: bodies sequenced, inner loops sunk as
// scf.for); a non-elided edge becomes par.redistribute (IV remap) or
// par.barrier.  IR-mutating; default off; constant shard extent only.
//===----------------------------------------------------------------------===//

/// If `loop`'s upper bound is `lb to %v` for a single SSA Value %v (the ub map
/// is a bare dim/symbol identity over one operand, e.g. the runtime batch size
/// `0 to %N`), return %v; else null.  Lets S2 shard a dynamic-extent axis.
static Value simpleDynUb(affine::AffineForOp loop) {
  if (loop.hasConstantUpperBound())
    return nullptr;
  AffineMap m = loop.getUpperBoundMap();
  if (m.getNumResults() != 1)
    return nullptr;
  AffineExpr e = m.getResult(0);
  auto ops = loop.getUpperBoundOperands();
  unsigned pos;
  if (auto d = dyn_cast<AffineDimExpr>(e))
    pos = d.getPosition();
  else if (auto s = dyn_cast<AffineSymbolExpr>(e))
    pos = m.getNumDims() + s.getPosition();
  else
    return nullptr; // not a bare identity (has coeff/offset): too complex
  if (pos >= ops.size())
    return nullptr;
  return ops[pos];
}

/// The perfect nest of `root` if it is an S2 shard band: outermost loop is the
/// (parallel) shard axis with constant lb/step and either the given constant
/// upper bound `ubConst` (dynUb null) or exactly the dynamic upper-bound Value
/// `dynUb` (so every band shards the SAME runtime extent); body materializable.
static SmallVector<affine::AffineForOp, 4>
spmdBand(affine::AffineForOp root, const ParAliasOracle &oracle, int64_t lb,
         int64_t step, int64_t ubConst, Value dynUb) {
  SmallVector<affine::AffineForOp, 4> band;
  affine::getPerfectlyNestedLoops(band, root);
  affine::AffineForOp s = band.front();
  if (classifyLoop(s.getOperation(), oracle) != AxisKind::Parallel)
    return {};
  if (!s.hasConstantLowerBound() || s.getConstantLowerBound() != lb ||
      s.getStepAsInt() != step)
    return {};
  if (dynUb) {
    if (simpleDynUb(s) != dynUb)
      return {};
  } else if (!s.hasConstantUpperBound() || s.getConstantUpperBound() != ubConst) {
    return {};
  }
  if (!bandMaterializable(band, /*dynOuterUb=*/dynUb != nullptr))
    return {};
  return band;
}

/// Every value the whole nest reads from outside still defined before `anchor`
/// (so it dominates a region inserted there).  Walks the FULL nest (unlike the
/// M3 depth-1 bodyOperandsDominate).
static bool nestOperandsDominate(affine::AffineForOp root, Operation *anchor) {
  bool ok = true;
  root.walk([&](Operation *op) {
    for (Value v : op->getOperands()) {
      if (isa<BlockArgument>(v))
        continue; // IV (remapped) / enclosing-scope arg (dominates)
      Operation *def = v.getDefiningOp();
      if (!def || root->isAncestor(def))
        continue; // defined inside the nest
      if (def->getBlock() == anchor->getBlock() &&
          !def->isBeforeInBlock(anchor))
        ok = false;
    }
  });
  return ok;
}

/// Make a run of shard bands contiguous by hoisting the inter-band "glue" --
/// scratch allocs, constants, pure index/view ops (the per-layer setup real
/// ONNX kernels interleave between layers) -- above the first band.  Sound: a
/// hoisted op is either side-effect-free (reads no memory, so no RAW with a
/// band store) or a fresh allocation (no ordering constraint with other
/// buffers), and bands have no SSA results, so nothing hoisted depends on a
/// band; moving it earlier preserves semantics.  Returns false (no mutation) if
/// any inter-band op is NOT hoistable (a load/store/call/dealloc/region op) --
/// then materialization bails.
static bool hoistInterBandGlue(ArrayRef<affine::AffineForOp> roots) {
  llvm::SmallPtrSet<Operation *, 16> bandSet;
  for (affine::AffineForOp r : roots)
    bandSet.insert(r.getOperation());
  affine::AffineForOp frontBand = roots.front(), backBand = roots.back();
  Operation *first = frontBand.getOperation();
  Operation *last = backBand.getOperation();
  SmallVector<Operation *> glue;
  for (Operation *op = first->getNextNode(); op && op != last;
       op = op->getNextNode()) {
    if (bandSet.count(op))
      continue; // an intervening band root
    bool alloc = isa<memref::AllocOp, memref::AllocaOp>(op);
    if (op->getNumRegions() != 0 || !(isMemoryEffectFree(op) || alloc))
      return false; // reads/writes memory or carries a region: not hoistable
    glue.push_back(op);
  }
  for (Operation *g : glue)
    g->moveBefore(first); // forward order -> relative order preserved
  return true;
}

/// Materialize a contiguous run of shard-axis bands (`roots`, perfect nests
/// `nests`) into ONE par.region over the shard space [lb,ub,step).  Maximal
/// ELIDE-connected sub-runs share a par.forall (bodies sequenced); a non-elided
/// boundary becomes par.redistribute (IV remap) or par.barrier.
static void
materializeSpmd(ArrayRef<affine::AffineForOp> roots,
                ArrayRef<SmallVector<affine::AffineForOp, 4>> nests, int64_t lb,
                int64_t step, int64_t ubConst, Value dynUb, unsigned &nForall,
                unsigned &nBarrier, unsigned &nRedist) {
  affine::AffineForOp first = roots.front();
  OpBuilder b(first);
  Location loc = first.getLoc();
  auto regionOp = b.create<par::RegionOp>(loc);
  Block *rblk = b.createBlock(&regionOp.getRegion());

  unsigned idx = 0, n = roots.size();
  affine::AffineForOp prevLast = nullptr;
  while (idx < n) {
    // Grow a maximal ELIDE-connected sub-run.
    unsigned k = idx + 1;
    while (k < n && classifyEdge(roots[k - 1], roots[k]) == EdgeKind::Elide)
      ++k;

    // Sync at the boundary from the previous sub-run (a non-elided edge).
    b.setInsertionPointToEnd(rblk);
    if (prevLast) {
      EdgeKind e = classifyEdge(prevLast, roots[idx]);
      std::optional<RedistDesc> rd;
      if (e == EdgeKind::Redistribute)
        rd = findReshuffle({prevLast}, {roots[idx]});
      if (rd) {
        b.create<par::RedistributeOp>(loc, rd->buf, b.getStringAttr(rd->from),
                                      b.getStringAttr(rd->to));
        ++nRedist;
      } else { // Halo / Barrier, or a Redistribute with no clean 1-D remap
        b.create<par::BarrierOp>(loc);
        ++nBarrier;
      }
    }

    // One forall over the shard axis for this sub-run.
    b.setInsertionPointToEnd(rblk);
    int64_t ubEntry = dynUb ? ShapedType::kDynamic : ubConst;
    SmallVector<Value> dynOps;
    if (dynUb)
      dynOps.push_back(dynUb);
    auto forall = b.create<par::ForallOp>(
        loc, TypeRange{}, b.getDenseI64ArrayAttr({lb}),
        b.getDenseI64ArrayAttr({ubEntry}), b.getDenseI64ArrayAttr({step}), dynOps,
        ValueRange{});
    Block *fblk = b.createBlock(&forall.getRegion());
    fblk->addArgument(b.getIndexType(), loc);
    Value iv = fblk->getArgument(0);
    b.setInsertionPointToStart(fblk);
    for (unsigned i = idx; i < k; ++i) {
      ArrayRef<affine::AffineForOp> band = nests[i];
      affine::AffineForOp shardLoop = band.front();
      IRMapping map;
      map.map(shardLoop.getInductionVar(), iv);
      buildSeq(b, band, /*idx=*/1, map); // inner loops -> scf.for, then body
    }
    b.create<par::YieldOp>(loc, ValueRange{});
    ++nForall;

    prevLast = roots[k - 1];
    idx = k;
  }

  b.setInsertionPointToEnd(rblk);
  b.create<par::YieldOp>(loc, ValueRange{});

  for (affine::AffineForOp r : roots)
    r.erase();
}

//===----------------------------------------------------------------------===//
// S7 — batch-1 within-sample per-band sharding.  Each top-level band shards its
// OWN outermost parallel loop (the output axis: oc / spatial / output-neuron),
// so the inner reduction (ic/kernel/k) stays within-shard -- no cross-shard
// reduce.  One par.region, a par.forall per band, par.barrier between bands.
//===----------------------------------------------------------------------===//

/// Per-band shard plan: the perfect nest, plus its outermost loop's bounds
/// (constant or a single runtime Value).  nullopt if the outer loop is not a
/// materializable parallel axis.
struct PerBandInfo {
  SmallVector<affine::AffineForOp, 4> nest;
  unsigned shardIdx; // which loop in the nest is the shard axis
  int64_t lb, step, ubConst;
  Value dynUb;  // shard upper bound is a bare runtime value
  bool dynExpr; // shard upper bound is an affine expression (e.g. N-1) to be
                // expanded to an SSA value at emit time
  bool perfect; // true: perfect nest (buildSeqSkip); false: imperfect body
                // (cloneDeAffine the shard loop's whole body)
};

/// A loop runs <=1 iteration (e.g. the batch axis at batch=1): sharding it
/// gives no parallelism, so the shard axis must look past it.
static bool degenerateExtent(affine::AffineForOp l) {
  return l.hasConstantLowerBound() && l.hasConstantUpperBound() &&
         (l.getConstantUpperBound() - l.getConstantLowerBound()) <=
             l.getStepAsInt();
}

static std::optional<PerBandInfo> perBandShard(affine::AffineForOp root,
                                               const ParAliasOracle &oracle) {
  PerBandInfo info;
  affine::getPerfectlyNestedLoops(info.nest, root);
  // Shard the OUTERMOST parallel loop with real extent (>1).  Degenerate outer
  // loops (e.g. the batch axis at batch=1) are skipped -- they nest inside the
  // forall (a trivial interchange, sound because they run once).  A
  // non-degenerate loop that ISN'T the shard axis above it would block the
  // interchange, so bail then.
  unsigned n = info.nest.size(), d = 0;
  for (; d < n; ++d) {
    affine::AffineForOp l = info.nest[d];
    bool par = classifyLoop(l.getOperation(), oracle) == AxisKind::Parallel;
    if (par && !degenerateExtent(l))
      break; // shard here
    if (!degenerateExtent(l))
      return std::nullopt; // non-degenerate, non-shardable loop outside: bail
  }
  if (d == n)
    return std::nullopt; // no parallel loop with real extent
  info.shardIdx = d;
  affine::AffineForOp shard = info.nest[d];
  if (!shard.hasConstantLowerBound())
    return std::nullopt;
  info.lb = shard.getConstantLowerBound();
  info.step = shard.getStepAsInt();
  info.dynUb = nullptr;
  info.dynExpr = false;
  info.ubConst = 0;
  if (shard.hasConstantUpperBound()) {
    info.ubConst = shard.getConstantUpperBound();
  } else if (d == 0) {
    // dynamic extent only supported on the outermost axis of the band
    info.dynUb = simpleDynUb(shard); // bare runtime value (fast path)
    if (!info.dynUb) {
      // a single-result affine bound (e.g. N-1): expand it to a value at emit.
      if (shard.getUpperBoundMap().getNumResults() != 1)
        return std::nullopt; // min-bound (multi-result): not handled
      info.dynExpr = true;
    }
  } else {
    return std::nullopt;
  }
  if (bandMaterializable(info.nest,
                         /*dynOuterUb=*/info.dynUb != nullptr || info.dynExpr)) {
    info.perfect = true;
  } else if (deAffinable(shard) && [&] {
               // The degenerate loops above the shard axis are mapped IV->lb by
               // emitForall (they run once); that is only sound if they carry no
               // iter_args (whose results we would otherwise drop).  Guaranteed
               // for the conv/batch prefix; bail to par.critical if not.
               for (unsigned k = 0; k < info.shardIdx; ++k)
                 if (info.nest[k].getNumResults() != 0)
                   return false;
               return true;
             }()) {
    // Imperfect body (sibling sub-loops / inner memref or iter_arg reduction).
    // Sound to shard because `shard` is oracle-PARALLEL (owner-computes); the
    // body is de-affined into the forall.  Any loops ABOVE the shard axis
    // (shardIdx>0) are guaranteed degenerate (extent<=step) by the selection
    // loop above -- they run once, so emitForall maps their IVs to their lower
    // bound and the de-affined shard body is emitted directly under the forall
    // (a sound trivial interchange past the once-iterating outer loops).
    info.perfect = false;
  } else {
    return std::nullopt;
  }
  return info;
}

/// Build the band's loops as scf.for, SKIPPING the shard loop (its IV is mapped
/// to the par.forall arg), then clone the innermost body.  The skipped shard
/// loop becomes the outer par.forall; degenerate loops above it nest inside
/// (sound interchange).
static void buildSeqSkip(OpBuilder &b, ArrayRef<affine::AffineForOp> band,
                         unsigned idx, unsigned shardIdx, IRMapping &map) {
  if (idx == band.size()) {
    affine::AffineForOp inner = band.back();
    for (Operation &op : inner.getBody()->without_terminator())
      cloneBodyOp(b, &op, map);
    return;
  }
  if (idx == shardIdx) {
    buildSeqSkip(b, band, idx + 1, shardIdx, map);
    return;
  }
  affine::AffineForOp l = band[idx];
  Location loc = l.getLoc();
  Value lb = b.create<arith::ConstantIndexOp>(loc, l.getConstantLowerBound());
  Value ub = b.create<arith::ConstantIndexOp>(loc, l.getConstantUpperBound());
  Value st = b.create<arith::ConstantIndexOp>(loc, l.getStepAsInt());
  auto forOp = b.create<scf::ForOp>(loc, lb, ub, st);
  map.map(l.getInductionVar(), forOp.getInductionVar());
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToStart(forOp.getBody());
  buildSeqSkip(b, band, idx + 1, shardIdx, map);
}

/// May `op` write or free memory?  Such ops can't be replicated across workers
/// (a concurrent write is a race) -> run single-worker.  Read-only / pure ops
/// replicate safely (every worker computes the same value, SSA visible to all).
static bool mayWriteOrFree(Operation *op) {
  if (isMemoryEffectFree(op))
    return false;
  auto iface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!iface)
    return true; // unknown effects (unregistered op / call): conservative
  SmallVector<MemoryEffects::EffectInstance> effects;
  iface.getEffects(effects);
  for (auto &e : effects)
    if (isa<MemoryEffects::Write, MemoryEffects::Free>(e.getEffect()))
      return true;
  return false;
}

/// A span value is used after the band span (so erasing the span would dangle).
static bool usedAfter(Operation *op, Operation *lastBand, Block &entry) {
  for (Value r : op->getResults())
    for (Operation *user : r.getUsers()) {
      Operation *top = (user->getBlock() == &entry)
                           ? user
                           : entry.findAncestorOpInBlock(*user);
      if (top && lastBand->isBeforeInBlock(top))
        return true;
    }
  return false;
}

/// A SEQUENTIAL outer band that wraps parallel inner bands (a time-stepped
/// stencil: `for t (seq) { spatial1 (par); spatial2 (par) }`, or any
/// sequential-outer / parallel-inner shape).  Materialized as `scf.for(t)`
/// (run redundantly by the whole team) wrapping one `par.forall` per inner band,
/// with the implicit end-of-wsloop barrier between them (no `par.barrier`, which
/// may not nest under scf.for).  True iff `root`'s own axis is non-degenerate
/// SEQUENTIAL and every immediate child is either a perBandShard-able parallel
/// band or pure (replicable) glue; conservative otherwise (-> par.critical).
static bool seqWrappable(affine::AffineForOp root, const ParAliasOracle &oracle) {
  if (classifyLoop(root.getOperation(), oracle) == AxisKind::Parallel)
    return false; // a parallel outer is a normal forall, not a seq wrapper
  if (degenerateExtent(root))
    return false;
  bool hasBand = false;
  for (Operation &child : root.getBody()->without_terminator()) {
    if (auto cb = dyn_cast<affine::AffineForOp>(&child)) {
      if (!perBandShard(cb, oracle))
        return false; // an inner band we can't shard: bail (stay critical)
      hasBand = true;
    } else if (!isMemoryEffectFree(&child)) {
      return false; // side-effecting glue we won't replicate per worker
    }
  }
  return hasBand;
}

/// Whole-function widening (PARALLEL_SPMD_SPEC.md §2/§7): wrap the band span of
/// `fn`'s entry block in ONE par.region.  Each span op is dispositioned:
///   MOVE        — allocs + pure metadata/index ops whose operands all dominate
///                 the region: hoisted before the team (computed once, shared,
///                 may escape post-span e.g. the output buffer/view);
///   FORALL      — a band shardable on its own output axis;
///   SEQWRAP     — a sequential outer band wrapping parallel inner bands
///                 (scf.for { par.forall ... }, implicit wsloop barriers);
///   CRITICAL    — a non-shardable band, or write/free glue: one worker, in
///                 order (effect visible after its implicit barrier);
///   REPLICATE   — read-only/pure glue depending on in-region data: cloned into
///                 the region (every worker recomputes it, SSA visible to all).
/// par.barrier precedes each band.  Bails (no mutation) only if an in-region
/// value (replicate/critical glue) escapes the span -- it can't (it lives
/// inside omp.parallel).
/// Emit a remark characterizing a CRITICAL (single-worker) band: per-level loop
/// bounds + AxisKind (why no parallel shard axis exists) + an op-type histogram
/// (what it computes).  Diagnostic only -- the leverage-point scout for the
/// serial floor.
static void describeCriticalBand(affine::AffineForOp band,
                                 const ParAliasOracle &oracle) {
  SmallVector<affine::AffineForOp, 4> nest;
  affine::getPerfectlyNestedLoops(nest, band);
  std::string s;
  llvm::raw_string_ostream os(s);
  os << "par-spmd-critical: loops[";
  for (unsigned i = 0; i < nest.size(); ++i) {
    affine::AffineForOp l = nest[i];
    if (i)
      os << ",";
    if (l.hasConstantLowerBound() && l.hasConstantUpperBound())
      os << l.getConstantLowerBound() << ":" << l.getConstantUpperBound();
    else
      os << "dyn";
    os << "(" << axisTag(classifyLoop(l.getOperation(), oracle)) << ")";
  }
  os << "]";
  if (nest.back().getNumResults() > 0)
    os << " iter_args=" << nest.back().getNumResults();
  // Op-type histogram over the whole band subtree (skip the loop/yield scaffold).
  llvm::MapVector<StringRef, unsigned> hist;
  band.getBody()->walk([&](Operation *op) {
    StringRef n = op->getName().getStringRef();
    if (n.starts_with("affine.for") || n.ends_with(".yield") ||
        n.starts_with("affine.apply"))
      return;
    hist[n]++;
  });
  os << " ops{";
  bool first = true;
  for (auto &kv : hist) {
    if (!first)
      os << " ";
    os << kv.first << ":" << kv.second;
    first = false;
  }
  os << "}";
  band->emitRemark(os.str());
}

static bool materializeWholeFunc(func::FuncOp fn, const ParAliasOracle &oracle,
                                 unsigned &nForall, unsigned &nCritical,
                                 unsigned &nReplicated, unsigned &nMoved,
                                 unsigned &nBarrier, unsigned &shardable,
                                 unsigned &nBands, unsigned &nElided, bool diag,
                                 StringRef &bailReason) {
  Block &entry = fn.getBody().front();
  Operation *firstBand = nullptr, *lastBand = nullptr;
  for (Operation &op : entry)
    if (isa<affine::AffineForOp>(op)) {
      if (!firstBand)
        firstBand = &op;
      lastBand = &op;
    }
  if (!firstBand)
    return false; // no bands

  SmallVector<Operation *> span;
  for (Operation *op = firstBand;; op = op->getNextNode()) {
    span.push_back(op);
    if (op == lastBand)
      break;
  }

  enum class Disp { Move, Forall, SeqWrap, Critical, Replicate };
  // Plan pass: classify; `moved` tracks ops hoisted before the region so a
  // pure op depending only on moved/pre-span values can be moved too.
  llvm::SmallPtrSet<Operation *, 32> moved;
  auto dominatesRegion = [&](Value v) {
    Operation *d = v.getDefiningOp();
    if (!d)
      return true; // block/func argument
    if (moved.count(d))
      return true; // hoisted before the region
    return d->getBlock() == &entry && d->isBeforeInBlock(firstBand);
  };
  SmallVector<Disp> disp;
  disp.reserve(span.size());
  for (Operation *op : span) {
    Disp d;
    if (auto band = dyn_cast<affine::AffineForOp>(op)) {
      d = perBandShard(band, oracle)  ? Disp::Forall
          : seqWrappable(band, oracle) ? Disp::SeqWrap
                                       : Disp::Critical;
    } else {
      bool operandsDominate = llvm::all_of(
          op->getOperands(), [&](Value v) { return dominatesRegion(v); });
      bool isAlloc = isa<memref::AllocOp, memref::AllocaOp>(op);
      if ((isAlloc || isMemoryEffectFree(op)) && operandsDominate) {
        moved.insert(op); // hoist before the team (computed once, shared)
        d = Disp::Move;
      } else if (isAlloc) {
        // A shared buffer whose (dynamic) size depends on an in-region value
        // can't be hoisted, and replicating it would make it thread-private
        // (unsound for a shared activation).  Bail.
        bailReason = "an alloc size depends on an in-region value";
        return false;
      } else if (mayWriteOrFree(op)) {
        d = Disp::Critical;
      } else {
        d = Disp::Replicate;
      }
    }
    // In-region ops (replicate / critical glue) produce values that live inside
    // omp.parallel -> they must not be used after the span.  Bands have no
    // results; moved ops may escape freely.
    if ((d == Disp::Replicate || (d == Disp::Critical && !isa<affine::AffineForOp>(op))) &&
        usedAfter(op, lastBand, entry)) {
      bailReason = "an in-region glue value is used after the band span";
      return false;
    }
    disp.push_back(d);
  }

  OpBuilder b(firstBand);
  Location loc = firstBand->getLoc();
  auto regionOp = b.create<par::RegionOp>(loc);
  Block *rblk = b.createBlock(&regionOp.getRegion());
  IRMapping map;

  auto cloneCritical = [&](Operation *op) {
    b.setInsertionPointToEnd(rblk);
    auto crit = b.create<par::CriticalOp>(loc);
    b.createBlock(&crit.getRegion());
    b.setInsertionPointToStart(crit.getBody());
    b.clone(*op, map);
    b.setInsertionPointToEnd(crit.getBody());
    b.create<par::YieldOp>(loc, ValueRange{});
    ++nCritical;
  };

  // Build ONE par.forall (no preceding barrier) at b's current insertion point,
  // sharding `info`'s axis; leaves b inside the forall body.  Shared by the
  // top-level Forall band and the SeqWrap inner bands.
  auto emitForall = [&](PerBandInfo info) {
    bool dyn = info.dynUb || info.dynExpr;
    int64_t ubEntry = dyn ? ShapedType::kDynamic : info.ubConst;
    SmallVector<Value> dynOps;
    if (info.dynExpr) {
      // Expand the shard loop's affine upper bound (e.g. N-1) to an SSA value
      // at the forall's insertion point (operands dominate the region).
      affine::AffineForOp shard = info.nest[info.shardIdx];
      dynOps.push_back(expandForBound(b, loc, shard.getUpperBoundMap(),
                                      shard.getUpperBoundOperands(),
                                      /*isUpper=*/true, map));
    } else if (info.dynUb) {
      dynOps.push_back(map.lookupOrDefault(info.dynUb));
    }
    auto forall = b.create<par::ForallOp>(
        loc, TypeRange{}, b.getDenseI64ArrayAttr({info.lb}),
        b.getDenseI64ArrayAttr({ubEntry}), b.getDenseI64ArrayAttr({info.step}),
        dynOps, ValueRange{});
    // Triangular / IV-dependent inner extent -> tag for a dynamic OpenMP
    // schedule (a static block schedule starves the low-IV workers).
    if (bandImbalanced(info.nest[info.shardIdx]))
      forall->setAttr("par.dynamic", b.getUnitAttr());
    Block *fblk = b.createBlock(&forall.getRegion());
    fblk->addArgument(b.getIndexType(), loc);
    map.map(info.nest[info.shardIdx].getInductionVar(), fblk->getArgument(0));
    b.setInsertionPointToStart(fblk);
    if (info.perfect) {
      buildSeqSkip(b, info.nest, /*idx=*/0, info.shardIdx, map);
    } else {
      // Degenerate loops above the shard axis run exactly once: map their IVs to
      // their constant lower bound so the de-affined shard body, which may
      // reference them, resolves correctly (sound trivial interchange).
      for (unsigned k = 0; k < info.shardIdx; ++k) {
        affine::AffineForOp dl = info.nest[k];
        Value c = b.create<arith::ConstantIndexOp>(
            loc, dl.getConstantLowerBound());
        map.map(dl.getInductionVar(), c);
      }
      buildShardedBody(b, info.nest[info.shardIdx], map);
    }
    b.create<par::YieldOp>(loc, ValueRange{});
    ++nForall;
    ++shardable;
  };

  // Barrier elision (PARALLEL_SPMD_SPEC.md §4).  A barrier before a Forall band
  // is needed iff, since the last barrier, some worker wrote a location that a
  // *different* worker accesses in this band.  Track every Forall band emitted
  // since the last barrier (`pending`); a non-Forall state producer (Critical
  // band/glue, SeqWrap) sets `forcesBarrier` so the next band always syncs (the
  // conservative case -- this matches the pre-elision behavior exactly).  A
  // Forall band B elides its barrier iff EVERY pending band A shards the SAME
  // owner partition (identical lb/ub/step => omp static schedule maps iteration
  // s to the same worker in both) AND A->B is owner-aligned on the shard axis
  // (classifyEdgeShard == Elide).  This only ever *removes* barriers the
  // pre-elision code emitted; every other sync site is unchanged.
  struct Pend {
    affine::AffineForOp shardLoop;
    Value shardIV;
    PerBandInfo info;
  };
  SmallVector<Pend, 32> pending;
  bool forcesBarrier = false;
  auto samePartition = [](const PerBandInfo &a, const PerBandInfo &b) {
    if (a.dynExpr || b.dynExpr)
      return false; // affine-expr extents: don't reason about the partition
    if (a.lb != b.lb || a.step != b.step)
      return false;
    if ((a.dynUb != nullptr) != (b.dynUb != nullptr))
      return false;
    if (a.dynUb)
      return a.dynUb == b.dynUb; // same runtime extent value
    return a.ubConst == b.ubConst;
  };
  // Decide whether a barrier must precede Forall band B (info `bInfo`).
  auto needBarrierForall = [&](const PerBandInfo &bInfo) -> bool {
    if (nForall + nCritical == 0)
      return false; // nothing emitted yet
    if (forcesBarrier)
      return true; // a non-Forall producer since the last barrier
    affine::AffineForOp bShard = bInfo.nest[bInfo.shardIdx];
    Value bIV = bShard.getInductionVar();
    // Elision relies on the omp STATIC schedule mapping iteration s to the same
    // worker in both bands.  An imbalanced band lowers to a DYNAMIC schedule
    // (par.dynamic) -> the owner of s is not statically determined -> no aligned
    // owner-computes guarantee; keep its sync.
    if (bandImbalanced(bShard))
      return true;
    for (Pend &p : pending) {
      if (bandImbalanced(p.shardLoop))
        return true;
      if (!samePartition(p.info, bInfo))
        return true; // different owner partition: can't prove owner-aligned
      if (classifyEdgeShard(p.shardLoop.getOperation(), p.shardIV,
                            bShard.getOperation(), bIV) != EdgeKind::Elide)
        return true; // a non-aligned cross-shard dependence
    }
    return false; // all pending bands are owner-aligned on B: elide
  };

  for (size_t i = 0, e = span.size(); i < e; ++i) {
    Operation *op = span[i];
    switch (disp[i]) {
    case Disp::Move:
      op->moveBefore(regionOp); // keep identity; dominates region + post-span
      ++nMoved;
      break;
    case Disp::Forall: {
      ++nBands;
      auto info = perBandShard(cast<affine::AffineForOp>(op), oracle);
      b.setInsertionPointToEnd(rblk);
      if (needBarrierForall(*info)) {
        b.create<par::BarrierOp>(loc);
        ++nBarrier;
        pending.clear();
        forcesBarrier = false;
      } else if (nForall + nCritical > 0) {
        ++nElided; // a barrier the pre-elision path would have emitted
      }
      emitForall(*info);
      affine::AffineForOp sl = info->nest[info->shardIdx];
      pending.push_back({sl, sl.getInductionVar(), *info});
      break;
    }
    case Disp::SeqWrap: {
      // Sequential outer band -> scf.for (run by the whole team); each inner
      // parallel band -> par.forall, synchronized by the implicit
      // end-of-wsloop barrier (par.barrier may not nest under scf.for).
      ++nBands;
      auto outer = cast<affine::AffineForOp>(op);
      b.setInsertionPointToEnd(rblk);
      if (nForall + nCritical > 0) {
        b.create<par::BarrierOp>(loc);
        ++nBarrier;
      }
      pending.clear();
      forcesBarrier = true; // its internal wsloop barriers don't fully sync it
      Value slb = expandForBound(b, loc, outer.getLowerBoundMap(),
                                 outer.getLowerBoundOperands(),
                                 /*isUpper=*/false, map);
      Value sub = expandForBound(b, loc, outer.getUpperBoundMap(),
                                 outer.getUpperBoundOperands(),
                                 /*isUpper=*/true, map);
      Value sst = b.create<arith::ConstantIndexOp>(loc, outer.getStepAsInt());
      auto sfor = b.create<scf::ForOp>(loc, slb, sub, sst);
      map.map(outer.getInductionVar(), sfor.getInductionVar());
      for (Operation &child : outer.getBody()->without_terminator()) {
        b.setInsertionPoint(sfor.getBody()->getTerminator());
        if (auto cb = dyn_cast<affine::AffineForOp>(&child))
          emitForall(*perBandShard(cb, oracle));
        else
          cloneDeAffine(b, &child, map); // pure glue, replicated in the loop
      }
      break;
    }
    case Disp::Critical:
      if (auto cband = dyn_cast<affine::AffineForOp>(op)) {
        ++nBands;
        if (diag)
          describeCriticalBand(cband, oracle);
        b.setInsertionPointToEnd(rblk);
        if (nForall + nCritical > 0) {
          b.create<par::BarrierOp>(loc);
          ++nBarrier;
        }
        pending.clear();
      }
      cloneCritical(op);
      // A critical band/glue op writes shared state single-worker; the next
      // band must barrier to see it (matches the pre-elision nCritical>0 gate).
      forcesBarrier = true;
      break;
    case Disp::Replicate:
      b.setInsertionPointToEnd(rblk);
      b.clone(*op, map);
      ++nReplicated;
      break;
    }
  }
  b.setInsertionPointToEnd(rblk);
  b.create<par::YieldOp>(loc, ValueRange{});
  // Erase the originals we cloned into the region (everything but moved ops),
  // in reverse program order.
  for (size_t i = span.size(); i-- > 0;)
    if (disp[i] != Disp::Move)
      span[i]->erase();
  return true;
}

/// S0 shard-axis selection: the operand-agnostic bound id (shardAxisId) covering
/// the most top-level parallel bands, preferring axes with real parallelism
/// (extent != 1); falls back to any if all candidates are degenerate.  Reports
/// the chosen axis `coverage` and the `total` parallel bands.  nullopt = none.
static std::optional<std::string>
selectShardAxis(ArrayRef<affine::AffineForOp> bands,
                const ParAliasOracle &oracle, unsigned &coverage,
                unsigned &total) {
  std::map<std::string, unsigned> idCount;
  std::set<std::string> degenerate; // extent==1 axes: no real parallelism
  total = 0;
  for (affine::AffineForOp band : bands)
    if (auto id = shardAxisId(band, oracle)) {
      idCount[*id]++;
      ++total;
      if (shardLabel(band) == "extent=1")
        degenerate.insert(*id);
    }
  if (idCount.empty())
    return std::nullopt;
  std::string shardId;
  coverage = 0;
  for (auto &kv : idCount)
    if (!degenerate.count(kv.first) && kv.second > coverage) {
      coverage = kv.second;
      shardId = kv.first;
    }
  if (coverage == 0)
    for (auto &kv : idCount)
      if (kv.second > coverage) {
        coverage = kv.second;
        shardId = kv.first;
      }
  return shardId;
}

//===----------------------------------------------------------------------===//

struct DrParBubblesPass
    : public impl::DrParBubblesPassBase<DrParBubblesPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    ParAliasOracle oracle;

    // M0 — per-loop axis classification.
    if (parTestDiagnostics) {
      module.walk([&](func::FuncOp fn) {
        if (fn.isExternal())
          return;
        fn.walk([&](Operation *op) {
          if (!isa<affine::AffineForOp, scf::ForOp>(op))
            return;
          AxisKind k = classifyLoop(op, oracle);
          LLVM_DEBUG(llvm::dbgs() << "seed bubble @" << op << " -> "
                                  << describe(k) << "\n");
          op->emitRemark(describe(k));
        });
      });
    }

    // M1 — maximal-region formation (climb + conformant fuse).
    if (parTestRegions) {
      module.walk([&](func::FuncOp fn) {
        if (fn.isExternal())
          return;
        llvm::DenseMap<Block *, SmallVector<affine::AffineForOp, 4>> byBlock;
        SmallVector<Block *, 8> order;
        fn.walk([&](affine::AffineForOp loop) {
          if (!isBandRoot(loop))
            return;
          auto &v = byBlock[loop->getBlock()];
          if (v.empty())
            order.push_back(loop->getBlock());
          v.push_back(loop);
        });
        for (Block *b : order)
          processBlock(byBlock[b], oracle);
      });
    }

    // M2/M3 — materialize maximal regions into the `par` dialect.  Single
    // climb bands -> par.region/par.forall (+ scf.for suffix); runs of
    // conformant depth-1 parallel siblings -> fused foralls / barrier groups.
    if (parMaterialize) {
      module.walk([&](func::FuncOp fn) {
        if (fn.isExternal())
          return;
        // Band roots per block, in program order (preserved by the walk).
        llvm::DenseMap<Block *, SmallVector<affine::AffineForOp, 4>> byBlock;
        SmallVector<Block *, 8> order;
        fn.walk([&](affine::AffineForOp loop) {
          if (!isBandRoot(loop))
            return;
          auto &v = byBlock[loop->getBlock()];
          if (v.empty())
            order.push_back(loop->getBlock());
          v.push_back(loop);
        });
        for (Block *blk : order)
          materializeBlock(byBlock[blk], oracle);
      });
    }

    // S0/S1 — whole-function shard-axis + barrier-elision spike (diagnostic).
    if (parTestSpmd) {
      module.walk([&](func::FuncOp fn) {
        if (fn.isExternal())
          return;
        SmallVector<affine::AffineForOp, 16> bands;
        fn.walk([&](affine::AffineForOp loop) {
          if (isBandRoot(loop))
            bands.push_back(loop);
        });
        // S0: pick the shard axis (by operand-agnostic bound identity)
        // covering the most top-level parallel bands.
        unsigned best = 0, total = 0;
        auto sel = selectShardAxis(bands, oracle, best, total);
        if (!sel)
          return;
        std::string shardId = *sel;
        std::string label;
        for (affine::AffineForOp band : bands)
          if (auto id = shardAxisId(band, oracle))
            if (*id == shardId) {
              label = shardLabel(band);
              break;
            }
        fn.emitRemark("par-shard: ") << label << " bands=" << best << "/"
                                     << total;

        // S1: classify each edge between consecutive shard-axis bands.
        unsigned elide = 0, halo = 0, redist = 0, barrier = 0;
        affine::AffineForOp prev = nullptr;
        for (affine::AffineForOp band : bands) {
          auto id = shardAxisId(band, oracle);
          if (!id || *id != shardId) {
            prev = nullptr; // off-axis band breaks the SPMD chain
            continue;
          }
          if (prev) {
            EdgeKind k = classifyEdge(prev, band);
            band->emitRemark("par-edge: ") << edgeName(k);
            switch (k) {
            case EdgeKind::Elide: ++elide; break;
            case EdgeKind::Halo: ++halo; break;
            case EdgeKind::Redistribute: ++redist; break;
            case EdgeKind::Barrier: ++barrier; break;
            }
          }
          prev = band;
        }
        unsigned edges = elide + halo + redist + barrier;
        fn.emitRemark("par-spmd: elide=")
            << elide << " halo=" << halo << " redistribute=" << redist
            << " barrier=" << barrier << " (of " << edges << " shard edges)";
      });
    }

    // S2 — whole-function widening + materialization into one par.region.
    if (parSpmd) {
      module.walk([&](func::FuncOp fn) {
        if (fn.isExternal())
          return;
        SmallVector<affine::AffineForOp, 16> bands;
        fn.walk([&](affine::AffineForOp loop) {
          if (isBandRoot(loop))
            bands.push_back(loop);
        });
        unsigned coverage = 0, total = 0;
        auto sel = selectShardAxis(bands, oracle, coverage, total);
        if (!sel)
          return;
        std::string shardId = *sel;

        // Shard-axis bounds from a representative on-axis band.  lb/step must be
        // constant; the upper bound is either a constant or a single runtime
        // Value (`0 to %N`, e.g. a dynamic batch size) -- spmdBand then requires
        // every on-axis band to share that SAME Value (owner-computes needs one
        // shard space).  A complex affine ub bails.
        int64_t lb = 0, ubConst = 0, step = 1;
        Value dynUb = nullptr;
        bool got = false;
        for (affine::AffineForOp band : bands)
          if (auto id = shardAxisId(band, oracle); id && *id == shardId) {
            if (!band.hasConstantLowerBound())
              break;
            lb = band.getConstantLowerBound();
            step = band.getStepAsInt();
            if (band.hasConstantUpperBound())
              ubConst = band.getConstantUpperBound();
            else if (!(dynUb = simpleDynUb(band)))
              break; // non-constant, non-simple bound: too complex
            got = true;
            break;
          }
        if (!got) {
          fn.emitRemark(
              "par-spmd: not materialized (non-constant lb / complex shard "
              "bound)");
          return;
        }

        // Materialize only the function entry block's top-level bands.  Every
        // one must be an on-axis materializable shard band; they must be
        // contiguous; and their bodies must depend only on values dominating
        // the first band (so the region is well-formed).  Any miss => bail
        // (sound: real kernels with per-layer scratch / off-axis reductions /
        // interleaved ops fall here -- the documented S2 blockers).
        Block &entry = fn.getBody().front();
        SmallVector<affine::AffineForOp, 8> roots;
        for (Operation &op : entry)
          if (auto f = dyn_cast<affine::AffineForOp>(&op))
            roots.push_back(f);
        if (roots.empty())
          return;

        SmallVector<SmallVector<affine::AffineForOp, 4>, 8> nests;
        for (affine::AffineForOp r : roots) {
          auto nest = spmdBand(r, oracle, lb, step, ubConst, dynUb);
          if (nest.empty()) {
            fn.emitRemark(
                "par-spmd: not materialized (off-axis or non-materializable "
                "band)");
            return;
          }
          nests.push_back(std::move(nest));
        }
        // Hoist movable inter-band glue (scratch allocs / pure index+view ops)
        // so the bands become contiguous; bail on a non-hoistable op.
        if (!hoistInterBandGlue(roots)) {
          fn.emitRemark(
              "par-spmd: not materialized (non-hoistable op between shard "
              "bands)");
          return;
        }
        for (affine::AffineForOp r : roots)
          if (!nestOperandsDominate(r, roots.front().getOperation())) {
            fn.emitRemark(
                "par-spmd: not materialized (inter-band value dependence)");
            return;
          }

        unsigned nForall = 0, nBarrier = 0, nRedist = 0;
        materializeSpmd(roots, nests, lb, step, ubConst, dynUb, nForall,
                        nBarrier, nRedist);
        fn.emitRemark("par-spmd: materialized foralls=")
            << nForall << " barriers=" << nBarrier
            << " redistributes=" << nRedist << " ("
            << (dynUb ? "dyn" : std::to_string(ubConst - lb)) << " extent, "
            << roots.size() << " bands)";
      });
    }

    // S7 — batch-1 within-sample per-band sharding (whole-function widening).
    if (parSpmdPerband) {
      module.walk([&](func::FuncOp fn) {
        if (fn.isExternal())
          return;
        unsigned nForall = 0, nCritical = 0, nReplicated = 0, nMoved = 0,
                 nBarrier = 0, shardable = 0, nBands = 0, nElided = 0;
        StringRef bail;
        if (!materializeWholeFunc(fn, oracle, nForall, nCritical, nReplicated,
                                  nMoved, nBarrier, shardable, nBands, nElided,
                                  parSpmdDiag, bail)) {
          if (!bail.empty())
            fn.emitRemark("par-spmd-perband: not materialized (") << bail << ")";
          return;
        }
        fn.emitRemark("par-spmd-perband: materialized foralls=")
            << nForall << " critical=" << nCritical
            << " replicated=" << nReplicated << " moved=" << nMoved
            << " barriers=" << nBarrier << " elided=" << nElided << " ("
            << nBands << " bands, " << shardable << " parallel)";
      });
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createDrParBubblesPass() {
  return std::make_unique<DrParBubblesPass>();
}
