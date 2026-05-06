//===- CrossFnOrdered.cpp - Strategy 4 family ----------------------------===//
//
//===----------------------------------------------------------------------===//

#include "drcompiler/Transforms/DataRecomputation/Strategies/CrossFnOrdered.h"

#include "drcompiler/Transforms/DataRecomputation/CacheCostModel.h"
#include "drcompiler/Transforms/DataRecomputation/RematKernel.h"
#include "drcompiler/Transforms/Utils/MemrefBaseAnalysis.h"
#include "drcompiler/Transforms/Utils/OpDispatchUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

using drcompiler::collectBaseMemrefs;
using drcompiler::getStoredValue;
using drcompiler::getLoadStoreMemref;

namespace dr::strategies {

namespace {

// Local copy of the purity check used in DataRecomputation.cpp. Keeps the
// CrossFnOrdered translation unit self-contained.
bool isPureRuntimeCall(mlir::Operation *op) {
  auto call = mlir::dyn_cast<mlir::CallOpInterface>(op);
  if (!call) return false;
  if (mlir::isMemoryEffectFree(op)) return true;
  auto callableRef = call.getCallableForCallee();
  auto sym = mlir::dyn_cast<mlir::SymbolRefAttr>(callableRef);
  if (!sym) return false;
  llvm::StringRef name = sym.getRootReference().getValue();
  static constexpr llvm::StringLiteral pureCalls[] = {
    "_Msin_f64_f64",       "_Mcos_f64_f64",        "_Mtan_f64_f64",
    "_Masin_f64_f64",      "_Macos_f64_f64",       "_Matan_f64_f64",
    "_Matan2_f64_f64_f64", "_Msinh_f64_f64",       "_Mcosh_f64_f64",
    "_Mtanh_f64_f64",      "_Mexp_f64_f64",        "_Mlog_f64_f64",
    "_Mlog10_f64_f64",     "_Msqrt_f64_f64",       "_Mpow_f64_f64_f64",
    "_Mabs_f64_f64",       "_Msign_f64_f64",       "_Mceil_f64_f64",
    "_Mfloor_f64_f64",
  };
  for (auto p : pureCalls)
    if (name == p) return true;
  return false;
}

} // namespace







LoadSubRecord::LoadSubRecord() = default;
LoadSubRecord::LoadSubRecord(LoadSubRecord &&) noexcept = default;
LoadSubRecord &
LoadSubRecord::operator=(LoadSubRecord &&) noexcept = default;
LoadSubRecord::~LoadSubRecord() = default;


/// Trace memref operand back to a memref.get_global op, accepting only the
/// trivial chain memref.get_global → load. Returns nullptr if not such a
/// trivial chain (e.g. through subview, cast, function arg).
mlir::Operation *traceLoadToGlobal(mlir::Value memref) {
  if (auto getGlobal =
          mlir::dyn_cast_or_null<mlir::memref::GetGlobalOp>(memref.getDefiningOp()))
    return getGlobal.getOperation();
  return nullptr;
}

/// Resolve a get_global op to its memref::GlobalOp definition.
mlir::Operation *
resolveGetGlobal(mlir::Operation *getGlobalOp, mlir::ModuleOp moduleOp) {
  auto gg = mlir::dyn_cast_or_null<mlir::memref::GetGlobalOp>(getGlobalOp);
  if (!gg) return nullptr;
  return mlir::SymbolTable::lookupNearestSymbolFrom(moduleOp,
                                                    gg.getNameAttr());
}

/// Helper: extract the memref operand of a load op (memref/affine).
mlir::Value getLoadMemref(mlir::Operation *def) {
  if (auto l = mlir::dyn_cast<mlir::memref::LoadOp>(def))
    return l.getMemref();
  if (auto l = mlir::dyn_cast<mlir::affine::AffineLoadOp>(def))
    return l.getMemref();
  return nullptr;
}

/// Forward decl: the unified entry point. depth and seenStores guard
/// recursion (Strategy D).
bool buildCrossFunctionRematPlan(
    mlir::Operation *storeOp,
    mlir::FunctionOpInterface writerFn,
    AllocationRoots &allocRootFor,
    LoadProvenanceMap &loadProv,
    llvm::function_ref<bool(mlir::Operation *, mlir::Operation *)>
        fnWritesGlobal,
    CrossFnRematPlan &plan,
    unsigned depth,
    llvm::SmallDenseSet<mlir::Operation *> seenStores);

/// Find the outermost scf.for / affine.for ancestor of \p op that is
/// nested inside \p funcOp. Returns nullptr if op is at the top level of
/// funcOp or not under any supported loop.
mlir::Operation *
findOutermostEnclosingLoop(mlir::Operation *op,
                           mlir::FunctionOpInterface funcOp) {
  mlir::Operation *outer = nullptr;
  mlir::Operation *cur = op->getParentOp();
  while (cur && cur != funcOp.getOperation()) {
    if (mlir::isa<mlir::scf::ForOp, mlir::affine::AffineForOp>(cur))
      outer = cur;
    else if (!mlir::isa<mlir::scf::ParallelOp>(cur)) {
      // Any other region op (scf.if, etc.) — unsupported.
      // We don't reject here because the loop might still wrap it; only
      // the eventual buildLoopPlan body-check enforces purity.
    }
    cur = cur->getParentOp();
  }
  return outer;
}

/// Verify that \p loop's bounds (lower, upper, step) reduce to entry-block
/// args of \p writerFn or compile-time constants. Returns true and sets
/// \p staticBounds=true if all three are constants.
bool loopBoundsAreEntryArgsOrConst(mlir::Operation *loop,
                                          mlir::FunctionOpInterface writerFn,
                                          bool &staticBounds) {
  mlir::Region *body = writerFn.getCallableRegion();
  if (!body || body->empty()) return false;
  mlir::Block *entry = &body->front();

  auto isAcceptable = [&](mlir::Value v) -> bool {
    if (!v) return false;
    if (auto barg = mlir::dyn_cast<mlir::BlockArgument>(v))
      return barg.getOwner() == entry;
    auto *def = v.getDefiningOp();
    return def && def->hasTrait<mlir::OpTrait::ConstantLike>();
  };

  if (auto sf = mlir::dyn_cast<mlir::scf::ForOp>(loop)) {
    bool lbC = sf.getLowerBound().getDefiningOp() &&
               sf.getLowerBound().getDefiningOp()->hasTrait<mlir::OpTrait::ConstantLike>();
    bool ubC = sf.getUpperBound().getDefiningOp() &&
               sf.getUpperBound().getDefiningOp()->hasTrait<mlir::OpTrait::ConstantLike>();
    bool stC = sf.getStep().getDefiningOp() &&
               sf.getStep().getDefiningOp()->hasTrait<mlir::OpTrait::ConstantLike>();
    staticBounds = lbC && ubC && stC;
    if (!isAcceptable(sf.getLowerBound())) return false;
    if (!isAcceptable(sf.getUpperBound())) return false;
    if (!isAcceptable(sf.getStep())) return false;
    if (!sf.getInitArgs().empty()) return false; // no iter_args
    return true;
  }
  if (auto af = mlir::dyn_cast<mlir::affine::AffineForOp>(loop)) {
    // Affine bounds may be constants or affine_maps over entry-block args.
    // Accept the constant lb/ub case + the case where map operands are
    // entry-block args.
    auto checkOperands = [&](mlir::OperandRange ops) {
      for (mlir::Value v : ops)
        if (!isAcceptable(v)) return false;
      return true;
    };
    if (!checkOperands(af.getLowerBoundOperands())) return false;
    if (!checkOperands(af.getUpperBoundOperands())) return false;
    staticBounds = af.hasConstantLowerBound() && af.hasConstantUpperBound();
    if (!af.getInits().empty()) return false; // no iter_args
    return true;
  }
  return false;
}

/// Straight-line plan-builder: walks storeOp's stored-value operand tree
/// at-the-top-level of writerFn (no enclosing loops). Same shape as the
/// pre-D code, plus: (a) on inner-load hoist failure, attempt recursive
/// sub-plan substitution via loadProv (Strategy D); (b) accepts loop-IV
/// block-args when \p inLoopBody is the parent region under which IVs are
/// legal (used by buildLoopPlan to share this walker).
bool walkStoredValueTree(
    mlir::Value root,
    mlir::FunctionOpInterface writerFn,
    mlir::Region *loopRegion, // nullable; iv-args of blocks under this are accepted
    AllocationRoots &allocRootFor,
    LoadProvenanceMap &loadProv,
    llvm::function_ref<bool(mlir::Operation *, mlir::Operation *)>
        fnWritesGlobal,
    CrossFnRematPlan &plan,
    unsigned depth,
    llvm::SmallDenseSet<mlir::Operation *> &seenStores) {
  mlir::Region *body = writerFn.getCallableRegion();
  if (!body || body->empty()) return false;
  mlir::Block *entryBlock = &body->front();
  mlir::ModuleOp moduleOp = writerFn->getParentOfType<mlir::ModuleOp>();

  llvm::SmallVector<mlir::Value, 8> worklist{root};
  llvm::SmallDenseSet<mlir::Value> visited;
  llvm::SmallDenseSet<mlir::Operation *> added;
  llvm::SmallDenseSet<unsigned> seenArgIdx;

  while (!worklist.empty()) {
    mlir::Value cur = worklist.pop_back_val();
    if (!visited.insert(cur).second) continue;

    if (auto barg = mlir::dyn_cast<mlir::BlockArgument>(cur)) {
      if (barg.getOwner() == entryBlock) {
        if (seenArgIdx.insert(barg.getArgNumber()).second)
          plan.writerArgs.push_back(barg);
        continue;
      }
      // Loop-IV block-arg: accept only if its owner block lives under
      // loopRegion (the loop we plan to clone). Otherwise unavailable
      // at caller.
      if (loopRegion && loopRegion->findAncestorBlockInRegion(
                            *barg.getOwner()) == barg.getOwner())
        continue; // IV reproduced via region clone at caller
      return false;
    }

    mlir::Operation *def = cur.getDefiningOp();
    if (!def) return false;

    if (mlir::isa<mlir::memref::LoadOp, mlir::affine::AffineLoadOp>(def)) {
      mlir::Value memref = getLoadMemref(def);
      mlir::Operation *getGlobalOp = traceLoadToGlobal(memref);
      if (!getGlobalOp) return false;
      mlir::Operation *globalOp = resolveGetGlobal(getGlobalOp, moduleOp);

      // Strategy D — try a recursive sub-plan first when loadProv has a
      // unique cross-fn store. Sub-plan is strictly more permissive at
      // the per-caller stage (the global itself may be written elsewhere
      // in caller's call subtree, as long as the inner-writer-call
      // window is interference-free). Fall back to hoist when sub-plan
      // is unavailable.
      auto trySubPlan = [&]() -> bool {
        if (depth + 1 >= kMaxCrossFnChainDepth) return false;
        auto provIt = loadProv.find(def);
        if (provIt == loadProv.end()) return false;
        const auto &provSet = provIt->second;
        if (provSet.size() != 1 || provSet.contains(nullptr)) return false;
        mlir::Operation *innerStore = *provSet.begin();
        if (!innerStore) return false;
        if (seenStores.contains(innerStore)) return false;
        auto innerWriterFn =
            innerStore->getParentOfType<mlir::FunctionOpInterface>();
        if (!innerWriterFn) return false;
        if (innerWriterFn.getOperation() == writerFn.getOperation())
          return false;
        mlir::Value innerStoredVal = getStoredValue(innerStore);
        if (!innerStoredVal) return false;
        if (innerStoredVal.getType() != def->getResult(0).getType())
          return false;
        LoadSubRecord sub;
        sub.loadOp = def;
        sub.innerStoreOp = innerStore;
        sub.innerWriterFn = innerWriterFn;
        sub.subPlan = std::make_unique<CrossFnRematPlan>();
        auto subSeen = seenStores;
        subSeen.insert(innerStore);
        if (!buildCrossFunctionRematPlan(innerStore, innerWriterFn,
                                         allocRootFor, loadProv,
                                         fnWritesGlobal, *sub.subPlan,
                                         depth + 1, subSeen))
          return false;
        mlir::Value innerStoreMemref =
            drcompiler::getLoadStoreMemref(innerStore);
        sub.innerGlobalOp =
            lookupAllocRoot(innerStoreMemref, allocRootFor);
        if (!sub.innerGlobalOp ||
            !mlir::isa<mlir::memref::GlobalOp>(sub.innerGlobalOp))
          return false;
        plan.loadSubs.push_back(std::move(sub));
        return true;
      };

      if (trySubPlan()) {
        // Don't add the load to opsToClone; don't walk its operands —
        // substitution is total at this load result.
        continue;
      }
      // Fall back to hoist path.
      bool canHoist = globalOp &&
                      !fnWritesGlobal(writerFn.getOperation(), globalOp);
      if (!canHoist) return false;
      plan.hoistedLoadGlobals.push_back(globalOp);
      if (added.insert(def).second) {
        plan.opsToClone.push_back(def);
        if (plan.opsToClone.size() > kMaxRematOps) return false;
      }
      for (mlir::Value operand : def->getOperands())
        worklist.push_back(operand);
      continue;
    }
    if (mlir::isa<mlir::LLVM::LoadOp>(def)) {
      return false; // LLVM dialect loads not yet supported
    }
    // get_global is a symbol reference with no value operands; safe to clone
    // at the caller even though it carries MemRead effect on the symbol table.
    if (mlir::isa<mlir::memref::GetGlobalOp>(def)) {
      if (added.insert(def).second) {
        plan.opsToClone.push_back(def);
        if (plan.opsToClone.size() > kMaxRematOps) return false;
      }
      continue;
    }
    if (mlir::isa<mlir::CallOpInterface>(def)) {
      if (!isPureRuntimeCall(def)) return false;
    } else if (!mlir::isMemoryEffectFree(def)) {
      return false;
    }

    if (added.insert(def).second) {
      plan.opsToClone.push_back(def);
      if (plan.opsToClone.size() > kMaxRematOps) return false;
    }
    for (mlir::Value operand : def->getOperands())
      worklist.push_back(operand);
  }
  return true;
}

/// Top-level (no enclosing loop) plan-builder.
bool buildStraightLinePlan(
    mlir::Operation *storeOp,
    mlir::FunctionOpInterface writerFn,
    AllocationRoots &allocRootFor,
    LoadProvenanceMap &loadProv,
    llvm::function_ref<bool(mlir::Operation *, mlir::Operation *)>
        fnWritesGlobal,
    CrossFnRematPlan &plan,
    unsigned depth,
    llvm::SmallDenseSet<mlir::Operation *> seenStores) {
  mlir::Value root = getStoredValue(storeOp);
  if (!root) return false;
  plan.rootValue = root;
  if (!walkStoredValueTree(root, writerFn, /*loopRegion=*/nullptr,
                           allocRootFor, loadProv, fnWritesGlobal, plan,
                           depth, seenStores))
    return false;
  std::reverse(plan.opsToClone.begin(), plan.opsToClone.end());
  return true;
}

/// Loop-cloning plan-builder (Strategy F.2). The store is nested inside an
/// scf.for / affine.for. We will clone the outermost enclosing loop into
/// the caller, redirecting writes from the original global to a scratch
/// buffer.
bool buildLoopPlan(
    mlir::Operation *storeOp,
    mlir::FunctionOpInterface writerFn,
    mlir::Operation *outerLoop,
    AllocationRoots &allocRootFor,
    LoadProvenanceMap &loadProv,
    llvm::function_ref<bool(mlir::Operation *, mlir::Operation *)>
        fnWritesGlobal,
    CrossFnRematPlan &plan,
    unsigned depth,
    llvm::SmallDenseSet<mlir::Operation *> seenStores) {
  mlir::ModuleOp moduleOp = writerFn->getParentOfType<mlir::ModuleOp>();

  // Bounds + iter_args check.
  bool staticBounds = false;
  if (!loopBoundsAreEntryArgsOrConst(outerLoop, writerFn, staticBounds))
    return false;

  // The store's memref must trace to a get_global directly (no
  // reinterpret_cast / strided sub-region for v1).
  mlir::Value storeMemref = drcompiler::getLoadStoreMemref(storeOp);
  if (!storeMemref) return false;
  mlir::Operation *getGlobalOp =
      mlir::dyn_cast_or_null<mlir::memref::GetGlobalOp>(
          storeMemref.getDefiningOp());
  // Marco emits:  %base = get_global @G  →  %rc = reinterpret_cast %base …
  //               memref.store %val, %rc[]
  // Trace through any chain of reinterpret_cast to find the base get_global.
  if (!getGlobalOp) {
    mlir::Value base = storeMemref;
    while (auto rc = mlir::dyn_cast_or_null<mlir::memref::ReinterpretCastOp>(
               base.getDefiningOp()))
      base = rc.getSource();
    getGlobalOp = mlir::dyn_cast_or_null<mlir::memref::GetGlobalOp>(
        base.getDefiningOp());
    if (!getGlobalOp) return false;
  }
  mlir::Operation *globalOp = resolveGetGlobal(getGlobalOp, moduleOp);
  if (!globalOp || !mlir::isa<mlir::memref::GlobalOp>(globalOp))
    return false;

  // Body purity check — walk every op inside outerLoop and ensure each
  // is one of: the targeted store, a pure arith/constant op, an scf.yield
  // / affine.yield, a hoistable load, a pure runtime call, or a nested
  // scf.for/affine.for satisfying the same recursive constraints. Reject
  // any other store (we don't want extra side effects in cloned region).
  bool bodyOk = true;
  outerLoop->walk([&](mlir::Operation *op) {
    if (op == outerLoop) return mlir::WalkResult::advance();
    if (op == storeOp) return mlir::WalkResult::advance();
    if (mlir::isa<mlir::scf::YieldOp, mlir::affine::AffineYieldOp,
                  mlir::scf::ForOp, mlir::affine::AffineForOp>(op))
      return mlir::WalkResult::advance();
    if (drcompiler::isAnyStoreOp(op)) {
      bodyOk = false;
      return mlir::WalkResult::interrupt();
    }
    if (mlir::isa<mlir::memref::LoadOp, mlir::affine::AffineLoadOp>(op))
      return mlir::WalkResult::advance(); // hoist-checked when walking value tree
    if (mlir::isa<mlir::memref::GetGlobalOp>(op))
      return mlir::WalkResult::advance(); // symbol ref, safe to clone
    if (mlir::isa<mlir::CallOpInterface>(op)) {
      if (!isPureRuntimeCall(op)) {
        bodyOk = false;
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    }
    if (!mlir::isMemoryEffectFree(op)) {
      bodyOk = false;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (!bodyOk) return false;

  // Collect captures: values used inside outerLoop but defined outside.
  // The region clone reproduces the loop body verbatim; only external
  // dependencies need to be cloned/mapped at the caller. Skip the
  // targeted get_global's result (we'll remap it to the scratch buffer).
  mlir::Value globalRef = getGlobalOp->getResult(0);
  llvm::SmallVector<mlir::Value, 8> captures;
  llvm::SmallDenseSet<mlir::Value> capSeen;
  outerLoop->walk([&](mlir::Operation *op) {
    for (mlir::Value v : op->getOperands()) {
      if (v == globalRef) continue;
      // Defined outside the loop?
      if (auto barg = mlir::dyn_cast<mlir::BlockArgument>(v)) {
        if (outerLoop->isAncestor(barg.getOwner()->getParentOp()))
          continue; // IV / nested block-arg owned by the loop region
      } else if (auto *def = v.getDefiningOp()) {
        if (outerLoop->isAncestor(def)) continue;
      }
      if (capSeen.insert(v).second) captures.push_back(v);
    }
  });

  // Walk each capture as a tree root (treated as straight-line — they're
  // external to the loop region).
  for (mlir::Value cap : captures) {
    if (!walkStoredValueTree(cap, writerFn, /*loopRegion=*/nullptr,
                             allocRootFor, loadProv, fnWritesGlobal, plan,
                             depth, seenStores))
      return false;
  }
  std::reverse(plan.opsToClone.begin(), plan.opsToClone.end());

  // rootValue retained for downstream type lookups (scalar element type
  // matches the global's element type).
  mlir::Value root = getStoredValue(storeOp);
  if (!root) return false;
  plan.rootValue = root;

  LoopMatRecord rec;
  rec.outerLoopOp = outerLoop;
  rec.storeOp = storeOp;
  rec.globalOp = globalOp;
  rec.getGlobalOp = getGlobalOp;
  rec.boundsAreStatic = staticBounds;

  // F.1 viability: each store index operand must be a pure loop IV (not an
  // affine expression over IVs, not arithmetic). Walk store indices and
  // verify each is the induction var of an scf.for/affine.for that's an
  // ancestor of storeOp.
  llvm::SmallVector<mlir::Value, 4> storeIdxOps;
  if (auto s = mlir::dyn_cast<mlir::memref::StoreOp>(storeOp))
    for (mlir::Value v : s.getIndices()) storeIdxOps.push_back(v);
  else if (auto s = mlir::dyn_cast<mlir::affine::AffineStoreOp>(storeOp))
    for (mlir::Value v : s.getIndices()) storeIdxOps.push_back(v);
  bool ivsOk = !storeIdxOps.empty();
  for (mlir::Value v : storeIdxOps) {
    auto barg = mlir::dyn_cast<mlir::BlockArgument>(v);
    if (!barg) { ivsOk = false; break; }
    auto *loopOp = barg.getOwner()->getParentOp();
    if (auto sf = mlir::dyn_cast<mlir::scf::ForOp>(loopOp)) {
      if (sf.getInductionVar() != v) { ivsOk = false; break; }
    } else if (auto af = mlir::dyn_cast<mlir::affine::AffineForOp>(loopOp)) {
      if (af.getInductionVar() != v) { ivsOk = false; break; }
    } else {
      ivsOk = false; break;
    }
    rec.storeIVs.push_back(barg);
  }
  rec.extractionViable = ivsOk;
  if (!ivsOk) rec.storeIVs.clear();

  plan.loopMat = rec;
  return true;
}

/// Walk \p storeOp's stored-value operand tree. Accept pure ops whose
/// leaves are constants or block args of \p writerFn's entry block.
/// Memref loads are accepted (a) as hoists when safe, or (b) via Strategy
/// D recursive sub-plans. Stores nested inside an scf.for / affine.for are
/// handled via Strategy F (loop materialization).
bool buildCrossFunctionRematPlan(
    mlir::Operation *storeOp,
    mlir::FunctionOpInterface writerFn,
    AllocationRoots &allocRootFor,
    LoadProvenanceMap &loadProv,
    llvm::function_ref<bool(mlir::Operation *, mlir::Operation *)>
        fnWritesGlobal,
    CrossFnRematPlan &plan,
    unsigned depth,
    llvm::SmallDenseSet<mlir::Operation *> seenStores) {
  plan.opsToClone.clear();
  plan.writerArgs.clear();
  plan.hoistedLoadGlobals.clear();
  plan.loadSubs.clear();
  plan.loopMat.reset();

  mlir::Operation *outerLoop =
      findOutermostEnclosingLoop(storeOp, writerFn);
  if (!outerLoop) {
    // Top-level store — straight-line.
    return buildStraightLinePlan(storeOp, writerFn, allocRootFor, loadProv,
                                 fnWritesGlobal, plan, depth, seenStores);
  }
  return buildLoopPlan(storeOp, writerFn, outerLoop, allocRootFor, loadProv,
                       fnWritesGlobal, plan, depth, seenStores);
}

/// Find the latest call to \p writerFn in \p caller that strictly dominates
/// \p readerCall and has no intervening writer to \p globalOp between them.
/// Returns nullptr if no such call exists.
mlir::Operation *
findEligibleWriterCall(mlir::FunctionOpInterface caller,
                       mlir::Operation *readerCall,
                       mlir::FunctionOpInterface writerFn,
                       mlir::Operation *globalOp,
                       mlir::DominanceInfo &domInfo,
                       const ModuleGlobalWrites &moduleGlobalWrites,
                       AllocationRoots &allocRootFor,
                       const llvm::DenseSet<mlir::Operation *> &globalAllocOps,
                       llvm::function_ref<bool(mlir::Operation *,
                                               mlir::Operation *)>
                           fnWritesGlobal) {
  mlir::Region *body = caller.getCallableRegion();
  if (!body || body->empty()) return nullptr;

  // Collect candidate writer calls within caller.
  llvm::SmallVector<mlir::Operation *, 4> candidates;
  body->walk([&](mlir::CallOpInterface call) {
    auto callableRef = call.getCallableForCallee();
    auto sym = mlir::dyn_cast<mlir::SymbolRefAttr>(callableRef);
    if (!sym) return;
    auto rootName = sym.getRootReference().getValue();
    if (rootName != writerFn.getName()) return;
    if (!domInfo.properlyDominates(call.getOperation(), readerCall)) return;
    candidates.push_back(call.getOperation());
  });

  if (candidates.empty()) return nullptr;

  // Pick the latest (closest dominator). Sort by program order: the one
  // that comes last and still dominates readerCall.
  mlir::Operation *latest = nullptr;
  for (mlir::Operation *c : candidates) {
    if (!latest || domInfo.properlyDominates(latest, c))
      latest = c;
  }
  if (!latest) return nullptr;

  // Verify no intervening op between (latest, readerCall) writes to
  // globalOp. We do a conservative scan:
  //   - any direct store whose root is globalOp is a writer
  //   - any call to a function (other than writerFn) that may write
  //     globalOp is a writer
  bool interfered = false;
  body->walk([&](mlir::Operation *op) {
    if (interfered) return mlir::WalkResult::skip();
    if (!domInfo.properlyDominates(latest, op)) return mlir::WalkResult::advance();
    if (!domInfo.properlyDominates(op, readerCall)) return mlir::WalkResult::advance();

    // Direct store?
    if (drcompiler::isAnyStoreOp(op)) {
      mlir::Value memref = drcompiler::getLoadStoreMemref(op);
      llvm::SmallVector<mlir::Value, 2> bases;
      drcompiler::collectBaseMemrefs(memref, allocRootFor, bases);
      for (mlir::Value base : bases) {
        auto it = allocRootFor.find(base);
        if (it != allocRootFor.end() && it->second == globalOp)
          interfered = true;
      }
    }
    // Call site?
    if (auto call = mlir::dyn_cast<mlir::CallOpInterface>(op)) {
      auto sym = mlir::dyn_cast<mlir::SymbolRefAttr>(call.getCallableForCallee());
      if (!sym) {
        interfered = true; // unknown callee
        return mlir::WalkResult::advance();
      }
      // Skip the writer-call itself; we already accept it as the writer.
      if (op == latest) return mlir::WalkResult::advance();
      // Resolve.
      auto *calleeOp = mlir::SymbolTable::lookupNearestSymbolFrom(
          op, sym);
      auto callee = mlir::dyn_cast_or_null<mlir::FunctionOpInterface>(calleeOp);
      if (!callee) {
        interfered = true; return mlir::WalkResult::advance();
      }
      if (fnWritesGlobal(callee.getOperation(), globalOp))
        interfered = true;
    }
    (void)moduleGlobalWrites;
    (void)globalAllocOps;
    return mlir::WalkResult::advance();
  });

  if (interfered) return nullptr;
  return latest;
}

/// Materialize the writer's expression at the caller, just after
/// \p writerCall. For straight-line plans, clones \p plan.opsToClone with
/// the writer's entry-block args remapped to writerCall's operands. For
/// loop-mat plans (Strategy F), allocates a scratch buffer and clones the
/// outer loop into it. For sub-plans (Strategy D), recurses into each
/// LoadSubRecord, anchored after a separately-resolved inner-writer-call
/// in the same caller function.
mlir::Value materializeAtCaller(mlir::Operation *writerCall,
                                       const CrossFnRematPlan &plan,
                                       mlir::OpBuilder &builder,
                                       const MatCtx &ctx);

/// Resolve an inner-writer-call in the same caller function that
/// dominates \p anchorCall and is interference-free for \p sub.
mlir::Operation *
resolveInnerWriterCall(mlir::Operation *anchorCall, const LoadSubRecord &sub,
                       const MatCtx &ctx) {
  auto callerFn =
      anchorCall->getParentOfType<mlir::FunctionOpInterface>();
  if (!callerFn) return nullptr;
  return findEligibleWriterCall(callerFn, anchorCall, sub.innerWriterFn,
                                sub.innerGlobalOp, *ctx.domInfo,
                                *ctx.moduleGlobalWrites, *ctx.allocRootFor,
                                *ctx.globalAllocs, ctx.fnWritesGlobal);
}

/// Strategy F.1 materializer: clones a single iteration's worth of the
/// writer's loop body at the caller, with each loop IV substituted by
/// the corresponding caller-side index value. Returns the cloned stored
/// value as a scalar — no scratch buffer, no loop. \p idxValues must be
/// in caller scope (already mapped through the call boundary).
mlir::Value
materializeExtraction(mlir::Operation *writerCall,
                      const CrossFnRematPlan &plan,
                      mlir::OpBuilder &builder,
                      llvm::ArrayRef<mlir::Value> idxValues) {
  const auto &lm = *plan.loopMat;
  if (idxValues.size() != lm.storeIVs.size()) return {};
  // Caller controls insertion point (already advanced past any const-op
  // clones made for idxValues). Do not reset.

  mlir::IRMapping mapping;
  for (mlir::BlockArgument arg : plan.writerArgs) {
    unsigned idx = arg.getArgNumber();
    if (idx >= writerCall->getNumOperands()) return {};
    mapping.map((mlir::Value)arg, writerCall->getOperand(idx));
  }
  for (auto [iv, val] : llvm::zip(lm.storeIVs, idxValues))
    mapping.map((mlir::Value)iv, val);

  // External captures (constants, hoistable loads etc.).
  for (mlir::Operation *op : plan.opsToClone) {
    mlir::Operation *cloned = builder.clone(*op, mapping);
    for (auto [oldRes, newRes] :
         llvm::zip(op->getResults(), cloned->getResults()))
      mapping.map(oldRes, newRes);
  }

  // Walk inner-loop body ops in pre-order; clone each one (skipping the
  // for-op wrappers, the yields, and the targeted store).
  llvm::SmallVector<mlir::Operation *, 8> bodyOps;
  lm.outerLoopOp->walk<mlir::WalkOrder::PreOrder>(
      [&](mlir::Operation *op) {
        if (op == lm.outerLoopOp) return mlir::WalkResult::advance();
        if (op == lm.storeOp) return mlir::WalkResult::advance();
        if (mlir::isa<mlir::scf::YieldOp, mlir::affine::AffineYieldOp,
                      mlir::scf::ForOp, mlir::affine::AffineForOp>(op))
          return mlir::WalkResult::advance();
        bodyOps.push_back(op);
        return mlir::WalkResult::advance();
      });
  for (mlir::Operation *op : bodyOps) {
    mlir::Operation *cloned = builder.clone(*op, mapping);
    for (auto [oldRes, newRes] :
         llvm::zip(op->getResults(), cloned->getResults()))
      mapping.map(oldRes, newRes);
  }

  return mapping.lookupOrNull(getStoredValue(lm.storeOp));
}

/// Strategy F materializer: clones the writer's outer loop into a scratch
/// buffer at the caller and returns the scratch memref.
static mlir::Value materializeLoop(mlir::Operation *writerCall,
                                   const CrossFnRematPlan &plan,
                                   mlir::OpBuilder &builder) {
  const auto &lm = *plan.loopMat;
  auto globalOp = mlir::cast<mlir::memref::GlobalOp>(lm.globalOp);
  auto memrefTy = mlir::cast<mlir::MemRefType>(globalOp.getType());

  // 1. Allocate scratch (memref.alloca with the global's static type).
  builder.setInsertionPointAfter(writerCall);
  if (!memrefTy.hasStaticShape()) return {};
  auto scratch = builder.create<mlir::memref::AllocaOp>(
      writerCall->getLoc(), memrefTy);

  // 2. Clone external dependencies (plan.opsToClone — captures of the loop
  //    region that come from outside).
  mlir::IRMapping mapping;
  for (mlir::BlockArgument arg : plan.writerArgs) {
    unsigned idx = arg.getArgNumber();
    if (idx >= writerCall->getNumOperands()) return {};
    mapping.map((mlir::Value)arg, writerCall->getOperand(idx));
  }
  // Map the writer's get_global result to the scratch memref so the cloned
  // store writes to scratch instead of the original global.
  mapping.map(lm.getGlobalOp->getResult(0), scratch.getResult());

  for (mlir::Operation *op : plan.opsToClone) {
    mlir::Operation *cloned = builder.clone(*op, mapping);
    for (auto [oldRes, newRes] :
         llvm::zip(op->getResults(), cloned->getResults()))
      mapping.map(oldRes, newRes);
  }
  // 3. Clone the outer loop op via region clone (deep-clones body).
  builder.clone(*lm.outerLoopOp, mapping);
  return scratch.getResult();
}

mlir::Value materializeAtCaller(mlir::Operation *writerCall,
                                       const CrossFnRematPlan &plan,
                                       mlir::OpBuilder &builder,
                                       const MatCtx &ctx) {
  if (plan.loopMat) return materializeLoop(writerCall, plan, builder);

  mlir::IRMapping mapping;
  // Strategy D — materialize each sub-plan at its own innerWC, then bind
  // the outer load's result to the materialized sub-value via the mapping.
  for (const LoadSubRecord &sub : plan.loadSubs) {
    mlir::Operation *innerWC = resolveInnerWriterCall(writerCall, sub, ctx);
    if (!innerWC) return {};
    mlir::OpBuilder subBuilder(innerWC);
    mlir::Value subVal =
        materializeAtCaller(innerWC, *sub.subPlan, subBuilder, ctx);
    if (!subVal) return {};
    mapping.map(sub.loadOp->getResult(0), subVal);
  }

  // Map writer's entry-block args to the caller's call operands.
  for (mlir::BlockArgument arg : plan.writerArgs) {
    unsigned idx = arg.getArgNumber();
    if (idx >= writerCall->getNumOperands()) return {};
    mapping.map((mlir::Value)arg, writerCall->getOperand(idx));
  }
  // Clone ops in topological order at the caller's insertion point.
  builder.setInsertionPointAfter(writerCall);
  for (mlir::Operation *op : plan.opsToClone) {
    mlir::Operation *cloned = builder.clone(*op, mapping);
    for (auto [oldRes, newRes] :
         llvm::zip(op->getResults(), cloned->getResults()))
      mapping.map(oldRes, newRes);
  }
  return mapping.lookupOrNull(plan.rootValue);
}

/// Add an extra entry-block argument of \p extraType to \p funcOp and
/// replace every op in \p loadOps with that argument. Mutates the function
/// signature in place. Returns the new argument.
mlir::BlockArgument
addParamAndReplaceLoads(mlir::FunctionOpInterface funcOp,
                        mlir::Type extraType,
                        llvm::ArrayRef<mlir::Operation *> loadOps) {
  mlir::Region *body = funcOp.getCallableRegion();
  mlir::Block *entry = &body->front();
  unsigned newArgIdx = entry->getNumArguments();
  mlir::BlockArgument newArg =
      entry->addArgument(extraType, funcOp.getLoc());

  // Update the function type.
  llvm::SmallVector<mlir::Type> newInputs(funcOp.getArgumentTypes().begin(),
                                          funcOp.getArgumentTypes().end());
  newInputs.push_back(extraType);
  auto newFnTy = mlir::FunctionType::get(funcOp.getContext(), newInputs,
                                         funcOp.getResultTypes());
  funcOp.setType(newFnTy);
  (void)newArgIdx;

  // Replace each load with the new arg, then erase.
  for (mlir::Operation *l : loadOps) {
    l->getResult(0).replaceAllUsesWith(newArg);
    l->erase();
  }
  return newArg;
}

/// Strategy F.2 reader-side rewrite: add an extra memref entry-block arg
/// and rewrite each load in \p loadOps to load from the new arg at the
/// same indices (instead of the original global). Returns the new arg.
mlir::BlockArgument
addMemrefParamAndRewriteLoads(mlir::FunctionOpInterface funcOp,
                              mlir::Type memrefTy,
                              llvm::ArrayRef<mlir::Operation *> loadOps) {
  mlir::Region *body = funcOp.getCallableRegion();
  mlir::Block *entry = &body->front();
  mlir::BlockArgument newArg =
      entry->addArgument(memrefTy, funcOp.getLoc());

  llvm::SmallVector<mlir::Type> newInputs(funcOp.getArgumentTypes().begin(),
                                           funcOp.getArgumentTypes().end());
  newInputs.push_back(memrefTy);
  auto newFnTy = mlir::FunctionType::get(funcOp.getContext(), newInputs,
                                         funcOp.getResultTypes());
  funcOp.setType(newFnTy);

  for (mlir::Operation *l : loadOps) {
    mlir::OpBuilder b(l);
    mlir::Value newRes;
    if (auto ml = mlir::dyn_cast<mlir::memref::LoadOp>(l)) {
      newRes = b.create<mlir::memref::LoadOp>(l->getLoc(), newArg,
                                               ml.getIndices())
                   .getResult();
    } else if (auto al = mlir::dyn_cast<mlir::affine::AffineLoadOp>(l)) {
      newRes = b.create<mlir::affine::AffineLoadOp>(
                    l->getLoc(), newArg, al.getAffineMap(), al.getIndices())
                   .getResult();
    } else {
      continue;
    }
    l->getResult(0).replaceAllUsesWith(newRes);
    l->erase();
  }
  return newArg;
}

/// Clone \p reader into a new private function with a "_dr_spec" suffix.
/// Caller must rewrite call sites to use the clone.
mlir::FunctionOpInterface
cloneReaderForSpec(mlir::FunctionOpInterface reader) {
  mlir::OpBuilder builder(reader.getOperation()->getContext());
  builder.setInsertionPointAfter(reader.getOperation());
  auto cloned = mlir::cast<mlir::FunctionOpInterface>(
      builder.clone(*reader.getOperation()));
  // Make it private + give a unique name.
  if (auto symOp = mlir::dyn_cast<mlir::SymbolOpInterface>(cloned.getOperation()))
    symOp.setVisibility(mlir::SymbolTable::Visibility::Private);
  static unsigned counter = 0;
  std::string newName =
      (reader.getName() + "__dr_spec_" + llvm::Twine(counter++)).str();
  if (auto symOp = mlir::dyn_cast<mlir::SymbolOpInterface>(cloned.getOperation()))
    symOp.setName(newName);
  return cloned;
}

/// Rewrite \p oldCall to \p newCallee, appending \p extraOperands.
void rewriteCallSite(mlir::Operation *oldCall,
                            mlir::FunctionOpInterface newCallee,
                            llvm::ArrayRef<mlir::Value> extraOperands) {
  mlir::OpBuilder builder(oldCall);
  llvm::SmallVector<mlir::Value> newOperands(oldCall->getOperands().begin(),
                                              oldCall->getOperands().end());
  newOperands.append(extraOperands.begin(), extraOperands.end());

  // Create a new call op of the same kind. We assume func.call.
  if (auto fc = mlir::dyn_cast<mlir::func::CallOp>(oldCall)) {
    auto newCall = builder.create<mlir::func::CallOp>(
        oldCall->getLoc(), newCallee.getName(),
        newCallee.getResultTypes(), newOperands);
    for (auto [oldRes, newRes] :
         llvm::zip(oldCall->getResults(), newCall->getResults()))
      oldRes.replaceAllUsesWith(newRes);
    oldCall->erase();
    return;
  }
  // Other call ops not supported in v1.
}


} // namespace dr::strategies
