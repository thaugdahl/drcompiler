//===- DataRecomputation.cpp - Load-store provenance analysis pass ---------===//
//
//===----------------------------------------------------------------------===//
//
// This pass computes load-store provenance: for each memref.load, it determines
// which memref.store operations could have been the last writer.
//
// Loads are classified into four categories:
//   SINGLE  — exactly one known store (recomputation candidate)
//   MULTI   — multiple possible stores
//   LEAKED  — provenance includes an external/unknown write (nullptr)
//   KILLED  — all reaching stores were killed (empty provenance set)
//
// The analysis phases are:
//   1. Allocation root collection
//   2. Base memref tracing (through view-like ops)
//   3. Store value dependency analysis (SSA chain → allocation roots)
//   4. Reaching store state updates (join, kill, apply)
//   5. Call site analysis (callee argument effects, global clobbering)
//   6. Region walking (core analysis: analyzeBlock / analyzeOp)
//
// See AnalysisState.h for the data model (type aliases and structs).
// See DotEmitter.h for GraphViz visualization of the results.
//
//===----------------------------------------------------------------------===//

#include "drcompiler/Transforms/DataRecomputation.h"
#include "drcompiler/Transforms/CpuCostModel.h"
#include "drcompiler/Transforms/DataRecomputation/AnalysisState.h"
#include "drcompiler/Transforms/DataRecomputation/CacheCostModel.h"
#include "drcompiler/Transforms/DataRecomputation/DotEmitter.h"
#include "drcompiler/Transforms/DataRecomputation/RematKernel.h"
#include "drcompiler/Transforms/DataRecomputation/Strategies/ConstantGlobalFold.h"
#include "drcompiler/Transforms/DataRecomputation/Strategies/CrossFnOrdered.h"
#include "drcompiler/Transforms/DataRecomputation/Strategies/DirectForward.h"
#include "drcompiler/Transforms/DataRecomputation/Strategies/FullRemat.h"
#include "drcompiler/Transforms/DataRecomputation/Strategies/InterprocRemat.h"
#include "drcompiler/Transforms/DataRecomputation/Strategies/PartialRemat.h"
#include "drcompiler/Transforms/DataRecomputation/Strategies/Strategy.h"
#include "drcompiler/Transforms/DataRecomputationIndexing.h"
#include "drcompiler/Transforms/Utils/MemrefBaseAnalysis.h"
#include "drcompiler/Transforms/Utils/OpDispatchUtils.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FileSystem.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/FormatVariadic.h"

// Gated by pass options dr-debug / dr-summary; both default OFF (silent).
// Set by runOnOperation before any DRDBG/DRSUM site fires.
namespace {
static bool drDebugEnabled = false;
static bool drSummaryEnabled = false;
} // namespace
#define DRDBG()                                                                \
  (drDebugEnabled ? llvm::dbgs()                                               \
                  : static_cast<llvm::raw_ostream &>(llvm::nulls()))           \
      << "DRCOMP: "
#define DRSUM()                                                                \
  (drSummaryEnabled ? llvm::errs()                                             \
                    : static_cast<llvm::raw_ostream &>(llvm::nulls()))         \
      << "DRSUM: "

namespace mlir {
#define GEN_PASS_DEF_DATARECOMPUTATIONPASS
#include "drcompiler/Transforms/Passes.h.inc"
} // namespace mlir

using namespace dr;

namespace {

/// Whitelist of Marco runtime intrinsics that are mathematically pure.
/// Defined here so it's visible to both callee-effect helpers and the
/// Strategy 4 plan-builder. Also accepts any call whose op advertises
/// MemoryEffectFree (canonical MLIR purity check).
static bool isPureRuntimeCall(mlir::Operation *op) {
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

// ===== Op Type Dispatch Helpers =====

template <class... OpTys>
struct OpTypePack {};

template <class... OpTys>
struct OpTypeHelper {
  static bool anyOf(mlir::Operation *op) { return mlir::isa<OpTys...>(op); }
};

template <class... OpTys>
struct OpTypeHelper<OpTypePack<OpTys...>> {
  static bool anyOf(mlir::Operation *op) {
    return OpTypeHelper<OpTys...>::anyOf(op);
  }
};

using GlobalStaticAllocOps =
    OpTypePack<mlir::memref::GlobalOp, mlir::LLVM::GlobalOp>;
using ViewLike =
    OpTypePack<mlir::memref::SubViewOp, mlir::memref::ReinterpretCastOp,
               mlir::memref::ViewOp, mlir::memref::CastOp>;

// ===== Phase 1: Allocation Root Collection =====

bool isGlobalStaticAllocationOp(mlir::Operation *op) {
  if (mlir::isa<mlir::SymbolOpInterface>(op)) {
    return OpTypeHelper<GlobalStaticAllocOps>::anyOf(op);
  }
  return false;
}

llvm::DenseSet<mlir::Operation *>
collectGlobalStaticAllocations(DRPassContext &passCtx) {
  llvm::DenseSet<mlir::Operation *> result{};
  mlir::ModuleOp moduleOp = passCtx.getModuleOp();
  llvm::for_each(moduleOp.getOps(), [&](mlir::Operation &op) {
    bool isGSA = isGlobalStaticAllocationOp(&op);
    bool allocatesOnSymbol = mlir::isa<mlir::SymbolOpInterface>(&op);
    if (isGSA && allocatesOnSymbol) {
      result.insert(&op);
    }
  });

  return result;
}

/// Walk \p rootOp and build a map from each allocated memref Value to the
/// operation that created it (alloc ops and memref.get_global).
static AllocationRoots collectAllocationRoots(DRPassContext &passCtx,
                                               mlir::Operation *rootOp) {
  AllocationRoots allocRootFor;

  rootOp->walk([&](mlir::Operation *op){
    if (mlir::hasEffect<mlir::MemoryEffects::Allocate>(op)) {
      for (mlir::Value res : op->getResults()) {
        if (mlir::isa<mlir::MemRefType, mlir::LLVM::LLVMPointerType>(
              res.getType())) {
          allocRootFor.try_emplace(res, op);
        }
      }
    }

    // Also collect get_globals
    if (auto getGlobalOp = mlir::dyn_cast<mlir::memref::GetGlobalOp>(op)) {
      auto *symOp = passCtx.getSymTabCollection().lookupNearestSymbolFrom(
        passCtx.getModuleOp(), getGlobalOp.getNameAttr());


      for (mlir::Value res : op->getResults()) {
        if (mlir::isa<mlir::MemRefType>(res.getType())) {
          allocRootFor.try_emplace(res, symOp);
        }
      }
    }

    // krnl.global is an unregistered op (we don't link the krnl dialect)
    // produced by onnx-mlir for constant tensors. Each instance produces a
    // memref result that behaves as a static, never-written global. Treat
    // the op itself as the allocation root, analogous to memref::GlobalOp.
    if (op->getName().getStringRef() == "krnl.global") {
      for (mlir::Value res : op->getResults()) {
        if (mlir::isa<mlir::MemRefType>(res.getType())) {
          allocRootFor.try_emplace(res, op);
        }
      }
    }
  });

  return allocRootFor;
}

/// Walk the entire module and collect every store whose base traces to a
/// `memref.global`.  The result is keyed on the global op and used to seed
/// per-function initial state, so loads of cross-function globals classify
/// as SINGLE/MULTI instead of being silently skipped.
///
/// Coverage is set to nullopt (may-write-anywhere) — we don't try to model
/// loop context across function boundaries.
static ModuleGlobalWrites
collectModuleGlobalWrites(mlir::Operation *rootOp,
                          AllocationRoots &allocRootFor) {
  ModuleGlobalWrites result;
  rootOp->walk([&](mlir::Operation *op) {
    if (!drcompiler::isAnyStoreOp(op)) return;
    mlir::Value memref = drcompiler::getLoadStoreMemref(op);
    llvm::SmallVector<mlir::Value, 2> bases;
    drcompiler::collectBaseMemrefs(memref, allocRootFor, bases);
    for (mlir::Value base : bases) {
      auto it = allocRootFor.find(base);
      if (it == allocRootFor.end()) continue;
      mlir::Operation *root = it->second;
      if (!mlir::isa<mlir::memref::GlobalOp>(root)) continue;
      result[root].push_back({op, std::nullopt});
    }
  });
  return result;
}

// ===== Phase 2: Base Memref Tracing =====

//< Take a value and trace it back via ViewLike Ops to find base memrefs.
//< Base Memref = The closest to the alloc result as possible.
//
// A dense map is used to memoize traces for earlier termination.
//< Example
/*
```mlir
%base = memref.alloc ...
%vv = memref.subview %base ...
%vv2 = memref.subview %vv
// vv2 will yield %base
```
*/
// Forwarder so existing call sites keep their unqualified name.
// Actual implementation lives in drcompiler/Transforms/Utils/MemrefBaseAnalysis.
inline void collectBaseMemrefs(
    mlir::Value v,
    llvm::DenseMap<mlir::Value, mlir::Operation *> &allocRootFor,
    llvm::SmallVectorImpl<mlir::Value> &bases) {
  drcompiler::collectBaseMemrefs(v, allocRootFor, bases);
}

// ===== Phase 3: Store Value Dependency Analysis =====

/// Trace each store's value operand through SSA to find all allocation roots
/// the value depends on via memory loads.
static StoreValueDeps computeStoreValueDeps(
    mlir::Operation *rootOp,
    llvm::DenseMap<mlir::Value, mlir::Operation *> &allocRootFor) {
  StoreValueDeps result;

  auto processStoreValue = [&](mlir::Operation *storeOp, mlir::Value storedVal) {
    llvm::SmallDenseSet<mlir::Operation *, 4> deps;
    llvm::SmallVector<mlir::Value, 8> worklist;
    llvm::SmallDenseSet<mlir::Value> visited;

    worklist.push_back(storedVal);

    while (!worklist.empty()) {
      mlir::Value current = worklist.pop_back_val();
      if (!visited.insert(current).second)
        continue;

      // BlockArgument cases
      if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(current)) {
        if (mlir::isa<mlir::MemRefType>(blockArg.getType())) {
          llvm::SmallVector<mlir::Value, 2> bases;
          collectBaseMemrefs(blockArg, allocRootFor, bases);

          for (mlir::Value base : bases) {
            auto it = allocRootFor.find(base);
            if (it != allocRootFor.end())
              deps.insert(it->second);
          }
        }
        continue;  // stop recursing for any block argument
      }

      mlir::Operation *defOp = current.getDefiningOp();
      if (!defOp)
        continue;

      // LoadOp: record the load's memref root as a dependency
      if (auto loadOp = mlir::dyn_cast<mlir::memref::LoadOp>(defOp)) {
        llvm::SmallVector<mlir::Value, 2> bases;
        collectBaseMemrefs(loadOp.getMemRef(), allocRootFor, bases);
        for (mlir::Value base : bases) {
          auto it = allocRootFor.find(base);
          if (it != allocRootFor.end())
            deps.insert(it->second);
        }
        continue;
      }

      // AffineLoadOp: record the load's memref root as a dependency
      if (auto affLoad = mlir::dyn_cast<mlir::affine::AffineLoadOp>(defOp)) {
        llvm::SmallVector<mlir::Value, 2> bases;
        collectBaseMemrefs(affLoad.getMemRef(), allocRootFor, bases);
        for (mlir::Value base : bases) {
          auto it = allocRootFor.find(base);
          if (it != allocRootFor.end())
            deps.insert(it->second);
        }
        continue;
      }

      // LLVM::LoadOp: record the load's pointer root as a dependency
      if (auto llvmLoad = mlir::dyn_cast<mlir::LLVM::LoadOp>(defOp)) {
        llvm::SmallVector<mlir::Value, 2> bases;
        collectBaseMemrefs(llvmLoad.getAddr(), allocRootFor, bases);
        for (mlir::Value base : bases) {
          auto it = allocRootFor.find(base);
          if (it != allocRootFor.end())
            deps.insert(it->second);
        }
        continue;
      }

      // CallOpInterface: conservatively depend on all memref/ptr-typed operands
      if (mlir::isa<mlir::CallOpInterface>(defOp)) {
        for (mlir::Value operand : defOp->getOperands()) {
          if (!mlir::isa<mlir::MemRefType, mlir::LLVM::LLVMPointerType>(
                  operand.getType()))
            continue;
          llvm::SmallVector<mlir::Value, 2> bases;
          collectBaseMemrefs(operand, allocRootFor, bases);
          for (mlir::Value base : bases) {
            auto it = allocRootFor.find(base);
            if (it != allocRootFor.end())
              deps.insert(it->second);
          }
        }
        continue;
      }

      // No operands (constants, etc.): stop
      if (defOp->getNumOperands() == 0)
        continue;

      // Otherwise: recurse into all operands
      for (mlir::Value operand : defOp->getOperands())
        worklist.push_back(operand);
    }

    if (!deps.empty())
      result[storeOp] = std::move(deps);
  };

  rootOp->walk([&](mlir::memref::StoreOp op) {
    processStoreValue(op.getOperation(), op.getValueToStore());
  });
  rootOp->walk([&](mlir::affine::AffineStoreOp op) {
    processStoreValue(op.getOperation(), op.getValueToStore());
  });
  rootOp->walk([&](mlir::LLVM::StoreOp op) {
    processStoreValue(op.getOperation(), op.getValue());
  });

  return result;
}

// ===== Phase 4: Reaching Store State Updates =====

/// Union-join two StoreMaps. Returns a new map containing,
/// for each root, the merged entries from both maps.
/// If the same store op appears in both maps, their coverages are unioned.
/// Union-join two StoreMaps.  When `injectAbsentSentinel` is true (the
/// default), roots present in only one side gain an unknown-write
/// sentinel `{nullptr, nullopt}` to model that the other path had no
/// reaching store for that root.  This is the correct join for control
/// flow where one branch may not execute (scf.if without else, may-zero
/// loops).  Set `injectAbsentSentinel=false` for loop-carry joins, where
/// every iteration past the first observes the body's prior writes.
///
/// `skipSentinelRoots` (optional) suppresses sentinel injection for the
/// listed roots even when `injectAbsentSentinel` is true.  Used by
/// `analyzeForOp` to skip sentinel injection for allocs whose dynamic
/// shape is bound to the loop's trip-count operand: when the loop runs
/// zero iterations the alloc has zero elements along the corresponding
/// dimension, making any later in-bounds access through that root
/// unreachable.  See BLOCKERS_CLAUDE.md §S1-A.
StoreMap joinStoreMaps(StoreMap result, const StoreMap &other,
                       bool injectAbsentSentinel = true,
                       const llvm::SmallDenseSet<mlir::Operation *>
                           *skipSentinelRoots = nullptr) {
  // Snapshot keys of `result` BEFORE merging.  Auto-vivify in the merge
  // loop would otherwise erase the "absent on this side" signal.
  llvm::SmallDenseSet<mlir::Operation *, 8> resultKeys;
  if (injectAbsentSentinel) {
    resultKeys.reserve(result.size());
    for (auto &kv : result)
      resultKeys.insert(kv.first);
  }

  auto ensureNullSentinel = [](IndexedStoreVec &entries) {
    bool hasNullSentinel = llvm::any_of(entries, [](const IndexedStore &e) {
      return e.storeOp == nullptr;
    });
    if (!hasNullSentinel)
      entries.push_back({nullptr, std::nullopt});
  };

  auto shouldSkip = [&](mlir::Operation *root) {
    return skipSentinelRoots && skipSentinelRoots->count(root);
  };

  for (auto &[root, entries] : other) {
    auto &resultEntries = result[root];
    if (injectAbsentSentinel && !resultKeys.count(root) && !shouldSkip(root))
      ensureNullSentinel(resultEntries);
    for (const auto &entry : entries) {
      auto it = llvm::find_if(resultEntries, [&](const IndexedStore &e) {
        return e.storeOp == entry.storeOp;
      });
      if (it != resultEntries.end()) {
        if (!it->coverage || !entry.coverage)
          it->coverage = std::nullopt;
        else
          *it->coverage += *entry.coverage;
      } else {
        resultEntries.push_back(entry);
      }
    }
  }

  if (injectAbsentSentinel) {
    for (auto root : resultKeys) {
      if (other.count(root))
        continue;
      if (shouldSkip(root))
        continue;
      ensureNullSentinel(result[root]);
    }
  }

  return result;
}

/// Remove from state any store whose value depends on clobberedRoot.
static void killDependentStores(
    mlir::Operation *clobberedRoot,
    StoreMap &state,
    const StoreValueDeps &storeValueDeps) {
  for (auto &[root, entries] : state) {
    llvm::erase_if(entries, [&](const IndexedStore &entry) {
      if (!entry.storeOp) return false;  // keep nullptr sentinel
      auto it = storeValueDeps.find(entry.storeOp);
      return it != storeValueDeps.end() && it->second.contains(clobberedRoot);
    });
  }
}

/// Try to fold an SSA value reaching an affine.store/load index into a
/// constant. Handles arith.constant, statically trip-1 affine.for IVs (the
/// IV must equal the constant lower bound on its single iteration), and
/// affine.apply chains whose operands all fold.
static std::optional<int64_t> tryFoldAffineValue(mlir::Value v) {
  if (auto cst = mlir::getConstantIntValue(v))
    return *cst;
  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(v)) {
    if (auto forOp = mlir::dyn_cast<mlir::affine::AffineForOp>(
            blockArg.getOwner()->getParentOp())) {
      if (blockArg == forOp.getInductionVar() &&
          forOp.hasConstantLowerBound() &&
          forOp.hasConstantUpperBound()) {
        int64_t lb = forOp.getConstantLowerBound();
        int64_t ub = forOp.getConstantUpperBound();
        int64_t step = forOp.getStepAsInt();
        // Statically trip-exactly-1: one iteration with IV == lb.
        if (step > 0 && ub > lb && (ub - lb) <= step)
          return lb;
      }
    }
    return std::nullopt;
  }
  if (auto applyOp = v.getDefiningOp<mlir::affine::AffineApplyOp>()) {
    llvm::SmallVector<mlir::Attribute, 4> attrs;
    auto idxTy = mlir::IndexType::get(applyOp.getContext());
    for (mlir::Value op : applyOp.getMapOperands()) {
      auto folded = tryFoldAffineValue(op);
      if (!folded) return std::nullopt;
      attrs.push_back(mlir::IntegerAttr::get(idxTy, *folded));
    }
    llvm::SmallVector<mlir::Attribute, 1> results;
    if (mlir::failed(applyOp.getAffineMap().constantFold(attrs, results)))
      return std::nullopt;
    if (results.size() != 1) return std::nullopt;
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(results[0]))
      return intAttr.getInt();
  }
  return std::nullopt;
}

/// Linearize integer coords on memref `v` into a single base-relative
/// element offset using `v`'s strided layout. Works uniformly through
/// view chains because MLIR's memref type carries the absolute strides
/// and offset from the underlying buffer's element 0. Returns nullopt on
/// dynamic strides/offset, rank mismatch, or non-MemRef type.
static std::optional<int64_t>
linearizeCoords(mlir::Value v, llvm::ArrayRef<int64_t> coords) {
  auto memrefTy = mlir::dyn_cast<mlir::MemRefType>(v.getType());
  if (!memrefTy)
    return std::nullopt;
  llvm::SmallVector<int64_t, 4> strides;
  int64_t offset;
  if (mlir::failed(memrefTy.getStridesAndOffset(strides, offset)))
    return std::nullopt;
  if (mlir::ShapedType::isDynamic(offset))
    return std::nullopt;
  if (strides.size() != coords.size())
    return std::nullopt;
  int64_t lin = offset;
  for (size_t i = 0; i < coords.size(); ++i) {
    if (mlir::ShapedType::isDynamic(strides[i]))
      return std::nullopt;
    lin += strides[i] * coords[i];
  }
  return lin;
}

/// Coverage for a memref.load/store, expressed as a base-linear single
/// point set. Uniform across direct accesses and accesses through any
/// chain of static-strided view-like ops. Returns nullopt if any index
/// is non-constant or the strided layout is dynamic.
static std::optional<PointSet> computeBaseLinearMemrefAccess(
    mlir::Value memref, mlir::Operation::operand_range indices) {
  llvm::SmallVector<int64_t, 4> coords;
  for (mlir::Value idx : indices) {
    auto constOp = idx.getDefiningOp<mlir::arith::ConstantOp>();
    if (!constOp)
      return std::nullopt;
    auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValue());
    if (!intAttr)
      return std::nullopt;
    coords.push_back(intAttr.getInt());
  }
  auto lin = linearizeCoords(memref, coords);
  if (!lin)
    return std::nullopt;
  llvm::SmallVector<int64_t, 1> linCoords{*lin};
  return PointSet::fromCoords(linCoords);
}

/// Coverage for an affine.load/store, expressed as a base-linear single
/// point set. Folds the affine map with `tryFoldAffineValue` to obtain
/// concrete multi-dim coords, then linearizes via the memref's strided
/// layout. Returns nullopt if any operand fails to fold or the layout is
/// dynamic.
static std::optional<PointSet> computeBaseLinearAffineAccess(
    mlir::Value memref, mlir::AffineMap map,
    mlir::Operation::operand_range mapOperands) {
  if (map.getNumInputs() !=
      static_cast<unsigned>(llvm::range_size(mapOperands)))
    return std::nullopt;
  auto idxTy = mlir::IndexType::get(map.getContext());
  llvm::SmallVector<mlir::Attribute, 4> operandAttrs;
  for (mlir::Value v : mapOperands) {
    auto folded = tryFoldAffineValue(v);
    if (!folded)
      return std::nullopt;
    operandAttrs.push_back(mlir::IntegerAttr::get(idxTy, *folded));
  }
  llvm::SmallVector<mlir::Attribute, 4> results;
  if (mlir::failed(map.constantFold(operandAttrs, results)))
    return std::nullopt;
  llvm::SmallVector<int64_t, 4> coords;
  for (mlir::Attribute a : results) {
    auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(a);
    if (!intAttr)
      return std::nullopt;
    coords.push_back(intAttr.getInt());
  }
  auto lin = linearizeCoords(memref, coords);
  if (!lin)
    return std::nullopt;
  llvm::SmallVector<int64_t, 1> linCoords{*lin};
  return PointSet::fromCoords(linCoords);
}

/// Update state to reflect a memref.store.
/// Rank-0 stores (no indices) kill all prior stores for that root.
/// Indexed stores with constant indices subtract their coverage from prior
/// entries and remove fully-killed entries. Dynamic-index stores are added
/// conservatively (nullopt coverage, cannot kill prior stores).
void applyStore(
    mlir::memref::StoreOp storeOp, StoreMap &state,
    llvm::DenseMap<mlir::Value, mlir::Operation *> &allocRootFor,
    const StoreValueDeps &storeValueDeps) {
  llvm::SmallVector<mlir::Value, 2> bases;
  collectBaseMemrefs(storeOp.getMemRef(), allocRootFor, bases);
  for (mlir::Value base : bases) {
    auto it = allocRootFor.find(base);
    // collectBaseMemrefs may return a function argument or other unmodelled
    // base. Stores through such bases are not tracked in our reaching-stores
    // state — skip silently and rely on the conservative call-clobber phase
    // to reflect the write at call boundaries.
    if (it == allocRootFor.end()) continue;

    // Kill stores whose values depended on the memory being overwritten
    killDependentStores(it->second, state, storeValueDeps);

    if (storeOp.getIndices().empty()) {
      // Rank-0 store: kill all prior entries and replace.
      state[it->second] = {{storeOp.getOperation(), std::nullopt}};
    } else {
      // Determine coverage as a base-linear point set. Works through any
      // chain of static-strided view-like ops (subview, reinterpret_cast,
      // cast, view) — falls back to nullopt only on dynamic indices or
      // dynamic strides/offset.
      auto coverage = computeBaseLinearMemrefAccess(
          storeOp.getMemRef(), storeOp.getIndices());

      if (coverage) {
        // Concrete coverage: subtract from prior entries, remove dead ones.
        llvm::erase_if(state[it->second], [&](IndexedStore &entry) {
          if (!entry.coverage) return false;  // can't subtract from universal
          *entry.coverage -= *coverage;
          return entry.coverage->empty();
        });
      }
      state[it->second].push_back({storeOp.getOperation(), coverage});
    }
  }
}

/// Update state to reflect an llvm.store.
/// Always uses nullopt coverage (flat pointer, no index information).
void applyLLVMStore(
    mlir::LLVM::StoreOp storeOp, StoreMap &state,
    llvm::DenseMap<mlir::Value, mlir::Operation *> &allocRootFor,
    const StoreValueDeps &storeValueDeps) {
  llvm::SmallVector<mlir::Value, 2> bases;
  collectBaseMemrefs(storeOp.getAddr(), allocRootFor, bases);
  for (mlir::Value base : bases) {
    auto it = allocRootFor.find(base);
    if (it == allocRootFor.end()) continue;
    killDependentStores(it->second, state, storeValueDeps);
    state[it->second].push_back({storeOp.getOperation(), std::nullopt});
  }
}

// ===== Phase 5: Call Site Analysis =====

/// Analyze how a defined callee uses a specific memref argument.
/// Returns whether the callee may write through that argument, whether it
/// passes the argument to further calls, and the direct store operations.
static CalleeArgEffect analyzeCalleeArg(
    mlir::FunctionOpInterface callee, unsigned argIdx) {
  CalleeArgEffect effect;
  mlir::Region *body = callee.getCallableRegion();
  if (!body || body->empty()) {
    // External callee: conservatively assume writes.
    effect.mayWrite = true;
    effect.passedToCall = true;
    return effect;
  }

  if (argIdx >= body->getNumArguments()) {
    // Out of bounds (e.g., variadic): conservative.
    effect.mayWrite = true;
    effect.passedToCall = true;
    return effect;
  }

  mlir::BlockArgument arg = body->getArgument(argIdx);
  if (!mlir::isa<mlir::MemRefType, mlir::LLVM::LLVMPointerType>(arg.getType()))
    return effect;

  // Collect all values that are views/casts of this argument.
  llvm::SmallDenseSet<mlir::Value> argValues;
  llvm::SmallVector<mlir::Value> worklist;
  argValues.insert(arg);
  worklist.push_back(arg);

  while (!worklist.empty()) {
    mlir::Value current = worklist.pop_back_val();
    for (mlir::OpOperand &use : current.getUses()) {
      mlir::Operation *user = use.getOwner();
      if (OpTypeHelper<ViewLike>::anyOf(user) ||
          mlir::isa<mlir::LLVM::GEPOp>(user) ||
          user->getName().getStringRef() == "polygeist.memref2pointer") {
        for (mlir::Value result : user->getResults()) {
          if (mlir::isa<mlir::MemRefType, mlir::LLVM::LLVMPointerType>(
                  result.getType())) {
            if (argValues.insert(result).second)
              worklist.push_back(result);
          }
        }
      }
    }
  }

  // Check uses for stores and calls.
  for (mlir::Value v : argValues) {
    for (mlir::OpOperand &use : v.getUses()) {
      mlir::Operation *user = use.getOwner();
      if (auto storeOp = mlir::dyn_cast<mlir::memref::StoreOp>(user)) {
        if (argValues.contains(storeOp.getMemRef())) {
          effect.mayWrite = true;
          effect.storeOps.push_back(storeOp.getOperation());
        }
      }
      if (auto affStore = mlir::dyn_cast<mlir::affine::AffineStoreOp>(user)) {
        if (argValues.contains(affStore.getMemRef())) {
          effect.mayWrite = true;
          effect.storeOps.push_back(affStore.getOperation());
        }
      }
      if (auto llvmStore = mlir::dyn_cast<mlir::LLVM::StoreOp>(user)) {
        if (argValues.contains(llvmStore.getAddr())) {
          effect.mayWrite = true;
          effect.storeOps.push_back(llvmStore.getOperation());
        }
      }
      if (mlir::isa<mlir::CallOpInterface>(user))
        effect.passedToCall = true;
    }
  }

  return effect;
}

/// Check if a callee function may access a given global variable.
/// External callees cannot access private globals.
/// Defined callees are scanned for direct memref.get_global usage.
static bool calleeMayAccessGlobal(
    mlir::FunctionOpInterface callee,
    mlir::Operation *globalOp) {
  auto globalSymbol = mlir::dyn_cast<mlir::SymbolOpInterface>(globalOp);
  if (!globalSymbol)
    return true;

  bool isPrivate =
      globalSymbol.getVisibility() == mlir::SymbolTable::Visibility::Private;

  mlir::Region *body = callee.getCallableRegion();
  if (!body || body->empty()) {
    // External callee: cannot access private globals.
    return !isPrivate;
  }

  // Defined callee: check for direct memref.get_global usage.
  llvm::StringRef globalName = globalSymbol.getName();
  bool directAccess = false;
  bool hasInnerCalls = false;

  body->walk([&](mlir::Operation *op) {
    if (auto getGlobal = mlir::dyn_cast<mlir::memref::GetGlobalOp>(op)) {
      if (getGlobal.getName() == globalName)
        directAccess = true;
    }
    if (auto call = mlir::dyn_cast<mlir::CallOpInterface>(op)) {
      // Pure runtime intrinsics don't access user globals.
      if (!isPureRuntimeCall(op))
        hasInnerCalls = true;
    }
  });

  if (directAccess) return true;
  if (!hasInnerCalls) return false;

  // Has non-pure inner calls but no direct access: conservatively assume
  // inner calls could access the global.
  return true;
}

/// Mark memref arguments of a call as clobbered, with refinements for
/// defined callees (read-only args skip clobbering, direct writes propagate
/// actual store ops) and global visibility (private globals unreachable by
/// external callees).
void applyCall(
    mlir::Operation *callOp, StoreMap &state,
    llvm::DenseMap<mlir::Value, mlir::Operation *> &allocRootFor,
    const StoreValueDeps &storeValueDeps,
    const llvm::DenseSet<mlir::Operation *> &globalAllocOps,
    mlir::FunctionOpInterface callee,
    InterproceduralOriginMap &interproceduralOrigins,
    const ModuleGlobalWrites &moduleGlobalWrites) {
  // Track which roots we've already clobbered to avoid redundant work.
  llvm::SmallDenseSet<mlir::Operation *, 8> clobbered;

  // Handle explicit memref / llvm.ptr arguments.
  for (auto [i, operand] : llvm::enumerate(callOp->getOperands())) {
    if (!mlir::isa<mlir::MemRefType, mlir::LLVM::LLVMPointerType>(
            operand.getType()))
      continue;
    llvm::SmallVector<mlir::Value, 2> bases;
    collectBaseMemrefs(operand, allocRootFor, bases);
    for (mlir::Value base : bases) {
      auto it = allocRootFor.find(base);
      if (it == allocRootFor.end()) continue;

      if (callee) {
        CalleeArgEffect effect = analyzeCalleeArg(callee, i);
        if (!effect.mayWrite && !effect.passedToCall) {
          // Callee only reads this argument: no clobber needed.
          continue;
        }

        killDependentStores(it->second, state, storeValueDeps);

        if (effect.mayWrite && !effect.passedToCall) {
          // S3-D: detect an unconditional rank-0 store in the callee's
          // entry block.  Such a store fully overwrites the root every
          // call, so it kills prior caller-side stores (analogous to the
          // rank-0 branch of applyStore).  Only the entry block counts —
          // stores nested inside scf.if / loops in the callee may not
          // execute, and treating them as kills would be unsound.
          mlir::Region *calleeBody = callee.getCallableRegion();
          mlir::Block *entryBlock = (calleeBody && !calleeBody->empty())
                                        ? &calleeBody->front()
                                        : nullptr;
          bool calleeRank0Kill = false;
          for (mlir::Operation *storeOp : effect.storeOps) {
            if (storeOp->getBlock() != entryBlock)
              continue;
            bool rank0 = false;
            if (auto m = mlir::dyn_cast<mlir::memref::StoreOp>(storeOp))
              rank0 = m.getIndices().empty();
            else if (auto a =
                         mlir::dyn_cast<mlir::affine::AffineStoreOp>(storeOp))
              rank0 = a.getIndices().empty();
            if (rank0) {
              calleeRank0Kill = true;
              break;
            }
          }
          if (calleeRank0Kill)
            state[it->second].clear();

          // Direct writes only: propagate actual store ops.
          for (mlir::Operation *storeOp : effect.storeOps) {
            state[it->second].push_back({storeOp, std::nullopt});
            interproceduralOrigins[storeOp].push_back({callOp, callee});
          }
        } else {
          // Passed to further call or external: conservative.
          state[it->second].push_back({nullptr, std::nullopt});
        }
      } else {
        // Unresolved callee: conservative clobber.
        killDependentStores(it->second, state, storeValueDeps);
        state[it->second].push_back({nullptr, std::nullopt});
      }
      clobbered.insert(it->second);
    }
  }

  // Clobber globals that the callee may access.
  for (mlir::Operation *globalOp : globalAllocOps) {
    if (clobbered.contains(globalOp))
      continue;

    if (callee && !calleeMayAccessGlobal(callee, globalOp))
      continue;

    killDependentStores(globalOp, state, storeValueDeps);
    // Globals whose writers are fully enumerated module-wide don't need a
    // conservative nullptr — every real writer is already in state via the
    // function-entry seed. Inject nullptr only for unenumerated globals
    // (e.g. if collectModuleGlobalWrites missed them, or if the global has
    // no writers at all in the module — in which case we can't know whether
    // some external linker-visible code might write to it).
    auto mit = moduleGlobalWrites.find(globalOp);
    if (mit == moduleGlobalWrites.end() || mit->second.empty())
      state[globalOp].push_back({nullptr, std::nullopt});
  }
}

// ===== Phase 6: Region Walking (Core Analysis) =====

// Forward declarations for the recursive analysis.
static void analyzeBlock(mlir::Block &block, StoreMap &state,
    AnalysisContext &ctx);

static void analyzeOp(mlir::Operation *op, StoreMap &state,
    AnalysisContext &ctx);

static void analyzeBlock(mlir::Block &block, StoreMap &state,
    AnalysisContext &ctx) {
  for (mlir::Operation &op : block)
    analyzeOp(&op, state, ctx);
}

// scf.if / affine.if: analyze each region from a copy of the entry state
// then join the branches.
static void analyzeIfOp(mlir::Operation *op, StoreMap &state,
                        AnalysisContext &ctx) {
  StoreMap thenState = state;
  mlir::Region &thenRegion = op->getRegion(0);
  if (!thenRegion.empty())
    analyzeBlock(thenRegion.front(), thenState, ctx);

  StoreMap elseState = state;
  if (op->getNumRegions() > 1) {
    mlir::Region &elseRegion = op->getRegion(1);
    if (!elseRegion.empty())
      analyzeBlock(elseRegion.front(), elseState, ctx);
  }

  state = joinStoreMaps(thenState, elseState);
}

// Canonical key for a dim-source SSA value. Two `memref.dim` ops on the
// same memref with the same constant index produce equal keys even though
// their results are distinct SSA values; they are observably equal at
// runtime. The first element is the underlying value (the memref or, for
// non-`memref.dim` values, the value itself); the second is the dim index
// (or -1 for non-`memref.dim` keys).
using DimKey = std::pair<mlir::Value, int64_t>;

static DimKey canonicalDimKey(mlir::Value v) {
  if (auto dimOp = v.getDefiningOp<mlir::memref::DimOp>()) {
    auto idx = dimOp.getConstantIndex();
    if (idx)
      return {dimOp.getSource(), *idx};
  }
  return {v, -1};
}

// Collect alloc roots whose dynamic shape is bound to one of the SSA values
// driving an affine.for's bounds. When the loop has zero trip count those
// allocs have zero elements along the corresponding dimension, so any later
// in-bounds access to the alloc is also unreachable on that path. This makes
// it sound to suppress the absent-path sentinel for these roots in the join.
//
// Bound operands are normalised through `memref.dim %M, %c<k>` to the
// canonical `(M, k)` key so that distinct SSA values from independent
// `memref.dim` calls on the same source compare equal.
static llvm::SmallDenseSet<mlir::Operation *>
collectTripCorrelatedRoots(mlir::affine::AffineForOp affineFor,
                           const StoreMap &candidates) {
  llvm::SmallDenseSet<mlir::Operation *> result;

  llvm::SmallDenseSet<DimKey, 4> boundKeys;
  auto addBoundOperands = [&](mlir::ValueRange operands) {
    for (mlir::Value v : operands)
      boundKeys.insert(canonicalDimKey(v));
  };
  if (!affineFor.hasConstantLowerBound())
    addBoundOperands(affineFor.getLowerBoundOperands());
  if (!affineFor.hasConstantUpperBound())
    addBoundOperands(affineFor.getUpperBoundOperands());
  if (boundKeys.empty())
    return result;

  auto allocDynSizes = [](mlir::Operation *op,
                          llvm::SmallVectorImpl<mlir::Value> &out) {
    if (auto a = mlir::dyn_cast<mlir::memref::AllocOp>(op))
      for (mlir::Value v : a.getDynamicSizes()) out.push_back(v);
    else if (auto a = mlir::dyn_cast<mlir::memref::AllocaOp>(op))
      for (mlir::Value v : a.getDynamicSizes()) out.push_back(v);
  };

  llvm::SmallVector<mlir::Value, 4> sizes;
  for (auto &kv : candidates) {
    sizes.clear();
    allocDynSizes(kv.first, sizes);
    for (mlir::Value sz : sizes) {
      if (boundKeys.count(canonicalDimKey(sz))) {
        result.insert(kv.first);
        break;
      }
    }
  }
  return result;
}

// scf.for / affine.for: two-pass analysis to model loop-carried deps.
// Pass 1 produces post-body state; pass 2 re-runs the body with the carried
// entry to capture stores visible from a prior iteration. Affine loops with
// statically-positive trip count skip the may-not-run join.
static void analyzeForOp(mlir::Operation *op, StoreMap &state,
                         AnalysisContext &ctx) {
  mlir::Region &bodyRegion = op->getRegion(0);

  StoreMap bodyState = state;
  if (!bodyRegion.empty())
    analyzeBlock(bodyRegion.front(), bodyState, ctx);

  // Carry join — iteration N>1 always observes prior body stores, so no
  // absent-path sentinel.
  StoreMap carriedEntry =
      joinStoreMaps(state, bodyState, /*injectAbsentSentinel=*/false);
  StoreMap bodyState2 = carriedEntry;
  if (!bodyRegion.empty())
    analyzeBlock(bodyRegion.front(), bodyState2, ctx);

  if (auto affineFor = mlir::dyn_cast<mlir::affine::AffineForOp>(op)) {
    if (affineFor.hasConstantLowerBound() &&
        affineFor.hasConstantUpperBound() &&
        affineFor.getConstantLowerBound() <
            affineFor.getConstantUpperBound()) {
      state = bodyState2;
      return;
    }
    // May-zero-trip affine.for: suppress the absent-path sentinel for alloc
    // roots whose dynamic shape is bound to the loop's trip-count operand.
    auto correlated = collectTripCorrelatedRoots(affineFor, bodyState2);
    state = joinStoreMaps(state, bodyState2, /*injectAbsentSentinel=*/true,
                          correlated.empty() ? nullptr : &correlated);
    return;
  }

  state = joinStoreMaps(state, bodyState2);
}

// scf.while: same two-pass treatment as analyzeForOp, but covers both the
// before (condition) and after (body) regions.
static void analyzeWhileOp(mlir::scf::WhileOp whileOp, StoreMap &state,
                           AnalysisContext &ctx) {
  mlir::Region &beforeRegion = whileOp.getBefore();
  mlir::Region &afterRegion = whileOp.getAfter();

  StoreMap condState = state;
  if (!beforeRegion.empty())
    analyzeBlock(beforeRegion.front(), condState, ctx);
  StoreMap bodyState = condState;
  if (!afterRegion.empty())
    analyzeBlock(afterRegion.front(), bodyState, ctx);

  StoreMap carriedEntry =
      joinStoreMaps(state, bodyState, /*injectAbsentSentinel=*/false);
  StoreMap condState2 = carriedEntry;
  if (!beforeRegion.empty())
    analyzeBlock(beforeRegion.front(), condState2, ctx);
  StoreMap bodyState2 = condState2;
  if (!afterRegion.empty())
    analyzeBlock(afterRegion.front(), bodyState2, ctx);

  state = joinStoreMaps(state, bodyState2);
}

// memref/affine/llvm store: dispatch to dialect-specific applier or, for
// affine.store, kill dependents and record nullopt coverage (affine maps
// make index extraction non-trivial).
static void analyzeStoreOp(mlir::Operation *op, StoreMap &state,
                           AnalysisContext &ctx) {
  auto &allocRootFor = ctx.allocRootFor;
  if (auto storeOp = mlir::dyn_cast<mlir::memref::StoreOp>(op)) {
    applyStore(storeOp, state, allocRootFor, ctx.storeValueDeps);
    return;
  }
  if (auto affStore = mlir::dyn_cast<mlir::affine::AffineStoreOp>(op)) {
    llvm::SmallVector<mlir::Value, 2> bases;
    collectBaseMemrefs(affStore.getMemRef(), allocRootFor, bases);
    for (mlir::Value base : bases) {
      auto it = allocRootFor.find(base);
      if (it == allocRootFor.end()) continue;
      killDependentStores(it->second, state, ctx.storeValueDeps);
      auto coverage = computeBaseLinearAffineAccess(
          affStore.getMemRef(), affStore.getAffineMap(),
          affStore.getMapOperands());
      if (coverage) {
        llvm::erase_if(state[it->second], [&](IndexedStore &entry) {
          if (!entry.coverage) return false;
          *entry.coverage -= *coverage;
          return entry.coverage->empty();
        });
      }
      state[it->second].push_back({op, coverage});
    }
    return;
  }
  if (auto llvmStore = mlir::dyn_cast<mlir::LLVM::StoreOp>(op)) {
    applyLLVMStore(llvmStore, state, allocRootFor, ctx.storeValueDeps);
    return;
  }
}

// Insert every reaching store for `loadOp`'s base allocation roots into
// loadProv[loadOp]. For memref.load with concrete indices we filter by
// PointSet overlap; affine.load and llvm.load record all reaching stores.
static void recordLoadProvenance(mlir::Operation *loadOp, mlir::Value memref,
                                 StoreMap &state, AnalysisContext &ctx) {
  auto &allocRootFor = ctx.allocRootFor;
  auto &loadProv = ctx.loadProv;
  llvm::SmallVector<mlir::Value, 2> bases;
  collectBaseMemrefs(memref, allocRootFor, bases);

  bool isMemrefLoad = mlir::isa<mlir::memref::LoadOp>(loadOp);
  for (mlir::Value base : bases) {
    auto it = allocRootFor.find(base);
    if (it == allocRootFor.end()) continue;
    auto rootIt = state.find(it->second);
    if (rootIt == state.end()) continue;

    // Ensure the load appears in provenance even if no stores match
    // (e.g. all stores were killed). An empty set classifies as MULTI.
    auto &provSet = loadProv[loadOp];

    std::optional<PointSet> loadCoverage;
    if (isMemrefLoad) {
      auto load = mlir::cast<mlir::memref::LoadOp>(loadOp);
      loadCoverage = load.getIndices().empty()
                         ? std::nullopt
                         : computeBaseLinearMemrefAccess(load.getMemRef(),
                                                         load.getIndices());
    } else if (auto affLoad =
                   mlir::dyn_cast<mlir::affine::AffineLoadOp>(loadOp)) {
      loadCoverage = computeBaseLinearAffineAccess(
          affLoad.getMemRef(), affLoad.getAffineMap(),
          affLoad.getMapOperands());
    }

    for (const auto &entry : rootIt->second) {
      if (loadCoverage && entry.coverage) {
        if (loadCoverage->overlaps(*entry.coverage))
          provSet.insert(entry.storeOp);
      } else {
        provSet.insert(entry.storeOp);
      }
    }
  }
}

// CallOpInterface: record enriched call edge for interproc analysis, then
// apply the conservative clobber.
static void analyzeCallOp(mlir::CallOpInterface callIface, StoreMap &state,
                          AnalysisContext &ctx) {
  mlir::Operation *op = callIface.getOperation();
  auto &allocRootFor = ctx.allocRootFor;

  mlir::FunctionOpInterface callee = nullptr;
  auto callableRef = callIface.getCallableForCallee();
  if (auto symbol = mlir::dyn_cast<mlir::SymbolRefAttr>(callableRef)) {
    auto *calleeOp = ctx.passCtx.getSymTabCollection().lookupSymbolIn(
        ctx.passCtx.getModuleOp().getOperation(), symbol);
    if (calleeOp)
      callee = mlir::dyn_cast<mlir::FunctionOpInterface>(calleeOp);
  }

  EnrichedCallEdge edge;
  edge.callSiteOp = op;
  edge.callee = callee;
  edge.argStores.resize(op->getNumOperands());

  for (auto [i, operand] : llvm::enumerate(op->getOperands())) {
    if (!mlir::isa<mlir::MemRefType, mlir::LLVM::LLVMPointerType>(
            operand.getType()))
      continue;

    llvm::SmallVector<mlir::Value, 2> bases;
    collectBaseMemrefs(operand, allocRootFor, bases);
    for (mlir::Value base : bases) {
      auto it = allocRootFor.find(base);
      if (it == allocRootFor.end()) continue;
      auto rootIt = state.find(it->second);
      if (rootIt != state.end())
        for (const auto &entry : rootIt->second)
          edge.argStores[i].insert(entry.storeOp);
    }
  }

  mlir::Operation *calleeOp = callee ? callee.getOperation() : nullptr;
  ctx.callGraph[calleeOp].push_back(std::move(edge));

  applyCall(op, state, allocRootFor, ctx.storeValueDeps,
            ctx.globalAllocOps, callee, ctx.interproceduralOrigins,
            ctx.moduleGlobalWrites);
}

// Generic op carrying regions: analyze each, then join across regions and
// fold into the outer state. Catch-all for region-bearing ops not otherwise
// handled.
static void analyzeGenericRegionOp(mlir::Operation *op, StoreMap &state,
                                   AnalysisContext &ctx) {
  StoreMap joined = state;
  bool first = true;
  for (mlir::Region &region : op->getRegions()) {
    if (region.empty()) continue;
    StoreMap regionState = state;
    analyzeBlock(region.front(), regionState, ctx);
    if (first) {
      joined = regionState;
      first = false;
    } else {
      joined = joinStoreMaps(joined, regionState);
    }
  }
  if (!first)
    state = joinStoreMaps(state, joined);
}

static void analyzeOp(mlir::Operation *op, StoreMap &state,
    AnalysisContext &ctx) {
  if (mlir::isa<mlir::scf::IfOp, mlir::affine::AffineIfOp>(op))
    return analyzeIfOp(op, state, ctx);

  if (mlir::isa<mlir::scf::ForOp, mlir::affine::AffineForOp>(op))
    return analyzeForOp(op, state, ctx);

  if (auto whileOp = mlir::dyn_cast<mlir::scf::WhileOp>(op))
    return analyzeWhileOp(whileOp, state, ctx);

  if (drcompiler::isAnyStoreOp(op))
    return analyzeStoreOp(op, state, ctx);

  if (drcompiler::isAnyLoadOp(op))
    return recordLoadProvenance(op, drcompiler::getLoadStoreMemref(op),
                                state, ctx);

  if (auto callIface = mlir::dyn_cast<mlir::CallOpInterface>(op))
    return analyzeCallOp(callIface, state, ctx);

  if (op->getNumRegions() > 0) {
    analyzeGenericRegionOp(op, state, ctx);
  }
}

// ===== Debug Utilities =====

// Simple Escape Analysis
llvm::DenseSet<mlir::FunctionOpInterface> getPotentialEscapingFunctions(DRPassContext &ctx)
{
  llvm::DenseSet<mlir::FunctionOpInterface> result{};

  mlir::ModuleOp moduleOp = ctx.getModuleOp();

  auto &symTabCollection = ctx.getSymTabCollection();

  // Find calls to functions
  moduleOp.walk([&](mlir::Operation *op) {
    if ( auto funcOpIface = mlir::dyn_cast<mlir::CallOpInterface>(op) ) {

      // Either Value or SymbolRefAttr
      auto callee = funcOpIface.getCallableForCallee();
      auto symbol = callee.dyn_cast<mlir::SymbolRefAttr>();

      if ( !symbol ) {
        return mlir::WalkResult::skip();
      }

      auto *calleeOp = symTabCollection.lookupSymbolIn(moduleOp.getOperation(), symbol);
      auto funcIface = mlir::dyn_cast<mlir::FunctionOpInterface>(calleeOp);

      if ( ! funcIface ) {
        return mlir::WalkResult::skip();
      }

      // If not private OR is external
      const bool external = funcIface.isExternal();
      const bool externallyVisible = funcIface.isPublic();

      const bool takesPtrLike = llvm::any_of(funcIface.getArgumentTypes(), [](mlir::Type t) {
        bool isPointerLike = mlir::isa<mlir::PtrLikeTypeInterface>(t);
        return isPointerLike;
      });

      if ( takesPtrLike && ( external || externallyVisible )  ) {
         result.insert(funcIface);
      }
    }

    return mlir::WalkResult::advance();
  });

  for ( auto &x : result ) {
    DRDBG() << "Leaky function: " << x.getName() << "\n";
  }

  return result;
}

/// Check if a value's SSA operand tree contains any function entry arguments.
static bool dependsOnFuncArg(mlir::Value val) {
  llvm::SmallVector<mlir::Value, 8> worklist;
  llvm::SmallDenseSet<mlir::Value> visited;
  worklist.push_back(val);
  while (!worklist.empty()) {
    mlir::Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(current)) {
      mlir::Block *block = blockArg.getOwner();
      if (block && block->isEntryBlock())
        if (mlir::isa_and_nonnull<mlir::FunctionOpInterface>(
                block->getParentOp()))
          return true;
      continue;
    }
    mlir::Operation *defOp = current.getDefiningOp();
    if (!defOp)
      continue;
    for (mlir::Value operand : defOp->getOperands())
      worklist.push_back(operand);
  }
  return false;
}

/// Recursively print the SSA operand tree of a value for debugging.
static void printOperandTree(mlir::Value val, unsigned indent = 0,
                             unsigned maxDepth = 8) {
  std::string prefix(indent * 2, ' ');
  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(val)) {
    DRDBG() << prefix << "block-arg #" << blockArg.getArgNumber()
            << " : " << blockArg.getType() << "\n";
    return;
  }
  mlir::Operation *defOp = val.getDefiningOp();
  if (!defOp) {
    DRDBG() << prefix << "<unknown value>\n";
    return;
  }
  DRDBG() << prefix << *defOp << "\n";
  if (indent >= maxDepth) {
    DRDBG() << prefix << "  ...\n";
    return;
  }
  for (mlir::Value operand : defOp->getOperands())
    printOperandTree(operand, indent + 1, maxDepth);
}

// Out-of-line helpers used by runOnOperation. The rematerialization
// kernels and cache cost model live in:
//   drcompiler/Transforms/DataRecomputation/CacheCostModel.h
//   drcompiler/Transforms/DataRecomputation/RematKernel.h
// Strategy classes live in
//   drcompiler/Transforms/DataRecomputation/Strategies/*.h
using drcompiler::getStoredValue;
using dr::CacheParams;
using dr::estimateBufferSizeBytes;
using dr::estimateInterveningFootprint;
using dr::estimateOperandReloadPenalty;
using dr::estimateLoadLatency;
using dr::decideBufferStrategy;
using dr::lookupAllocRoot;
using dr::buildRootWriteMap;
using dr::RootWriteMap;
using dr::estimateComputeCost;


// ===== Pass Class =====

class DataRecomputationPass final
    : public mlir::impl::DataRecomputationPassBase<DataRecomputationPass> {
public:
  using DataRecomputationPassBase<
      DataRecomputationPass>::DataRecomputationPassBase;

  void runOnOperation() override;

private:
  mlir::ModuleOp moduleOp;
};

} // namespace

// Strategy 4 (cross-function ordered recomputation) helpers, plan
// builder, materializer, and reader-rewrite all live in
// drcompiler/Transforms/DataRecomputation/Strategies/CrossFnOrdered.h.

void DataRecomputationPass::runOnOperation() {
  using namespace dr::strategies;
  moduleOp = this->getOperation();

  // Bind module-static debug/summary gates from pass options.
  drDebugEnabled = drDebug;
  drSummaryEnabled = drSummary;

  // Load CPU cost model (from file or built-in defaults).
  drcompiler::CpuCostModel costModel =
      cpuCostModelFile.empty()
          ? drcompiler::CpuCostModel::getDefault()
          : drcompiler::CpuCostModel::loadFromFile(cpuCostModelFile);

  mlir::MLIRContext *context = &getContext();
  mlir::SymbolTableCollection symTabCollection{};

  DRPassContext passCtx{context, symTabCollection, moduleOp};

  // Strategy 0: constant-global fold (pre-analysis peephole).
  dr::strategies::runConstantGlobalFold(moduleOp, symTabCollection,
                                        drTestDiagnostics, drSummary);

  auto globalAllocs = collectGlobalStaticAllocations(passCtx);

  // Collect allocation roots for the whole module so the analysis can resolve
  // memref bases that are allocated in one function and passed to another.
  auto allocRootFor = collectAllocationRoots(passCtx, moduleOp);

  // Log functions that may let memory escape to external callers.
  for (auto &f : getPotentialEscapingFunctions(passCtx))
    DRDBG() << "Potential escaper: " << *f.getOperation() << "\n";

  StoreValueDeps storeValueDeps = computeStoreValueDeps(moduleOp, allocRootFor);

  // Module-wide writers per global. Seeds each function's initial state.
  ModuleGlobalWrites moduleGlobalWrites =
      collectModuleGlobalWrites(moduleOp, allocRootFor);

  // Direct caller map: callee Operation* -> set of FunctionOpInterface ops
  // containing call sites. Phase-aware seeding uses this to filter out
  // writes from functions that don't share a static caller with the
  // reader, so disjoint execution phases (e.g. Marco's IC vs dynamic
  // drivers) don't pollute each other's reaching-stores.
  llvm::DenseMap<mlir::Operation *, llvm::SmallDenseSet<mlir::Operation *, 4>>
      directCallersOf;
  llvm::DenseMap<mlir::Operation *, llvm::SmallDenseSet<mlir::Operation *, 4>>
      directCalleesOf;
  moduleOp.walk([&](mlir::CallOpInterface call) {
    mlir::Operation *callable = call.resolveCallable();
    if (!callable) return;
    auto callerFn = call->getParentOfType<mlir::FunctionOpInterface>();
    if (!callerFn) return;
    directCallersOf[callable].insert(callerFn.getOperation());
    directCalleesOf[callerFn.getOperation()].insert(callable);
  });

  // Per-global writer-function set, derived from moduleGlobalWrites. Used
  // by fnTransitivelyWritesGlobal to detect transitive writes.
  llvm::DenseMap<mlir::Operation *, llvm::SmallDenseSet<mlir::Operation *, 4>>
      writersPerGlobal;
  for (auto &kv : moduleGlobalWrites) {
    for (auto &entry : kv.second) {
      if (!entry.storeOp) continue;
      auto wf =
          entry.storeOp->getParentOfType<mlir::FunctionOpInterface>();
      if (wf) writersPerGlobal[kv.first].insert(wf.getOperation());
    }
  }

  // Lazy transitive-callees cache (incl. self) for any function. Shared by
  // phase-root computation, hoist safety check, and write-interference
  // refinement so we never recompute reach.
  llvm::DenseMap<mlir::Operation *, llvm::SmallDenseSet<mlir::Operation *, 8>>
      transCalleesCache;
  auto transitiveCalleesOf = [&](mlir::Operation *root)
      -> llvm::SmallDenseSet<mlir::Operation *, 8> & {
    auto [it, inserted] = transCalleesCache.try_emplace(root);
    if (!inserted) return it->second;
    auto &result = it->second;
    llvm::SmallVector<mlir::Operation *, 8> wl{root};
    while (!wl.empty()) {
      mlir::Operation *f = wl.pop_back_val();
      if (!result.insert(f).second) continue;
      auto cIt = directCalleesOf.find(f);
      if (cIt == directCalleesOf.end()) continue;
      for (mlir::Operation *callee : cIt->second) wl.push_back(callee);
    }
    return result;
  };

  // Sound transitive-write check: does the static call-tree rooted at
  // \p fn (including fn itself) contain any function that stores to
  // \p globalOp? Closes the soundness hole left by walking only fn's
  // direct stores or only fn's immediate callees.
  auto fnTransitivelyWritesGlobal = [&](mlir::Operation *fn,
                                        mlir::Operation *globalOp) -> bool {
    auto wIt = writersPerGlobal.find(globalOp);
    if (wIt == writersPerGlobal.end()) return false;
    auto &reach = transitiveCalleesOf(fn);
    for (mlir::Operation *w : wIt->second)
      if (reach.contains(w)) return true;
    return false;
  };

  // Phase roots: functions never called inside the module (public drivers
  // / external entry points). Two functions are in the same phase iff they
  // share a phase root that transitively reaches both — strictly tighter
  // than direct-caller sharing and sound by construction.
  llvm::SmallVector<mlir::Operation *, 4> phaseRoots;
  moduleOp.walk([&](mlir::FunctionOpInterface fn) {
    auto cIt = directCallersOf.find(fn.getOperation());
    if (cIt == directCallersOf.end() || cIt->second.empty())
      phaseRoots.push_back(fn.getOperation());
  });
  llvm::DenseMap<mlir::Operation *, llvm::SmallDenseSet<mlir::Operation *, 2>>
      phaseRootsReaching;
  for (mlir::Operation *root : phaseRoots) {
    auto &reach = transitiveCalleesOf(root);
    for (mlir::Operation *f : reach)
      phaseRootsReaching[f].insert(root);
  }

  LoadProvenanceMap loadProv;
  EnrichedCallGraph callGraph;
  InterproceduralOriginMap interproceduralOrigins;
  AnalysisContext ctx{passCtx,        allocRootFor,
                      loadProv,       callGraph,
                      {},             storeValueDeps,
                      globalAllocs,   interproceduralOrigins,
                      moduleGlobalWrites};

  // Analyze each top-level function. Seed initial state with module-wide
  // global writers from OTHER functions, filtered by phase: include a
  // writer's stores only if the writer-function shares a direct caller
  // with the reader (or one is a direct caller of the other). Functions
  // with no callers fall back to coarse module-wide seeding.
  moduleOp.walk([&](mlir::FunctionOpInterface funcOp) {
    mlir::Region *body = funcOp.getCallableRegion();
    if (!body || body->empty()) return;

    StoreMap state;

    // Find globals referenced in this function via memref.get_global.
    llvm::DenseSet<mlir::Operation *> referencedGlobals;
    funcOp.walk([&](mlir::memref::GetGlobalOp g) {
      auto *symOp = symTabCollection.lookupNearestSymbolFrom(
          moduleOp, g.getNameAttr());
      if (symOp) referencedGlobals.insert(symOp);
    });

    auto readerRootsIt = phaseRootsReaching.find(funcOp.getOperation());
    bool readerHasRoots = readerRootsIt != phaseRootsReaching.end() &&
                          !readerRootsIt->second.empty();

    // Sound phase filter: writer and reader must share a transitive
    // ancestor that has no internal callers (a phase root). Direct-caller
    // sharing was unsound — a writer reachable through a different chain
    // could still execute between unrelated direct calls in a common
    // grand-ancestor.
    auto sharesPhase = [&](mlir::Operation *writerFn) -> bool {
      if (!readerHasRoots) return true; // unreachable from any root: fallback
      auto wIt = phaseRootsReaching.find(writerFn);
      if (wIt == phaseRootsReaching.end()) return false;
      for (mlir::Operation *r : wIt->second)
        if (readerRootsIt->second.count(r)) return true;
      return false;
    };

    for (mlir::Operation *globalOp : referencedGlobals) {
      auto it = moduleGlobalWrites.find(globalOp);
      if (it == moduleGlobalWrites.end()) continue;
      auto &dest = state[globalOp];
      for (auto &entry : it->second) {
        if (!entry.storeOp) continue;
        auto enclosing =
            entry.storeOp->getParentOfType<mlir::FunctionOpInterface>();
        if (!enclosing) continue;
        if (enclosing.getOperation() == funcOp.getOperation())
          continue; // own writes added precisely by dataflow
        if (!sharesPhase(enclosing.getOperation()))
          continue;
        dest.push_back(entry);
      }
    }

    analyzeBlock(body->front(), state, ctx);
  });

  // ===== Interprocedural Load Propagation =====
  //
  // The per-function analysis tracks loads within each function independently.
  // Loads inside a callee from block-argument memrefs won't have provenance
  // because the callee's arguments aren't in allocRootFor.
  //
  // This phase uses the enriched call graph: for each call site where we know
  // (a) which allocation root maps to which callee argument, and
  // (b) which stores reached that argument at the call site,
  // we walk the callee body, find loads from that argument, and record
  // their provenance using the caller's reaching stores.

  for (auto &[calleeOp, edges] : callGraph) {
    if (!calleeOp) continue;
    auto callee = mlir::dyn_cast<mlir::FunctionOpInterface>(calleeOp);
    if (!callee) continue;
    mlir::Region *body = callee.getCallableRegion();
    if (!body || body->empty()) continue;

    for (auto &edge : edges) {
      // For each memref argument position, find loads inside the callee
      // from that argument and propagate provenance.
      for (auto [argIdx, argStores] : llvm::enumerate(edge.argStores)) {
        if (argStores.empty()) continue;
        if (argIdx >= body->getNumArguments()) continue;

        mlir::Value calleeArg = body->getArgument(argIdx);
        if (!mlir::isa<mlir::MemRefType>(calleeArg.getType()))
          continue;

        // Walk all uses of this argument inside the callee to find loads.
        llvm::SmallVector<mlir::Value, 8> worklist{calleeArg};
        llvm::SmallDenseSet<mlir::Value> visited;
        while (!worklist.empty()) {
          mlir::Value current = worklist.pop_back_val();
          if (!visited.insert(current).second) continue;

          for (mlir::Operation *user : current.getUsers()) {
            // Track through view-like ops and casts.
            if (OpTypeHelper<ViewLike>::anyOf(user)) {
              for (mlir::Value result : user->getResults())
                worklist.push_back(result);
              continue;
            }

            // Found a load from the argument — propagate provenance.
            if (mlir::isa<mlir::memref::LoadOp, mlir::affine::AffineLoadOp>(
                    user)) {
              auto &prov = loadProv[user];
              for (mlir::Operation *store : argStores)
                prov.insert(store);
            }
          }
        }
      }
    }
  }

  // Classify loads into SINGLE, MULTI, LEAKED, and KILLED categories.
  // LEAKED  = provenance contains nullptr (external/unknown write).
  // KILLED  = all reaching stores were killed (empty provenance set).
  // SINGLE  = exactly one non-null provenance store (and not leaked).
  // MULTI   = multiple provenance stores (and not leaked).
  llvm::SmallVector<std::pair<mlir::Operation *, StoreSet *>> singleLoads;
  llvm::SmallVector<std::pair<mlir::Operation *, StoreSet *>> multiLoads;
  llvm::SmallVector<std::pair<mlir::Operation *, StoreSet *>> leakedLoads;
  llvm::SmallVector<std::pair<mlir::Operation *, StoreSet *>> killedLoads;

  for (auto &[loadOp, stores] : loadProv) {
    if (stores.contains(nullptr)) {
      leakedLoads.push_back({loadOp, &stores});
    } else if (stores.empty()) {
      killedLoads.push_back({loadOp, &stores});
    } else if (stores.size() == 1) {
      singleLoads.push_back({loadOp, &stores});
    } else {
      multiLoads.push_back({loadOp, &stores});
    }
  }

  // Emit MLIR remarks for test diagnostics (used with -verify-diagnostics).
  if (drTestDiagnostics) {
    for (auto &[loadOp, stores] : singleLoads)
      loadOp->emitRemark("load: SINGLE");
    for (auto &[loadOp, stores] : multiLoads)
      loadOp->emitRemark("load: MULTI");
    for (auto &[loadOp, stores] : leakedLoads)
      loadOp->emitRemark("load: LEAKED");
    for (auto &[loadOp, stores] : killedLoads)
      loadOp->emitRemark("load: KILLED");
  }

  // Helper to print a provenance store with ARGUMENT flag and operand tree.
  auto printProvenance = [](mlir::Operation *s) {
    if (!s) {
      DRDBG() << "  Provenance: <external write>\n";
      return;
    }
    if (auto storeOp = mlir::dyn_cast<mlir::memref::StoreOp>(s)) {
      bool fromArg = dependsOnFuncArg(storeOp.getValueToStore());
      DRDBG() << "  Provenance store: " << *s
              << (fromArg ? " (ARGUMENT)" : "") << "\n";
      DRDBG() << "  Operand tree:\n";
      printOperandTree(storeOp.getValueToStore(), 2);
    } else if (auto affStore = mlir::dyn_cast<mlir::affine::AffineStoreOp>(s)) {
      bool fromArg = dependsOnFuncArg(affStore.getValueToStore());
      DRDBG() << "  Provenance store: " << *s
              << (fromArg ? " (ARGUMENT)" : "") << "\n";
      DRDBG() << "  Operand tree:\n";
      printOperandTree(affStore.getValueToStore(), 2);
    } else if (auto llvmStore = mlir::dyn_cast<mlir::LLVM::StoreOp>(s)) {
      bool fromArg = dependsOnFuncArg(llvmStore.getValue());
      DRDBG() << "  Provenance store: " << *s
              << (fromArg ? " (ARGUMENT)" : "") << "\n";
      DRDBG() << "  Operand tree:\n";
      printOperandTree(llvmStore.getValue(), 2);
    } else {
      DRDBG() << "  Provenance store: " << *s << "\n";
    }
  };

  DRDBG() << "=== SINGLE (" << singleLoads.size() << ") ===\n";
  for (auto &[loadOp, stores] : singleLoads) {
    DRDBG() << "Load: " << *loadOp << "\n";
    printProvenance(*stores->begin());
  }

  DRDBG() << "=== MULTI (" << multiLoads.size() << ") ===\n";
  for (auto &[loadOp, stores] : multiLoads) {
    DRDBG() << "Load: " << *loadOp << "\n";
    for (mlir::Operation *s : *stores)
      printProvenance(s);
  }

  DRDBG() << "=== LEAKED (" << leakedLoads.size() << ") ===\n";
  for (auto &[loadOp, stores] : leakedLoads) {
    DRDBG() << "Load: " << *loadOp << "\n";
    for (mlir::Operation *s : *stores)
      printProvenance(s);
  }

  DRDBG() << "=== KILLED (" << killedLoads.size() << ") ===\n";
  for (auto &[loadOp, stores] : killedLoads) {
    DRDBG() << "Load: " << *loadOp << "\n";
  }

  if (!drDotFile.empty())
    dr::emitProvenanceDot(drDotFile, singleLoads, multiLoads, leakedLoads,
                          killedLoads);

  // Log store value dependencies
  for (auto &[storeOp, deps] : storeValueDeps) {
    DRDBG() << "Store value deps: " << *storeOp << "\n";
    for (mlir::Operation *root : deps)
      DRDBG() << "  depends on alloc root: " << *root << "\n";
  }

  // Log enriched call graph edges
  for (auto &[calleeOp, edges] : callGraph) {
    if (calleeOp)
      DRDBG() << "Callee: " << calleeOp->getName() << "\n";
    else
      DRDBG() << "Callee: <indirect/external>\n";
    for (auto &edge : edges) {
      DRDBG() << "  Call site: " << *edge.callSiteOp << "\n";
      for (auto [i, stores] : llvm::enumerate(edge.argStores)) {
        if (stores.empty()) continue;
        DRDBG() << "    arg[" << i << "] reached by " << stores.size()
                << " store(s)\n";
      }
    }
  }

  // ===== Transformation Phase =====
  if (drRecompute) {
    mlir::DominanceInfo domInfo(moduleOp);

    CacheParams cache{drL1Size,    drL2Size,    drL3Size,
                      drL1Latency, drL2Latency, drL3Latency,
                      drMemLatency, drCacheLineSize};

    // S3-A: optional cache hierarchy parameters from the cost-model JSON
    // override CLI defaults.  Per-field: only fields the file actually
    // contained (parsed successfully) are applied.
    const auto &jsonCache = costModel.cacheParams();
    if (jsonCache.l1Size)    cache.l1Size    = *jsonCache.l1Size;
    if (jsonCache.l2Size)    cache.l2Size    = *jsonCache.l2Size;
    if (jsonCache.l3Size)    cache.l3Size    = *jsonCache.l3Size;
    if (jsonCache.l1Latency) cache.l1Latency = *jsonCache.l1Latency;
    if (jsonCache.l2Latency) cache.l2Latency = *jsonCache.l2Latency;
    if (jsonCache.l3Latency) cache.l3Latency = *jsonCache.l3Latency;
    if (jsonCache.memLatency) cache.memLatency = *jsonCache.memLatency;

    // Partial remat requires cost model — warn if misconfigured.
    bool partialRematEnabled = drPartialRemat && drCostModel;
    if (drPartialRemat && !drCostModel)
      moduleOp->emitWarning(
          "dr-partial-remat requires dr-cost-model; partial remat disabled");

    // Module-wide per-root write summary used by partial remat leaf-safety.
    RootWriteMap rootWrites;
    if (partialRematEnabled)
      rootWrites = buildRootWriteMap(moduleOp, allocRootFor);

    // Collect load/store pairs to process (avoid iterator invalidation).
    // Handles memref.load/store and affine.load/store uniformly.
    struct SingleLoadEntry {
      mlir::Operation *loadOp;
      mlir::Operation *storeOp;
    };
    llvm::SmallVector<SingleLoadEntry> toProcess;

    for (auto &[loadOp, stores] : singleLoads) {
      auto *onlyStore = *stores->begin();
      mlir::Value storedVal = getStoredValue(onlyStore);
      if (!storedVal)
        continue;
      toProcess.push_back({loadOp, onlyStore});
    }

    // Helper: resolve a load op's memref to an allocation root, tracing
    // through casts and interprocedurally through call arguments.
    auto findAllocRoot =
        [&](mlir::Operation *loadOp) -> mlir::Operation * {
      mlir::Value loadMemref;
      if (auto l = mlir::dyn_cast<mlir::memref::LoadOp>(loadOp))
        loadMemref = l.getMemRef();
      else if (auto l = mlir::dyn_cast<mlir::affine::AffineLoadOp>(loadOp))
        loadMemref = l.getMemRef();
      if (!loadMemref) return nullptr;

      // Direct lookup.
      auto rootIt = allocRootFor.find(loadMemref);
      if (rootIt != allocRootFor.end()) return rootIt->second;

      // Trace through casts/views.
      llvm::SmallVector<mlir::Value, 4> bases;
      collectBaseMemrefs(loadMemref, allocRootFor, bases);
      for (mlir::Value base : bases) {
        auto it = allocRootFor.find(base);
        if (it != allocRootFor.end()) return it->second;
      }

      // Interprocedural: the memref traces to a block argument.
      // Walk the bases to find a block argument, then use the call graph
      // to find the corresponding caller operand.
      for (mlir::Value base : bases) {
        auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(base);
        if (!blockArg) continue;

        auto *parentOp = blockArg.getOwner()->getParentOp();
        auto parentFunc =
            mlir::dyn_cast_or_null<mlir::FunctionOpInterface>(parentOp);
        if (!parentFunc) continue;

        unsigned argIdx = blockArg.getArgNumber();

        // Search the call graph for call sites to this function.
        auto cgIt = callGraph.find(parentFunc.getOperation());
        if (cgIt == callGraph.end()) continue;

        for (auto &edge : cgIt->second) {
          if (argIdx >= edge.callSiteOp->getNumOperands()) continue;
          mlir::Value callerOperand = edge.callSiteOp->getOperand(argIdx);

          auto it2 = allocRootFor.find(callerOperand);
          if (it2 != allocRootFor.end()) return it2->second;

          llvm::SmallVector<mlir::Value, 4> callerBases;
          collectBaseMemrefs(callerOperand, allocRootFor, callerBases);
          for (mlir::Value cb : callerBases) {
            auto it3 = allocRootFor.find(cb);
            if (it3 != allocRootFor.end()) return it3->second;
          }
        }
      }
      return nullptr;
    };

    // When the cost model is enabled, compute per-buffer decisions.
    // Group SINGLE loads by allocation root and decide per-buffer.
    llvm::DenseSet<mlir::Operation *> skipBuffers; // buffers to keep
    if (drCostModel) {
      // Group loads by allocation root.
      llvm::DenseMap<mlir::Operation *, unsigned> bufferConsumerCount;
      llvm::DenseMap<mlir::Operation *, unsigned> bufferComputeCost;

      for (auto &entry : toProcess) {
        mlir::Value storedVal = getStoredValue(entry.storeOp);
        if (!storedVal)
          continue;

        mlir::Operation *allocRoot = findAllocRoot(entry.loadOp);
        if (!allocRoot)
          continue;

        bufferConsumerCount[allocRoot]++;

        // Track the max compute cost across stores to this buffer.
        unsigned cost = estimateComputeCost(storedVal, costModel);
        auto &existing = bufferComputeCost[allocRoot];
        existing = std::max(existing, cost);
      }

      // Compute per-buffer intervening footprints and operand penalties
      // when footprint analysis is on.
      llvm::DenseMap<mlir::Operation *, int64_t> bufferStoreToLoadFP;
      llvm::DenseMap<mlir::Operation *, unsigned> bufferOperandPenalty;

      if (drFootprintAnalysis) {
        for (auto &entry : toProcess) {
          mlir::Operation *root = findAllocRoot(entry.loadOp);
          if (!root)
            continue;
          int64_t fp = estimateInterveningFootprint(entry.storeOp, entry.loadOp,
                                                    cache, callGraph);

          auto &existing = bufferStoreToLoadFP[root];
          existing = std::max(existing, fp);

          // Per-operand penalty: check which operand memrefs are still warm
          // (re-accessed between store and load) vs cold (evicted).
          mlir::Value storedVal = getStoredValue(entry.storeOp);
          if (storedVal) {
            unsigned penalty = estimateOperandReloadPenalty(
                storedVal, entry.storeOp, entry.loadOp, fp, cache);
            auto &existingPenalty = bufferOperandPenalty[root];
            existingPenalty = std::max(existingPenalty, penalty);
          }
        }
      }

      // Now decide per-buffer.
      for (auto &[allocRoot, numConsumers] : bufferConsumerCount) {
        unsigned computeCost = bufferComputeCost.lookup(allocRoot);
        auto bufSize = estimateBufferSizeBytes(allocRoot);
        int64_t sizeBytes = bufSize.value_or((int64_t)cache.l2Size + 1);

        unsigned loadLat = estimateLoadLatency(sizeBytes, cache);

        int64_t storeToLoadFP = bufferStoreToLoadFP.lookup(allocRoot);
        unsigned opPenalty = bufferOperandPenalty.lookup(allocRoot);

        auto decision = decideBufferStrategy(
            computeCost, /*leafLoadCost=*/0, loadLat, numConsumers, sizeBytes,
            storeToLoadFP, opPenalty, cache);

        DRDBG() << "=== Cost model for " << *allocRoot << " ===\n";
        DRDBG() << "  buffer size: "
                << (bufSize ? std::to_string(*bufSize) + " bytes" : "dynamic")
                << "\n";
        DRDBG() << "  compute cost: " << computeCost << " cycles\n";
        DRDBG() << "  load latency: " << decision.loadLatency << " cycles\n";
        DRDBG() << "  consumers: " << numConsumers << "\n";
        if (drFootprintAnalysis) {
          DRDBG() << "  storeToLoad footprint: " << storeToLoadFP
                  << " bytes\n";
          DRDBG() << "  operand penalty: " << opPenalty << " cycles\n";
        }
        DRDBG() << "  decision: "
                << (decision.recompute ? "RECOMPUTE" : "KEEP BUFFER") << "\n";

        if (drSummaryEnabled) {
          DRSUM() << "buffer ";
          allocRoot->getLoc().print(llvm::errs());
          llvm::errs() << ": "
                       << (decision.recompute ? "RECOMPUTE" : "KEEP")
                       << " (compute=" << computeCost
                       << ", load=" << decision.loadLatency
                       << ", consumers=" << numConsumers
                       << ", size="
                       << (bufSize ? std::to_string(*bufSize) : "dynamic");
          if (drFootprintAnalysis)
            llvm::errs() << ", storeToLoadFP=" << storeToLoadFP
                         << ", operandPenalty=" << opPenalty;
          llvm::errs() << ")\n";
        }

        if (drTestDiagnostics) {
          auto diag = allocRoot->emitRemark()
              << "cost-model: "
              << (decision.recompute ? "RECOMPUTE" : "KEEP")
              << " (compute=" << computeCost
              << ", load=" << decision.loadLatency
              << ", consumers=" << numConsumers
              << ", size=" << (bufSize ? std::to_string(*bufSize) : "dynamic");
          if (drFootprintAnalysis)
            diag << ", storeToLoadFP=" << storeToLoadFP
                 << ", operandPenalty=" << opPenalty;
          diag << ")";
        }

        if (!decision.recompute)
          skipBuffers.insert(allocRoot);
      }
    }

    // Per-load strategy pipeline. Strategies fire in order, first match
    // wins; each strategy decides whether it consumes the candidate.
    dr::strategies::StrategyEnv env{
        domInfo,
        loadProv,
        interproceduralOrigins,
        allocRootFor,
        rootWrites,
        cache,
        costModel,
        partialRematEnabled,
        drPartialMaxLeaves,
        (bool)drTestDiagnostics,
        (bool)drSummary};

    dr::strategies::DirectForward directForward;
    dr::strategies::FullRemat fullRemat;
    dr::strategies::PartialRemat partialIntra(
        dr::strategies::PartialRemat::Mode::Intraprocedural);
    dr::strategies::InterprocRemat interprocRemat;

    for (auto &entry : toProcess) {
      mlir::Operation *loadOp = entry.loadOp;
      mlir::Operation *storeOp = entry.storeOp;
      mlir::Value storedVal = getStoredValue(storeOp);

      // Cost model: skip loads whose buffer the cost model decided to keep.
      if (drCostModel && !skipBuffers.empty()) {
        mlir::Operation *root = findAllocRoot(loadOp);
        if (root && skipBuffers.contains(root)) {
          if (drTestDiagnostics)
            loadOp->emitRemark() << "cost-model: SKIP_LOAD (buffer kept)";
          if (drSummaryEnabled) {
            DRSUM() << "load ";
            loadOp->getLoc().print(llvm::errs());
            llvm::errs() << ": SKIP_COST (buffer kept)\n";
          }
          continue;
        }
      }

      dr::strategies::LoadCandidate cand{loadOp, storeOp, storedVal};
      auto loadFn = loadOp->getParentOfType<mlir::FunctionOpInterface>();
      auto storeFn = storeOp->getParentOfType<mlir::FunctionOpInterface>();

      if (loadFn == storeFn) {
        if (directForward.tryApply(cand, env) ==
            dr::strategies::Outcome::Accepted) continue;
        if (fullRemat.tryApply(cand, env) ==
            dr::strategies::Outcome::Accepted) continue;
        partialIntra.tryApply(cand, env);
      } else {
        interprocRemat.tryApply(cand, env);
      }
    }

    // Strategies 1-3 may have erased some loads in toProcess (UAF if we
    // touch those pointers). Build a fresh set of live load ops by walking
    // the module post-strategies-1-3.
    llvm::SmallDenseSet<mlir::Operation *> liveLoads;
    moduleOp.walk([&](mlir::Operation *op) {
      if (mlir::isa<mlir::memref::LoadOp, mlir::affine::AffineLoadOp>(op))
        liveLoads.insert(op);
    });

    // ===== Strategy 4: cross-function ordered recomputation =====
    //
    // For SINGLE loads where load and store are in different functions,
    // try to find a common caller in which the writer call dominates the
    // reader call. If found, materialize the writer's expression at the
    // caller and pass it as an extra arg to the reader (in-place when the
    // reader is private + single-caller, otherwise via a private clone).
    //
    // Group SINGLE loads per (readerFn, storeOp) so loads sharing the
    // same store can share one specialization / one extra arg.
    struct CrossFnKey {
      mlir::Operation *readerFn;
      mlir::Operation *storeOp;
      bool operator==(const CrossFnKey &o) const {
        return readerFn == o.readerFn && storeOp == o.storeOp;
      }
    };
    struct CrossFnKeyInfo {
      static CrossFnKey getEmptyKey() {
        return {llvm::DenseMapInfo<mlir::Operation *>::getEmptyKey(), nullptr};
      }
      static CrossFnKey getTombstoneKey() {
        return {llvm::DenseMapInfo<mlir::Operation *>::getTombstoneKey(),
                nullptr};
      }
      static unsigned getHashValue(const CrossFnKey &k) {
        return llvm::hash_combine(k.readerFn, k.storeOp);
      }
      static bool isEqual(const CrossFnKey &a, const CrossFnKey &b) {
        return a == b;
      }
    };
    llvm::DenseMap<CrossFnKey,
                   llvm::SmallVector<mlir::Operation *>, CrossFnKeyInfo>
        crossGroups;
    for (auto &entry : toProcess) {
      // Skip dangling pointers from erased loads.
      if (!liveLoads.contains(entry.loadOp)) continue;
      auto loadFn = entry.loadOp->getParentOfType<mlir::FunctionOpInterface>();
      auto storeFn =
          entry.storeOp->getParentOfType<mlir::FunctionOpInterface>();
      if (!loadFn || !storeFn || loadFn == storeFn) continue;
      crossGroups[{loadFn.getOperation(), entry.storeOp}].push_back(
          entry.loadOp);
    }

    mlir::DominanceInfo domInfoTop(moduleOp);

    for (auto &kv : crossGroups) {
      mlir::Operation *readerFnOp = kv.first.readerFn;
      mlir::Operation *storeOp = kv.first.storeOp;
      auto &loadOps = kv.second;
      auto readerFn = mlir::cast<mlir::FunctionOpInterface>(readerFnOp);
      auto writerFn = storeOp->getParentOfType<mlir::FunctionOpInterface>();

      // Cost gate: if the cost model decided to KEEP this load's buffer,
      // skip Strategy 4 too — recomputing across functions can't beat a
      // load the model already deemed cheaper than recomputation.
      if (drCostModel && !skipBuffers.empty()) {
        bool skipForCost = false;
        for (auto *l : loadOps) {
          mlir::Operation *root = findAllocRoot(l);
          if (root && skipBuffers.contains(root)) {
            skipForCost = true;
            break;
          }
        }
        if (skipForCost) {
          if (drTestDiagnostics)
            for (auto *l : loadOps)
              l->emitRemark() << "interproc-cross: REJECT_COST";
          continue;
        }
      }

      // Build remat plan from the writer's stored value.
      CrossFnRematPlan plan;
      llvm::SmallDenseSet<mlir::Operation *> seenStores{storeOp};
      if (!buildCrossFunctionRematPlan(storeOp, writerFn, allocRootFor,
                                       loadProv,
                                       fnTransitivelyWritesGlobal, plan,
                                       /*depth=*/0, seenStores)) {
        if (drTestDiagnostics)
          for (auto *l : loadOps)
            l->emitRemark() << "interproc-cross: REJECT_PLAN";
        continue;
      }

      // Determine the global the load reads.
      mlir::Value loadMemref =
          drcompiler::getLoadStoreMemref(loadOps.front());
      mlir::Operation *globalOp =
          lookupAllocRoot(loadMemref, allocRootFor);
      if (!globalOp || !mlir::isa<mlir::memref::GlobalOp>(globalOp)) {
        if (drTestDiagnostics)
          for (auto *l : loadOps)
            l->emitRemark() << "interproc-cross: REJECT_NOTGLOBAL";
        continue;
      }

      // Collect live call sites via a fresh module walk. The callGraph is
      // built before any IR rewrites; prior rewriteCallSite() calls erase
      // and replace call-site ops, leaving callGraph entries stale. A walk
      // over the live IR finds only the ops that are currently in scope.
      llvm::SmallVector<mlir::Operation *, 4> liveCallSites;
      moduleOp.walk([&](mlir::func::CallOp call) {
        auto sym =
            mlir::dyn_cast<mlir::SymbolRefAttr>(call.getCallableForCallee());
        if (sym && sym.getRootReference().getValue() == readerFn.getName())
          liveCallSites.push_back(call.getOperation());
      });
      if (liveCallSites.empty()) {
        if (drTestDiagnostics)
          for (auto *l : loadOps)
            l->emitRemark() << "interproc-cross: REJECT_NOCALLERS";
        continue;
      }

      // For each caller, find an eligible writer call.
      struct ConformingSite {
        mlir::Operation *readerCall;
        mlir::Operation *writerCall;
      };
      llvm::SmallVector<ConformingSite, 4> conforming;
      llvm::SmallVector<mlir::Operation *, 4> nonConforming;

      // Cache transitive callees per caller (used to verify that no writer
      // to a hoisted-load global is reachable from caller's call subtree).
      llvm::DenseMap<mlir::Operation *,
                     llvm::SmallDenseSet<mlir::Operation *, 8>>
          callerCalleesCache;
      auto computeTransClos = [&](mlir::Operation *root)
          -> llvm::SmallDenseSet<mlir::Operation *, 8> & {
        auto [it, inserted] = callerCalleesCache.try_emplace(root);
        if (!inserted) return it->second;
        auto &result = it->second;
        llvm::SmallVector<mlir::Operation *, 8> wl{root};
        while (!wl.empty()) {
          mlir::Operation *f = wl.pop_back_val();
          if (!result.insert(f).second) continue;
          auto cIt = directCalleesOf.find(f);
          if (cIt == directCalleesOf.end()) continue;
          for (mlir::Operation *callee : cIt->second)
            wl.push_back(callee);
        }
        return result;
      };

      // Hoisted-load safety per caller: writers of any hoisted global must
      // not be reachable from the caller's static call subtree. Recurses
      // through sub-plans (Strategy D) so that chained inner plans are
      // also checked.
      std::function<bool(mlir::Operation *, const CrossFnRematPlan &)>
          hoistedLoadsSafeAt =
              [&](mlir::Operation *callerOp,
                  const CrossFnRematPlan &p) -> bool {
        auto &callees = computeTransClos(callerOp);
        for (mlir::Operation *gOp : p.hoistedLoadGlobals) {
          auto wIt = moduleGlobalWrites.find(gOp);
          if (wIt == moduleGlobalWrites.end()) continue;
          for (auto &e : wIt->second) {
            if (!e.storeOp) continue;
            auto wFn =
                e.storeOp->getParentOfType<mlir::FunctionOpInterface>();
            if (!wFn) continue;
            if (callees.count(wFn.getOperation())) return false;
          }
        }
        for (const LoadSubRecord &sub : p.loadSubs) {
          if (!hoistedLoadsSafeAt(callerOp, *sub.subPlan)) return false;
        }
        return true;
      };

      // Strategy D — verify each sub-plan resolves to an eligible
      // inner-writer-call inside this caller, anchored before
      // \p anchorCall (the next-outer materialization point).
      std::function<bool(mlir::FunctionOpInterface,
                         mlir::Operation *,
                         const CrossFnRematPlan &)>
          chainedSubsValidAt =
              [&](mlir::FunctionOpInterface callerFn,
                  mlir::Operation *anchorCall,
                  const CrossFnRematPlan &p) -> bool {
        for (const LoadSubRecord &sub : p.loadSubs) {
          mlir::Operation *innerWC = findEligibleWriterCall(
              callerFn, anchorCall, sub.innerWriterFn, sub.innerGlobalOp,
              domInfoTop, moduleGlobalWrites, allocRootFor, globalAllocs,
              fnTransitivelyWritesGlobal);
          if (!innerWC) return false;
          if (!chainedSubsValidAt(callerFn, innerWC, *sub.subPlan))
            return false;
        }
        return true;
      };

      for (mlir::Operation *siteOp : liveCallSites) {
        auto callerFn = siteOp->getParentOfType<mlir::FunctionOpInterface>();
        if (!callerFn) { nonConforming.push_back(siteOp); continue; }
        if (!hoistedLoadsSafeAt(callerFn.getOperation(), plan)) {
          nonConforming.push_back(siteOp);
          continue;
        }
        auto *writerCall = findEligibleWriterCall(
            callerFn, siteOp, writerFn, globalOp, domInfoTop,
            moduleGlobalWrites, allocRootFor, globalAllocs,
            fnTransitivelyWritesGlobal);
        if (!writerCall) {
          nonConforming.push_back(siteOp);
          continue;
        }
        // Strategy D — verify all sub-plans resolve at this caller.
        if (!plan.loadSubs.empty() &&
            !chainedSubsValidAt(callerFn, writerCall, plan)) {
          nonConforming.push_back(siteOp);
          continue;
        }
        conforming.push_back({siteOp, writerCall});
      }
      if (conforming.empty()) {
        if (drTestDiagnostics)
          for (auto *l : loadOps)
            l->emitRemark() << "interproc-cross: REJECT_NO_ORDERED_CALLER";
        continue;
      }

      // Decide: in-place rewrite or specialize?
      bool isPrivate = false;
      if (auto symOp = mlir::dyn_cast<mlir::SymbolOpInterface>(readerFnOp))
        isPrivate = symOp.getVisibility() ==
                    mlir::SymbolTable::Visibility::Private;
      bool inPlace = isPrivate && nonConforming.empty() &&
                     liveCallSites.size() == conforming.size() &&
                     conforming.size() == 1;

      // Plan kind drives the extra-arg type and reader-side rewrite path.
      // Straight-line / D plans pass a scalar; F.2 plans pass a memref
      // (the scratch buffer); F.1 (single-iteration extraction) passes a
      // scalar but materializes per-call-site by cloning the loop body
      // with IV→reader-index substitution.
      bool isLoopMat = plan.loopMat.has_value();
      bool useF1 = false;
      // Per-load-index ref: either a reader entry-block-arg index, or a
      // constant op cloned at the caller. Computed only for F.1.
      struct IdxRef {
        bool isReaderArg = false;
        unsigned readerArgIdx = 0;
        mlir::Operation *constOp = nullptr;
      };
      llvm::SmallVector<IdxRef, 4> f1IdxRefs;
      if (isLoopMat && plan.loopMat->extractionViable &&
          loadOps.size() == 1) {
        // Resolve the single load's indices against the reader's entry
        // block. If every index is either an entry-block arg or a
        // constant-defining op inside the reader, F.1 is viable.
        mlir::Operation *load = loadOps.front();
        llvm::SmallVector<mlir::Value, 4> loadIdxOps;
        if (auto l = mlir::dyn_cast<mlir::memref::LoadOp>(load))
          for (mlir::Value v : l.getIndices()) loadIdxOps.push_back(v);
        else if (auto l = mlir::dyn_cast<mlir::affine::AffineLoadOp>(load))
          for (mlir::Value v : l.getIndices()) loadIdxOps.push_back(v);
        mlir::Block *readerEntry =
            &readerFn.getCallableRegion()->front();
        bool ok = (loadIdxOps.size() == plan.loopMat->storeIVs.size());
        for (mlir::Value v : loadIdxOps) {
          if (!ok) break;
          IdxRef ref;
          if (auto barg = mlir::dyn_cast<mlir::BlockArgument>(v)) {
            if (barg.getOwner() != readerEntry) { ok = false; break; }
            ref.isReaderArg = true;
            ref.readerArgIdx = barg.getArgNumber();
          } else if (auto *def = v.getDefiningOp()) {
            if (!def->hasTrait<mlir::OpTrait::ConstantLike>()) {
              ok = false; break;
            }
            ref.constOp = def;
          } else {
            ok = false; break;
          }
          f1IdxRefs.push_back(ref);
        }
        useF1 = ok;
      }
      mlir::Type extraTy =
          isLoopMat && !useF1
              ? mlir::cast<mlir::memref::GlobalOp>(plan.loopMat->globalOp)
                    .getType()
              : plan.rootValue.getType();

      MatCtx matCtx{&domInfoTop, &moduleGlobalWrites, &allocRootFor,
                    &globalAllocs, fnTransitivelyWritesGlobal};

      // Helper to materialize for a given site, dispatching by plan kind.
      auto materializeForSite = [&](mlir::Operation *writerCall,
                                    mlir::Operation *readerCall,
                                    mlir::OpBuilder &b) -> mlir::Value {
        if (!useF1) return materializeAtCaller(writerCall, plan, b, matCtx);
        // F.1: build idxValues for this site.
        llvm::SmallVector<mlir::Value, 4> idxValues;
        b.setInsertionPointAfter(writerCall);
        for (const IdxRef &ref : f1IdxRefs) {
          if (ref.isReaderArg) {
            if (ref.readerArgIdx >= readerCall->getNumOperands()) return {};
            idxValues.push_back(readerCall->getOperand(ref.readerArgIdx));
          } else {
            mlir::Operation *cloned = b.clone(*ref.constOp);
            idxValues.push_back(cloned->getResult(0));
          }
        }
        return materializeExtraction(writerCall, plan, b, idxValues);
      };

      if (inPlace) {
        // Mutate readerFn in place. F.1 uses scalar arg (loop body
        // extracts a single value); F.2 uses memref arg (scratch buffer).
        if (isLoopMat && !useF1)
          addMemrefParamAndRewriteLoads(readerFn, extraTy, loadOps);
        else
          addParamAndReplaceLoads(readerFn, extraTy, loadOps);
        auto &site = conforming.front();
        mlir::OpBuilder builder(site.writerCall);
        mlir::Value matVal =
            materializeForSite(site.writerCall, site.readerCall, builder);
        if (!matVal) continue;
        if (drTestDiagnostics)
          site.readerCall->emitRemark() << "interproc-cross: ACCEPT_INPLACE";
        if (drSummaryEnabled) {
          DRSUM() << "cross-fn ";
          site.readerCall->getLoc().print(llvm::errs());
          llvm::errs() << ": ACCEPT_INPLACE (loads=" << loadOps.size() << ")\n";
        }
        rewriteCallSite(site.readerCall, readerFn, {matVal});
      } else {
        // Specialize.
        auto spec = cloneReaderForSpec(readerFn);
        // Find equivalent loadOps inside the clone (parallel walk).
        llvm::SmallVector<mlir::Operation *> clonedLoads;
        {
          mlir::Region *origBody = readerFn.getCallableRegion();
          mlir::Region *specBody = spec.getCallableRegion();
          llvm::DenseMap<mlir::Operation *, unsigned> idxOfLoad;
          unsigned i = 0;
          origBody->walk([&](mlir::Operation *op) {
            if (mlir::isa<mlir::memref::LoadOp, mlir::affine::AffineLoadOp>(op))
              idxOfLoad[op] = i++;
          });
          llvm::SmallVector<mlir::Operation *> clonedLoadOrder;
          specBody->walk([&](mlir::Operation *op) {
            if (mlir::isa<mlir::memref::LoadOp, mlir::affine::AffineLoadOp>(op))
              clonedLoadOrder.push_back(op);
          });
          for (mlir::Operation *l : loadOps) {
            auto it = idxOfLoad.find(l);
            if (it == idxOfLoad.end()) continue;
            if (it->second < clonedLoadOrder.size())
              clonedLoads.push_back(clonedLoadOrder[it->second]);
          }
        }
        if (isLoopMat && !useF1)
          addMemrefParamAndRewriteLoads(spec, extraTy, clonedLoads);
        else
          addParamAndReplaceLoads(spec, extraTy, clonedLoads);

        // F.2 / D / straight-line plans cache materialization per
        // writerCall; F.1 must not cache (each call site has its own
        // index values).
        llvm::DenseMap<mlir::Operation *, mlir::Value> matCache;
        for (auto &site : conforming) {
          mlir::Value matVal;
          auto cIt = useF1 ? matCache.end() : matCache.find(site.writerCall);
          if (cIt != matCache.end()) {
            matVal = cIt->second;
          } else {
            mlir::OpBuilder builder(site.writerCall);
            matVal =
                materializeForSite(site.writerCall, site.readerCall, builder);
            if (!matVal) continue;
            if (!useF1) matCache[site.writerCall] = matVal;
          }
          if (drTestDiagnostics)
            site.readerCall->emitRemark()
                << "interproc-cross: ACCEPT_SPECIALIZED";
          if (drSummaryEnabled) {
            DRSUM() << "cross-fn ";
            site.readerCall->getLoc().print(llvm::errs());
            llvm::errs() << ": ACCEPT_SPECIALIZED (loads="
                         << loadOps.size() << ")\n";
          }
          rewriteCallSite(site.readerCall, spec, {matVal});
        }
      }
    }
  }
}

namespace mlir {
std::unique_ptr<Pass> createDataRecomputationPass() {
  return std::make_unique<DataRecomputationPass>();
}
} // namespace mlir
