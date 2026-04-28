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
#include "drcompiler/Transforms/DataRecomputation/DotEmitter.h"
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

#define DRDBG() llvm::dbgs() << "DRCOMP: "

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
/// See BLOCKERS_CLAUDE.md §S1-A.
StoreMap joinStoreMaps(StoreMap result, const StoreMap &other,
                       bool injectAbsentSentinel = true) {
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

  for (auto &[root, entries] : other) {
    auto &resultEntries = result[root];
    if (injectAbsentSentinel && !resultKeys.count(root))
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

/// Try to extract concrete constant indices from a store/load's index operands.
/// Returns a PointSet containing the single accessed point, or nullopt if
/// any index is non-constant.
static std::optional<PointSet> computeAccessIndices(
    mlir::Operation::operand_range indices) {
  llvm::SmallVector<int64_t> coords;
  for (mlir::Value idx : indices) {
    auto constOp = idx.getDefiningOp<mlir::arith::ConstantOp>();
    if (!constOp) return std::nullopt;
    auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValue());
    if (!intAttr) return std::nullopt;
    coords.push_back(intAttr.getInt());
  }
  return PointSet::fromCoords(coords);
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
      // Determine coverage: concrete if all indices are constant and the
      // store is not through a ViewLike op; nullopt otherwise.
      bool throughView = (base != storeOp.getMemRef());
      auto coverage = throughView
                          ? std::nullopt
                          : computeAccessIndices(storeOp.getIndices());

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
      state[it->second].push_back({op, std::nullopt});
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
      bool throughView = (base != load.getMemRef());
      loadCoverage = (load.getIndices().empty() || throughView)
                         ? std::nullopt
                         : computeAccessIndices(load.getIndices());
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

// ===== Recomputation Helpers =====

/// Returns true if the store dominates the load (same function, direct forward).
static bool isSafeToForward(mlir::Operation *storeOp,
                            mlir::Operation *loadOp,
                            mlir::DominanceInfo &domInfo) {
  return domInfo.dominates(storeOp, loadOp);
}

using drcompiler::getStoredValue;

// ===== Footprint Analysis =====

/// Default trip count when bounds cannot be statically determined. Picked
/// to match a common per-loop iteration count seen in SPEC inner loops:
/// big enough that small constants (1, 4) don't dominate cache estimates,
/// small enough that footprint × trip stays within typical L2 sizes.
/// Tuning beyond ±2× makes little difference to the cost model decisions
/// downstream (cache-fit vs. memory-bound is the binary outcome).
constexpr int64_t kDefaultTripCount = 128;

/// Trace a value to a compile-time constant integer, walking through
/// index_cast ops and (one level of) call-site argument forwarding.
static std::optional<int64_t> traceToConstant(
    mlir::Value val, const EnrichedCallGraph &callGraph) {
  // Direct arith.constant / arith.constant index.
  if (mlir::Operation *defOp = val.getDefiningOp()) {
    if (auto cIdx = mlir::dyn_cast<mlir::arith::ConstantIndexOp>(defOp))
      return cIdx.value();
    if (auto cInt = mlir::dyn_cast<mlir::arith::ConstantIntOp>(defOp))
      return cInt.value();
    // Walk through index_cast.
    if (auto cast = mlir::dyn_cast<mlir::arith::IndexCastOp>(defOp))
      return traceToConstant(cast.getIn(), callGraph);
    return std::nullopt;
  }

  // Block argument: check call graph for constant forwarding.
  auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(val);
  if (!blockArg)
    return std::nullopt;

  auto *parentOp = blockArg.getOwner()->getParentOp();
  auto funcOp = mlir::dyn_cast_or_null<mlir::FunctionOpInterface>(parentOp);
  if (!funcOp)
    return std::nullopt;

  unsigned argIdx = blockArg.getArgNumber();
  auto cgIt = callGraph.find(funcOp.getOperation());
  if (cgIt == callGraph.end() || cgIt->second.empty())
    return std::nullopt;

  std::optional<int64_t> commonVal;
  for (const auto &edge : cgIt->second) {
    if (argIdx >= edge.callSiteOp->getNumOperands())
      return std::nullopt;
    mlir::Value callerOperand = edge.callSiteOp->getOperand(argIdx);
    // One-level trace: resolve the caller operand to a constant directly.
    mlir::Operation *callerDef = callerOperand.getDefiningOp();
    std::optional<int64_t> traced;
    if (callerDef) {
      if (auto c = mlir::dyn_cast<mlir::arith::ConstantIndexOp>(callerDef))
        traced = c.value();
      else if (auto c = mlir::dyn_cast<mlir::arith::ConstantIntOp>(callerDef))
        traced = c.value();
      else if (auto cast = mlir::dyn_cast<mlir::arith::IndexCastOp>(callerDef))
        if (auto *inner = cast.getIn().getDefiningOp()) {
          if (auto c = mlir::dyn_cast<mlir::arith::ConstantIndexOp>(inner))
            traced = c.value();
          else if (auto c = mlir::dyn_cast<mlir::arith::ConstantIntOp>(inner))
            traced = c.value();
        }
    }
    if (!traced)
      return std::nullopt;
    if (commonVal && *commonVal != *traced)
      return std::nullopt;
    commonVal = traced;
  }
  return commonVal;
}

/// Estimate the trip count of a loop op (affine.for or scf.for).
/// Returns nullopt if the trip count cannot be determined.
static std::optional<int64_t> estimateTripCount(
    mlir::Operation *loopOp, const EnrichedCallGraph &callGraph) {
  // affine.for
  if (auto affFor = mlir::dyn_cast<mlir::affine::AffineForOp>(loopOp)) {
    int64_t step = affFor.getStepAsInt();
    if (step <= 0) return std::nullopt;
    // Both bounds constant.
    if (affFor.hasConstantLowerBound() && affFor.hasConstantUpperBound()) {
      int64_t lb = affFor.getConstantLowerBound();
      int64_t ub = affFor.getConstantUpperBound();
      if (ub <= lb) return 0;
      return (ub - lb + step - 1) / step;
    }
    // Constant lower bound, try to trace dynamic upper bound.
    if (affFor.hasConstantLowerBound()) {
      auto ubOperands = affFor.getUpperBoundOperands();
      if (ubOperands.size() == 1) {
        auto ubVal = traceToConstant(ubOperands[0], callGraph);
        if (ubVal) {
          int64_t lb = affFor.getConstantLowerBound();
          if (*ubVal <= lb) return 0;
          return (*ubVal - lb + step - 1) / step;
        }
      }
    }
    return std::nullopt;
  }

  // scf.for
  if (auto scfFor = mlir::dyn_cast<mlir::scf::ForOp>(loopOp)) {
    auto lbVal = traceToConstant(scfFor.getLowerBound(), callGraph);
    auto ubVal = traceToConstant(scfFor.getUpperBound(), callGraph);
    auto stepVal = traceToConstant(scfFor.getStep(), callGraph);
    if (lbVal && ubVal && stepVal && *stepVal > 0) {
      if (*ubVal <= *lbVal) return 0;
      return (*ubVal - *lbVal + *stepVal - 1) / *stepVal;
    }
    return std::nullopt;
  }

  return std::nullopt;
}

// ===== Cost Model =====

/// Cache hierarchy parameters for the cost model.
struct CacheParams {
  unsigned l1Size;    // bytes
  unsigned l2Size;    // bytes
  unsigned l3Size;    // bytes (0 = unknown / not modeled)
  unsigned l1Latency; // cycles
  unsigned l2Latency;
  unsigned l3Latency;
  unsigned memLatency;
  unsigned cacheLineSize; // bytes
};

/// Estimate the ALU cost of recomputing a value by walking its SSA operand
/// tree.  Each operation is weighted via the CpuCostModel.  Loads and
/// block arguments are free (they are inputs, not recomputed).
static unsigned estimateComputeCost(mlir::Value val,
                                    const drcompiler::CpuCostModel &costModel) {
  llvm::SmallVector<mlir::Value, 8> worklist;
  llvm::SmallDenseSet<mlir::Value> visited;
  unsigned cost = 0;
  worklist.push_back(val);

  while (!worklist.empty()) {
    mlir::Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    if (mlir::isa<mlir::BlockArgument>(current))
      continue;
    mlir::Operation *defOp = current.getDefiningOp();
    if (!defOp)
      continue;

    // Loads are inputs — they represent values already in registers or memory.
    if (mlir::isa<mlir::memref::LoadOp, mlir::affine::AffineLoadOp,
                  mlir::LLVM::LoadOp>(defOp))
      continue;

    cost += costModel.opCost(defOp);

    for (mlir::Value operand : defOp->getOperands())
      worklist.push_back(operand);
  }
  return cost;
}

/// Estimate the size of an allocation in bytes. Returns nullopt when the
/// size cannot be determined statically:
///   - allocOp is not a memref::AllocOp / AllocaOp (e.g. get_global, an
///     unmodelled allocator, or a function-arg memref);
///   - the memref shape contains dynamic dimensions;
///   - the element type has zero bit-width (opaque/unknown type).
/// Callers fall back to a conservative size (typically `cache.l2Size + 1`,
/// i.e. assume the buffer misses cache).
static std::optional<int64_t> estimateBufferSizeBytes(mlir::Operation *allocOp) {
  mlir::MemRefType memrefTy;

  if (auto alloc = mlir::dyn_cast<mlir::memref::AllocOp>(allocOp))
    memrefTy = alloc.getType();
  else if (auto alloca = mlir::dyn_cast<mlir::memref::AllocaOp>(allocOp))
    memrefTy = alloca.getType();
  else
    return std::nullopt;

  if (!memrefTy.hasStaticShape())
    return std::nullopt;

  int64_t numElements = 1;
  for (int64_t dim : memrefTy.getShape())
    numElements *= dim;

  unsigned elementBits = memrefTy.getElementTypeBitWidth();
  if (elementBits == 0)
    return std::nullopt;

  return numElements * (elementBits / 8);
}

/// Estimate the cost of one load from a buffer, given its size and the cache
/// hierarchy.  Uses a simple model: if the buffer fits in Lk, assume Lk
/// latency.
static unsigned estimateLoadLatency(int64_t bufferSizeBytes,
                                    const CacheParams &cache) {
  if (bufferSizeBytes <= (int64_t)cache.l1Size)
    return cache.l1Latency;
  if (bufferSizeBytes <= (int64_t)cache.l2Size)
    return cache.l2Latency;
  if (cache.l3Size > 0 && bufferSizeBytes <= (int64_t)cache.l3Size)
    return cache.l3Latency;
  return cache.memLatency;
}

// ===== Per-Op and Block Footprint Estimation =====

/// Estimate the memory footprint (bytes) of a single execution of an op.
static int64_t estimateOpFootprintBytes(mlir::Operation *op,
                                        const CacheParams &cache) {
  // memref / affine / llvm scalar load and store: derive element type from
  // the memref/pointer operand.
  if (drcompiler::isAnyLoadOp(op) || drcompiler::isAnyStoreOp(op)) {
    mlir::Type elemTy = drcompiler::getLoadStoreElementType(op);
    if (elemTy && elemTy.isIntOrFloat())
      return elemTy.getIntOrFloatBitWidth() / 8;
    return 8; // conservative fallback for opaque LLVM pointer types
  }

  // vector.load / vector.store / vector.transfer_read / vector.transfer_write
  for (mlir::Value result : op->getResults()) {
    if (auto vecTy = mlir::dyn_cast<mlir::VectorType>(result.getType())) {
      if (mlir::isa<mlir::vector::LoadOp, mlir::vector::TransferReadOp>(op))
        return (vecTy.getNumElements() * vecTy.getElementTypeBitWidth()) / 8;
    }
  }
  if (mlir::isa<mlir::vector::StoreOp, mlir::vector::TransferWriteOp>(op)) {
    mlir::Value vecVal = op->getOperand(0);
    if (auto vecTy = mlir::dyn_cast<mlir::VectorType>(vecVal.getType()))
      return (vecTy.getNumElements() * vecTy.getElementTypeBitWidth()) / 8;
  }

  // CallOpInterface with memref args: sum of buffer sizes.
  if (auto callOp = mlir::dyn_cast<mlir::CallOpInterface>(op)) {
    int64_t total = 0;
    for (mlir::Value operand : callOp->getOperands()) {
      auto memrefTy = mlir::dyn_cast<mlir::MemRefType>(operand.getType());
      if (!memrefTy)
        continue;
      if (memrefTy.hasStaticShape()) {
        int64_t numElems = 1;
        for (int64_t dim : memrefTy.getShape())
          numElems *= dim;
        unsigned elemBits = memrefTy.getElementTypeBitWidth();
        total += elemBits > 0 ? numElems * (elemBits / 8) : 0;
      } else {
        total += cache.l2Size; // conservative for dynamic shapes
      }
    }
    return total;
  }

  return 0; // non-memory op
}

// Forward declaration.
static int64_t estimateBlockFootprintBytes(mlir::Block &block,
                                           const CacheParams &cache,
                                           const EnrichedCallGraph &callGraph);

/// Estimate total memory footprint of all ops in a block, accounting for
/// nested loops (body × trip count) and branches (max of alternatives).
static int64_t estimateBlockFootprintBytes(mlir::Block &block,
                                           const CacheParams &cache,
                                           const EnrichedCallGraph &callGraph) {
  int64_t total = 0;

  for (mlir::Operation &op : block) {
    // Loops: body footprint × trip count.
    if (mlir::isa<mlir::affine::AffineForOp, mlir::scf::ForOp>(&op)) {
      mlir::Region &bodyRegion = op.getRegion(0);
      if (!bodyRegion.empty()) {
        int64_t bodyFP =
            estimateBlockFootprintBytes(bodyRegion.front(), cache, callGraph);
        auto tc = estimateTripCount(&op, callGraph);
        if (tc)
          total += bodyFP * *tc;
        else
          total += bodyFP * kDefaultTripCount;
      }
      continue;
    }

    // If/else: max of branches.
    if (mlir::isa<mlir::scf::IfOp, mlir::affine::AffineIfOp>(&op)) {
      int64_t thenFP = 0, elseFP = 0;
      mlir::Region &thenRegion = op.getRegion(0);
      if (!thenRegion.empty())
        thenFP = estimateBlockFootprintBytes(thenRegion.front(), cache,
                                             callGraph);
      if (op.getNumRegions() > 1) {
        mlir::Region &elseRegion = op.getRegion(1);
        if (!elseRegion.empty())
          elseFP = estimateBlockFootprintBytes(elseRegion.front(), cache,
                                               callGraph);
      }
      total += std::max(thenFP, elseFP);
      continue;
    }

    // scf.while: body × default trip count.
    if (mlir::isa<mlir::scf::WhileOp>(&op)) {
      if (op.getNumRegions() > 1) {
        mlir::Region &bodyRegion = op.getRegion(1);
        if (!bodyRegion.empty()) {
          int64_t bodyFP =
              estimateBlockFootprintBytes(bodyRegion.front(), cache, callGraph);
          total += bodyFP * kDefaultTripCount;
        }
      }
      continue;
    }

    // Leaf op.
    total += estimateOpFootprintBytes(&op, cache);
  }

  return total;
}

// ===== Intervening Footprint Estimation =====

/// Find the ancestor of `op` that lives directly in `targetBlock`.
/// Returns nullptr if `op` is not nested inside `targetBlock`.
static mlir::Operation *findAncestorInBlock(mlir::Operation *op,
                                            mlir::Block *targetBlock) {
  mlir::Operation *current = op;
  while (current) {
    if (current->getBlock() == targetBlock)
      return current;
    current = current->getParentOp();
  }
  return nullptr;
}

/// Sum the footprint of ops strictly between `from` and `to` in the same block.
/// Both must be in the same block; `from` must precede `to`.
static int64_t sumFootprintBetween(mlir::Operation *from, mlir::Operation *to,
                                   const CacheParams &cache,
                                   const EnrichedCallGraph &callGraph) {
  int64_t total = 0;
  for (mlir::Operation *it = from->getNextNode(); it && it != to;
       it = it->getNextNode()) {
    // If the op has regions (loop, if, etc.), estimate its full footprint.
    if (it->getNumRegions() > 0) {
      // Loops.
      if (mlir::isa<mlir::affine::AffineForOp, mlir::scf::ForOp>(it)) {
        mlir::Region &body = it->getRegion(0);
        if (!body.empty()) {
          int64_t bodyFP =
              estimateBlockFootprintBytes(body.front(), cache, callGraph);
          auto tc = estimateTripCount(it, callGraph);
          total += tc ? bodyFP * *tc : bodyFP * kDefaultTripCount;
        }
        continue;
      }
      // If/else.
      if (mlir::isa<mlir::scf::IfOp, mlir::affine::AffineIfOp>(it)) {
        int64_t thenFP = 0, elseFP = 0;
        if (!it->getRegion(0).empty())
          thenFP = estimateBlockFootprintBytes(it->getRegion(0).front(), cache,
                                               callGraph);
        if (it->getNumRegions() > 1 && !it->getRegion(1).empty())
          elseFP = estimateBlockFootprintBytes(it->getRegion(1).front(), cache,
                                               callGraph);
        total += std::max(thenFP, elseFP);
        continue;
      }
      // While.
      if (mlir::isa<mlir::scf::WhileOp>(it)) {
        if (it->getNumRegions() > 1 && !it->getRegion(1).empty()) {
          int64_t bodyFP = estimateBlockFootprintBytes(
              it->getRegion(1).front(), cache, callGraph);
          total += bodyFP * kDefaultTripCount;
        }
        continue;
      }
      // Other region-bearing ops: sum all regions.
      for (mlir::Region &region : it->getRegions())
        if (!region.empty())
          total +=
              estimateBlockFootprintBytes(region.front(), cache, callGraph);
      continue;
    }
    total += estimateOpFootprintBytes(it, cache);
  }
  return total;
}

/// Sum the footprint of all ops after `op` in its block.
static int64_t sumFootprintAfter(mlir::Operation *op, const CacheParams &cache,
                                 const EnrichedCallGraph &callGraph) {
  // Use the block terminator as the sentinel.
  mlir::Block *block = op->getBlock();
  if (!block)
    return 0;
  return sumFootprintBetween(op, block->getTerminator(), cache, callGraph);
}

/// Sum the footprint of all ops before `op` in its block.
static int64_t sumFootprintBefore(mlir::Operation *op,
                                  const CacheParams &cache,
                                  const EnrichedCallGraph &callGraph) {
  mlir::Block *block = op->getBlock();
  if (!block || block->empty())
    return 0;
  // Walk from the first op up to (but not including) `op`.
  int64_t total = 0;
  for (mlir::Operation &it : *block) {
    if (&it == op)
      break;
    if (it.getNumRegions() > 0) {
      if (mlir::isa<mlir::affine::AffineForOp, mlir::scf::ForOp>(&it)) {
        mlir::Region &body = it.getRegion(0);
        if (!body.empty()) {
          int64_t bodyFP =
              estimateBlockFootprintBytes(body.front(), cache, callGraph);
          auto tc = estimateTripCount(&it, callGraph);
          total += tc ? bodyFP * *tc : bodyFP * kDefaultTripCount;
        }
        continue;
      }
      // Simplified: delegate to estimateBlockFootprintBytes for other regions.
      for (mlir::Region &region : it.getRegions())
        if (!region.empty())
          total +=
              estimateBlockFootprintBytes(region.front(), cache, callGraph);
      continue;
    }
    total += estimateOpFootprintBytes(&it, cache);
  }
  return total;
}

/// Estimate the total memory footprint of operations between `storeOp` and
/// `loadOp` in program order.  This is the "interjected" memory traffic that
/// determines whether the stored value is still in cache when the load runs.
static int64_t estimateInterveningFootprint(
    mlir::Operation *storeOp, mlir::Operation *loadOp,
    const CacheParams &cache, const EnrichedCallGraph &callGraph) {
  // Case 0: different functions — conservative.
  auto storeFn = storeOp->getParentOfType<mlir::FunctionOpInterface>();
  auto loadFn = loadOp->getParentOfType<mlir::FunctionOpInterface>();
  if (storeFn != loadFn)
    return cache.l2Size;

  mlir::Block *storeBlock = storeOp->getBlock();
  mlir::Block *loadBlock = loadOp->getBlock();

  // Case 1: same block.
  if (storeBlock == loadBlock)
    return sumFootprintBetween(storeOp, loadOp, cache, callGraph);

  // Case 2: store is in a nested scope below load's block.
  if (mlir::Operation *storeAnc =
          findAncestorInBlock(storeOp, loadBlock)) {
    int64_t fp = sumFootprintAfter(storeOp, cache, callGraph);
    // Walk upward from store's block through enclosing scopes.
    mlir::Operation *cur = storeOp->getParentOp();
    while (cur && cur != storeAnc) {
      mlir::Block *curBlock = cur->getBlock();
      if (curBlock)
        fp += sumFootprintAfter(cur, cache, callGraph);
      cur = cur->getParentOp();
    }
    fp += sumFootprintBetween(storeAnc, loadOp, cache, callGraph);
    return fp;
  }

  // Case 3: load is in a nested scope below store's block.
  if (mlir::Operation *loadAnc =
          findAncestorInBlock(loadOp, storeBlock)) {
    int64_t fp = sumFootprintBetween(storeOp, loadAnc, cache, callGraph);
    fp += sumFootprintBefore(loadOp, cache, callGraph);
    // Walk upward from load's block through enclosing scopes.
    mlir::Operation *cur = loadOp->getParentOp();
    while (cur && cur != loadAnc) {
      mlir::Block *curBlock = cur->getBlock();
      if (curBlock)
        fp += sumFootprintBefore(cur, cache, callGraph);
      cur = cur->getParentOp();
    }
    return fp;
  }

  // Case 4: both in nested scopes — find common ancestor block.
  // Walk store upward until we find a block that contains an ancestor of load.
  for (mlir::Operation *sp = storeOp; sp; sp = sp->getParentOp()) {
    mlir::Block *spBlock = sp->getBlock();
    if (!spBlock)
      continue;
    mlir::Operation *loadAnc = findAncestorInBlock(loadOp, spBlock);
    if (!loadAnc)
      continue;
    mlir::Operation *storeAnc = sp;
    // Now storeAnc and loadAnc are both in spBlock.
    int64_t fp = sumFootprintAfter(storeOp, cache, callGraph);
    // Walk from store up to storeAnc.
    mlir::Operation *cur = storeOp->getParentOp();
    while (cur && cur != storeAnc) {
      mlir::Block *curBlock = cur->getBlock();
      if (curBlock)
        fp += sumFootprintAfter(cur, cache, callGraph);
      cur = cur->getParentOp();
    }
    fp += sumFootprintBetween(storeAnc, loadAnc, cache, callGraph);
    // Walk from loadAnc down to load.
    cur = loadOp->getParentOp();
    while (cur && cur != loadAnc) {
      mlir::Block *curBlock = cur->getBlock();
      if (curBlock)
        fp += sumFootprintBefore(cur, cache, callGraph);
      cur = cur->getParentOp();
    }
    fp += sumFootprintBefore(loadOp, cache, callGraph);
    return fp;
  }

  // Fallback: conservative.
  return cache.l2Size;
}

/// Collect the memref values that the SSA operand tree of `val` loads from.
/// These are the operands that would need to be re-loaded during recomputation.
static void collectOperandMemrefs(
    mlir::Value val,
    llvm::SmallDenseSet<mlir::Value> &memrefs) {
  llvm::SmallVector<mlir::Value, 8> worklist;
  llvm::SmallDenseSet<mlir::Value> visited;
  worklist.push_back(val);

  while (!worklist.empty()) {
    mlir::Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    if (mlir::isa<mlir::BlockArgument>(current))
      continue;
    mlir::Operation *defOp = current.getDefiningOp();
    if (!defOp)
      continue;

    if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(defOp)) {
      memrefs.insert(load.getMemRef());
      continue;
    }
    if (auto load = mlir::dyn_cast<mlir::affine::AffineLoadOp>(defOp)) {
      memrefs.insert(load.getMemRef());
      continue;
    }
    if (mlir::isa<mlir::LLVM::LoadOp>(defOp))
      continue; // can't extract memref from llvm.load

    for (mlir::Value operand : defOp->getOperands())
      worklist.push_back(operand);
  }
}

/// Check whether any operation in the block range (from, to) in the same block
/// accesses the given memref value.  Recurses into nested regions.
static bool memrefAccessedInRange(mlir::Operation *from, mlir::Operation *to,
                                  mlir::Value memref);

/// Check whether a memref is accessed anywhere inside an operation's regions.
static bool memrefAccessedInOp(mlir::Operation *op, mlir::Value memref) {
  bool found = false;
  op->walk([&](mlir::Operation *inner) {
    if (found) return mlir::WalkResult::interrupt();
    if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(inner)) {
      if (load.getMemRef() == memref) { found = true; return mlir::WalkResult::interrupt(); }
    } else if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(inner)) {
      if (store.getMemRef() == memref) { found = true; return mlir::WalkResult::interrupt(); }
    } else if (auto load = mlir::dyn_cast<mlir::affine::AffineLoadOp>(inner)) {
      if (load.getMemRef() == memref) { found = true; return mlir::WalkResult::interrupt(); }
    } else if (auto store = mlir::dyn_cast<mlir::affine::AffineStoreOp>(inner)) {
      if (store.getMemRef() == memref) { found = true; return mlir::WalkResult::interrupt(); }
    }
    return mlir::WalkResult::advance();
  });
  return found;
}

static bool memrefAccessedInRange(mlir::Operation *from, mlir::Operation *to,
                                  mlir::Value memref) {
  for (mlir::Operation *it = from->getNextNode(); it && it != to;
       it = it->getNextNode()) {
    if (memrefAccessedInOp(it, memref))
      return true;
  }
  return false;
}

/// Estimate the operand reload penalty for recomputation.
///
/// Walks the SSA operand tree of the stored value to find which memrefs
/// recomputation would need to read.  For each, checks whether that memref
/// is re-accessed in the intervening ops between store and load.  If it is,
/// the operand is likely still in cache (warm); if not, the full intervening
/// footprint applies and the operand may have been evicted.
///
/// Returns a per-element penalty in cycles that should be added to the
/// recomputation cost.
static unsigned estimateOperandReloadPenalty(
    mlir::Value storedVal,
    mlir::Operation *storeOp, mlir::Operation *loadOp,
    int64_t storeToLoadFootprint,
    const CacheParams &cache) {
  // Collect memrefs needed by the recomputation chain.
  llvm::SmallDenseSet<mlir::Value> operandMemrefs;
  collectOperandMemrefs(storedVal, operandMemrefs);

  if (operandMemrefs.empty())
    return 0; // pure computation, no memory operands

  // Check if store and load are in the same function (for range scanning).
  auto storeFn = storeOp->getParentOfType<mlir::FunctionOpInterface>();
  auto loadFn = loadOp->getParentOfType<mlir::FunctionOpInterface>();

  unsigned totalPenalty = 0;
  unsigned coldCount = 0;

  for (mlir::Value memref : operandMemrefs) {
    bool warm = false;

    // If same function and same block, check the intervening range.
    if (storeFn == loadFn && storeOp->getBlock() == loadOp->getBlock()) {
      warm = memrefAccessedInRange(storeOp, loadOp, memref);
    } else if (storeFn == loadFn) {
      // Different blocks: check if the memref is accessed in the load's block
      // before the load (most recent access).
      mlir::Block *loadBlock = loadOp->getBlock();
      if (loadBlock && !loadBlock->empty()) {
        // Check ops before the load in its block.
        for (mlir::Operation &it : *loadBlock) {
          if (&it == loadOp) break;
          if (memrefAccessedInOp(&it, memref)) {
            warm = true;
            break;
          }
        }
      }
    }
    // Cross-function: conservatively assume cold.

    if (!warm)
      coldCount++;
  }

  if (coldCount == 0)
    return 0; // all operands are warm

  // For cold operands, estimate the reload cost based on intervening footprint.
  // The penalty is the latency delta above L1 (since we assumed L1 in the base cost).
  if (storeToLoadFootprint > (int64_t)cache.l1Size) {
    unsigned evictedLatency = estimateLoadLatency(storeToLoadFootprint, cache);
    unsigned delta = evictedLatency > cache.l1Latency
                         ? evictedLatency - cache.l1Latency
                         : 0;
    // Scale by fraction of cold operands.
    totalPenalty = (delta * coldCount) / operandMemrefs.size();
  }

  return totalPenalty;
}

/// Per-buffer materialization decision.
struct MaterializationDecision {
  bool recompute;              // true = eliminate buffer, false = keep it
  unsigned aluCost;            // ALU cost to recompute one element
  unsigned leafLoadCost;       // per-element cost of cloned partial leaf loads
  unsigned loadLatency;        // estimated load latency (from cache model)
  unsigned numConsumers;       // number of SINGLE loads from this buffer
  int64_t bufferSizeBytes;
  int64_t storeToLoadFootprint;   // intervening bytes between store and load
  unsigned operandPenalty;        // per-element operand reload penalty (cycles)
};

/// Decide whether to recompute or keep a buffer.
///
/// With leafLoadCost = 0 and footprint analysis disabled this reduces to
/// the original formula:
///   keepCost      = aluCost + 1 + numConsumers × loadLatency
///   recomputeCost = numConsumers × aluCost
///
/// With leafLoadCost > 0 (partial rematerialization), each recomputed
/// element also pays the cost of re-issuing its cloned leaf loads.
///
/// When footprint analysis is on:
///   - storeToLoadFootprint shifts the effective load latency (the buffer
///     may have been evicted by intervening memory traffic).
///   - operandPenalty (from estimateOperandReloadPenalty) accounts for the
///     cost of re-fetching operands needed for recomputation, weighted by
///     how many of those operands are still warm vs cold in cache.
static MaterializationDecision
decideBufferStrategy(unsigned aluCost, unsigned leafLoadCost,
                     unsigned loadLatency, unsigned numConsumers,
                     int64_t bufferSizeBytes, int64_t storeToLoadFootprint,
                     unsigned operandPenalty, const CacheParams &cache) {
  // Effective load latency: account for cache eviction from intervening traffic.
  unsigned effectiveLoadLatency = loadLatency;
  if (storeToLoadFootprint > 0) {
    int64_t workingSet = bufferSizeBytes + storeToLoadFootprint;
    effectiveLoadLatency = estimateLoadLatency(workingSet, cache);
  }

  unsigned keepCost = aluCost + 1 + numConsumers * effectiveLoadLatency;
  unsigned recomputeCost =
      numConsumers * (aluCost + leafLoadCost + operandPenalty);

  bool recompute = recomputeCost <= keepCost;

  return MaterializationDecision{recompute,        aluCost,
                                 leafLoadCost,     effectiveLoadLatency,
                                 numConsumers,    bufferSizeBytes,
                                 storeToLoadFootprint, operandPenalty};
}

// Forward declaration; definition follows the partial-remat helpers below.
static mlir::Operation *lookupAllocRoot(
    mlir::Value memref,
    llvm::DenseMap<mlir::Value, mlir::Operation *> &allocRootFor);

/// Return the induction variable of the innermost enclosing loop of op,
/// or a null Value when op is not inside any affine/scf loop.
static mlir::Value innermostEnclosingIV(mlir::Operation *op) {
  mlir::Operation *parent = op ? op->getParentOp() : nullptr;
  while (parent) {
    if (auto f = mlir::dyn_cast<mlir::affine::AffineForOp>(parent))
      return f.getInductionVar();
    if (auto f = mlir::dyn_cast<mlir::scf::ForOp>(parent))
      return f.getInductionVar();
    parent = parent->getParentOp();
  }
  return {};
}

/// Return the linear coefficient of dim `pos` in an affine expression.
/// Returns std::nullopt when the expression is not affine-linear in that
/// dim (e.g. uses mod/div/floordiv/ceildiv or multiplies the dim by a
/// non-constant).
static std::optional<int64_t> affineLinearCoef(mlir::AffineExpr expr,
                                               unsigned pos) {
  if (auto c = mlir::dyn_cast<mlir::AffineConstantExpr>(expr))
    return (int64_t)0;
  if (auto d = mlir::dyn_cast<mlir::AffineDimExpr>(expr))
    return (d.getPosition() == pos) ? (int64_t)1 : (int64_t)0;
  if (mlir::isa<mlir::AffineSymbolExpr>(expr))
    return (int64_t)0;
  auto bin = mlir::dyn_cast<mlir::AffineBinaryOpExpr>(expr);
  if (!bin)
    return std::nullopt;
  auto lhs = affineLinearCoef(bin.getLHS(), pos);
  auto rhs = affineLinearCoef(bin.getRHS(), pos);
  if (!lhs || !rhs)
    return std::nullopt;
  switch (bin.getKind()) {
  case mlir::AffineExprKind::Add:
    return *lhs + *rhs;
  case mlir::AffineExprKind::Mul: {
    if (auto c = mlir::dyn_cast<mlir::AffineConstantExpr>(bin.getLHS()))
      return c.getValue() * *rhs;
    if (auto c = mlir::dyn_cast<mlir::AffineConstantExpr>(bin.getRHS()))
      return *lhs * c.getValue();
    return std::nullopt;
  }
  default:
    return std::nullopt; // mod/floordiv/ceildiv
  }
}

/// Return true when value `v` transitively depends on `iv`.  Walks defining
/// ops; stops on block arguments (other than `iv` itself).
static bool dependsOnValue(mlir::Value v, mlir::Value iv) {
  if (!v || !iv)
    return false;
  llvm::SmallVector<mlir::Value, 8> worklist;
  llvm::SmallDenseSet<mlir::Value> visited;
  worklist.push_back(v);
  while (!worklist.empty()) {
    mlir::Value cur = worklist.pop_back_val();
    if (!visited.insert(cur).second)
      continue;
    if (cur == iv)
      return true;
    mlir::Operation *defOp = cur.getDefiningOp();
    if (!defOp)
      continue;
    for (mlir::Value o : defOp->getOperands())
      worklist.push_back(o);
  }
  return false;
}

/// Estimate the per-iteration access stride (in elements) of a load/store
/// access relative to the induction variable `iv`.  Returns:
///   - 0 when the access is invariant in iv (same address every iteration).
///   - a positive integer when the stride is statically determinable.
///   - std::nullopt when the stride cannot be determined; callers should
///     treat this as pessimistically full-cache-line.
/// Handles both memref.load/store and affine.load/store.
static std::optional<int64_t>
estimateAccessStrideElements(mlir::Operation *accessOp, mlir::Value iv) {
  if (!accessOp || !iv)
    return std::nullopt;

  mlir::MemRefType memrefTy;
  mlir::ValueRange rawIndices;
  mlir::AffineMap map;
  mlir::ValueRange mapOperands;

  if (auto op = mlir::dyn_cast<mlir::memref::LoadOp>(accessOp)) {
    memrefTy = op.getMemRefType();
    rawIndices = op.getIndices();
  } else if (auto op = mlir::dyn_cast<mlir::memref::StoreOp>(accessOp)) {
    memrefTy = op.getMemRefType();
    rawIndices = op.getIndices();
  } else if (auto op = mlir::dyn_cast<mlir::affine::AffineLoadOp>(accessOp)) {
    memrefTy = op.getMemRefType();
    map = op.getAffineMap();
    mapOperands = op.getMapOperands();
  } else if (auto op = mlir::dyn_cast<mlir::affine::AffineStoreOp>(accessOp)) {
    memrefTy = op.getMemRefType();
    map = op.getAffineMap();
    mapOperands = op.getMapOperands();
  } else {
    return std::nullopt;
  }

  unsigned rank = memrefTy.getRank();
  if (rank == 0)
    return (int64_t)0; // scalar memref: always same address

  llvm::ArrayRef<int64_t> shape = memrefTy.getShape();
  // Trailing-dim-product for each dim gives the element stride that a
  // coefficient of 1 in that dim contributes (row-major).
  llvm::SmallVector<int64_t, 4> trailing(rank, 1);
  for (int i = (int)rank - 2; i >= 0; --i) {
    if (shape[i + 1] < 0)
      return std::nullopt; // dynamic trailing dim
    trailing[i] = trailing[i + 1] * shape[i + 1];
  }

  int64_t totalStride = 0;

  if (map) {
    // affine.load/store: analyze each result expression.
    int ivPos = -1;
    for (unsigned i = 0; i < mapOperands.size(); ++i) {
      if (mapOperands[i] == iv) {
        ivPos = (int)i;
        break;
      }
    }
    if (ivPos < 0)
      return (int64_t)0; // iv not among the map operands

    // In MLIR, affine map operands are laid out as [dims..., symbols...].
    // AffineDimExpr positions reference the dim portion. If the iv is
    // passed as a symbol, treat as unknown.
    unsigned numDims = map.getNumDims();
    if ((unsigned)ivPos >= numDims)
      return std::nullopt;

    for (unsigned i = 0; i < rank; ++i) {
      auto coef = affineLinearCoef(map.getResult(i), (unsigned)ivPos);
      if (!coef)
        return std::nullopt;
      totalStride += *coef * trailing[i];
    }
  } else {
    // memref.load/store: inspect each index.
    for (unsigned i = 0; i < rank; ++i) {
      mlir::Value idx = rawIndices[i];
      if (idx == iv) {
        totalStride += trailing[i];
        continue;
      }
      if (!dependsOnValue(idx, iv))
        continue; // invariant in this dim
      // Simple pattern: idx = iv * c  or  iv * c + k  (linear integer
      // arithmetic via arith.muli/addi).  Walk and try to extract a
      // constant coefficient; bail out otherwise.
      std::function<std::optional<int64_t>(mlir::Value)> coefOf =
          [&](mlir::Value v) -> std::optional<int64_t> {
        if (v == iv)
          return (int64_t)1;
        if (!dependsOnValue(v, iv))
          return (int64_t)0;
        mlir::Operation *d = v.getDefiningOp();
        if (!d)
          return std::nullopt;
        if (mlir::isa<mlir::arith::AddIOp>(d)) {
          auto lhs = coefOf(d->getOperand(0));
          auto rhs = coefOf(d->getOperand(1));
          if (!lhs || !rhs)
            return std::nullopt;
          return *lhs + *rhs;
        }
        if (mlir::isa<mlir::arith::MulIOp>(d)) {
          auto getConst = [](mlir::Value x) -> std::optional<int64_t> {
            auto *dx = x.getDefiningOp();
            if (!dx)
              return std::nullopt;
            if (auto c = mlir::dyn_cast<mlir::arith::ConstantIndexOp>(dx))
              return (int64_t)c.value();
            if (auto c = mlir::dyn_cast<mlir::arith::ConstantIntOp>(dx))
              return (int64_t)c.value();
            return std::nullopt;
          };
          auto c0 = getConst(d->getOperand(0));
          auto c1 = getConst(d->getOperand(1));
          if (c0) {
            auto r = coefOf(d->getOperand(1));
            if (!r)
              return std::nullopt;
            return *c0 * *r;
          }
          if (c1) {
            auto l = coefOf(d->getOperand(0));
            if (!l)
              return std::nullopt;
            return *l * *c1;
          }
          return std::nullopt;
        }
        if (auto cast = mlir::dyn_cast<mlir::arith::IndexCastOp>(d))
          return coefOf(cast.getOperand());
        if (auto cast = mlir::dyn_cast<mlir::arith::IndexCastUIOp>(d))
          return coefOf(cast.getOperand());
        return std::nullopt;
      };
      auto c = coefOf(idx);
      if (!c)
        return std::nullopt;
      totalStride += *c * trailing[i];
    }
  }

  // Normalize: negative strides access in reverse but still touch one line
  // per iteration conservatively; use magnitude.
  if (totalStride < 0)
    totalStride = -totalStride;
  return totalStride;
}

/// Return the element size (bytes) of a load/store op's memref.
static unsigned accessElementBytes(mlir::Operation *accessOp) {
  mlir::Type elemTy;
  if (auto o = mlir::dyn_cast<mlir::memref::LoadOp>(accessOp))
    elemTy = o.getMemRefType().getElementType();
  else if (auto o = mlir::dyn_cast<mlir::memref::StoreOp>(accessOp))
    elemTy = o.getMemRefType().getElementType();
  else if (auto o = mlir::dyn_cast<mlir::affine::AffineLoadOp>(accessOp))
    elemTy = o.getMemRefType().getElementType();
  else if (auto o = mlir::dyn_cast<mlir::affine::AffineStoreOp>(accessOp))
    elemTy = o.getMemRefType().getElementType();
  if (!elemTy || !elemTy.isIntOrFloat())
    return 8; // conservative
  unsigned bits = elemTy.getIntOrFloatBitWidth();
  return bits > 0 ? bits / 8 : 8;
}

/// Stride-aware per-issuance cost for a load/store at its current (or
/// intended) site.  Scales the cold-miss latency by the fraction of a
/// cache line touched per iteration of the innermost enclosing loop at
/// `siteOp`:
///
///   effBytes = min(cacheLine, stride*elemSize + 1)
///   cost     = ceil(missLatency * effBytes / cacheLine)
///
/// The `+1` pessimizes unaligned strided access that may straddle a line
/// boundary.  When there is no enclosing loop, or the stride cannot be
/// determined, the full miss latency is charged.
static unsigned estimateAccessLatency(mlir::Operation *accessOp,
                                      mlir::Operation *siteOp,
                                      int64_t bufferBytes,
                                      const CacheParams &cache) {
  unsigned missLat = estimateLoadLatency(bufferBytes, cache);
  unsigned cacheLine = cache.cacheLineSize ? cache.cacheLineSize : 64;

  mlir::Value iv = innermostEnclosingIV(siteOp ? siteOp : accessOp);
  if (!iv)
    return missLat; // one-shot: charge a full cold miss.

  auto strideOpt = estimateAccessStrideElements(accessOp, iv);
  if (!strideOpt)
    return missLat; // unknown stride → pessimize as full line per iter.

  unsigned elemBytes = accessElementBytes(accessOp);
  uint64_t strideBytes = (uint64_t)(*strideOpt) * elemBytes + 1;
  uint64_t effBytes = std::min<uint64_t>(cacheLine, strideBytes);
  uint64_t num = (uint64_t)missLat * effBytes;
  unsigned scaled = (unsigned)((num + cacheLine - 1) / cacheLine);
  return scaled ? scaled : 1; // at least 1 cycle of cost.
}

/// Estimate the per-issuance cost of the partial-leaf loads cloned during
/// partial rematerialization.  Each leaf is priced against its buffer-size
/// cache tier and scaled by stride-aware cache-line amortization relative
/// to the innermost enclosing loop of `insertionPoint` (where the clones
/// will execute).
static unsigned
estimateLeafLoadsCost(llvm::ArrayRef<mlir::Operation *> partialLeaves,
                      mlir::Operation *insertionPoint,
                      const CacheParams &cache,
                      llvm::DenseMap<mlir::Value, mlir::Operation *> &allocRootFor) {
  unsigned total = 0;
  for (mlir::Operation *leaf : partialLeaves) {
    mlir::Value memref;
    if (auto l = mlir::dyn_cast<mlir::memref::LoadOp>(leaf))
      memref = l.getMemRef();
    else if (auto l = mlir::dyn_cast<mlir::affine::AffineLoadOp>(leaf))
      memref = l.getMemRef();
    if (!memref) {
      total += cache.memLatency;
      continue;
    }
    mlir::Operation *root = lookupAllocRoot(memref, allocRootFor);
    int64_t sizeBytes = cache.l2Size + 1; // conservative if unknown
    if (root) {
      if (auto bs = estimateBufferSizeBytes(root))
        sizeBytes = *bs;
    }
    total += estimateAccessLatency(leaf, insertionPoint, sizeBytes, cache);
  }
  return total;
}

/// Maximum number of load→store chains to follow during rematerialization.
constexpr unsigned kMaxChainDepth = 8;

/// Maximum number of ops that may be cloned for a single rematerialization.
/// Prevents blowup on deeply chained arithmetic (e.g. SHA-256 rounds).
constexpr unsigned kMaxRematOps = 64;

/// Per-allocation-root write summary used by partial rematerialization.
///
/// writeCount: number of distinct memref/affine store ops whose base traces
/// back to the root. A value > 1 makes the root ineligible for partial remat
/// (we cannot prove re-reading yields the same value without tracking
/// per-index coverage).
///
/// singleWriter: the one store op when writeCount == 1. Used to verify the
/// writer dominates the partial remat insertion point.
///
/// escapesToCall: true if the root's memref (or a view of it) is passed to
/// any call in the function. Escaping roots cannot be treated as write-once
/// because an external callee may mutate them.
struct RootWriteSummary {
  unsigned writeCount = 0;
  mlir::Operation *singleWriter = nullptr;
  bool escapesToCall = false;
};

using RootWriteMap = llvm::DenseMap<mlir::Operation *, RootWriteSummary>;

/// Build a write summary per allocation root, covering every function in the
/// module. Stores are attributed to roots by tracing their memref through
/// view-like ops via collectBaseMemrefs.
static RootWriteMap
buildRootWriteMap(mlir::ModuleOp moduleOp,
                  llvm::DenseMap<mlir::Value, mlir::Operation *> &allocRootFor) {
  RootWriteMap result;

  auto attribute = [&](mlir::Value memref, mlir::Operation *storeOp) {
    llvm::SmallVector<mlir::Value, 2> bases;
    collectBaseMemrefs(memref, allocRootFor, bases);
    llvm::SmallDenseSet<mlir::Operation *, 2> attributedRoots;
    for (mlir::Value base : bases) {
      auto it = allocRootFor.find(base);
      if (it == allocRootFor.end())
        continue;
      if (!attributedRoots.insert(it->second).second)
        continue;
      auto &summary = result[it->second];
      summary.writeCount++;
      summary.singleWriter = (summary.writeCount == 1) ? storeOp : nullptr;
    }
  };

  moduleOp.walk([&](mlir::Operation *op) {
    if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(op))
      attribute(store.getMemRef(), op);
    else if (auto store = mlir::dyn_cast<mlir::affine::AffineStoreOp>(op))
      attribute(store.getMemRef(), op);
    else if (auto call = mlir::dyn_cast<mlir::CallOpInterface>(op)) {
      for (mlir::Value operand : call->getOperands()) {
        if (!mlir::isa<mlir::MemRefType, mlir::LLVM::LLVMPointerType>(
                operand.getType()))
          continue;
        llvm::SmallVector<mlir::Value, 2> bases;
        collectBaseMemrefs(operand, allocRootFor, bases);
        for (mlir::Value base : bases) {
          auto it = allocRootFor.find(base);
          if (it != allocRootFor.end())
            result[it->second].escapesToCall = true;
        }
      }
    }
  });

  return result;
}

/// Look up an allocation root for a load-like op's memref value. Traces
/// through view-like casts so subview/cast users resolve to their root.
static mlir::Operation *lookupAllocRoot(
    mlir::Value memref,
    llvm::DenseMap<mlir::Value, mlir::Operation *> &allocRootFor) {
  auto it = allocRootFor.find(memref);
  if (it != allocRootFor.end())
    return it->second;
  llvm::SmallVector<mlir::Value, 2> bases;
  collectBaseMemrefs(memref, allocRootFor, bases);
  for (mlir::Value base : bases) {
    auto it2 = allocRootFor.find(base);
    if (it2 != allocRootFor.end())
      return it2->second;
  }
  return nullptr;
}

/// Validate that an SSA value used by a partial leaf load is available at
/// insertionPoint. Mirrors the block-arg/dominance checks in
/// isRematerializable.
static bool isValueAvailableAt(mlir::Value operand,
                               mlir::Operation *insertionPoint,
                               mlir::DominanceInfo &domInfo,
                               const mlir::IRMapping *argMapping) {
  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
    if (argMapping) {
      mlir::Block *block = blockArg.getOwner();
      bool isEntryArg = block && block->isEntryBlock() &&
                        mlir::isa_and_nonnull<mlir::FunctionOpInterface>(
                            block->getParentOp());
      if (!isEntryArg)
        return false;
      if (!argMapping->contains(blockArg))
        return false;
      return domInfo.dominates(argMapping->lookup(blockArg), insertionPoint);
    }
    return domInfo.dominates(operand, insertionPoint);
  }
  mlir::Operation *defOp = operand.getDefiningOp();
  if (!defOp)
    return false;
  return domInfo.properlyDominates(defOp, insertionPoint);
}

/// Returns true if the writer's execution is guaranteed to have completed
/// by the time control reaches insertionPoint. MLIR's DominanceInfo is
/// conservative about ops inside loops/branches; we walk up the ancestor
/// chain until we find a common-region ancestor and check sequential
/// ordering there.
///
/// This is sound for partial rematerialization because the pass only
/// considers replacing loads that the original program actually executed
/// (SINGLE-provenance loads reaching insertionPoint) — if the writer is
/// statically reachable and we observe the downstream load firing, any
/// in-loop or in-branch wrapping of the writer must also have fired.
static bool writerReachesInsertionPoint(mlir::Operation *writer,
                                        mlir::Operation *insertionPoint) {
  // Find the nearest enclosing block common to both writer and the
  // insertion point, then compare the isBeforeInBlock order of the
  // ancestor of `writer` against the ancestor of `insertionPoint` at
  // that level.  This handles the case where the IP is in a deeper
  // nested region than the writer (e.g. writer in a sibling top-level
  // loop, IP inside a later loop body).
  for (mlir::Operation *ipAncestor = insertionPoint; ipAncestor;
       ipAncestor = ipAncestor->getParentOp()) {
    mlir::Block *ipBlock = ipAncestor->getBlock();
    if (!ipBlock)
      return false;
    mlir::Operation *wAncestor = writer;
    while (wAncestor && wAncestor->getBlock() != ipBlock)
      wAncestor = wAncestor->getParentOp();
    if (wAncestor)
      return wAncestor->isBeforeInBlock(ipAncestor);
  }
  return false;
}

/// Why a partial leaf candidate was rejected. Consumed by diagnostics.
enum class PartialLeafReject {
  None,
  NoAllocRoot,
  AllocaOutOfScope,
  Escapes,
  MultiWrite,
  WriterDoesNotDominate,
  OperandNotLive,
};

/// Determine whether a non-chainable leaf load can be safely re-issued at
/// insertionPoint. A safe leaf's clone reads the same memref cell as the
/// original load did. See plan §3 for the write-once proxy rationale.
static bool isSafeLeafAt(
    mlir::Operation *loadOp, mlir::Operation *insertionPoint,
    mlir::DominanceInfo &domInfo,
    const mlir::IRMapping *argMapping,
    llvm::DenseMap<mlir::Value, mlir::Operation *> &allocRootFor,
    const RootWriteMap &rootWrites,
    PartialLeafReject &reject) {
  reject = PartialLeafReject::None;

  mlir::Value memref;
  if (auto l = mlir::dyn_cast<mlir::memref::LoadOp>(loadOp))
    memref = l.getMemRef();
  else if (auto l = mlir::dyn_cast<mlir::affine::AffineLoadOp>(loadOp))
    memref = l.getMemRef();
  if (!memref)
    return false;

  mlir::Operation *root = lookupAllocRoot(memref, allocRootFor);
  if (!root) {
    reject = PartialLeafReject::NoAllocRoot;
    return false;
  }

  // Scoped alloca: the alloca's parent region must enclose insertionPoint.
  if (mlir::isa<mlir::memref::AllocaOp>(root)) {
    mlir::Region *allocRegion = root->getParentRegion();
    mlir::Region *ipRegion = insertionPoint->getParentRegion();
    if (!allocRegion || !ipRegion || !allocRegion->isAncestor(ipRegion)) {
      reject = PartialLeafReject::AllocaOutOfScope;
      return false;
    }
  }

  auto it = rootWrites.find(root);
  if (it == rootWrites.end()) {
    reject = PartialLeafReject::MultiWrite;
    return false;
  }
  const RootWriteSummary &summary = it->second;
  if (summary.escapesToCall) {
    reject = PartialLeafReject::Escapes;
    return false;
  }
  if (summary.writeCount != 1 || !summary.singleWriter) {
    reject = PartialLeafReject::MultiWrite;
    return false;
  }
  if (!writerReachesInsertionPoint(summary.singleWriter, insertionPoint)) {
    reject = PartialLeafReject::WriterDoesNotDominate;
    return false;
  }
  // S2-A: block-order reachability accepts a writer wrapped in a
  // conditional (scf.if / affine.if) because the conditional itself
  // sits before insertionPoint -- but the writer may never have fired
  // if the condition was false.  Reject if any ancestor between the
  // writer and the common-block level is a conditional region op.
  // Loops are still accepted; S1-A's join-lattice sentinel ensures a
  // may-zero loop yields LEAKED rather than SINGLE upstream.
  {
    for (mlir::Operation *ipAnc = insertionPoint; ipAnc;
         ipAnc = ipAnc->getParentOp()) {
      mlir::Operation *wAnc = summary.singleWriter;
      while (wAnc && wAnc->getBlock() != ipAnc->getBlock())
        wAnc = wAnc->getParentOp();
      if (wAnc) {
        for (mlir::Operation *cur = summary.singleWriter->getParentOp();
             cur; cur = cur->getParentOp()) {
          if (mlir::isa<mlir::scf::IfOp, mlir::affine::AffineIfOp>(cur)) {
            reject = PartialLeafReject::WriterDoesNotDominate;
            return false;
          }
          if (cur == wAnc)
            break;
        }
        break;
      }
    }
  }
  (void)domInfo;

  // Every operand of the leaf load (including memref and indices/map operands)
  // must be available at insertionPoint.
  for (mlir::Value operand : loadOp->getOperands()) {
    if (!isValueAvailableAt(operand, insertionPoint, domInfo, argMapping)) {
      reject = PartialLeafReject::OperandNotLive;
      return false;
    }
  }

  return true;
}

/// Checks whether the SSA operand tree rooted at rootVal can be cloned
/// (rematerialized) just before insertionPoint.
///
/// argMapping: optional mapping from callee entry block arguments to caller
/// values (for interprocedural use).
///
/// loadProv: if non-null, enables chaining through SINGLE-provenance loads.
/// When the operand tree encounters a memref.load whose provenance is SINGLE,
/// the load is substituted with the unique store's value and recursion
/// continues into that value's operand tree.
///
/// loadSubs: output pairs of (loadResult, storeValue) for each load that was
/// chained through. Used by rematerializeAt to wire up the substitutions.
///
/// On success, opsToClone is filled in topological (def-before-use) order.
/// Options controlling the partial-rematerialization path in
/// isRematerializable. When enabled, non-chainable leaf loads may be
/// cloned verbatim if isSafeLeafAt accepts them and the leaf budget
/// has room.
struct PartialRematOpts {
  bool allow = false;
  unsigned maxLeaves = 0;
  llvm::DenseMap<mlir::Value, mlir::Operation *> *allocRootFor = nullptr;
  const RootWriteMap *rootWrites = nullptr;
  llvm::SmallVectorImpl<mlir::Operation *> *leaves = nullptr;
  PartialLeafReject lastReject = PartialLeafReject::None;
};

static bool isRematerializable(
    mlir::Value rootVal, mlir::Operation *insertionPoint,
    mlir::DominanceInfo &domInfo,
    llvm::SmallVectorImpl<mlir::Operation *> &opsToClone,
    const mlir::IRMapping *argMapping = nullptr,
    const LoadProvenanceMap *loadProv = nullptr,
    llvm::SmallVectorImpl<std::pair<mlir::Value, mlir::Value>> *loadSubs =
        nullptr,
    PartialRematOpts *partial = nullptr) {
  opsToClone.clear();
  llvm::SmallVector<mlir::Value, 8> worklist;
  llvm::SmallDenseSet<mlir::Value> visited;
  llvm::SmallDenseSet<mlir::Operation *> added;
  unsigned chainDepth = 0;
  worklist.push_back(rootVal);

  while (!worklist.empty()) {
    mlir::Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;

    // Block argument: check dominance (possibly through argMapping).
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(current)) {
      if (argMapping) {
        // Interprocedural: check if this is an entry block arg of a function.
        mlir::Block *block = blockArg.getOwner();
        bool isEntryArg =
            block && block->isEntryBlock() &&
            mlir::isa_and_nonnull<mlir::FunctionOpInterface>(
                block->getParentOp());
        if (!isEntryArg)
          return false; // Loop IV or similar — not available in caller.
        if (!argMapping->contains(blockArg))
          return false; // Unmapped argument.
        mlir::Value mapped = argMapping->lookup(blockArg);
        if (!domInfo.dominates(mapped, insertionPoint))
          return false;
      } else {
        if (!domInfo.dominates(current, insertionPoint))
          return false;
      }
      continue;
    }

    mlir::Operation *defOp = current.getDefiningOp();
    if (!defOp)
      return false;

    // Special case: handle memref.load / affine.load.
    if (mlir::isa<mlir::memref::LoadOp, mlir::affine::AffineLoadOp>(defOp)) {
      // Partial mode: prefer cloning as a partial leaf (no recursion into
      // the load's upstream store value). This lets the pass rematerialize
      // expressions whose chain would otherwise dead-end at a non-chainable
      // inner load — Strategy 2 already tried the full chain path.
      if (partial && partial->allow && partial->leaves &&
          partial->rootWrites && partial->allocRootFor) {
        if (partial->leaves->size() >= partial->maxLeaves)
          return false;
        PartialLeafReject why = PartialLeafReject::None;
        if (isSafeLeafAt(defOp, insertionPoint, domInfo, argMapping,
                         *partial->allocRootFor, *partial->rootWrites, why)) {
          if (added.insert(defOp).second)
            opsToClone.push_back(defOp);
          partial->leaves->push_back(defOp);
          continue;
        }
        partial->lastReject = why;
        // Fall through to chain path as a last resort.
      }

      // Strict-chain path: chain through SINGLE-provenance loads only.
      if (loadProv && loadSubs && chainDepth < kMaxChainDepth) {
        auto provIt = loadProv->find(defOp);
        if (provIt != loadProv->end()) {
          const auto &provSet = provIt->second;
          if (provSet.size() == 1 && !provSet.contains(nullptr)) {
            auto *uniqueStore = *provSet.begin();
            mlir::Value chainedVal = getStoredValue(uniqueStore);
            if (chainedVal) {
              loadSubs->push_back({defOp->getResult(0), chainedVal});
              worklist.push_back(chainedVal);
              ++chainDepth;
              continue;
            }
          }
        }
      }
      // Not SINGLE, not resolvable, or depth exceeded.
      return false;
    }

    // Reject ops with memory effects or calls.
    if (!mlir::isMemoryEffectFree(defOp))
      return false;
    if (mlir::isa<mlir::CallOpInterface>(defOp))
      return false;

    // Leaf op (no operands): e.g., constants.
    if (defOp->getNumOperands() == 0) {
      if (added.insert(defOp).second)
        opsToClone.push_back(defOp);
      continue;
    }

    // Interior op: add and recurse into operands.
    if (added.insert(defOp).second) {
      opsToClone.push_back(defOp);
      if (opsToClone.size() > kMaxRematOps)
        return false;
    }
    for (mlir::Value operand : defOp->getOperands())
      worklist.push_back(operand);
  }

  // Reverse for topological order (defs before uses).
  std::reverse(opsToClone.begin(), opsToClone.end());
  return true;
}

/// Clone the ops in opsToClone at insertionPoint and return the rematerialized
/// value corresponding to originalVal.
///
/// loadSubs: pairs of (loadResult, storeValue) from isRematerializable.
/// Each load result is mapped to the (possibly cloned) store value so that
/// downstream cloned ops see the substituted value instead of the load.
static mlir::Value rematerializeAt(
    mlir::Value originalVal, mlir::Operation *insertionPoint,
    llvm::ArrayRef<mlir::Operation *> opsToClone,
    const mlir::IRMapping *argMapping = nullptr,
    llvm::ArrayRef<std::pair<mlir::Value, mlir::Value>> loadSubs = {}) {
  mlir::OpBuilder builder(insertionPoint);
  mlir::IRMapping mapping;
  if (argMapping) {
    // Copy callee-arg → caller-value mappings.
    for (auto &[from, to] :
         llvm::make_range(argMapping->getValueMap().begin(),
                          argMapping->getValueMap().end()))
      mapping.map(from, to);
  }
  for (mlir::Operation *op : opsToClone) {
    // Eagerly resolve load substitutions whose store value is now available.
    for (auto &[loadResult, storeValue] : loadSubs) {
      if (!mapping.contains(loadResult)) {
        mlir::Value resolved = mapping.lookupOrDefault(storeValue);
        if (resolved != storeValue || !storeValue.getDefiningOp()) {
          // storeValue was cloned or is a dominating value — resolve now.
          mapping.map(loadResult, resolved);
        }
      }
    }
    builder.clone(*op, mapping);
  }
  // Final pass: resolve any loadSubs for dominating values not in opsToClone.
  for (auto &[loadResult, storeValue] : loadSubs) {
    if (!mapping.contains(loadResult))
      mapping.map(loadResult, mapping.lookupOrDefault(storeValue));
  }
  return mapping.lookupOrDefault(originalVal);
}

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

// ===== Strategy 4: Cross-function ordered recomputation =====
//
// Triggers when a SINGLE load L in reader R reads a global G whose unique
// reaching store S lives in a different function W, AND there exists a
// caller C in which the call to W strictly precedes the call to R, with no
// intervening writer to G. Recompute S's stored value at C and substitute
// it for L inside R.
//
// Two rewrite modes:
//   (a) IN-PLACE: R is private and has exactly one call site (the
//       conforming one). Mutate R's signature in place.
//   (b) SPECIALIZE: clone R into a private R'; rewrite only conforming
//       call sites to call R'. Original R retained for non-conforming
//       callers.

namespace {

struct CrossFnRematPlan;

/// Recursive sub-plan record: an inner memref.load whose hoist-path failed
/// is substituted with a sub-plan that materializes the inner store's
/// value at the outer caller, anchored just-after a separately-resolved
/// inner-writer-call. (Strategy D — chained loads.)
struct LoadSubRecord {
  mlir::Operation *loadOp = nullptr;       // the inner load in writer body
  mlir::Operation *innerStoreOp = nullptr; // S_inner
  mlir::Operation *innerGlobalOp = nullptr;// G_inner (the memref::GlobalOp)
  mlir::FunctionOpInterface innerWriterFn{};
  std::unique_ptr<CrossFnRematPlan> subPlan;

  LoadSubRecord();
  LoadSubRecord(LoadSubRecord &&) noexcept;
  LoadSubRecord &operator=(LoadSubRecord &&) noexcept;
  ~LoadSubRecord();
};

/// Loop-cloning materialization record (Strategy F — loop-bounded stores).
/// When set, materialization clones a loop nest (writing to a scratch
/// buffer at the caller) instead of cloning a straight-line expression
/// tree. The reader's matching loads of the original global are redirected
/// to load from the scratch buffer.
struct LoopMatRecord {
  mlir::Operation *outerLoopOp = nullptr;  // outermost scf.for/affine.for to clone
  mlir::Operation *storeOp = nullptr;      // targeted store within the loop body
  mlir::Operation *globalOp = nullptr;     // base memref::GlobalOp the store writes
  mlir::Operation *getGlobalOp = nullptr;  // the get_global op inside the writer
  bool boundsAreStatic = false;
};

struct CrossFnRematPlan {
  /// Ops to clone, topologically ordered (defs before uses).
  llvm::SmallVector<mlir::Operation *, 4> opsToClone;
  /// Block-args of the writer function that the cloned expression depends on.
  /// Each appears at the position they should be substituted at the caller.
  llvm::SmallVector<mlir::BlockArgument, 4> writerArgs;
  /// Globals read by hoisted memref.load ops in opsToClone. The caller must
  /// have NO writers to these globals reachable in its static call subtree
  /// between the materialization point and the writer call (verified by
  /// the driver before accepting the plan for a given caller).
  llvm::SmallVector<mlir::Operation *, 4> hoistedLoadGlobals;
  /// Strategy D — per-load substitution records. The substituted load is
  /// NOT cloned; its result is mapped to the sub-plan's materialized value.
  llvm::SmallVector<LoadSubRecord, 2> loadSubs;
  /// Strategy F — when set, materializeAtCaller clones a loop nest into a
  /// scratch buffer and returns the scratch memref instead of a scalar.
  std::optional<LoopMatRecord> loopMat;
  /// The writer's original stored value (root of the operand tree). For
  /// loop-mat plans this is the storeOp's stored value (used downstream
  /// for type checks).
  mlir::Value rootValue;
};

LoadSubRecord::LoadSubRecord() = default;
LoadSubRecord::LoadSubRecord(LoadSubRecord &&) noexcept = default;
LoadSubRecord &
LoadSubRecord::operator=(LoadSubRecord &&) noexcept = default;
LoadSubRecord::~LoadSubRecord() = default;

/// Recursion cap for chained sub-plans (Strategy D).
constexpr unsigned kMaxCrossFnChainDepth = 3;

/// Trace memref operand back to a memref.get_global op, accepting only the
/// trivial chain memref.get_global → load. Returns nullptr if not such a
/// trivial chain (e.g. through subview, cast, function arg).
static mlir::Operation *traceLoadToGlobal(mlir::Value memref) {
  if (auto getGlobal =
          mlir::dyn_cast_or_null<mlir::memref::GetGlobalOp>(memref.getDefiningOp()))
    return getGlobal.getOperation();
  return nullptr;
}

/// Resolve a get_global op to its memref::GlobalOp definition.
static mlir::Operation *
resolveGetGlobal(mlir::Operation *getGlobalOp, mlir::ModuleOp moduleOp) {
  auto gg = mlir::dyn_cast_or_null<mlir::memref::GetGlobalOp>(getGlobalOp);
  if (!gg) return nullptr;
  return mlir::SymbolTable::lookupNearestSymbolFrom(moduleOp,
                                                    gg.getNameAttr());
}

/// Helper: extract the memref operand of a load op (memref/affine).
static mlir::Value getLoadMemref(mlir::Operation *def) {
  if (auto l = mlir::dyn_cast<mlir::memref::LoadOp>(def))
    return l.getMemref();
  if (auto l = mlir::dyn_cast<mlir::affine::AffineLoadOp>(def))
    return l.getMemref();
  return nullptr;
}

/// Forward decl: the unified entry point. depth and seenStores guard
/// recursion (Strategy D).
static bool buildCrossFunctionRematPlan(
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
static mlir::Operation *
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
static bool loopBoundsAreEntryArgsOrConst(mlir::Operation *loop,
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
static bool walkStoredValueTree(
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
static bool buildStraightLinePlan(
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
static bool buildLoopPlan(
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
  if (!getGlobalOp) return false;
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
  plan.loopMat = rec;
  return true;
}

/// Walk \p storeOp's stored-value operand tree. Accept pure ops whose
/// leaves are constants or block args of \p writerFn's entry block.
/// Memref loads are accepted (a) as hoists when safe, or (b) via Strategy
/// D recursive sub-plans. Stores nested inside an scf.for / affine.for are
/// handled via Strategy F (loop materialization).
static bool buildCrossFunctionRematPlan(
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
static mlir::Operation *
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

/// Materialization context — bundles caller-side analysis state needed
/// for sub-plan resolution (Strategy D).
struct MatCtx {
  mlir::DominanceInfo *domInfo = nullptr;
  const ModuleGlobalWrites *moduleGlobalWrites = nullptr;
  AllocationRoots *allocRootFor = nullptr;
  const llvm::DenseSet<mlir::Operation *> *globalAllocs = nullptr;
  llvm::function_ref<bool(mlir::Operation *, mlir::Operation *)>
      fnWritesGlobal;
};

/// Materialize the writer's expression at the caller, just after
/// \p writerCall. For straight-line plans, clones \p plan.opsToClone with
/// the writer's entry-block args remapped to writerCall's operands. For
/// loop-mat plans (Strategy F), allocates a scratch buffer and clones the
/// outer loop into it. For sub-plans (Strategy D), recurses into each
/// LoadSubRecord, anchored after a separately-resolved inner-writer-call
/// in the same caller function.
static mlir::Value materializeAtCaller(mlir::Operation *writerCall,
                                       const CrossFnRematPlan &plan,
                                       mlir::OpBuilder &builder,
                                       const MatCtx &ctx);

/// Resolve an inner-writer-call in the same caller function that
/// dominates \p anchorCall and is interference-free for \p sub.
static mlir::Operation *
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

static mlir::Value materializeAtCaller(mlir::Operation *writerCall,
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
static mlir::BlockArgument
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
static mlir::BlockArgument
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
static mlir::FunctionOpInterface
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
static void rewriteCallSite(mlir::Operation *oldCall,
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

} // namespace

void DataRecomputationPass::runOnOperation() {
  moduleOp = this->getOperation();

  // Load CPU cost model (from file or built-in defaults).
  drcompiler::CpuCostModel costModel =
      cpuCostModelFile.empty()
          ? drcompiler::CpuCostModel::getDefault()
          : drcompiler::CpuCostModel::loadFromFile(cpuCostModelFile);

  mlir::MLIRContext *context = &getContext();
  mlir::SymbolTableCollection symTabCollection{};

  DRPassContext passCtx{context, symTabCollection, moduleOp};

  // Strategy 0: constant-global direct-forward. memref.global with
  // `constant` set never gets written, so any load can be replaced by
  // the global's initial value. We only handle splat-dense / scalar
  // initializers here; non-splat tables fall through to the dataflow
  // strategies (where they classify KILLED and are left alone).
  {
    llvm::SmallVector<mlir::Operation *, 8> toErase;
    moduleOp.walk([&](mlir::Operation *op) {
      mlir::Value memref;
      mlir::ValueRange rawIndices;
      if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(op)) {
        memref = load.getMemref();
        rawIndices = load.getIndices();
      } else if (auto load =
                     mlir::dyn_cast<mlir::affine::AffineLoadOp>(op)) {
        memref = load.getMemref();
        rawIndices = load.getIndices();
      } else {
        return;
      }
      auto gg = memref.getDefiningOp<mlir::memref::GetGlobalOp>();
      if (!gg) return;
      auto *symOp = symTabCollection.lookupNearestSymbolFrom(
          moduleOp, gg.getNameAttr());
      auto globalOp =
          mlir::dyn_cast_or_null<mlir::memref::GlobalOp>(symOp);
      if (!globalOp || !globalOp.getConstant()) return;
      auto initAttrOpt = globalOp.getInitialValue();
      if (!initAttrOpt) return;
      auto memrefTy =
          mlir::cast<mlir::MemRefType>(globalOp.getType());
      mlir::Type elemTy = memrefTy.getElementType();
      mlir::Attribute valAttr;
      if (auto dense =
              mlir::dyn_cast<mlir::DenseElementsAttr>(*initAttrOpt)) {
        if (!dense.isSplat()) return; // skip indexed lookup for now
        if (auto fp = mlir::dyn_cast<mlir::DenseFPElementsAttr>(dense)) {
          valAttr = mlir::FloatAttr::get(elemTy, fp.getSplatValue<llvm::APFloat>());
        } else if (auto i =
                       mlir::dyn_cast<mlir::DenseIntElementsAttr>(dense)) {
          valAttr = mlir::IntegerAttr::get(elemTy, i.getSplatValue<llvm::APInt>());
        } else {
          return;
        }
      } else {
        return;
      }
      mlir::OpBuilder builder(op);
      auto cst = builder.create<mlir::arith::ConstantOp>(
          op->getLoc(), elemTy, mlir::cast<mlir::TypedAttr>(valAttr));
      op->getResult(0).replaceAllUsesWith(cst);
      if (drTestDiagnostics)
        op->emitRemark() << "constant-global-fold: ACCEPT";
      toErase.push_back(op);
      (void)rawIndices;
    });
    for (mlir::Operation *op : toErase) op->erase();
  }

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

    for (auto &entry : toProcess) {
      mlir::Operation *loadOp = entry.loadOp;
      mlir::Operation *storeOp = entry.storeOp;
      mlir::Value storedVal = getStoredValue(storeOp);

      // If cost model says keep this buffer, skip rematerialization.
      if (drCostModel && !skipBuffers.empty()) {
        mlir::Operation *root = findAllocRoot(loadOp);
        if (root && skipBuffers.contains(root)) {
          if (drTestDiagnostics)
            loadOp->emitRemark() << "cost-model: SKIP_LOAD (buffer kept)";
          continue;
        }
      }

      auto loadFn = loadOp->getParentOfType<mlir::FunctionOpInterface>();
      auto storeFn = storeOp->getParentOfType<mlir::FunctionOpInterface>();

      if (loadFn == storeFn) {
        // --- Intraprocedural ---

        // Strategy 1: Direct value forwarding (store dominates load).
        if (isSafeToForward(storeOp, loadOp, domInfo)) {
          if (storedVal.getType() != loadOp->getResult(0).getType()) {
            if (drTestDiagnostics)
              loadOp->emitRemark() << "direct-forward: REJECT_TYPE";
            continue;
          }
          if (drTestDiagnostics)
            loadOp->emitRemark() << "direct-forward: ACCEPT";
          loadOp->getResult(0).replaceAllUsesWith(storedVal);
          loadOp->erase();
          continue;
        }

        // Strategy 2: Rematerialization (with load chaining).
        llvm::SmallVector<mlir::Operation *> opsToClone;
        llvm::SmallVector<std::pair<mlir::Value, mlir::Value>> loadSubs;
        if (isRematerializable(storedVal, loadOp, domInfo,
                               opsToClone, /*argMapping=*/nullptr,
                               &loadProv, &loadSubs)) {
          mlir::Value clonedVal = rematerializeAt(
              storedVal, loadOp, opsToClone,
              /*argMapping=*/nullptr, loadSubs);
          if (clonedVal.getType() != loadOp->getResult(0).getType()) {
            if (drTestDiagnostics)
              loadOp->emitRemark() << "full-remat: REJECT_TYPE";
            continue;
          }
          if (drTestDiagnostics)
            loadOp->emitRemark() << "full-remat: ACCEPT";
          loadOp->getResult(0).replaceAllUsesWith(clonedVal);
          loadOp->erase();
          continue;
        } else if (drTestDiagnostics) {
          loadOp->emitRemark() << "full-remat: REJECT_UNSAFE";
        }

        // Strategy 2b: Partial rematerialization.
        if (partialRematEnabled) {
          llvm::SmallVector<mlir::Operation *> partialOps;
          llvm::SmallVector<std::pair<mlir::Value, mlir::Value>> partialSubs;
          llvm::SmallVector<mlir::Operation *> partialLeaves;
          PartialRematOpts partial;
          partial.allow = true;
          partial.maxLeaves = drPartialMaxLeaves;
          partial.allocRootFor = &allocRootFor;
          partial.rootWrites = &rootWrites;
          partial.leaves = &partialLeaves;
          if (isRematerializable(storedVal, loadOp, domInfo, partialOps,
                                 /*argMapping=*/nullptr, &loadProv,
                                 &partialSubs, &partial)) {
            unsigned alu = estimateComputeCost(storedVal, costModel);
            unsigned leaf = estimateLeafLoadsCost(partialLeaves, loadOp,
                                                  cache, allocRootFor);
            mlir::Value loadMemref =
                drcompiler::getLoadStoreMemref(loadOp);
            mlir::Operation *loadRoot =
                lookupAllocRoot(loadMemref, allocRootFor);
            int64_t sizeBytes = cache.l2Size + 1;
            if (loadRoot)
              if (auto bs = estimateBufferSizeBytes(loadRoot))
                sizeBytes = *bs;
            unsigned loadLat =
                estimateAccessLatency(loadOp, loadOp, sizeBytes, cache);
            // Partial remat replaces one original load with a cloned
            // expression of equal or greater memory traffic. Gate strictly:
            // only fire if re-execution is cheaper than the original load.
            if (alu + leaf >= loadLat) {
              if (drTestDiagnostics)
                loadOp->emitRemark()
                    << "partial-remat: REJECT_COST (alu=" << alu
                    << ", leaves=" << leaf << ", keep=" << loadLat << ")";
              continue;
            }
            mlir::Value clonedVal = rematerializeAt(storedVal, loadOp,
                                                    partialOps, nullptr,
                                                    partialSubs);
            if (clonedVal.getType() != loadOp->getResult(0).getType()) {
              if (drTestDiagnostics)
                loadOp->emitRemark() << "partial-remat: REJECT_TYPE";
              continue;
            }
            if (drTestDiagnostics)
              loadOp->emitRemark()
                  << "partial-remat: ACCEPT (alu=" << alu << ", leaves=" << leaf
                  << ", load=" << loadLat << ")";
            loadOp->getResult(0).replaceAllUsesWith(clonedVal);
            loadOp->erase();
            continue;
          } else if (drTestDiagnostics &&
                     partial.lastReject != PartialLeafReject::None) {
            const char *reason = "unknown";
            switch (partial.lastReject) {
            case PartialLeafReject::None: reason = "none"; break;
            case PartialLeafReject::NoAllocRoot: reason = "no-alloc-root"; break;
            case PartialLeafReject::AllocaOutOfScope:
              reason = "alloca-out-of-scope"; break;
            case PartialLeafReject::Escapes: reason = "escapes-to-call"; break;
            case PartialLeafReject::MultiWrite:
              reason = "intervening-write"; break;
            case PartialLeafReject::WriterDoesNotDominate:
              reason = "writer-does-not-dominate"; break;
            case PartialLeafReject::OperandNotLive:
              reason = "index-not-live"; break;
            }
            loadOp->emitRemark()
                << "partial-remat: REJECT_UNSAFE (reason=" << reason << ")";
          }
        }
      } else {
        // --- Interprocedural (Strategy 3) ---
        auto origIt = interproceduralOrigins.find(storeOp);
        if (origIt == interproceduralOrigins.end())
          continue;

        if (origIt->second.size() > 1) {
          if (drTestDiagnostics)
            loadOp->emitRemark() << "interproc: SKIP_AMBIGUOUS_ORIGIN";
          continue;
        }

        bool replaced = false;
        for (auto &origin : origIt->second) {
          if (!domInfo.dominates(origin.callSiteOp, loadOp))
            continue;

          mlir::Region *calleeBody = origin.callee.getCallableRegion();
          if (!calleeBody || calleeBody->empty())
            continue;

          // Build argument mapping: callee entry args → call operands.
          mlir::IRMapping argMapping;
          for (auto [calleeArg, callOperand] :
               llvm::zip(calleeBody->getArguments(),
                         origin.callSiteOp->getOperands()))
            argMapping.map(calleeArg, callOperand);

          llvm::SmallVector<mlir::Operation *> opsToClone;
          llvm::SmallVector<std::pair<mlir::Value, mlir::Value>> loadSubs;
          if (isRematerializable(storedVal, loadOp, domInfo,
                                 opsToClone, &argMapping,
                                 &loadProv, &loadSubs)) {
            mlir::Value clonedVal = rematerializeAt(
                storedVal, loadOp, opsToClone, &argMapping,
                loadSubs);
            if (clonedVal.getType() != loadOp->getResult(0).getType()) {
              if (drTestDiagnostics)
                loadOp->emitRemark() << "interproc-remat: REJECT_TYPE";
              continue;
            }
            if (drTestDiagnostics)
              loadOp->emitRemark() << "interproc-remat: ACCEPT";
            loadOp->getResult(0).replaceAllUsesWith(clonedVal);
            loadOp->erase();

            replaced = true;
            break;
          } else if (drTestDiagnostics) {
            loadOp->emitRemark() << "interproc-remat: REJECT_UNSAFE";
          }

          // Strategy 2b (interprocedural): partial rematerialization.
          if (partialRematEnabled) {
            llvm::SmallVector<mlir::Operation *> partialOps;
            llvm::SmallVector<std::pair<mlir::Value, mlir::Value>> partialSubs;
            llvm::SmallVector<mlir::Operation *> partialLeaves;
            PartialRematOpts partial;
            partial.allow = true;
            partial.maxLeaves = drPartialMaxLeaves;
            partial.allocRootFor = &allocRootFor;
            partial.rootWrites = &rootWrites;
            partial.leaves = &partialLeaves;
            if (isRematerializable(storedVal, loadOp, domInfo, partialOps,
                                   &argMapping, &loadProv, &partialSubs,
                                   &partial)) {
              unsigned alu = estimateComputeCost(storedVal, costModel);
              unsigned leaf = estimateLeafLoadsCost(partialLeaves, loadOp,
                                                    cache, allocRootFor);
              mlir::Value loadMemref =
                  drcompiler::getLoadStoreMemref(loadOp);
              mlir::Operation *loadRoot =
                  lookupAllocRoot(loadMemref, allocRootFor);
              int64_t sizeBytes = cache.l2Size + 1;
              if (loadRoot)
                if (auto bs = estimateBufferSizeBytes(loadRoot))
                  sizeBytes = *bs;
              unsigned loadLat =
                  estimateAccessLatency(loadOp, loadOp, sizeBytes, cache);
              auto dec = decideBufferStrategy(alu, leaf, loadLat,
                                              /*numConsumers=*/1, sizeBytes,
                                              /*storeToLoadFootprint=*/0,
                                              /*operandPenalty=*/0, cache);
              if (!dec.recompute) {
                if (drTestDiagnostics)
                  loadOp->emitRemark()
                      << "partial-remat: REJECT_COST (alu=" << alu
                      << ", leaves=" << leaf << ", keep=" << loadLat << ")";
                continue;
              }
              mlir::Value clonedVal = rematerializeAt(
                  storedVal, loadOp, partialOps, &argMapping, partialSubs);
              if (clonedVal.getType() != loadOp->getResult(0).getType()) {
                if (drTestDiagnostics)
                  loadOp->emitRemark() << "partial-remat: REJECT_TYPE";
                continue;
              }
              if (drTestDiagnostics)
                loadOp->emitRemark()
                    << "partial-remat: ACCEPT (alu=" << alu
                    << ", leaves=" << leaf << ", load=" << loadLat << ")";
              loadOp->getResult(0).replaceAllUsesWith(clonedVal);
              loadOp->erase();
              replaced = true;
              break;
            }
          }
        }
        (void)replaced;
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

      // Collect call sites of readerFn from the call graph.
      auto cgIt = callGraph.find(readerFn.getOperation());
      if (cgIt == callGraph.end() || cgIt->second.empty()) {
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

      for (auto &edge : cgIt->second) {
        auto callerFn = edge.callSiteOp
                            ->getParentOfType<mlir::FunctionOpInterface>();
        if (!callerFn) { nonConforming.push_back(edge.callSiteOp); continue; }
        if (!hoistedLoadsSafeAt(callerFn.getOperation(), plan)) {
          nonConforming.push_back(edge.callSiteOp);
          continue;
        }
        auto *writerCall = findEligibleWriterCall(
            callerFn, edge.callSiteOp, writerFn, globalOp, domInfoTop,
            moduleGlobalWrites, allocRootFor, globalAllocs,
            fnTransitivelyWritesGlobal);
        if (!writerCall) {
          nonConforming.push_back(edge.callSiteOp);
          continue;
        }
        // Strategy D — verify all sub-plans resolve at this caller.
        if (!plan.loadSubs.empty() &&
            !chainedSubsValidAt(callerFn, writerCall, plan)) {
          nonConforming.push_back(edge.callSiteOp);
          continue;
        }
        conforming.push_back({edge.callSiteOp, writerCall});
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
                     cgIt->second.size() == conforming.size() &&
                     conforming.size() == 1;

      // Plan kind drives the extra-arg type and reader-side rewrite path.
      // Straight-line / D plans pass a scalar; F.2 plans pass a memref
      // (the scratch buffer) and rewrite reader's loads via memref swap.
      bool isLoopMat = plan.loopMat.has_value();
      mlir::Type extraTy =
          isLoopMat ? mlir::cast<mlir::memref::GlobalOp>(plan.loopMat->globalOp)
                          .getType()
                    : plan.rootValue.getType();

      MatCtx matCtx{&domInfoTop, &moduleGlobalWrites, &allocRootFor,
                    &globalAllocs, fnTransitivelyWritesGlobal};

      if (inPlace) {
        // Mutate readerFn in place.
        if (isLoopMat)
          addMemrefParamAndRewriteLoads(readerFn, extraTy, loadOps);
        else
          addParamAndReplaceLoads(readerFn, extraTy, loadOps);
        auto &site = conforming.front();
        mlir::OpBuilder builder(site.writerCall);
        mlir::Value matVal =
            materializeAtCaller(site.writerCall, plan, builder, matCtx);
        if (!matVal) continue;
        if (drTestDiagnostics)
          site.readerCall->emitRemark() << "interproc-cross: ACCEPT_INPLACE";
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
        if (isLoopMat)
          addMemrefParamAndRewriteLoads(spec, extraTy, clonedLoads);
        else
          addParamAndReplaceLoads(spec, extraTy, clonedLoads);

        // Cache one materialization per writerCall.
        llvm::DenseMap<mlir::Operation *, mlir::Value> matCache;
        for (auto &site : conforming) {
          mlir::Value matVal;
          auto cIt = matCache.find(site.writerCall);
          if (cIt != matCache.end()) {
            matVal = cIt->second;
          } else {
            mlir::OpBuilder builder(site.writerCall);
            matVal =
                materializeAtCaller(site.writerCall, plan, builder, matCtx);
            if (!matVal) continue;
            matCache[site.writerCall] = matVal;
          }
          if (drTestDiagnostics)
            site.readerCall->emitRemark()
                << "interproc-cross: ACCEPT_SPECIALIZED";
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
