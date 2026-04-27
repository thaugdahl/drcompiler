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
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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
void collectBaseMemrefs(
  mlir::Value v, llvm::DenseMap<mlir::Value, mlir::Operation *> &allocRootFor,
  llvm::SmallVectorImpl<mlir::Value> &bases) {

  llvm::SmallVector<mlir::Value, 4> worklist{v};
  llvm::SmallDenseSet<mlir::Value> visited{};

  while (!worklist.empty()) {
    auto current = worklist.pop_back_val();
    if (!visited.insert(current).second) {
      continue;
    }

    // Base memref found
    if (allocRootFor.contains(current) ||
      mlir::isa<mlir::BlockArgument>(current)) {
      bases.push_back(current);
      continue;
    }

    // If the current value has a defining operation, and the operation
    // is a view-taking operation, we get the first operand and add it to the
    // worklist.
    if (mlir::Operation *def = current.getDefiningOp()) {
      if (OpTypeHelper<ViewLike>::anyOf(def)) {
        worklist.push_back(def->getOperand(0));
        continue;
      }

      // llvm.getelementptr: base pointer is operand 0
      if (mlir::isa<mlir::LLVM::GEPOp>(def)) {
        worklist.push_back(def->getOperand(0));
        continue;
      }

      // polygeist.memref2pointer: source memref is operand 0
      if (def->getName().getStringRef() == "polygeist.memref2pointer") {
        worklist.push_back(def->getOperand(0));
        continue;
      }
    }

    // Fallback: Treat as a base if no other method of determining base
    bases.push_back(current);
  }
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
    if (mlir::isa<mlir::CallOpInterface>(op))
      hasInnerCalls = true;
  });

  if (directAccess) return true;
  if (!hasInnerCalls) return false;

  // Has inner calls but no direct access: conservatively assume
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
    InterproceduralOriginMap &interproceduralOrigins) {
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

static void analyzeOp(mlir::Operation *op, StoreMap &state,
    AnalysisContext &ctx) {
  auto &allocRootFor = ctx.allocRootFor;
  auto &loadProv = ctx.loadProv;

  // --- scf.if / affine.if: branch + join ---
  if (mlir::isa<mlir::scf::IfOp>(op) ||
      mlir::isa<mlir::affine::AffineIfOp>(op)) {
    // Then region (always present, region 0)
    StoreMap thenState = state;
    mlir::Region &thenRegion = op->getRegion(0);
    if (!thenRegion.empty())
      analyzeBlock(thenRegion.front(), thenState, ctx);

    // Else region (region 1, may be empty or absent)
    StoreMap elseState = state;
    if (op->getNumRegions() > 1) {
      mlir::Region &elseRegion = op->getRegion(1);
      if (!elseRegion.empty())
        analyzeBlock(elseRegion.front(), elseState, ctx);
    }

    state = joinStoreMaps(thenState, elseState);
    return;
  }

  // --- scf.for / affine.for: may-execute loop ---
  if (mlir::isa<mlir::scf::ForOp>(op) ||
      mlir::isa<mlir::affine::AffineForOp>(op)) {
    mlir::Region &bodyRegion = op->getRegion(0);

    // First pass: body starts from pre-loop state.
    StoreMap bodyState = state;
    if (!bodyRegion.empty())
      analyzeBlock(bodyRegion.front(), bodyState, ctx);

    // Second pass: model loop-carried dependencies.  A store in the body can
    // reach a load in the NEXT iteration of the same body, so join the
    // post-body state back into the entry state and re-analyze.  This gives
    // every load inside the body a conservative view of all stores that may
    // have run before it (pre-loop init OR a prior iteration's body store).
    // No absent-path sentinel here: this is a carry join, not a may-not-run
    // join — iteration N>1 always observes prior body stores.
    StoreMap carriedEntry =
        joinStoreMaps(state, bodyState, /*injectAbsentSentinel=*/false);
    StoreMap bodyState2 = carriedEntry;
    if (!bodyRegion.empty())
      analyzeBlock(bodyRegion.front(), bodyState2, ctx);

    // For affine.for with provably-positive trip count (static lb < ub),
    // the body always executes at least once — use only the body state.
    if (auto affineFor = mlir::dyn_cast<mlir::affine::AffineForOp>(op)) {
      if (affineFor.hasConstantLowerBound() &&
          affineFor.hasConstantUpperBound() &&
          affineFor.getConstantLowerBound() <
              affineFor.getConstantUpperBound()) {
        state = bodyState2;
        return;
      }
    }

    // Loop may not execute: join body-result with pre-loop state
    state = joinStoreMaps(state, bodyState2);
    return;
  }

  // --- scf.while: may-execute while loop ---
  if (auto whileOp = mlir::dyn_cast<mlir::scf::WhileOp>(op)) {
    mlir::Region &beforeRegion = whileOp.getBefore();
    mlir::Region &afterRegion  = whileOp.getAfter();

    // First pass.
    StoreMap condState = state;
    if (!beforeRegion.empty())
      analyzeBlock(beforeRegion.front(), condState, ctx);
    StoreMap bodyState = condState;
    if (!afterRegion.empty())
      analyzeBlock(afterRegion.front(), bodyState, ctx);

    // Second pass: model loop-carried deps (body store visible on next iter).
    // Carry join — no absent-path sentinel.
    StoreMap carriedEntry =
        joinStoreMaps(state, bodyState, /*injectAbsentSentinel=*/false);
    StoreMap condState2 = carriedEntry;
    if (!beforeRegion.empty())
      analyzeBlock(beforeRegion.front(), condState2, ctx);
    StoreMap bodyState2 = condState2;
    if (!afterRegion.empty())
      analyzeBlock(afterRegion.front(), bodyState2, ctx);

    // Join with pre-loop (may not execute)
    state = joinStoreMaps(state, bodyState2);
    return;
  }

  // --- memref.store / affine.store ---
  if (auto storeOp = mlir::dyn_cast<mlir::memref::StoreOp>(op)) {
    applyStore(storeOp, state, allocRootFor, ctx.storeValueDeps);
    return;
  }
  if (auto affStore = mlir::dyn_cast<mlir::affine::AffineStoreOp>(op)) {
    // Treat affine.store conservatively (nullopt coverage — affine maps
    // make index extraction non-trivial).
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

  // --- llvm.store: flat (no index coverage) ---
  if (auto llvmStore = mlir::dyn_cast<mlir::LLVM::StoreOp>(op)) {
    applyLLVMStore(llvmStore, state, allocRootFor, ctx.storeValueDeps);
    return;
  }

  // --- llvm.load: record provenance (flat, all stores match) ---
  if (auto llvmLoad = mlir::dyn_cast<mlir::LLVM::LoadOp>(op)) {
    llvm::SmallVector<mlir::Value, 2> bases;
    collectBaseMemrefs(llvmLoad.getAddr(), allocRootFor, bases);
    for (mlir::Value base : bases) {
      auto it = allocRootFor.find(base);
      if (it == allocRootFor.end()) continue;
      auto rootIt = state.find(it->second);
      if (rootIt == state.end()) continue;

      auto &provSet = loadProv[op];
      for (const auto &entry : rootIt->second)
        provSet.insert(entry.storeOp);
    }
    return;
  }

  // --- affine.load: record provenance (flat / nullopt coverage) ---
  if (auto affLoad = mlir::dyn_cast<mlir::affine::AffineLoadOp>(op)) {
    llvm::SmallVector<mlir::Value, 2> bases;
    collectBaseMemrefs(affLoad.getMemRef(), allocRootFor, bases);
    for (mlir::Value base : bases) {
      auto it = allocRootFor.find(base);
      if (it == allocRootFor.end()) continue;
      auto rootIt = state.find(it->second);
      if (rootIt == state.end()) continue;

      auto &provSet = loadProv[op];
      for (const auto &entry : rootIt->second)
        provSet.insert(entry.storeOp);
    }
    return;
  }

  // --- memref.load: record provenance (offset-sensitive) ---
  if (auto loadOp = mlir::dyn_cast<mlir::memref::LoadOp>(op)) {
    llvm::SmallVector<mlir::Value, 2> bases;
    collectBaseMemrefs(loadOp.getMemRef(), allocRootFor, bases);
    for (mlir::Value base : bases) {
      auto it = allocRootFor.find(base);
      if (it == allocRootFor.end()) continue;
      auto rootIt = state.find(it->second);
      if (rootIt == state.end()) continue;

      // Ensure the load appears in provenance even if no stores match
      // (e.g., all stores were killed). An empty provenance set is
      // classified as MULTI by the downstream classifier.
      auto &provSet = loadProv[op];

      bool throughView = (base != loadOp.getMemRef());
      auto loadCoverage = (loadOp.getIndices().empty() || throughView)
                              ? std::nullopt
                              : computeAccessIndices(loadOp.getIndices());

      for (const auto &entry : rootIt->second) {
        if (loadCoverage && entry.coverage) {
          // Both have concrete coverage: only include if they overlap.
          if (loadCoverage->overlaps(*entry.coverage))
            provSet.insert(entry.storeOp);
        } else {
          // Either is nullopt: conservative, always overlaps.
          provSet.insert(entry.storeOp);
        }
      }
    }
    return;
  }

  // --- CallOpInterface: record enriched edge, then conservative clobber ---
  if (auto callIface = mlir::dyn_cast<mlir::CallOpInterface>(op)) {
    // Resolve callee
    mlir::FunctionOpInterface callee = nullptr;
    auto callableRef = callIface.getCallableForCallee();
    if (auto symbol = mlir::dyn_cast<mlir::SymbolRefAttr>(callableRef)) {
      auto *calleeOp = ctx.passCtx.getSymTabCollection().lookupSymbolIn(
          ctx.passCtx.getModuleOp().getOperation(), symbol);
      if (calleeOp)
        callee = mlir::dyn_cast<mlir::FunctionOpInterface>(calleeOp);
    }

    // Build enriched call edge
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

    // Record edge (keyed by callee op, or null for indirect/external)
    mlir::Operation *calleeOp = callee ? callee.getOperation() : nullptr;
    ctx.callGraph[calleeOp].push_back(std::move(edge));

    // Apply clobber with callee-aware refinements
    applyCall(op, state, allocRootFor, ctx.storeValueDeps,
              ctx.globalAllocOps, callee, ctx.interproceduralOrigins);
    return;
  }

  // --- Generic op with regions (catch-all) ---
  if (op->getNumRegions() > 0) {
    // Conservatively iterate all regions sequentially and join results.
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

/// Extract the value written by a store-like op (memref.store, affine.store).
static mlir::Value getStoredValue(mlir::Operation *op) {
  if (auto s = mlir::dyn_cast<mlir::memref::StoreOp>(op))
    return s.getValueToStore();
  if (auto s = mlir::dyn_cast<mlir::affine::AffineStoreOp>(op))
    return s.getValueToStore();
  return {};
}

// ===== Footprint Analysis =====

/// Default trip count when bounds cannot be statically determined.
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

/// Estimate the size of an allocation in bytes.  Returns nullopt if the size
/// cannot be determined statically (e.g. dynamic dimensions).
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
  // memref.store / memref.load
  if (auto s = mlir::dyn_cast<mlir::memref::StoreOp>(op))
    return s.getMemRefType().getElementTypeBitWidth() / 8;
  if (auto l = mlir::dyn_cast<mlir::memref::LoadOp>(op))
    return l.getMemRefType().getElementTypeBitWidth() / 8;

  // affine.store / affine.load
  if (auto s = mlir::dyn_cast<mlir::affine::AffineStoreOp>(op))
    return s.getMemRefType().getElementTypeBitWidth() / 8;
  if (auto l = mlir::dyn_cast<mlir::affine::AffineLoadOp>(op))
    return l.getMemRefType().getElementTypeBitWidth() / 8;

  // llvm.store / llvm.load
  if (auto s = mlir::dyn_cast<mlir::LLVM::StoreOp>(op)) {
    unsigned bits = 0;
    mlir::Type valTy = s.getValue().getType();
    if (valTy.isIntOrFloat())
      bits = valTy.getIntOrFloatBitWidth();
    return bits > 0 ? bits / 8 : 8; // conservative 8 bytes
  }
  if (auto l = mlir::dyn_cast<mlir::LLVM::LoadOp>(op)) {
    unsigned bits = 0;
    mlir::Type resTy = l.getResult().getType();
    if (resTy.isIntOrFloat())
      bits = resTy.getIntOrFloatBitWidth();
    return bits > 0 ? bits / 8 : 8;
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

  auto globalAllocs = collectGlobalStaticAllocations(passCtx);

  // Collect allocation roots for the whole module so the analysis can resolve
  // memref bases that are allocated in one function and passed to another.
  auto allocRootFor = collectAllocationRoots(passCtx, moduleOp);

  // Log functions that may let memory escape to external callers.
  for (auto &f : getPotentialEscapingFunctions(passCtx))
    DRDBG() << "Potential escaper: " << *f.getOperation() << "\n";

  StoreValueDeps storeValueDeps = computeStoreValueDeps(moduleOp, allocRootFor);

  LoadProvenanceMap loadProv;
  EnrichedCallGraph callGraph;
  InterproceduralOriginMap interproceduralOrigins;
  AnalysisContext ctx{passCtx, allocRootFor, loadProv, callGraph, {},
                      storeValueDeps, globalAllocs, interproceduralOrigins};

  // Analyze each top-level function
  moduleOp.walk([&](mlir::FunctionOpInterface funcOp) {
    mlir::Region *body = funcOp.getCallableRegion();
    if (!body || body->empty()) return;
    StoreMap state;
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
        if (root && skipBuffers.contains(root))
          continue; // cost model says keep — skip this load
      }

      auto loadFn = loadOp->getParentOfType<mlir::FunctionOpInterface>();
      auto storeFn = storeOp->getParentOfType<mlir::FunctionOpInterface>();

      if (loadFn == storeFn) {
        // --- Intraprocedural ---

        // Strategy 1: Direct value forwarding (store dominates load).
        if (isSafeToForward(storeOp, loadOp, domInfo)) {
          if (storedVal.getType() != loadOp->getResult(0).getType())
            continue; // type mismatch (e.g. i32 store → i8 load via reinterpret)
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
          if (clonedVal.getType() != loadOp->getResult(0).getType())
            continue; // type mismatch after rematerialization
          loadOp->getResult(0).replaceAllUsesWith(clonedVal);
          loadOp->erase();
          continue;
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
                mlir::isa<mlir::memref::LoadOp>(loadOp)
                    ? mlir::cast<mlir::memref::LoadOp>(loadOp).getMemRef()
                    : mlir::cast<mlir::affine::AffineLoadOp>(loadOp).getMemRef();
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
            if (clonedVal.getType() != loadOp->getResult(0).getType())
              continue;
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
            if (clonedVal.getType() != loadOp->getResult(0).getType())
              continue; // type mismatch after interprocedural rematerialization
            loadOp->getResult(0).replaceAllUsesWith(clonedVal);
            loadOp->erase();

            replaced = true;
            break;
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
                  mlir::isa<mlir::memref::LoadOp>(loadOp)
                      ? mlir::cast<mlir::memref::LoadOp>(loadOp).getMemRef()
                      : mlir::cast<mlir::affine::AffineLoadOp>(loadOp)
                            .getMemRef();
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
              if (clonedVal.getType() != loadOp->getResult(0).getType())
                continue;
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
  }
}

namespace mlir {
std::unique_ptr<Pass> createDataRecomputationPass() {
  return std::make_unique<DataRecomputationPass>();
}
} // namespace mlir
