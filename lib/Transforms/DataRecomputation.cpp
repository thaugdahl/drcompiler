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
StoreMap joinStoreMaps(StoreMap result, const StoreMap &other) {
  for (auto &[root, entries] : other) {
    auto &resultEntries = result[root];
    for (const auto &entry : entries) {
      // Deduplicate: if the same store op exists, union coverages.
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
          mlir::isa<mlir::LLVM::GEPOp>(user)) {
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
    StoreMap bodyState = state;
    mlir::Region &bodyRegion = op->getRegion(0);
    if (!bodyRegion.empty())
      analyzeBlock(bodyRegion.front(), bodyState, ctx);

    // For affine.for with provably-positive trip count (static lb < ub),
    // the body always executes at least once — use only the body state.
    if (auto affineFor = mlir::dyn_cast<mlir::affine::AffineForOp>(op)) {
      if (affineFor.hasConstantLowerBound() &&
          affineFor.hasConstantUpperBound() &&
          affineFor.getConstantLowerBound() <
              affineFor.getConstantUpperBound()) {
        state = bodyState;
        return;
      }
    }

    // Loop may not execute: join body-result with pre-loop state
    state = joinStoreMaps(state, bodyState);
    return;
  }

  // --- scf.while: may-execute while loop ---
  if (auto whileOp = mlir::dyn_cast<mlir::scf::WhileOp>(op)) {
    // before-region (condition)
    StoreMap condState = state;
    mlir::Region &beforeRegion = whileOp.getBefore();
    if (!beforeRegion.empty())
      analyzeBlock(beforeRegion.front(), condState, ctx);

    // after-region (body)
    StoreMap bodyState = condState;
    mlir::Region &afterRegion = whileOp.getAfter();
    if (!afterRegion.empty())
      analyzeBlock(afterRegion.front(), bodyState, ctx);

    // Join with pre-loop (may not execute)
    state = joinStoreMaps(state, bodyState);
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
static void printOperandTree(mlir::Value val, unsigned indent = 0) {
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
  for (mlir::Value operand : defOp->getOperands())
    printOperandTree(operand, indent + 1);
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

// ===== Cost Model =====

/// Cache hierarchy parameters for the cost model.
struct CacheParams {
  unsigned l1Size;    // bytes
  unsigned l2Size;    // bytes
  unsigned l1Latency; // cycles
  unsigned l2Latency;
  unsigned l3Latency;
  unsigned memLatency;
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
  return cache.l3Latency;
}

/// Per-buffer materialization decision.
struct MaterializationDecision {
  bool recompute;         // true = eliminate buffer, false = keep it
  unsigned computeCost;   // ALU cost to recompute one element
  unsigned loadLatency;   // estimated load latency (from cache model)
  unsigned numConsumers;  // number of SINGLE loads from this buffer
  int64_t bufferSizeBytes;
};

/// Decide whether to recompute or keep a buffer.
///
/// Cost of keeping:   store_cost + numConsumers × loadLatency
/// Cost of recomputing: numConsumers × computeCost
///
/// We recompute when recomputing is cheaper.  The store cost is approximated
/// as equal to computeCost (you computed the value to store it).
static MaterializationDecision
decideBufferStrategy(unsigned computeCost, unsigned loadLatency,
                     unsigned numConsumers, int64_t bufferSizeBytes) {
  // Cost of keeping the buffer: we already computed the value (computeCost)
  // plus stored it (~1 cycle write to L1), plus each consumer pays loadLatency.
  unsigned keepCost = computeCost + 1 + numConsumers * loadLatency;

  // Cost of recomputing: each consumer recomputes from scratch.
  unsigned recomputeCost = numConsumers * computeCost;

  bool recompute = recomputeCost <= keepCost;

  return MaterializationDecision{recompute, computeCost, loadLatency,
                                 numConsumers, bufferSizeBytes};
}

/// Maximum number of load→store chains to follow during rematerialization.
constexpr unsigned kMaxChainDepth = 8;

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
static bool isRematerializable(
    mlir::Value rootVal, mlir::Operation *insertionPoint,
    mlir::DominanceInfo &domInfo,
    llvm::SmallVectorImpl<mlir::Operation *> &opsToClone,
    const mlir::IRMapping *argMapping = nullptr,
    const LoadProvenanceMap *loadProv = nullptr,
    llvm::SmallVectorImpl<std::pair<mlir::Value, mlir::Value>> *loadSubs =
        nullptr) {
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

    // Special case: chain through SINGLE-provenance load ops
    // (memref.load or affine.load).
    if (mlir::isa<mlir::memref::LoadOp, mlir::affine::AffineLoadOp>(defOp)) {
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
    if (added.insert(defOp).second)
      opsToClone.push_back(defOp);
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

    CacheParams cache{drL1Size, drL2Size, drL1Latency,
                      drL2Latency, drL3Latency, drMemLatency};

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

      // Now decide per-buffer.
      for (auto &[allocRoot, numConsumers] : bufferConsumerCount) {
        unsigned computeCost = bufferComputeCost.lookup(allocRoot);
        auto bufSize = estimateBufferSizeBytes(allocRoot);
        int64_t sizeBytes = bufSize.value_or((int64_t)cache.l2Size + 1);

        unsigned loadLat = estimateLoadLatency(sizeBytes, cache);
        auto decision =
            decideBufferStrategy(computeCost, loadLat, numConsumers, sizeBytes);

        DRDBG() << "=== Cost model for " << *allocRoot << " ===\n";
        DRDBG() << "  buffer size: "
                << (bufSize ? std::to_string(*bufSize) + " bytes" : "dynamic")
                << "\n";
        DRDBG() << "  compute cost: " << computeCost << " cycles\n";
        DRDBG() << "  load latency: " << loadLat << " cycles\n";
        DRDBG() << "  consumers: " << numConsumers << "\n";
        DRDBG() << "  decision: "
                << (decision.recompute ? "RECOMPUTE" : "KEEP BUFFER") << "\n";

        if (drTestDiagnostics)
          allocRoot->emitRemark()
              << "cost-model: "
              << (decision.recompute ? "RECOMPUTE" : "KEEP")
              << " (compute=" << computeCost
              << ", load=" << loadLat
              << ", consumers=" << numConsumers
              << ", size=" << (bufSize ? std::to_string(*bufSize) : "dynamic")
              << ")";

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
          loadOp->getResult(0).replaceAllUsesWith(clonedVal);
          loadOp->erase();
          continue;
        }
      } else {
        // --- Interprocedural (Strategy 3) ---
        auto origIt = interproceduralOrigins.find(storeOp);
        if (origIt == interproceduralOrigins.end())
          continue;

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
            loadOp->getResult(0).replaceAllUsesWith(clonedVal);
            loadOp->erase();
            replaced = true;
            break;
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
