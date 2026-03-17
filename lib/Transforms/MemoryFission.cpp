//===- MemoryFission.cpp - Cache-aware loop fission -----------------------===//
//
// Identifies expensive computation duplicated across multiple loops and
// materializes it into a buffer when the cost model says keeping the buffer
// is cheaper than recomputing.
//
// This is the inverse of loop fusion: it undoes greedy fusion decisions
// (e.g. from cgeist -O2) where the result is redundant expensive ALU work.
//
//===----------------------------------------------------------------------===//

#include "drcompiler/Transforms/MemoryFission.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include <map>
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DRDBG() llvm::dbgs() << "DRCOMP-FISSION: "

namespace mlir {
#define GEN_PASS_DEF_MEMORYFISSIONPASS
#include "drcompiler/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

// ===== Cost Model (shared constants with DataRecomputation) =====

static unsigned opCost(Operation *op) {
  if (isa<arith::ConstantOp>(op))
    return 0;
  if (isa<arith::AddIOp, arith::AddFOp, arith::SubIOp, arith::SubFOp,
          arith::XOrIOp, arith::AndIOp, arith::OrIOp, arith::ShRSIOp,
          arith::ShRUIOp, arith::ShLIOp, arith::SelectOp, arith::CmpIOp,
          arith::CmpFOp>(op))
    return 1;
  if (isa<arith::MulIOp, arith::MulFOp>(op))
    return 3;
  if (isa<arith::DivSIOp, arith::DivUIOp, arith::DivFOp, arith::RemSIOp,
          arith::RemUIOp, arith::RemFOp>(op))
    return 15;
  if (isa<math::SqrtOp, math::ExpOp, math::LogOp, math::SinOp, math::CosOp,
          math::TanhOp, math::PowFOp>(op))
    return 20;
  if (isa<arith::SIToFPOp, arith::FPToSIOp, arith::ExtSIOp, arith::ExtUIOp,
          arith::TruncIOp, arith::IndexCastOp, arith::BitcastOp>(op))
    return 1;
  return 5;
}

// ===== Computation Chain Fingerprinting =====

/// A computation chain: the ops between an affine.load (input) and the
/// "tip" value (where the chain feeds into the loop's reduction/output).
struct CompChain {
  Value sourceMemref; // the memref loaded from
  Value tipValue;     // the final computed value
  unsigned cost = 0;  // total ALU cost
  SmallVector<Operation *> ops; // ops in the chain (topological order)

  /// Structural fingerprint: captures op names and constant values,
  /// abstracting over SSA names and loop IVs.
  std::string fingerprint;
};

/// Recursively build a fingerprint string for a value's computation tree.
/// IVs (block arguments of affine.for) normalize to "IV".
/// Loads from the source memref normalize to "LOAD".
/// Constants include their value.  Other ops include their name.
static void buildFingerprint(Value val, Value sourceMemref,
                             llvm::raw_string_ostream &os,
                             llvm::SmallDenseSet<Value> &visited) {
  if (!visited.insert(val).second) {
    os << "REF";
    return;
  }

  // Block argument — loop IV or function arg.
  if (auto blockArg = dyn_cast<BlockArgument>(val)) {
    if (auto forOp = dyn_cast<affine::AffineForOp>(
            blockArg.getOwner()->getParentOp())) {
      if (blockArg == forOp.getInductionVar()) {
        os << "IV";
        return;
      }
    }
    os << "ARG" << blockArg.getArgNumber();
    return;
  }

  Operation *defOp = val.getDefiningOp();
  if (!defOp) {
    os << "?";
    return;
  }

  // Load from the source memref.
  if (auto loadOp = dyn_cast<affine::AffineLoadOp>(defOp)) {
    if (loadOp.getMemRef() == sourceMemref) {
      os << "LOAD";
      return;
    }
  }

  // Constant.
  if (auto constOp = dyn_cast<arith::ConstantOp>(defOp)) {
    os << "C(";
    constOp.getValue().print(os, /*elideType=*/false);
    os << ")";
    return;
  }

  // General op: name + operand fingerprints.
  os << defOp->getName().getStringRef() << "(";
  for (auto [i, operand] : llvm::enumerate(defOp->getOperands())) {
    if (i > 0) os << ",";
    buildFingerprint(operand, sourceMemref, os, visited);
  }
  os << ")";
}

/// Extract computation chains from an affine.for loop body.
/// Strategy: find expensive operations (sqrt, div, transcendentals) in the
/// loop body, trace backward to find the full computation subtree feeding
/// each, and fingerprint it.  This captures the shared expensive computation
/// regardless of what consumer-specific ops use its result.
static SmallVector<CompChain>
extractChains(affine::AffineForOp forOp) {
  SmallVector<CompChain> chains;
  DenseSet<Operation *> alreadyTipped;

  // Find expensive ops in the loop body.
  SmallVector<Operation *> expensiveOps;
  for (Operation &op : *forOp.getBody()) {
    if (opCost(&op) >= 15)
      expensiveOps.push_back(&op);
  }

  for (Operation *expOp : expensiveOps) {
    if (expOp->getNumResults() == 0) continue;

    // Walk forward from the expensive op: if it has a single user that is
    // another expensive pure op (e.g., sqrt→divf), extend the tip to cover
    // both.  Only extend when there is a sole user — if the tip has multiple
    // users it is a branch point and each branch should get its own chain.
    Value tip = expOp->getResult(0);
    if (tip.hasOneUse()) {
      Operation *user = *tip.getUsers().begin();
      if (user->getNumResults() == 1 && isMemoryEffectFree(user) &&
          opCost(user) >= 10 &&
          !isa<affine::AffineYieldOp>(user)) {
        tip = user->getResult(0);
      }
    }

    if (alreadyTipped.contains(tip.getDefiningOp()))
      continue;
    alreadyTipped.insert(tip.getDefiningOp());

    // Trace backward from the tip to collect the subtree and find the
    // source memref.
    SmallVector<Operation *> subtreeOps;
    Value sourceMemref;
    unsigned cost = 0;
    DenseSet<Value> visited;

    std::function<void(Value)> traceBack = [&](Value val) {
      if (!visited.insert(val).second) return;
      if (isa<BlockArgument>(val)) return;
      Operation *def = val.getDefiningOp();
      if (!def) return;
      if (auto loadOp = dyn_cast<affine::AffineLoadOp>(def)) {
        Value memref = loadOp.getMemRef();
        if (memref.getDefiningOp() &&
            isa<memref::AllocOp, memref::AllocaOp>(memref.getDefiningOp()))
          return;
        if (!sourceMemref)
          sourceMemref = memref;
        return;
      }
      if (!isMemoryEffectFree(def)) return;
      cost += opCost(def);
      subtreeOps.push_back(def);
      for (Value operand : def->getOperands())
        traceBack(operand);
    };

    traceBack(tip);

    if (subtreeOps.empty() || !sourceMemref || cost < 15)
      continue;

    // Reverse for topological order (defs before uses) since traceBack
    // walks backward and appends in reverse order.
    std::reverse(subtreeOps.begin(), subtreeOps.end());

    // Build fingerprint.
    std::string fp;
    llvm::raw_string_ostream os(fp);
    llvm::SmallDenseSet<Value> fpVisited;
    buildFingerprint(tip, sourceMemref, os, fpVisited);

    CompChain chain;
    chain.sourceMemref = sourceMemref;
    chain.tipValue = tip;
    chain.cost = cost;
    chain.ops = std::move(subtreeOps);
    chain.fingerprint = std::move(fp);
    chains.push_back(std::move(chain));
  }

  return chains;
}

/// Information about a fission candidate: a group of loops computing
/// the same expensive expression from the same source memref.
struct FissionCandidate {
  std::string fingerprint;
  Value sourceMemref;
  Type elementType;
  unsigned computeCost;
  SmallVector<std::pair<affine::AffineForOp, CompChain>> loops;
};

// ===== Memory Fission Pass =====

class MemoryFissionPass final
    : public impl::MemoryFissionPassBase<MemoryFissionPass> {
public:
  using MemoryFissionPassBase<MemoryFissionPass>::MemoryFissionPassBase;

  void runOnOperation() override;
};

} // namespace

void MemoryFissionPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();

  moduleOp.walk([&](mlir::FunctionOpInterface funcOp) {
    Region *body = funcOp.getCallableRegion();
    if (!body || body->empty())
      return;

    // Step 1: Collect computation chains from all affine.for loops.
    DenseMap<Value, SmallVector<std::pair<affine::AffineForOp, CompChain>>>
        memrefGroups;

    funcOp->walk([&](affine::AffineForOp forOp) {
      auto chains = extractChains(forOp);
      DRDBG() << "Loop " << forOp.getLoc() << ": " << chains.size()
              << " chains\n";
      for (auto &chain : chains) {
        DRDBG() << "  fingerprint: " << chain.fingerprint
                << " cost=" << chain.cost
                << " tip=" << *chain.tipValue.getDefiningOp() << "\n";
        memrefGroups[chain.sourceMemref].push_back({forOp, std::move(chain)});
      }
    });

    // Step 2: Group by (sourceMemref, fingerprint) and find duplicates.
    SmallVector<FissionCandidate> candidates;

    for (auto &[memref, entries] : memrefGroups) {
      // Group entries by (fingerprint, parent_region) — only match loops
      // at the same nesting level in the same parent.
      using GroupKey = std::pair<std::string, Region *>;
      std::map<GroupKey, SmallVector<size_t>> fpGroups;
      for (auto [i, entry] : llvm::enumerate(entries)) {
        Region *parent = entry.first->getParentRegion();
        fpGroups[{entry.second.fingerprint, parent}].push_back(i);
      }

      DRDBG() << "Memref " << memref << ": " << entries.size()
              << " chains, " << fpGroups.size() << " unique fingerprints\n";

      for (auto &[key, indices] : fpGroups) {
        auto &fp = key.first;
        DRDBG() << "  fp=\"" << fp << "\" appears " << indices.size()
                << " time(s)\n";
        if (indices.size() < 2)
          continue; // No duplication — skip.

        FissionCandidate cand;
        cand.fingerprint = fp;
        cand.sourceMemref = memref;
        cand.elementType = entries[indices[0]].second.tipValue.getType();
        cand.computeCost = 0;
        for (size_t idx : indices) {
          auto &entry = entries[idx];
          cand.computeCost =
              std::max(cand.computeCost, entry.second.cost);
          cand.loops.push_back({entry.first, std::move(entry.second)});
        }
        candidates.push_back(std::move(cand));
      }
    }

    if (candidates.empty())
      return;

    // Step 3: Apply cost model and transform.
    for (auto &cand : candidates) {
      unsigned numConsumers = cand.loops.size();
      unsigned computeCost = cand.computeCost;

      // Estimate buffer size: we don't know the trip count statically in
      // general, but the buffer element size × trip count determines cache
      // behavior.  For the cost model, use per-element analysis:
      //   recompute_cost = numConsumers × computeCost
      //   keep_cost      = computeCost + 1 + numConsumers × loadLatency
      // We assume the buffer fits in L1 (since we don't know trip count).
      unsigned loadLatency = l1Latency;

      unsigned recomputeCost = numConsumers * computeCost;
      unsigned keepCost = computeCost + 1 + numConsumers * loadLatency;

      bool shouldFission = keepCost < recomputeCost;

      DRDBG() << "Candidate: " << numConsumers << " loops, cost="
              << computeCost << ", fingerprint=" << cand.fingerprint << "\n";
      DRDBG() << "  recompute=" << recomputeCost << " vs keep=" << keepCost
              << " → " << (shouldFission ? "FISSION" : "SKIP") << "\n";

      if (testDiagnostics) {
        // Emit on the first loop in the group.
        cand.loops[0].first->emitRemark()
            << "memory-fission: "
            << (shouldFission ? "FISSION" : "SKIP")
            << " (compute=" << computeCost
            << ", consumers=" << numConsumers << ")";
      }

      if (!shouldFission)
        continue;

      // Safety: skip if any loop has a complex body (nested affine.for).
      auto &[firstLoop, firstChain] = cand.loops[0];
      bool allSimple = true;
      for (auto &[loop, chain] : cand.loops) {
        // Skip if this loop contains nested affine.for (complex body).
        bool hasInnerLoop = false;
        loop.getBody()->walk([&](affine::AffineForOp) {
          hasInnerLoop = true;
        });
        if (hasInnerLoop) {
          allSimple = false;
          break;
        }
      }
      if (!allSimple) {
        DRDBG() << "  Skipping: nested loops not supported\n";
        continue;
      }

      // Safety: all loops must have the same upper bound operands.
      // Skip if bounds differ (e.g., loop 0..n vs 1..n).
      bool boundsMatch = true;
      for (size_t i = 1; i < cand.loops.size(); ++i) {
        auto &[loop, chain] = cand.loops[i];
        if (loop.getUpperBoundMap() != firstLoop.getUpperBoundMap() ||
            loop.getUpperBoundOperands() != firstLoop.getUpperBoundOperands() ||
            loop.getLowerBoundMap() != firstLoop.getLowerBoundMap()) {
          boundsMatch = false;
          break;
        }
      }
      if (!boundsMatch) {
        DRDBG() << "  Skipping: loop bounds differ\n";
        continue;
      }

      // Step 4: Transform — create producer loop + buffer.
      OpBuilder builder(firstLoop);

      // Get the trip count operands from the first loop (use its bounds).
      // The producer loop will iterate over the same range.
      // For the buffer, we need a dynamic memref since trip count may be
      // dynamic.

      // Create buffer: memref<?xElementType>
      Value tripCount;
      // Get upper bound as a value.
      if (firstLoop.hasConstantUpperBound()) {
        int64_t ub = firstLoop.getConstantUpperBound();
        int64_t lb = firstLoop.hasConstantLowerBound()
                         ? firstLoop.getConstantLowerBound()
                         : 0;
        int64_t size = ub - lb;
        tripCount = builder.create<arith::ConstantIndexOp>(
            firstLoop.getLoc(), size);
      } else {
        // Dynamic upper bound — extract the trip count operand.
        auto ubOperands = firstLoop.getUpperBoundOperands();
        if (ubOperands.size() == 1) {
          tripCount = ubOperands[0];
        } else {
          DRDBG() << "  Skipping: can't extract trip count\n";
          continue;
        }
      }

      auto bufType = MemRefType::get({ShapedType::kDynamic}, cand.elementType);
      Value buf = builder.create<memref::AllocOp>(firstLoop.getLoc(), bufType,
                                                  ValueRange{tripCount});

      // Create producer loop: clone the computation from the first loop,
      // store results into the buffer.
      auto producerLoop = builder.create<affine::AffineForOp>(
          firstLoop.getLoc(),
          firstLoop.getLowerBoundOperands(), firstLoop.getLowerBoundMap(),
          firstLoop.getUpperBoundOperands(), firstLoop.getUpperBoundMap());
      {
        OpBuilder bodyBuilder =
            OpBuilder::atBlockBegin(producerLoop.getBody());
        Value iv = producerLoop.getInductionVar();

        // Clone the computation chain from the first loop.
        IRMapping mapping;
        mapping.map(firstLoop.getInductionVar(), iv);

        // Clone ops that the chain depends on (constants, the load, chain ops).
        // First, clone the load.
        auto &chain = firstChain;

        // Find the affine.load in the first loop that starts this chain.
        affine::AffineLoadOp srcLoad;
        firstLoop.getBody()->walk([&](affine::AffineLoadOp loadOp) {
          if (loadOp.getMemRef() == chain.sourceMemref)
            srcLoad = loadOp;
        });

        if (!srcLoad) {
          DRDBG() << "  Skipping: can't find source load\n";
          buf.getDefiningOp()->erase();
          producerLoop->erase();
          continue;
        }

        // Clone the load with mapped IV.
        bodyBuilder.clone(*srcLoad, mapping);

        // Clone each chain op in order.
        for (Operation *op : chain.ops)
          bodyBuilder.clone(*op, mapping);

        // Store the tip value into the buffer.
        Value clonedTip = mapping.lookupOrDefault(chain.tipValue);

        // Compute the buffer index (IV - lower bound).
        Value storeIdx = iv;
        if (firstLoop.hasConstantLowerBound() &&
            firstLoop.getConstantLowerBound() != 0) {
          Value lb = bodyBuilder.create<arith::ConstantIndexOp>(
              firstLoop.getLoc(), firstLoop.getConstantLowerBound());
          storeIdx = bodyBuilder.create<arith::SubIOp>(firstLoop.getLoc(),
                                                       iv, lb);
        }

        bodyBuilder.create<affine::AffineStoreOp>(
            firstLoop.getLoc(), clonedTip, buf, ValueRange{storeIdx});
      }

      // Step 5: In each consumer loop, replace the computation chain
      // with a load from the buffer.
      for (auto &[loop, chain] : cand.loops) {
        Value loopIV = loop.getInductionVar();

        // Find the load index relative to buffer start.
        OpBuilder loopBuilder(chain.tipValue.getDefiningOp());
        Value loadIdx = loopIV;
        if (loop.hasConstantLowerBound() &&
            loop.getConstantLowerBound() != 0) {
          Value lb = loopBuilder.create<arith::ConstantIndexOp>(
              loop.getLoc(), loop.getConstantLowerBound());
          loadIdx = loopBuilder.create<arith::SubIOp>(loop.getLoc(),
                                                      loopIV, lb);
        }

        Value loaded = loopBuilder.create<affine::AffineLoadOp>(
            loop.getLoc(), buf, ValueRange{loadIdx});

        // Replace all uses of the chain tip with the buffer load.
        chain.tipValue.replaceAllUsesWith(loaded);

        // Erase the chain ops (in reverse topological order).
        for (auto it = chain.ops.rbegin(); it != chain.ops.rend(); ++it) {
          Operation *op = *it;
          if (op->use_empty())
            op->erase();
        }
      }

      // Insert dealloc after the last consumer loop.
      Operation *lastOp = cand.loops.back().first;
      builder.setInsertionPointAfter(lastOp);
      builder.create<memref::DeallocOp>(firstLoop.getLoc(), buf);

      if (testDiagnostics) {
        buf.getDefiningOp()->emitRemark()
            << "materialized buffer for " << numConsumers << " consumers";
      }
    }
  });
}

namespace mlir {
std::unique_ptr<Pass> createMemoryFissionPass() {
  return std::make_unique<MemoryFissionPass>();
}
} // namespace mlir
