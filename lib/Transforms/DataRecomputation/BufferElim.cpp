//===- BufferElim.cpp - Per-allocation buffer-elimination verdicts -------===//

#include "drcompiler/Transforms/DataRecomputation/BufferElim.h"
#include "drcompiler/Transforms/DataRecomputation/CacheCostModel.h"
#include "drcompiler/Transforms/Utils/MemrefBaseAnalysis.h"
#include "drcompiler/Transforms/Utils/OpDispatchUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/Hashing.h"

#include <algorithm>

namespace dr {

namespace {

/// Conservative oracle: a callee escapes a memref argument unless its body
/// is available AND every use of the corresponding block argument resolves
/// to a load/store/dealloc/view-chain (i.e., never passed to another call,
/// never returned, never stored as a pointer). This is a simplified mirror
/// of analyzeCalleeArg in DataRecomputation.cpp — kept local so we can run
/// the gate without restructuring the existing static helpers.
bool calleeEscapesArg(mlir::CallOpInterface call, unsigned argIdx,
                      mlir::SymbolTableCollection &symTab) {
  auto symAttr = mlir::dyn_cast<mlir::SymbolRefAttr>(
      call.getCallableForCallee());
  if (!symAttr) return true;
  auto *symOp = symTab.lookupNearestSymbolFrom(
      call->getParentOfType<mlir::ModuleOp>(), symAttr);
  auto callee = mlir::dyn_cast_or_null<mlir::FunctionOpInterface>(symOp);
  if (!callee) return true;
  mlir::Region *body = callee.getCallableRegion();
  if (!body || body->empty()) return true; // external
  if (argIdx >= body->getNumArguments()) return true;

  mlir::BlockArgument arg = body->getArgument(argIdx);
  if (!mlir::isa<mlir::MemRefType, mlir::LLVM::LLVMPointerType>(arg.getType()))
    return false;

  // Walk uses of the block-arg through view-like ops; reject if anything
  // other than load/store/dealloc consumes it.
  llvm::SmallVector<mlir::Value, 4> wl{arg};
  llvm::SmallDenseSet<mlir::Value> seen;
  seen.insert(arg);

  while (!wl.empty()) {
    mlir::Value cur = wl.pop_back_val();
    for (mlir::OpOperand &use : cur.getUses()) {
      mlir::Operation *user = use.getOwner();

      if (drcompiler::isViewLikeOp(user)) {
        for (mlir::Value r : user->getResults())
          if (seen.insert(r).second)
            wl.push_back(r);
        continue;
      }

      if (mlir::isa<mlir::memref::LoadOp, mlir::affine::AffineLoadOp,
                    mlir::LLVM::LoadOp, mlir::memref::StoreOp,
                    mlir::affine::AffineStoreOp, mlir::LLVM::StoreOp,
                    mlir::memref::DeallocOp, mlir::memref::CopyOp>(user))
        continue;

      // Any other consumer (return, another call, ptrtoint, ...) escapes.
      return true;
    }
  }

  return false;
}

/// Count nodes in the SSA tree rooted at v: every defining op except
/// constants and loads. Capped to avoid runaway on cyclic-looking trees.
unsigned treeSize(mlir::Value v, llvm::SmallDenseSet<mlir::Value> &seen,
                   unsigned depth = 0) {
  if (depth > 8) return 0;
  if (!seen.insert(v).second) return 0;
  mlir::Operation *def = v.getDefiningOp();
  if (!def) return 0;
  if (mlir::isa<mlir::arith::ConstantOp>(def)) return 0;
  if (mlir::isa<mlir::memref::LoadOp, mlir::affine::AffineLoadOp,
                mlir::LLVM::LoadOp>(def))
    return 1; // a leaf load still costs 1 cloned op
  unsigned s = 1;
  for (mlir::Value o : def->getOperands())
    s += treeSize(o, seen, depth + 1);
  return s;
}

unsigned treeSize(mlir::Value v) {
  llvm::SmallDenseSet<mlir::Value> seen;
  return treeSize(v, seen, 0);
}

/// Commutative-aware structural hash of an SSA tree. Memoized via `memo`
/// so DAGs with shared subexpressions visit each node once; this keeps the
/// hash linear in node count instead of exponential. Depth-capped as a
/// safety net for adversarial inputs.
llvm::hash_code
structuralHash(mlir::Value v,
               llvm::DenseMap<mlir::Value, llvm::hash_code> &memo,
               unsigned depth = 0) {
  if (depth > 64) return llvm::hash_value("depth-limit");
  if (auto it = memo.find(v); it != memo.end()) return it->second;

  llvm::hash_code result;

  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(v)) {
    result = llvm::hash_combine("blockarg", blockArg.getArgNumber(),
                                blockArg.getOwner());
    memo[v] = result;
    return result;
  }

  mlir::Operation *def = v.getDefiningOp();
  if (!def) {
    result = llvm::hash_value("nodef");
    memo[v] = result;
    return result;
  }

  if (auto c = mlir::dyn_cast<mlir::arith::ConstantOp>(def)) {
    result = llvm::hash_combine("const", mlir::hash_value(c.getValueAttr()));
    memo[v] = result;
    return result;
  }

  llvm::StringRef name = def->getName().getStringRef();
  bool commutative = (name == "arith.addf" || name == "arith.mulf" ||
                      name == "arith.addi" || name == "arith.muli" ||
                      name == "arith.andi" || name == "arith.ori" ||
                      name == "arith.xori" || name == "arith.maxf" ||
                      name == "arith.minf" || name == "arith.maximumf" ||
                      name == "arith.minimumf");

  // Reserve a sentinel before recursing so cyclic uses (which can occur
  // via region-attached operands) terminate safely.
  memo[v] = llvm::hash_value("inprogress");

  llvm::SmallVector<llvm::hash_code, 4> operandHashes;
  operandHashes.reserve(def->getNumOperands());
  for (mlir::Value o : def->getOperands())
    operandHashes.push_back(structuralHash(o, memo, depth + 1));
  if (commutative)
    std::sort(operandHashes.begin(), operandHashes.end());

  llvm::hash_code h = llvm::hash_value(name);
  for (auto oh : operandHashes)
    h = llvm::hash_combine(h, oh);
  memo[v] = h;
  return h;
}

llvm::hash_code structuralHash(mlir::Value v) {
  llvm::DenseMap<mlir::Value, llvm::hash_code> memo;
  return structuralHash(v, memo, 0);
}

} // namespace

BufferElimCostDecision
computeBufferElimCost(mlir::Operation *allocRoot,
                      const PreElimRootStats &stats,
                      llvm::ArrayRef<mlir::Value> storedVals,
                      const CacheParams &cache,
                      const drcompiler::CpuCostModel &cpu,
                      const BufferElimTuning &tuning) {
  auto bufSize = estimateBufferSizeBytes(allocRoot);
  int64_t sizeBytes = bufSize.value_or((int64_t)cache.l2Size + 1);

  BufferElimCostInputs in;
  in.bufferSizeBytes = sizeBytes;
  in.numLoads = stats.loads;
  in.numStores = stats.stores;
  in.loadLatency = estimateLoadLatency(sizeBytes, cache);
  in.storeLatency = std::max(1u, in.loadLatency / 2);

  bool isAlloca = mlir::isa<mlir::memref::AllocaOp>(allocRoot);
  in.allocOverheadCycles = isAlloca ? 0u : 200u;

  if (sizeBytes > (int64_t)cache.l1Size) {
    unsigned ll = estimateLoadLatency(sizeBytes, cache);
    in.capacityPenaltyCycles = ll > cache.l1Latency ? ll - cache.l1Latency : 0;
  }

  unsigned perElemCost = 0;
  unsigned distinctTrees = 0;
  unsigned maxTreeSize = 0;
  llvm::SmallDenseSet<llvm::hash_code> hashes;
  for (mlir::Value sv : storedVals) {
    perElemCost = std::max(perElemCost, estimateComputeCost(sv, cpu));
    maxTreeSize = std::max(maxTreeSize, treeSize(sv));
    hashes.insert(structuralHash(sv));
  }
  distinctTrees = hashes.size();

  in.perElemComputeCost = perElemCost;
  in.numDistinctComputes =
      std::min<unsigned>(stats.loads,
                         distinctTrees ? distinctTrees : stats.loads);

  unsigned bloatOps = in.numDistinctComputes * maxTreeSize;
  in.codeBloatPenalty = bloatOps > tuning.icacheSoftBudget
                            ? bloatOps - tuning.icacheSoftBudget
                            : 0;

  if (maxTreeSize > tuning.regBudget) {
    unsigned excess = maxTreeSize - tuning.regBudget;
    in.regPressurePenalty =
        excess * tuning.spillCycles * in.numDistinctComputes;
  }

  return decideBufferElimination(in);
}

llvm::SmallVector<BufferElimVerdict>
computeBufferElimVerdicts(
    mlir::ModuleOp moduleOp,
    AllocationRoots &allocRootFor,
    const llvm::DenseSet<mlir::Operation *> &keepBuffers,
    const llvm::DenseMap<mlir::Operation *, PreElimRootStats> &preStats,
    const BufferStoredValues &storedValues,
    const llvm::SmallDenseSet<mlir::Operation *> &liveLoads,
    mlir::SymbolTableCollection &symTab,
    const CacheParams &cache,
    const drcompiler::CpuCostModel &cpu,
    const BufferElimTuning &tuning) {

  // Group allocOps. allocRootFor maps each result Value to its alloc Op;
  // we want one verdict per alloc Op (dedup over result values).
  llvm::SmallDenseSet<mlir::Operation *> allocOps;
  for (auto &kv : allocRootFor)
    if (kv.second)
      allocOps.insert(kv.second);

  // Count remaining loads per alloc root by walking the live IR. Safe
  // because liveLoads contains only ops still present in the module.
  llvm::DenseMap<mlir::Operation *, unsigned> remainingByRoot;
  for (mlir::Operation *op : liveLoads) {
    mlir::Value memref = drcompiler::getLoadStoreMemref(op);
    if (!memref) continue;
    llvm::SmallVector<mlir::Value, 2> bases;
    drcompiler::collectBaseMemrefs(memref, allocRootFor, bases);
    for (mlir::Value base : bases) {
      auto rIt = allocRootFor.find(base);
      if (rIt == allocRootFor.end()) continue;
      remainingByRoot[rIt->second]++;
    }
  }

  auto oracleLambda = [&](mlir::CallOpInterface call, unsigned argIdx) {
    return calleeEscapesArg(call, argIdx, symTab);
  };
  drcompiler::CallEscapeOracle oracle{oracleLambda};

  llvm::SmallVector<BufferElimVerdict> out;
  out.reserve(allocOps.size());

  for (mlir::Operation *alloc : allocOps) {
    BufferElimVerdict v;
    v.allocRoot = alloc;

    auto sIt = preStats.find(alloc);
    PreElimRootStats ps = (sIt != preStats.end()) ? sIt->second
                                                  : PreElimRootStats{};
    unsigned remaining = remainingByRoot.lookup(alloc);

    v.loadCount = ps.loads;
    v.storeCount = ps.stores;
    v.remainingLoads = remaining;
    v.multiLoadCount = ps.multi;
    v.leakedLoadCount = ps.leaked;
    v.killedLoadCount = ps.killed;

    // Globals (memref.global / krnl.global) aren't candidates for elimination
    // — they're module-level constants/shared. Report but mark infeasible.
    if (mlir::isa<mlir::memref::GlobalOp>(alloc) ||
        alloc->getName().getStringRef() == "krnl.global" ||
        mlir::isa<mlir::LLVM::GlobalOp>(alloc)) {
      v.escape.kind = drcompiler::EscapeKind::EscapesViaReturn;
      v.feasible = false;
      out.push_back(v);
      continue;
    }

    v.escape = drcompiler::analyzeAllocEscape(alloc, oracle);
    v.allLoadsReplaced = (remaining == 0);

    // Per-load cost-model veto: if any load is on a kept buffer, infeasible.
    bool kept = keepBuffers.contains(alloc);

    v.feasible = !kept &&
                 v.escape.kind == drcompiler::EscapeKind::NoEscape &&
                 v.allLoadsReplaced &&
                 v.multiLoadCount == 0 &&
                 v.leakedLoadCount == 0;

    auto bufSize = estimateBufferSizeBytes(alloc);
    v.bufferSizeBytes = bufSize.value_or((int64_t)cache.l2Size + 1);

    llvm::ArrayRef<mlir::Value> svs;
    auto svIt = storedValues.find(alloc);
    if (svIt != storedValues.end()) svs = svIt->second;

    auto dec = computeBufferElimCost(alloc, ps, svs, cache, cpu, tuning);
    v.keepCost = dec.keepCost;
    v.elimCost = dec.elimCost;
    v.costApproved = dec.eliminate;

    // Feasibility now also requires the cost rollup to approve.
    v.feasible = v.feasible && v.costApproved;

    out.push_back(v);
  }

  return out;
}

} // namespace dr
