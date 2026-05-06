//===- CacheCostModel.cpp - Cache-aware cost model -----------------------===//
//
//===----------------------------------------------------------------------===//

#include "drcompiler/Transforms/DataRecomputation/CacheCostModel.h"

#include "drcompiler/Transforms/Utils/OpDispatchUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/SmallVector.h"

namespace dr {

unsigned estimateComputeCost(mlir::Value val,
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

std::optional<int64_t> estimateBufferSizeBytes(mlir::Operation *allocOp) {
  mlir::MemRefType memrefTy;

  if (auto alloc = mlir::dyn_cast<mlir::memref::AllocOp>(allocOp))
    memrefTy = alloc.getType();
  else if (auto alloca = mlir::dyn_cast<mlir::memref::AllocaOp>(allocOp))
    memrefTy = alloca.getType();
  else if (auto g = mlir::dyn_cast<mlir::memref::GlobalOp>(allocOp))
    memrefTy = g.getType();
  else if (allocOp && allocOp->getNumResults() == 1 &&
           mlir::isa<mlir::MemRefType>(allocOp->getResult(0).getType()) &&
           (allocOp->getName().getStringRef() == "krnl.global"))
    // krnl.global is unregistered but its single memref result carries the
    // static type we need.
    memrefTy = mlir::cast<mlir::MemRefType>(allocOp->getResult(0).getType());
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

unsigned estimateLoadLatency(int64_t bufferSizeBytes,
                             const CacheParams &cache) {
  if (bufferSizeBytes <= (int64_t)cache.l1Size)
    return cache.l1Latency;
  if (bufferSizeBytes <= (int64_t)cache.l2Size)
    return cache.l2Latency;
  if (cache.l3Size > 0 && bufferSizeBytes <= (int64_t)cache.l3Size)
    return cache.l3Latency;
  return cache.memLatency;
}

std::optional<int64_t> traceToConstant(mlir::Value val,
                                       const EnrichedCallGraph &callGraph) {
  if (mlir::Operation *defOp = val.getDefiningOp()) {
    if (auto cIdx = mlir::dyn_cast<mlir::arith::ConstantIndexOp>(defOp))
      return cIdx.value();
    if (auto cInt = mlir::dyn_cast<mlir::arith::ConstantIntOp>(defOp))
      return cInt.value();
    if (auto cast = mlir::dyn_cast<mlir::arith::IndexCastOp>(defOp))
      return traceToConstant(cast.getIn(), callGraph);
    return std::nullopt;
  }

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

std::optional<int64_t> estimateTripCount(mlir::Operation *loopOp,
                                         const EnrichedCallGraph &callGraph) {
  if (auto affFor = mlir::dyn_cast<mlir::affine::AffineForOp>(loopOp)) {
    int64_t step = affFor.getStepAsInt();
    if (step <= 0) return std::nullopt;
    if (affFor.hasConstantLowerBound() && affFor.hasConstantUpperBound()) {
      int64_t lb = affFor.getConstantLowerBound();
      int64_t ub = affFor.getConstantUpperBound();
      if (ub <= lb) return 0;
      return (ub - lb + step - 1) / step;
    }
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

namespace {

int64_t estimateOpFootprintBytes(mlir::Operation *op,
                                 const CacheParams &cache) {
  if (drcompiler::isAnyLoadOp(op) || drcompiler::isAnyStoreOp(op)) {
    mlir::Type elemTy = drcompiler::getLoadStoreElementType(op);
    if (elemTy && elemTy.isIntOrFloat())
      return elemTy.getIntOrFloatBitWidth() / 8;
    return 8;
  }

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
        total += cache.l2Size;
      }
    }
    return total;
  }

  return 0;
}

int64_t estimateBlockFootprintBytes(mlir::Block &block,
                                    const CacheParams &cache,
                                    const EnrichedCallGraph &callGraph);

int64_t estimateBlockFootprintBytes(mlir::Block &block,
                                    const CacheParams &cache,
                                    const EnrichedCallGraph &callGraph) {
  int64_t total = 0;

  for (mlir::Operation &op : block) {
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

    total += estimateOpFootprintBytes(&op, cache);
  }

  return total;
}

mlir::Operation *findAncestorInBlock(mlir::Operation *op,
                                     mlir::Block *targetBlock) {
  mlir::Operation *current = op;
  while (current) {
    if (current->getBlock() == targetBlock)
      return current;
    current = current->getParentOp();
  }
  return nullptr;
}

int64_t sumFootprintBetween(mlir::Operation *from, mlir::Operation *to,
                            const CacheParams &cache,
                            const EnrichedCallGraph &callGraph) {
  int64_t total = 0;
  for (mlir::Operation *it = from->getNextNode(); it && it != to;
       it = it->getNextNode()) {
    if (it->getNumRegions() > 0) {
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
      if (mlir::isa<mlir::scf::WhileOp>(it)) {
        if (it->getNumRegions() > 1 && !it->getRegion(1).empty()) {
          int64_t bodyFP = estimateBlockFootprintBytes(
              it->getRegion(1).front(), cache, callGraph);
          total += bodyFP * kDefaultTripCount;
        }
        continue;
      }
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

int64_t sumFootprintAfter(mlir::Operation *op, const CacheParams &cache,
                          const EnrichedCallGraph &callGraph) {
  mlir::Block *block = op->getBlock();
  if (!block)
    return 0;
  return sumFootprintBetween(op, block->getTerminator(), cache, callGraph);
}

int64_t sumFootprintBefore(mlir::Operation *op,
                           const CacheParams &cache,
                           const EnrichedCallGraph &callGraph) {
  mlir::Block *block = op->getBlock();
  if (!block || block->empty())
    return 0;
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

bool memrefAccessedInOp(mlir::Operation *op, mlir::Value memref) {
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

bool memrefAccessedInRange(mlir::Operation *from, mlir::Operation *to,
                           mlir::Value memref) {
  for (mlir::Operation *it = from->getNextNode(); it && it != to;
       it = it->getNextNode()) {
    if (memrefAccessedInOp(it, memref))
      return true;
  }
  return false;
}

} // namespace

int64_t estimateInterveningFootprint(mlir::Operation *storeOp,
                                     mlir::Operation *loadOp,
                                     const CacheParams &cache,
                                     const EnrichedCallGraph &callGraph) {
  auto storeFn = storeOp->getParentOfType<mlir::FunctionOpInterface>();
  auto loadFn = loadOp->getParentOfType<mlir::FunctionOpInterface>();
  if (storeFn != loadFn)
    return cache.l2Size;

  mlir::Block *storeBlock = storeOp->getBlock();
  mlir::Block *loadBlock = loadOp->getBlock();

  if (storeBlock == loadBlock)
    return sumFootprintBetween(storeOp, loadOp, cache, callGraph);

  if (mlir::Operation *storeAnc =
          findAncestorInBlock(storeOp, loadBlock)) {
    int64_t fp = sumFootprintAfter(storeOp, cache, callGraph);
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

  if (mlir::Operation *loadAnc =
          findAncestorInBlock(loadOp, storeBlock)) {
    int64_t fp = sumFootprintBetween(storeOp, loadAnc, cache, callGraph);
    fp += sumFootprintBefore(loadOp, cache, callGraph);
    mlir::Operation *cur = loadOp->getParentOp();
    while (cur && cur != loadAnc) {
      mlir::Block *curBlock = cur->getBlock();
      if (curBlock)
        fp += sumFootprintBefore(cur, cache, callGraph);
      cur = cur->getParentOp();
    }
    return fp;
  }

  for (mlir::Operation *sp = storeOp; sp; sp = sp->getParentOp()) {
    mlir::Block *spBlock = sp->getBlock();
    if (!spBlock)
      continue;
    mlir::Operation *loadAnc = findAncestorInBlock(loadOp, spBlock);
    if (!loadAnc)
      continue;
    mlir::Operation *storeAnc = sp;
    int64_t fp = sumFootprintAfter(storeOp, cache, callGraph);
    mlir::Operation *cur = storeOp->getParentOp();
    while (cur && cur != storeAnc) {
      mlir::Block *curBlock = cur->getBlock();
      if (curBlock)
        fp += sumFootprintAfter(cur, cache, callGraph);
      cur = cur->getParentOp();
    }
    fp += sumFootprintBetween(storeAnc, loadAnc, cache, callGraph);
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

  return cache.l2Size;
}

void collectOperandMemrefs(mlir::Value val,
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
      continue;

    for (mlir::Value operand : defOp->getOperands())
      worklist.push_back(operand);
  }
}

unsigned estimateOperandReloadPenalty(mlir::Value storedVal,
                                      mlir::Operation *storeOp,
                                      mlir::Operation *loadOp,
                                      int64_t storeToLoadFootprint,
                                      const CacheParams &cache) {
  llvm::SmallDenseSet<mlir::Value> operandMemrefs;
  collectOperandMemrefs(storedVal, operandMemrefs);

  if (operandMemrefs.empty())
    return 0;

  auto storeFn = storeOp->getParentOfType<mlir::FunctionOpInterface>();
  auto loadFn = loadOp->getParentOfType<mlir::FunctionOpInterface>();

  unsigned totalPenalty = 0;
  unsigned coldCount = 0;

  for (mlir::Value memref : operandMemrefs) {
    bool warm = false;

    if (storeFn == loadFn && storeOp->getBlock() == loadOp->getBlock()) {
      warm = memrefAccessedInRange(storeOp, loadOp, memref);
    } else if (storeFn == loadFn) {
      mlir::Block *loadBlock = loadOp->getBlock();
      if (loadBlock && !loadBlock->empty()) {
        for (mlir::Operation &it : *loadBlock) {
          if (&it == loadOp) break;
          if (memrefAccessedInOp(&it, memref)) {
            warm = true;
            break;
          }
        }
      }
    }

    if (!warm)
      coldCount++;
  }

  if (coldCount == 0)
    return 0;

  if (storeToLoadFootprint > (int64_t)cache.l1Size) {
    unsigned evictedLatency = estimateLoadLatency(storeToLoadFootprint, cache);
    unsigned delta = evictedLatency > cache.l1Latency
                         ? evictedLatency - cache.l1Latency
                         : 0;
    totalPenalty = (delta * coldCount) / operandMemrefs.size();
  }

  return totalPenalty;
}

MaterializationDecision decideBufferStrategy(unsigned aluCost,
                                             unsigned leafLoadCost,
                                             unsigned loadLatency,
                                             unsigned numConsumers,
                                             int64_t bufferSizeBytes,
                                             int64_t storeToLoadFootprint,
                                             unsigned operandPenalty,
                                             const CacheParams &cache) {
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
                                 numConsumers,     bufferSizeBytes,
                                 storeToLoadFootprint, operandPenalty};
}

} // namespace dr
