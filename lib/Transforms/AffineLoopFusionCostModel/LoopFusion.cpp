//===- LoopFusion.cpp - Code to perform loop fusion -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements affine fusion.
//
//===----------------------------------------------------------------------===//

// DR-DIVERGE: ours first so the GEN_PASS_DECL_DRAFFINELOOPFUSIONPASS struct
// is in scope when the GEN_PASS_DEF expansion references it.
#include "drcompiler/Transforms/AffineLoopFusion.h"
#include "drcompiler/Analysis/ArchHandler.h"
#include "drcompiler/Analysis/RegisterPressureAnalysis.h"
#include "drcompiler/Analysis/SpillStrategy.h"
#include "drcompiler/Transforms/CpuCostModel.h"
#include "drcompiler/Transforms/DataRecomputation/CacheCostModel.h"

#if __has_include("mlir/Dialect/Affine/Passes.h")
#include "mlir/Dialect/Affine/Passes.h"
#else
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#endif

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/LoopFusionUtils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/raw_ostream.h"
#include <iomanip>
#include <optional>
#include <sstream>

namespace mlir {
// DR-DIVERGE: tablegen-defined Base lives in `mlir::` (not `mlir::affine::`).
// We open the `affine` namespace too so the FusionMode enum referenced by
// the tablegen-generated option is found unqualified.
using affine::FusionMode;
#define GEN_PASS_DEF_DRAFFINELOOPFUSIONPASS
#include "drcompiler/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "dr-affine-fusion"

using namespace mlir;
using namespace mlir::affine;

// DR-DIVERGE: file-scoped active context for the unified cost model.  Set in
// `runOnOperation` before calling into upstream's static fusion logic; read
// by our replacement of `isFusionProfitable`'s placeholder decision.
namespace dr_fusion {
struct UnifiedConfig {
  bool useUnifiedCostModel = false;
  bool emitRationale = false;
  std::unique_ptr<drcompiler::ArchHandler> archHandler;
  drcompiler::ArchParams archParams;
  drcompiler::RegisterParams regParams;
  drcompiler::SpillStrategy spillStrategy =
      drcompiler::SpillStrategy::ExcessHot;
  // DR-DIVERGE: cache hierarchy params used by `bytesToMemCycles` to
  // convert raw byte counts into cycle estimates comparable to the
  // ALU/register components of the combiner. Unified Zen4 geometry,
  // matching DataRecomputation/MemoryFission/AffineRegisterBlock (one
  // machine, one geometry) — was {.,262144,0,.} (l2 4x too small, and
  // l3=0 disabled the L3 tier so every 256 KB–∞ working set was priced
  // at mem=200 instead of L3=40).
  dr::CacheParams cache{32768, 1048576, 33554432, 4, 12, 40, 200, 64};
};
static const UnifiedConfig *gActive = nullptr;

/// Convert a memory footprint (bytes) to a per-iteration cycle estimate
/// using the configured cache hierarchy.  This makes memCycles unit-
/// comparable to ALU and register-spill cycles inside the combiner.
inline uint64_t bytesToMemCycles(int64_t bytes) {
  if (!gActive || bytes <= 0)
    return 0;
  // estimateLoadLatency returns per-access latency for data of this size.
  // The TOTAL memory cycles needed to process this many bytes are (lines
  // touched) * (per-access latency).  Treating per-access latency as the
  // total was the bug that made fused/unfused mem comparisons negligible
  // against ALU on programs like gemver.
  unsigned perAccess =
      dr::estimateLoadLatency(bytes, gActive->cache);
  uint64_t lineSize = std::max<uint64_t>(gActive->cache.cacheLineSize, 1u);
  uint64_t lines = (static_cast<uint64_t>(bytes) + lineSize - 1) / lineSize;
  return lines * static_cast<uint64_t>(perAccess);
}

/// Combined cost via the active arch handler.  When no arch is configured we
/// fall back to a plain sum so call sites can use the result uniformly.
inline unsigned combine(unsigned memCycles, unsigned regCycles,
                        unsigned aluCycles) {
  if (gActive && gActive->archHandler)
    return gActive->archHandler->combineCosts(memCycles, regCycles, aluCycles,
                                              gActive->archParams);
  return memCycles + regCycles + aluCycles;
}

/// Collect every op inside `srcRegion` that transitively feeds `srcStoreOp`.
/// This is the "slice" that fusion would clone into the destination body.
/// Walking the full slice (not just the store) gives RPA a realistic view
/// of the live-set the fused body would carry.
inline void collectSliceOps(mlir::Operation *srcStoreOp,
                            mlir::Region &srcRegion,
                            llvm::SmallVectorImpl<mlir::Operation *> &out) {
  if (!srcStoreOp)
    return;
  llvm::DenseSet<mlir::Operation *> visited;
  llvm::SmallVector<mlir::Operation *, 16> worklist{srcStoreOp};
  while (!worklist.empty()) {
    mlir::Operation *op = worklist.pop_back_val();
    if (!visited.insert(op).second)
      continue;
    // Restrict to ops inside the source loop's region; we don't want to
    // pull in iv math or constants defined in the enclosing function.
    if (!srcRegion.isAncestor(op->getParentRegion()))
      continue;
    out.push_back(op);
    for (mlir::Value v : op->getOperands()) {
      if (auto *def = v.getDefiningOp())
        worklist.push_back(def);
    }
  }
}

/// Hypothetical-fusion register-pressure approximation: walk the source
/// store's full def-use slice and replay it as cloned ops on top of the
/// destination body's pressure trace.
inline uint64_t estimateRegCyclesForFusion(AffineForOp srcForOp,
                                            AffineForOp dstForOp,
                                            Operation *srcStoreOp) {
  if (!gActive || !gActive->archHandler)
    return 0;
  llvm::SmallVector<mlir::Operation *, 16> clonedOps;
  collectSliceOps(srcStoreOp, srcForOp.getRegion(), clonedOps);
  drcompiler::PressureQuery q;
  q.params = gActive->regParams;
  q.strategy = gActive->spillStrategy;
  q.tripCount = 1;
  auto res = drcompiler::RegisterPressureAnalysis::analyzeHypotheticalStatic(
      dstForOp.getRegion(), clonedOps, *gActive->archHandler,
      gActive->archParams, q);
  return res.totalSpillCycles;
}

/// Per-memref bounding-box footprint of every affine access in `forOp`
/// (regions unioned per memref at the nest root's depth), or std::nullopt
/// if any region fails to compute.
inline std::optional<llvm::DenseMap<mlir::Value, int64_t>>
perMemrefFootprintBytes(AffineForOp forOp) {
  llvm::DenseMap<mlir::Value, std::unique_ptr<mlir::affine::MemRefRegion>>
      regions;
  unsigned depth = mlir::affine::getNestingDepth(forOp);
  bool error = false;
  forOp->walk([&](mlir::Operation *op) -> mlir::WalkResult {
    if (!mlir::isa<mlir::affine::AffineReadOpInterface,
                   mlir::affine::AffineWriteOpInterface>(op))
      return mlir::WalkResult::advance();
    auto region = std::make_unique<mlir::affine::MemRefRegion>(op->getLoc());
    if (failed(region->compute(op, depth))) {
      error = true;
      return mlir::WalkResult::interrupt();
    }
    auto it = regions.find(region->memref);
    if (it == regions.end()) {
      regions[region->memref] = std::move(region);
    } else if (failed(it->second->unionBoundingBox(*region))) {
      error = true;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (error)
    return std::nullopt;
  llvm::DenseMap<mlir::Value, int64_t> sizes;
  for (auto &kv : regions) {
    std::optional<int64_t> size = kv.second->getRegionSize();
    if (!size)
      return std::nullopt;
    sizes[kv.first] = *size;
  }
  return sizes;
}

/// True if an op with side effects we cannot attribute to a specific memref
/// (an opaque call such as a benchmark timer) sits between the two nests in
/// block order.  Fusion would relocate computation across it — legal when
/// the data doesn't escape, but it dissolves the program's phase structure
/// (e.g. moving array init into a timed region).  Different blocks are
/// conservatively treated as separated.
inline bool sideEffectingOpBetween(mlir::Operation *a, mlir::Operation *b) {
  if (a == b)
    return false;
  if (a->getBlock() != b->getBlock())
    return true;
  mlir::Operation *first = a->isBeforeInBlock(b) ? a : b;
  mlir::Operation *last = first == a ? b : a;
  for (mlir::Operation *op = first->getNextNode(); op && op != last;
       op = op->getNextNode()) {
    if (mlir::isMemoryEffectFree(op))
      continue;
    // Effects the MDG models precisely: affine accesses, allocations,
    // deallocations, and whole affine nests (their accesses carry edges).
    if (mlir::isa<mlir::affine::AffineForOp, mlir::affine::AffineIfOp,
                  mlir::affine::AffineReadOpInterface,
                  mlir::affine::AffineWriteOpInterface, mlir::memref::AllocOp,
                  mlir::memref::AllocaOp, mlir::memref::DeallocOp>(op))
      continue;
    return true;
  }
  return false;
}

/// Bytes of data the two nests both touch: for each memref accessed by both,
/// the smaller of the two per-nest footprints (a bounding-box overlap upper
/// bound — good enough for a fraction-of-traffic gate).  std::nullopt when
/// either side is unanalyzable.
inline std::optional<int64_t> sharedTrafficBytes(AffineForOp srcForOp,
                                                 AffineForOp dstForOp) {
  auto src = perMemrefFootprintBytes(srcForOp);
  auto dst = perMemrefFootprintBytes(dstForOp);
  if (!src || !dst)
    return std::nullopt;
  int64_t shared = 0;
  for (auto &kv : *src) {
    auto it = dst->find(kv.first);
    if (it != dst->end())
      shared += std::min(kv.second, it->second);
  }
  return shared;
}
} // namespace dr_fusion

namespace {
/// Loop fusion pass. This pass currently supports a greedy fusion policy,
/// which fuses loop nests with single-writer/single-reader memref dependences
/// with the goal of improving locality.
// TODO: Support fusion of source loop nests which write to multiple
// memrefs, where each memref can have multiple users (if profitable).
// DR-DIVERGE: pass class renamed to keep upstream affine-loop-fusion
// available alongside our fork (we register both in dr-opt).
struct DrAffineLoopFusionPass
    : public impl::DrAffineLoopFusionPassBase<DrAffineLoopFusionPass> {
  DrAffineLoopFusionPass() = default;
  DrAffineLoopFusionPass(unsigned fastMemorySpace,
                          uint64_t localBufSizeThresholdBytes,
                          bool maximalFusion,
                          enum FusionMode affineFusionMode) {
    this->fastMemorySpace = fastMemorySpace;
    this->localBufSizeThreshold = localBufSizeThresholdBytes / 1024;
    this->maximalFusion = maximalFusion;
    this->affineFusionMode = affineFusionMode;
  }

  void runOnBlock(Block *block);
  void runOnOperation() override;
};

} // namespace

// DR-DIVERGE (maximality fix): upstream ComputationSliceState::isMaximal()
// mis-aligns constraint dimensions whenever the slice contains loops that are
// not single-iteration equalities on destination IVs — e.g. the common outer
// time loop of two stencil nests fused inside it, whose slice bounds are the
// full constant range.  Its `consumerIVs` vector then has fewer real entries
// than slice dims; the padding dims stay unconstrained in the slice set, the
// integer-set difference src\slice comes out empty, and a NON-maximal slice
// is reported maximal.  canRemoveSrcNodeAfterFusion then erases the src nest
// even though the fused slice covers a strict subset of its iteration space —
// a miscompile.  Observed on PolyBench fdtd-2d: ex-update (i in [0,1000))
// fused into the hz nest (i in [0,999)) under the time loop loses the last
// ex row.
//
// Conservative replacement used for the removal decision only: maximality is
// decided purely from constant loop bounds; any non-constant or non-trivial
// form is "unknown", which blocks src removal (correct, possibly redundant).
static std::optional<bool> drIsSliceMaximal(const ComputationSliceState &slice) {
  for (unsigned i = 0, e = slice.lbs.size(); i < e; ++i) {
    AffineMap lbMap = slice.lbs[i];
    AffineMap ubMap = slice.ubs[i];
    AffineForOp srcLoop = getForInductionVarOwner(slice.ivs[i]);
    if (!srcLoop || !srcLoop.hasConstantLowerBound() ||
        !srcLoop.hasConstantUpperBound() || srcLoop.getStep() != 1)
      return std::nullopt;
    int64_t srcLb = srcLoop.getConstantLowerBound();
    int64_t srcUb = srcLoop.getConstantUpperBound();
    if (!lbMap || !ubMap || lbMap.getNumResults() != 1 ||
        ubMap.getNumResults() != 1)
      return std::nullopt;
    AffineExpr lbExpr = lbMap.getResult(0);
    AffineExpr ubExpr = ubMap.getResult(0);

    if (auto lbConst = dyn_cast<AffineConstantExpr>(lbExpr)) {
      auto ubConst = dyn_cast<AffineConstantExpr>(ubExpr);
      if (!ubConst)
        return std::nullopt;
      // Constant slice window, executed as-is for every dst iteration:
      // maximal in this dim iff it covers the whole src loop range.
      if (lbConst.getValue() > srcLb || ubConst.getValue() < srcUb)
        return false;
      continue;
    }

    if (lbExpr + 1 == ubExpr) {
      // Single-iteration equality with a dst IV: the union over the dst loop
      // is the dst loop's range; maximal in this dim iff that range covers
      // the src loop's.
      auto dimExpr = dyn_cast<AffineDimExpr>(lbExpr);
      if (!dimExpr)
        return std::nullopt;
      AffineForOp dstLoop = getForInductionVarOwner(
          slice.lbOperands[i][dimExpr.getPosition()]);
      if (!dstLoop || !dstLoop.hasConstantLowerBound() ||
          !dstLoop.hasConstantUpperBound() || dstLoop.getStep() != 1)
        return std::nullopt;
      if (dstLoop.getConstantLowerBound() > srcLb ||
          dstLoop.getConstantUpperBound() < srcUb)
        return false;
      continue;
    }

    return std::nullopt;
  }
  return true;
}

/// Returns true if node 'srcId' can be removed after fusing it with node
/// 'dstId'. The node can be removed if any of the following conditions are met:
///   1. 'srcId' has no output dependences after fusion and no escaping memrefs.
///   2. 'srcId' has no output dependences after fusion, has escaping memrefs
///       and the fusion slice is maximal.
///   3. 'srcId' has output dependences after fusion, the fusion slice is
///      maximal and the fusion insertion point dominates all the dependences.
static bool canRemoveSrcNodeAfterFusion(
    unsigned srcId, unsigned dstId, const ComputationSliceState &fusionSlice,
    Operation *fusedLoopInsPoint, const DenseSet<Value> &escapingMemRefs,
    const MemRefDependenceGraph &mdg) {

  Operation *dstNodeOp = mdg.getNode(dstId)->op;
  bool hasOutDepsAfterFusion = false;

  for (auto &outEdge : mdg.outEdges.lookup(srcId)) {
    Operation *depNodeOp = mdg.getNode(outEdge.id)->op;
    // Skip dependence with dstOp since it will be removed after fusion.
    if (depNodeOp == dstNodeOp)
      continue;

    // Only fusion within the same block is supported. Use domination analysis
    // when needed.
    if (depNodeOp->getBlock() != dstNodeOp->getBlock())
      return false;

    // Check if the insertion point of the fused loop dominates the dependence.
    // Otherwise, the src loop can't be removed.
    if (fusedLoopInsPoint != depNodeOp &&
        !fusedLoopInsPoint->isBeforeInBlock(depNodeOp)) {
      LDBG() << "Src loop can't be removed: dst loop doesn't "
             << "dominate dependence";
      return false;
    }

    hasOutDepsAfterFusion = true;
  }

  // If src loop has dependences after fusion or it writes to an live-out or
  // escaping memref, we can only remove it if the fusion slice is maximal so
  // that all the dependences are preserved.
  if (hasOutDepsAfterFusion || !escapingMemRefs.empty()) {
    // DR-DIVERGE (maximality fix): use the conservative constant-bounds
    // check instead of the broken ComputationSliceState::isMaximal() (see
    // drIsSliceMaximal above).
    std::optional<bool> isMaximal = drIsSliceMaximal(fusionSlice);
    if (!isMaximal) {
      LDBG() << "Src loop can't be removed: can't determine "
             << "if fusion is maximal";
      return false;
    }

    if (!*isMaximal) {
      LDBG() << "Src loop can't be removed: fusion is not maximal";
      return false;
    }
  }

  return true;
}

/// Returns in 'srcIdCandidates' the producer fusion candidates for consumer
/// 'dstId'. Candidates are sorted by node id order. This order corresponds to
/// the program order when the 'mdg' is created. However, program order is not
/// guaranteed and must not be required by the client. Program order won't be
/// held if the 'mdg' is reused from a previous fusion step or if the node
/// creation order changes in the future to support more advance cases.
// TODO: Move this to a loop fusion utility once 'mdg' is also moved.
static void getProducerCandidates(unsigned dstId,
                                  const MemRefDependenceGraph &mdg,
                                  SmallVectorImpl<unsigned> &srcIdCandidates) {
  // Skip if no input edges along which to fuse.
  if (mdg.inEdges.count(dstId) == 0)
    return;

  // Gather memrefs from loads in 'dstId'.
  auto *dstNode = mdg.getNode(dstId);
  DenseSet<Value> consumedMemrefs;
  for (Operation *load : dstNode->loads)
    consumedMemrefs.insert(cast<AffineReadOpInterface>(load).getMemRef());

  // Traverse 'dstId' incoming edges and gather the nodes that contain a store
  // to one of the consumed memrefs.
  for (const auto &srcEdge : mdg.inEdges.lookup(dstId)) {
    const auto *srcNode = mdg.getNode(srcEdge.id);
    // Skip if 'srcNode' is not a loop nest.
    if (!isa<AffineForOp>(srcNode->op))
      continue;

    if (any_of(srcNode->stores, [&](Operation *op) {
          auto storeOp = cast<AffineWriteOpInterface>(op);
          return consumedMemrefs.count(storeOp.getMemRef()) > 0;
        }))
      srcIdCandidates.push_back(srcNode->id);
  }

  llvm::sort(srcIdCandidates);
  srcIdCandidates.erase(llvm::unique(srcIdCandidates), srcIdCandidates.end());
}

/// Returns in 'producerConsumerMemrefs' the memrefs involved in a
/// producer-consumer dependence between 'srcId' and 'dstId'.
static void
gatherProducerConsumerMemrefs(unsigned srcId, unsigned dstId,
                              const MemRefDependenceGraph &mdg,
                              DenseSet<Value> &producerConsumerMemrefs) {
  auto *dstNode = mdg.getNode(dstId);
  auto *srcNode = mdg.getNode(srcId);
  gatherProducerConsumerMemrefs(srcNode->stores, dstNode->loads,
                                producerConsumerMemrefs);
}

/// A memref escapes in the context of the fusion pass if either:
///   1. it (or its alias) is a block argument, or
///   2. created by an op not known to guarantee alias freedom,
///   3. it (or its alias) are used by ops other than affine dereferencing ops
///   (e.g., by call op, memref load/store ops, alias creating ops, unknown ops,
///   terminator ops, etc.); such ops do not deference the memref in an affine
///   way.
static bool isEscapingMemref(Value memref, Block *block) {
  Operation *defOp = memref.getDefiningOp();
  // Check if 'memref' is a block argument.
  if (!defOp)
    return true;

  // Check if this is defined to be an alias of another memref.
  if (auto viewOp = dyn_cast<mlir::ViewLikeOpInterface>(defOp))
    if (memref == viewOp.getViewDest() &&
        isEscapingMemref(viewOp.getViewSource(), block))
      return true;

  // Any op besides allocating ops wouldn't guarantee alias freedom
  if (!hasSingleEffect<mlir::MemoryEffects::Allocate>(defOp, memref))
    return true;

  // Check if 'memref' is used by a non-deferencing op (including unknown ones)
  // (e.g., call ops, alias creating ops, etc.).
  return llvm::any_of(memref.getUsers(), [&](Operation *user) {
    // Ignore users outside of `block`.
    Operation *ancestorOp = block->getParent()->findAncestorOpInRegion(*user);
    if (!ancestorOp)
      return true;
    if (ancestorOp->getBlock() != block)
      return false;
    return !isa<AffineMapAccessInterface>(*user);
  });
}

/// Returns in 'escapingMemRefs' the memrefs from affine store ops in node 'id'
/// that escape the block or are accessed in a non-affine way.
static void gatherEscapingMemrefs(unsigned id, const MemRefDependenceGraph &mdg,
                                  DenseSet<Value> &escapingMemRefs) {
  auto *node = mdg.getNode(id);
  for (Operation *storeOp : node->stores) {
    auto memref = cast<AffineWriteOpInterface>(storeOp).getMemRef();
    if (escapingMemRefs.count(memref))
      continue;
    if (isEscapingMemref(memref, &mdg.block))
      escapingMemRefs.insert(memref);
  }
}

// DR-DIVERGE (frame-shift fix): upstream's mlir::affine::sinkSequentialLoops
// mis-indexes dependence components whenever the band is nested under outer
// loops.  checkMemrefAccessDependence returns one component per COMMON
// surrounding loop, outermost first — so for a band under e.g. a stencil time
// loop, component 0 belongs to the time loop, and every band loop reads its
// outer neighbour's component.  On PolyBench heat-3d/fdtd-2d the store's
// write-write self-dependence carried by the time loop (components
// [t: >=1, i: 0, j: 0, k: 0]) lands on the parallel i-loop, mislabels it
// sequential, and rotates the nest so the unit-stride dimension goes
// outermost — 6-10x slowdowns, even when no fusion ends up being performed
// (this runs unconditionally on every fusion-destination candidate).
//
// The local replacement below gathers dependences only at the band's own
// depths (d in [shift+1, shift+bandDepth]) and indexes components at
// shift+j.  Dependences carried by the enclosing loops are invariant to any
// intra-band permutation, so excluding them is exact, not a relaxation.
// Behaviour for top-level bands (shift == 0) is identical to upstream.

// Checks each dependence component against the permutation to see if the
// desired loop interchange would violate dependences by making the
// dependence component lexicographically negative.  Copy of upstream's
// static checkLoopInterchangeDependences with the band's nesting depth
// (`shift`) applied to the component index.
static bool
drCheckBandInterchangeDependences(
    const std::vector<SmallVector<DependenceComponent, 2>> &depCompsVec,
    ArrayRef<AffineForOp> loops, ArrayRef<unsigned> loopPermMap,
    unsigned shift) {
  unsigned maxLoopDepth = loops.size();
  SmallVector<unsigned, 4> loopPermMapInv;
  loopPermMapInv.resize(maxLoopDepth);
  for (unsigned i = 0; i < maxLoopDepth; ++i)
    loopPermMapInv[loopPermMap[i]] = i;

  for (const auto &depComps : depCompsVec) {
    assert(depComps.size() >= shift + maxLoopDepth);
    // Check if the first non-zero dependence component is positive.
    // This iterates through loops in the desired order.
    for (unsigned j = 0; j < maxLoopDepth; ++j) {
      unsigned permIndex = loopPermMapInv[j];
      assert(depComps[shift + permIndex].lb);
      int64_t depCompLb = *depComps[shift + permIndex].lb;
      if (depCompLb > 0)
        break;
      if (depCompLb < 0)
        return false;
    }
  }
  return true;
}

// Gathers dependence components for all load/store pairs in the perfect band
// rooted at `bandRoot`, at the band's own loop depths only: d in
// [shift+1, shift+bandDepth].  Dependences carried by the `shift` enclosing
// loops are deliberately excluded — they are preserved by construction under
// any permutation that keeps the band inside those loops.
static void drGetBandDependenceComponents(
    AffineForOp bandRoot, unsigned shift, unsigned bandDepth,
    std::vector<SmallVector<DependenceComponent, 2>> *depCompsVec) {
  SmallVector<Operation *, 8> loadAndStoreOps;
  bandRoot->walk([&](Operation *op) {
    if (isa<AffineReadOpInterface, AffineWriteOpInterface>(op))
      loadAndStoreOps.push_back(op);
  });

  unsigned numOps = loadAndStoreOps.size();
  for (unsigned d = shift + 1; d <= shift + bandDepth; ++d) {
    for (unsigned i = 0; i < numOps; ++i) {
      MemRefAccess srcAccess(loadAndStoreOps[i]);
      for (unsigned j = 0; j < numOps; ++j) {
        MemRefAccess dstAccess(loadAndStoreOps[j]);
        SmallVector<DependenceComponent, 2> depComps;
        DependenceResult result = checkMemrefAccessDependence(
            srcAccess, dstAccess, d, /*dependenceConstraints=*/nullptr,
            &depComps);
        if (hasDependence(result))
          depCompsVec->push_back(depComps);
      }
    }
  }
}

// Sinks all sequential loops to the innermost levels (while preserving
// relative order among them) and moves all parallel loops to the
// outermost (while again preserving relative order among them).
// This can increase the loop depth at which we can fuse a slice, since we are
// pushing loop carried dependence to a greater depth in the loop nest.
// DR-DIVERGE: frame-shift-corrected replacement for
// mlir::affine::sinkSequentialLoops (see comment block above).
static AffineForOp drSinkSequentialLoops(AffineForOp forOp) {
  SmallVector<AffineForOp, 4> loops;
  getPerfectlyNestedLoops(loops, forOp);
  if (loops.size() < 2)
    return forOp;

  // Number of loops enclosing the band: the dependence components of every
  // access pair inside the band start with one entry per enclosing loop.
  unsigned shift = getNestingDepth(loops[0]);
  unsigned maxLoopDepth = loops.size();
  std::vector<SmallVector<DependenceComponent, 2>> depCompsVec;
  drGetBandDependenceComponents(loops[0], shift, maxLoopDepth, &depCompsVec);

  // Mark loops as either parallel or sequential.
  SmallVector<bool, 8> isParallelLoop(maxLoopDepth, true);
  for (auto &depComps : depCompsVec) {
    assert(depComps.size() >= shift + maxLoopDepth);
    for (unsigned j = 0; j < maxLoopDepth; ++j) {
      DependenceComponent &depComp = depComps[shift + j];
      assert(depComp.lb.has_value() && depComp.ub.has_value());
      if (*depComp.lb != 0 || *depComp.ub != 0)
        isParallelLoop[j] = false;
    }
  }

  unsigned numParallelLoops = llvm::count(isParallelLoop, true);

  // Compute permutation of loops that sinks sequential loops (and thus raises
  // parallel loops) while preserving relative order.
  SmallVector<unsigned, 4> loopPermMap(maxLoopDepth);
  unsigned nextSequentialLoop = numParallelLoops;
  unsigned nextParallelLoop = 0;
  for (unsigned i = 0; i < maxLoopDepth; ++i) {
    if (isParallelLoop[i]) {
      loopPermMap[i] = nextParallelLoop++;
    } else {
      loopPermMap[i] = nextSequentialLoop++;
    }
  }

  // Check if permutation 'loopPermMap' would violate dependences.
  if (!drCheckBandInterchangeDependences(depCompsVec, loops, loopPermMap,
                                         shift))
    return forOp;
  // Perform loop interchange according to permutation 'loopPermMap'.
  unsigned loopNestRootIndex = permuteLoops(loops, loopPermMap);
  return loops[loopNestRootIndex];
}

static void sinkSequentialLoops(MemRefDependenceGraph::Node *node) {
  assert(isa<AffineForOp>(node->op));
  // DR-DIVERGE: call the frame-shift-corrected local version, not
  // mlir::affine::sinkSequentialLoops.
  AffineForOp newRootForOp = drSinkSequentialLoops(cast<AffineForOp>(node->op));
  node->op = newRootForOp;
}

/// Get the operation that should act as a dominance filter while replacing
/// memref uses with a private memref for which `producerStores` and
/// `sliceInsertionBlock` are provided. This effectively determines in what
/// part of the IR we should be performing the replacement.
static Operation *
getDominanceFilterForPrivateMemRefRepl(Block *sliceInsertionBlock,
                                       ArrayRef<Operation *> producerStores) {
  assert(!producerStores.empty() && "expected producer store");

  // We first find the common block that contains the producer stores and
  // the slice computation. The first ancestor among the ancestors of the
  // producer stores in that common block is the dominance filter to use for
  // replacement.
  Block *commonBlock = nullptr;
  // Find the common block of all relevant operations.
  for (Operation *store : producerStores) {
    Operation *otherOp =
        !commonBlock ? &*sliceInsertionBlock->begin() : &*commonBlock->begin();
    commonBlock = findInnermostCommonBlockInScope(store, otherOp);
  }
  assert(commonBlock &&
         "common block of producer stores and slice should exist");

  // Find the first ancestor among the ancestors of `producerStores` in
  // `commonBlock`.
  Operation *firstAncestor = nullptr;
  for (Operation *store : producerStores) {
    Operation *ancestor = commonBlock->findAncestorOpInBlock(*store);
    assert(ancestor && "producer store should be contained in common block");
    firstAncestor = !firstAncestor || ancestor->isBeforeInBlock(firstAncestor)
                        ? ancestor
                        : firstAncestor;
  }
  return firstAncestor;
}

/// Returns the amount of additional (redundant) computation that will be done
/// as a fraction of the total computation if `srcForOp` is fused into
/// `dstForOp` at depth `depth`. The method returns the compute cost of the
/// slice and the fused nest's compute cost in the trailing output arguments.
static std::optional<double> getAdditionalComputeFraction(
    AffineForOp srcForOp, AffineForOp dstForOp, unsigned depth,
    ArrayRef<ComputationSliceState> depthSliceUnions, int64_t &sliceCost,
    int64_t &fusedLoopNestComputeCost) {
  LDBG() << "Determining additional compute fraction...";
  // Compute cost of sliced and unsliced src loop nest.
  // Walk src loop nest and collect stats.
  LoopNestStats srcLoopNestStats;
  if (!getLoopNestStats(srcForOp, &srcLoopNestStats)) {
    LDBG() << "Failed to get source loop nest stats.";
    return std::nullopt;
  }

  // Compute cost of dst loop nest.
  LoopNestStats dstLoopNestStats;
  if (!getLoopNestStats(dstForOp, &dstLoopNestStats)) {
    LDBG() << "Failed to get destination loop nest stats.";
    return std::nullopt;
  }

  // Compute op instance count for the src loop nest without iteration slicing.
  uint64_t srcLoopNestCost = getComputeCost(srcForOp, srcLoopNestStats);

  // Compute op cost for the dst loop nest.
  uint64_t dstLoopNestCost = getComputeCost(dstForOp, dstLoopNestStats);

  const ComputationSliceState &slice = depthSliceUnions[depth - 1];
  // Skip slice union if it wasn't computed for this depth.
  if (slice.isEmpty()) {
    LDBG() << "Slice wasn't computed.";
    return std::nullopt;
  }

  if (!getFusionComputeCost(srcForOp, srcLoopNestStats, dstForOp,
                            dstLoopNestStats, slice,
                            &fusedLoopNestComputeCost)) {
    LDBG() << "Unable to compute fusion compute cost";
    return std::nullopt;
  }

  double additionalComputeFraction =
      fusedLoopNestComputeCost /
          (static_cast<double>(srcLoopNestCost) + dstLoopNestCost) -
      1;

  return additionalComputeFraction;
}

// Creates and returns a private (single-user) memref for fused loop rooted at
// 'forOp', with (potentially reduced) memref size based on the memref region
// written to by `storeOps` at depth 'dstLoopDepth'. 'sliceInsertionBlock'
// specifies the block in which the slice was/will be inserted. The method
// expects that all stores ops to the memref have the same access function.
// Returns nullptr if the creation failed.
static Value createPrivateMemRef(AffineForOp forOp,
                                 ArrayRef<Operation *> storeOps,
                                 unsigned dstLoopDepth,
                                 std::optional<unsigned> fastMemorySpace,
                                 Block *sliceInsertionBlock,
                                 uint64_t localBufSizeThreshold) {
  assert(!storeOps.empty() && "no source stores supplied");

  // Check if all stores have the same access function; we only support this
  // case.
  // TODO: Use union of memref write regions to compute private memref footprint
  // for store ops with different access functions.
  if (storeOps.size() > 1 &&
      !std::equal(std::next(storeOps.begin()), storeOps.end(), storeOps.begin(),
                  [](Operation *a, Operation *b) {
                    MemRefAccess aM(cast<AffineWriteOpInterface>(a));
                    MemRefAccess bM(cast<AffineWriteOpInterface>(b));
                    return aM == bM;
                  })) {
    LDBG() << "Private memref creation unsupported for multiple producer "
           << "stores with different access functions.";
    return nullptr;
  }

  Operation *srcStoreOp = storeOps[0];

  // Create builder to insert alloc op just before 'forOp'.
  OpBuilder b(forOp);
  // Builder to create constants at the top level.
  OpBuilder top(forOp->getParentRegion());
  // Create new memref type based on slice bounds.
  auto oldMemRef = cast<AffineWriteOpInterface>(srcStoreOp).getMemRef();
  auto oldMemRefType = cast<MemRefType>(oldMemRef.getType());
  unsigned rank = oldMemRefType.getRank();

  // Compute MemRefRegion for 'srcStoreOpInst' at depth 'dstLoopDepth'.
  MemRefRegion region(srcStoreOp->getLoc());
  bool validRegion = succeeded(
      region.compute(srcStoreOp, dstLoopDepth, /*sliceState=*/nullptr,
                     /*addMemRefDimBounds=*/true, /*dropLocalVars=*/false));

  (void)validRegion;
  assert(validRegion && "unexpected memref region failure");
  SmallVector<int64_t, 4> newShape;
  SmallVector<AffineMap, 4> lbs;
  lbs.reserve(rank);
  // Query 'region' for 'newShape' and lower bounds of MemRefRegion accessed
  // by 'srcStoreOpInst' at depth 'dstLoopDepth'.
  std::optional<int64_t> numElements =
      region.getConstantBoundingSizeAndShape(&newShape, &lbs);
  assert(numElements && "non-constant number of elts in local buffer");

  const FlatAffineValueConstraints *cst = region.getConstraints();
  // 'outerIVs' holds the values that this memory region is symbolic/parametric
  // on; this would correspond to loop IVs surrounding the level at which the
  // slice is being materialized.
  SmallVector<Value, 8> outerIVs;
  cst->getValues(rank, cst->getNumDimAndSymbolVars(), &outerIVs);

  // Build 'rank' AffineExprs from MemRefRegion 'lbs'
  SmallVector<AffineExpr, 4> offsets;
  offsets.reserve(rank);

  // Outer IVs are considered symbols during memref region computation. Replace
  // them uniformly with dims so that valid IR is guaranteed.
  SmallVector<AffineExpr> replacements;
  for (unsigned j = 0, e = lbs[0].getNumSymbols(); j < e; ++j)
    replacements.push_back(mlir::getAffineDimExpr(j, forOp.getContext()));
  for (unsigned d = 0; d < rank; ++d) {
    assert(lbs[d].getNumResults() == 1 &&
           "invalid private memref bound calculation");
    offsets.push_back(lbs[d].getResult(0).replaceSymbols(replacements));
  }

  // Create 'newMemRefType' using 'newShape' from MemRefRegion accessed
  // by 'srcStoreOpInst'.
  auto eltSize = getMemRefIntOrFloatEltSizeInBytes(oldMemRefType);
  assert(eltSize && "memrefs with size elt types expected");
  uint64_t bufSize = *eltSize * *numElements;
  Attribute newMemSpace;
  if (bufSize <= localBufSizeThreshold && fastMemorySpace.has_value()) {
    newMemSpace = b.getI64IntegerAttr(*fastMemorySpace);
  } else {
    newMemSpace = oldMemRefType.getMemorySpace();
  }
  auto newMemRefType = MemRefType::get(newShape, oldMemRefType.getElementType(),
                                       /*map=*/AffineMap(), newMemSpace);

  // Create new private memref for fused loop 'forOp'. 'newShape' is always
  // a constant shape.
  // TODO: Create/move alloc ops for private memrefs closer to their
  // consumer loop nests to reduce their live range. Currently they are added
  // at the beginning of the block, because loop nests can be reordered
  // during the fusion pass.
  Value newMemRef = memref::AllocOp::create(top, forOp.getLoc(), newMemRefType);

  // Build an AffineMap to remap access functions based on lower bound offsets.
  SmallVector<AffineExpr, 4> remapExprs;
  remapExprs.reserve(rank);
  for (unsigned i = 0; i < rank; i++) {
    auto dimExpr = b.getAffineDimExpr(outerIVs.size() + i);

    auto remapExpr =
        simplifyAffineExpr(dimExpr - offsets[i], outerIVs.size() + rank, 0);
    remapExprs.push_back(remapExpr);
  }

  auto indexRemap =
      AffineMap::get(outerIVs.size() + rank, 0, remapExprs, forOp.getContext());

  // Replace all users of 'oldMemRef' with 'newMemRef'.
  Operation *domFilter =
      getDominanceFilterForPrivateMemRefRepl(sliceInsertionBlock, storeOps);
  auto userFilterFn = [&](Operation *user) {
    auto domInfo = std::make_unique<DominanceInfo>(
        domFilter->getParentOfType<FunctionOpInterface>());
    return domInfo->dominates(domFilter, user);
  };
  LogicalResult res = replaceAllMemRefUsesWith(
      oldMemRef, newMemRef, /*extraIndices=*/{}, indexRemap,
      /*extraOperands=*/outerIVs,
      /*symbolOperands=*/{}, userFilterFn);
  assert(succeeded(res) &&
         "replaceAllMemrefUsesWith should always succeed here");
  (void)res;
  LDBG() << "Created private memref of type: " << newMemRefType;
  return newMemRef;
}

// Checks the profitability of fusing a backwards slice of the loop nest
// `srcForOp` into the loop nest surrounding 'dstLoadOpInsts'. The argument
// 'srcStoreOpInst' is used to calculate the storage reduction on the memref
// being produced and consumed, which is an input to the cost model. For
// producer-consumer fusion, 'srcStoreOpInst' will be the same as 'srcOpInst',
// as we are slicing w.r.t to that producer. For input-reuse fusion, 'srcOpInst'
// will be the src loop nest LoadOp which reads from the same memref as dst loop
// nest load ops, and 'srcStoreOpInst' will be the unique store op in the src
// node, which will be used to check that the write region is the same after
// input-reuse fusion. Computation slices are provided in 'depthSliceUnions' for
// each legal fusion depth. The maximal depth at which fusion is legal is
// provided in 'maxLegalFusionDepth'. Returns true if it is profitable to fuse
// the candidate loop nests. Returns false otherwise. `dstLoopDepth` is set to
// the most profitable depth at which to materialize the source loop nest slice.
// The profitability model executes the following steps:
// *) Computes the backward computation slice at 'srcOpInst'. This
//    computation slice of the loop nest surrounding 'srcOpInst' is
//    represented by modified src loop bounds in 'sliceState', which are
//    functions of loop IVs in the loop nest surrounding 'srcOpInst'.
// *) Computes the cost of unfused src/dst loop nests (currently the cost of a
//    loop nest is the total number of dynamic operation instances in the loop
//    nest).
// *) Computes the cost of fusing a slice of the src loop nest into the dst
//    loop nest at various values of dst loop depth, attempting to fuse
//    the largest computation slice at the maximal dst loop depth (closest to
//    the load) to minimize reuse distance and potentially enable subsequent
//    load/store forwarding.
//    NOTE: 'dstLoopDepth' refers to the loop depth within the destination loop
//    nest, at which the src computation slice is inserted/fused.
//    NOTE: We attempt to maximize the dst loop depth, but there are cases
//    where a particular setting for 'dstLoopNest' might fuse an unsliced
//    loop (within the src computation slice) at a depth which results in
//    excessive recomputation (see unit tests for examples).
// *) Compares the total cost of the unfused loop nests to the min cost fused
//    loop nest computed in the previous step, and returns true if the latter
//    is lower.
// TODO: Extend profitability analysis to support scenarios with multiple
// stores.
static bool isFusionProfitable(AffineForOp srcForOp,
                               ArrayRef<Operation *> producerStores,
                               AffineForOp dstForOp,
                               ArrayRef<ComputationSliceState> depthSliceUnions,
                               unsigned maxLegalFusionDepth,
                               unsigned *dstLoopDepth,
                               double computeToleranceThreshold) {
  LDBG() << "Checking whether fusion is profitable between source nest:";
  LDBG() << ' ' << srcForOp << " and destination nest:";
  LDBG() << dstForOp;

  if (maxLegalFusionDepth == 0) {
    LDBG() << "Can't fuse: maxLegalFusionDepth is 0";
    return false;
  }

  // Compute cost of sliced and unsliced src loop nest.

  // Walk src loop nest and collect stats.
  LoopNestStats srcLoopNestStats;
  if (!getLoopNestStats(srcForOp, &srcLoopNestStats))
    return false;

  // Compute cost of dst loop nest.
  LoopNestStats dstLoopNestStats;
  if (!getLoopNestStats(dstForOp, &dstLoopNestStats))
    return false;

  // We limit profitability analysis to only scenarios with
  // a single producer store for now. Note that some multi-store
  // producer scenarios will still go through profitability analysis
  // if only one of the stores is involved in the producer-consumer
  // relationship of the candidate loops.
  // TODO: Suppport multiple producer stores in profitability
  // analysis.
  if (producerStores.size() > 1) {
    LDBG() << "Limited profitability analysis. Not "
           << "supported for multiple producer store case.";
    int64_t sliceCost;
    int64_t fusedLoopNestComputeCost;
    // We will still fuse if fusion obeys the specified compute
    // tolerance at the max legal depth.
    auto fraction = getAdditionalComputeFraction(
        srcForOp, dstForOp, maxLegalFusionDepth, depthSliceUnions, sliceCost,
        fusedLoopNestComputeCost);
    if (!fraction || fraction > computeToleranceThreshold) {
      LDBG() << "Additional computation exceeds "
             << "compute tolerance. Not fusing.";
      return false;
    }
    LDBG() << "Considering fusion profitable at max legal depth.";
    return true;
  }

  Operation *srcStoreOp = producerStores.front();

  // Search for min cost value for 'dstLoopDepth'. At each value of
  // 'dstLoopDepth' from 'maxLegalLoopDepth' to '1', compute computation slice
  // bounds between 'srcOpInst' and each op in 'dstOpinsts' (taking the union
  // of these bounds). Next the union slice bounds are used to calculate
  // the cost of the slice and the cost of the slice inserted into the dst
  // loop nest at 'dstLoopDepth'.
  uint64_t minFusedLoopNestComputeCost = std::numeric_limits<uint64_t>::max();
  double maxStorageReduction = 0.0;
  std::optional<uint64_t> sliceMemEstimate;

  // The best loop depth at which to materialize the slice.
  std::optional<unsigned> bestDstLoopDepth;

  // Compute src loop nest write region size.
  MemRefRegion srcWriteRegion(srcStoreOp->getLoc());
  if (failed(srcWriteRegion.compute(srcStoreOp, /*loopDepth=*/0))) {
    LDBG() << "Unable to compute MemRefRegion for source operation";
    return false;
  }

  std::optional<int64_t> maybeSrcWriteRegionSizeBytes =
      srcWriteRegion.getRegionSize();
  if (!maybeSrcWriteRegionSizeBytes.has_value())
    return false;
  int64_t srcWriteRegionSizeBytes = *maybeSrcWriteRegionSizeBytes;

  // Compute op instance count for the src loop nest without iteration slicing.
  uint64_t srcLoopNestCost = getComputeCost(srcForOp, srcLoopNestStats);

  // Compute op instance count for the destination loop nest.
  uint64_t dstLoopNestCost = getComputeCost(dstForOp, dstLoopNestStats);

  // DR-DIVERGE: pre-loop unified-path state.  Carry the running min total
  // across loop iterations as a local (was `static` previously, which leaked
  // across isFusionProfitable invocations).
  uint64_t bestUnifiedTotal = std::numeric_limits<uint64_t>::max();
  // DR-DIVERGE: compute unfused baseline once for the unfused vs fused
  // comparison done after the per-depth search.
  uint64_t unfusedTotal = 0;
  if (dr_fusion::gActive && dr_fusion::gActive->useUnifiedCostModel) {
    auto srcMem = getMemoryFootprintBytes(srcForOp);
    auto dstMem = getMemoryFootprintBytes(dstForOp);
    // DR-DIVERGE: convert raw byte footprints to cycle estimates via the
    // cache hierarchy so all aspects (mem/reg/alu) of `combineCosts` are
    // in the same units.
    uint64_t unfusedMem =
        dr_fusion::bytesToMemCycles(srcMem.value_or(0)) +
        dr_fusion::bytesToMemCycles(dstMem.value_or(0));
    uint64_t unfusedAlu = srcLoopNestCost + dstLoopNestCost;
    uint64_t unfusedReg = 0;
    if (dr_fusion::gActive->archHandler) {
      drcompiler::PressureQuery q;
      q.params = dr_fusion::gActive->regParams;
      q.strategy = dr_fusion::gActive->spillStrategy;
      q.tripCount = 1;
      auto rs =
          drcompiler::RegisterPressureAnalysis::analyzeRegionStatic(
              srcForOp.getRegion(), *dr_fusion::gActive->archHandler,
              dr_fusion::gActive->archParams, q);
      auto rd =
          drcompiler::RegisterPressureAnalysis::analyzeRegionStatic(
              dstForOp.getRegion(), *dr_fusion::gActive->archHandler,
              dr_fusion::gActive->archParams, q);
      unfusedReg = rs.totalSpillCycles + rd.totalSpillCycles;
    }
    unfusedTotal = dr_fusion::combine(static_cast<unsigned>(unfusedMem),
                                       static_cast<unsigned>(unfusedReg),
                                       static_cast<unsigned>(unfusedAlu));
  }

  // Evaluate all depth choices for materializing the slice in the destination
  // loop nest.
  for (unsigned i = maxLegalFusionDepth; i >= 1; --i) {
    const ComputationSliceState &slice = depthSliceUnions[i - 1];
    // Skip slice union if it wasn't computed for this depth.
    if (slice.isEmpty())
      continue;

    // Compute cost of the slice separately, i.e, the compute cost of the slice
    // if all outer trip counts are one.
    int64_t sliceCost;

    int64_t fusedLoopNestComputeCost;

    auto mayAdditionalComputeFraction =
        getAdditionalComputeFraction(srcForOp, dstForOp, i, depthSliceUnions,
                                     sliceCost, fusedLoopNestComputeCost);
    if (!mayAdditionalComputeFraction) {
      LDBG() << "Can't determine additional compute fraction.";
      continue;
    }
    double additionalComputeFraction = *mayAdditionalComputeFraction;

    // Determine what the slice write MemRefRegion would be, if the src loop
    // nest slice 'slice' were to be inserted into the dst loop nest at loop
    // depth 'i'.
    MemRefRegion sliceWriteRegion(srcStoreOp->getLoc());
    if (failed(sliceWriteRegion.compute(srcStoreOp, /*loopDepth=*/0, &slice))) {
      LDBG() << "Failed to compute slice write region at loopDepth: " << i;
      continue;
    }

    std::optional<int64_t> maybeSliceWriteRegionSizeBytes =
        sliceWriteRegion.getRegionSize();
    if (!maybeSliceWriteRegionSizeBytes.has_value() ||
        *maybeSliceWriteRegionSizeBytes == 0) {
      LDBG() << "Failed to get slice write region size at loopDepth: " << i;
      continue;
    }
    int64_t sliceWriteRegionSizeBytes = *maybeSliceWriteRegionSizeBytes;

    double storageReduction = static_cast<double>(srcWriteRegionSizeBytes) /
                              static_cast<double>(sliceWriteRegionSizeBytes);

    LLVM_DEBUG({
      std::stringstream msg;
      msg << "  evaluating fusion profitability at depth : " << i << "\n"
          << std::fixed << std::setprecision(2)
          << "   additional compute fraction: "
          << 100.0 * additionalComputeFraction << "%\n"
          << "   storage reduction factor: " << storageReduction << "x\n"
          << "   fused nest cost: " << fusedLoopNestComputeCost << "\n"
          << "   src write region size: " << srcWriteRegionSizeBytes << "\n"
          << "   slice write region size: " << sliceWriteRegionSizeBytes;
      LDBG() << msg.str();
    });

    // DR-DIVERGE: replace upstream's placeholder cost model with the unified
    // memory+register+ALU combiner when the unified path is active.  Among
    // legal depths, pick the one that minimises arch.combineCosts(...).
    if (dr_fusion::gActive && dr_fusion::gActive->useUnifiedCostModel) {
      // bestUnifiedTotal is now a function-scope local (see above) so the
      // running minimum is correctly per-isFusionProfitable-call.

      // Aspect costs:
      //   memCycles ~ slice memory footprint, converted to cycles via
      //               the cache hierarchy (`estimateLoadLatency`).
      //   aluCycles ~ fused compute cost
      //   regCycles ~ RPA estimate over the destination body augmented with
      //               the source slice's ops (proxy for hypothetical fusion).
      uint64_t memCycles =
          dr_fusion::bytesToMemCycles(sliceWriteRegionSizeBytes);
      uint64_t aluCycles = static_cast<uint64_t>(fusedLoopNestComputeCost);
      uint64_t regCycles = dr_fusion::estimateRegCyclesForFusion(
          srcForOp, dstForOp, srcStoreOp);
      // Compute-tolerance still applies as a hard guard so we don't fuse
      // unboundedly redundant work even if mem+reg drop.
      if (additionalComputeFraction > computeToleranceThreshold)
        continue;
      unsigned total = dr_fusion::combine(static_cast<unsigned>(memCycles),
                                          static_cast<unsigned>(regCycles),
                                          static_cast<unsigned>(aluCycles));
      if (total < bestUnifiedTotal) {
        bestUnifiedTotal = total;
        bestDstLoopDepth = i;
        maxStorageReduction =
            static_cast<double>(srcWriteRegionSizeBytes) /
            static_cast<double>(sliceWriteRegionSizeBytes);
        minFusedLoopNestComputeCost = fusedLoopNestComputeCost;
        sliceMemEstimate = sliceWriteRegionSizeBytes;
      }
    } else {
      // Upstream's placeholder: maximise storage reduction subject to
      // computeToleranceThreshold.
      if ((storageReduction > maxStorageReduction) &&
          (additionalComputeFraction <= computeToleranceThreshold)) {
        maxStorageReduction = storageReduction;
        bestDstLoopDepth = i;
        minFusedLoopNestComputeCost = fusedLoopNestComputeCost;
        sliceMemEstimate = sliceWriteRegionSizeBytes;
      }
    }
  }

  // A simple cost model: fuse if it reduces the memory footprint.

  // DR-DIVERGE: under the unified path, reject fusion when the best fused
  // total is *not* better than running the two nests separately.  This is
  // the missing piece from plan §7.3 — upstream's placeholder never
  // compares against an unfused baseline.
  if (dr_fusion::gActive && dr_fusion::gActive->useUnifiedCostModel &&
      bestDstLoopDepth && bestUnifiedTotal >= unfusedTotal) {
    LDBG() << "Unified cost model rejects fusion: fused total "
           << bestUnifiedTotal << " >= unfused total " << unfusedTotal;
    if (dr_fusion::gActive->emitRationale) {
      std::string buf;
      llvm::raw_string_ostream os(buf);
      os << "fusion-rationale: REJECT fused_total=" << bestUnifiedTotal
         << " >= unfused_total=" << unfusedTotal;
      srcForOp->emitRemark(buf);
    }
    return false;
  }
  // DR-DIVERGE (shared-traffic gate): the only cost fusion genuinely
  // removes is memory traffic on data BOTH nests touch (a producer's stores
  // forwarded to consumer loads, or two nests sweeping the same array).
  // The cycle totals above cannot see what fusion breaks in the backend
  // (interleaved bodies that no longer vectorize): measured on atax at
  // EXTRALARGE, a fusion the totals scored 23% better ran 5x slower.  So
  // require the overlap to be a substantial fraction of the nests' combined
  // footprint before fusing; with nothing real to save, keep them apart.
  if (dr_fusion::gActive && dr_fusion::gActive->useUnifiedCostModel &&
      bestDstLoopDepth) {
    std::optional<int64_t> shared =
        dr_fusion::sharedTrafficBytes(srcForOp, dstForOp);
    auto srcFp = getMemoryFootprintBytes(srcForOp);
    auto dstFp = getMemoryFootprintBytes(dstForOp);
    if (shared && srcFp && dstFp && *srcFp + *dstFp > 0) {
      double fraction = static_cast<double>(*shared) /
                        static_cast<double>(*srcFp + *dstFp);
      if (fraction < 0.05) {
        LDBG() << "Unified cost model rejects fusion: shared traffic "
               << *shared << " bytes is only " << (100.0 * fraction)
               << "% of combined footprint";
        if (dr_fusion::gActive->emitRationale) {
          std::string buf;
          llvm::raw_string_ostream os(buf);
          os << "fusion-rationale: REJECT reason=no-shared-traffic shared="
             << *shared << " combined=" << (*srcFp + *dstFp);
          srcForOp->emitRemark(buf);
        }
        return false;
      }
    }
  }
  if (dr_fusion::gActive && dr_fusion::gActive->useUnifiedCostModel &&
      dr_fusion::gActive->emitRationale && bestDstLoopDepth) {
    std::string buf;
    llvm::raw_string_ostream os(buf);
    os << "fusion-rationale: FUSE depth=" << *bestDstLoopDepth
       << " fused_total=" << bestUnifiedTotal
       << " unfused_total=" << unfusedTotal;
    srcForOp->emitRemark(buf);
  }

  if (!bestDstLoopDepth) {
    LDBG() << "All fusion choices involve more than the threshold amount of "
           << "redundant computation; NOT fusing.";
    return false;
  }

  if (!bestDstLoopDepth) {
    LDBG() << "no fusion depth could be evaluated.";
    return false;
  }

  // Set dstLoopDepth based on best values from search.
  *dstLoopDepth = *bestDstLoopDepth;

  LDBG() << " LoopFusion fusion stats:";
  LDBG() << "  best loop depth: " << bestDstLoopDepth;
  LDBG() << "  src loop nest compute cost: " << srcLoopNestCost;
  LDBG() << "  dst loop nest compute cost: " << dstLoopNestCost;
  LDBG() << "  fused loop nest compute cost: " << minFusedLoopNestComputeCost;

  auto dstMemSize = getMemoryFootprintBytes(dstForOp);
  auto srcMemSize = getMemoryFootprintBytes(srcForOp);

  std::optional<double> storageReduction;

  if (!dstMemSize || !srcMemSize) {
    LDBG() << "  fusion memory benefit cannot be evaluated; NOT fusing.";
    return false;
  }

  auto srcMemSizeVal = *srcMemSize;
  auto dstMemSizeVal = *dstMemSize;

  assert(sliceMemEstimate && "expected value");
  auto fusedMem = dstMemSizeVal + *sliceMemEstimate;

  LDBG() << "   src mem: " << srcMemSizeVal;
  LDBG() << "   dst mem: " << dstMemSizeVal;
  LDBG() << "   fused mem: " << fusedMem;
  LDBG() << "   slice mem: " << sliceMemEstimate;

  if (static_cast<long>(fusedMem) > srcMemSizeVal + dstMemSizeVal) {
    LDBG() << "Fusion is not profitable; NOT fusing.";
    return false;
  }
  storageReduction =
      100.0 *
      (1.0 - fusedMem / (static_cast<double>(srcMemSizeVal) + dstMemSizeVal));

  double additionalComputeFraction =
      100.0 * (minFusedLoopNestComputeCost /
                   (static_cast<double>(srcLoopNestCost) + dstLoopNestCost) -
               1);
  (void)additionalComputeFraction;
  LLVM_DEBUG({
    std::stringstream msg;
    msg << " fusion is most profitable at depth " << *dstLoopDepth << " with "
        << std::setprecision(2) << additionalComputeFraction
        << "% redundant computation and a ";
    msg << (storageReduction ? std::to_string(*storageReduction) : "<unknown>");
    msg << "% storage reduction.";
    LDBG() << msg.str();
  });

  return true;
}

namespace {

// GreedyFusion greedily fuses loop nests which have a producer/consumer or
// input-reuse relationship on a memref, with the goal of improving locality.
//
// The steps of the producer-consumer fusion algorithm are as follows:
//
// *) A worklist is initialized with node ids from the dependence graph.
// *) For each node id in the worklist:
//   *) Pop an AffineForOp of the worklist. This 'dstAffineForOp' will be a
//      candidate destination AffineForOp into which fusion will be attempted.
//   *) Add each LoadOp currently in 'dstAffineForOp' into list 'dstLoadOps'.
//   *) For each LoadOp in 'dstLoadOps' do:
//      *) Look up dependent loop nests which have a single store op to the same
//         memref.
//      *) Check if dependences would be violated by the fusion.
//      *) Get a computation slice of 'srcLoopNest', which adjusts its loop
//         bounds to be functions of 'dstLoopNest' IVs and symbols.
//      *) Fuse the 'srcLoopNest' computation slice into the 'dstLoopNest',
//         at a loop depth determined by the cost model in 'isFusionProfitable'.
//      *) Add the newly fused load/store operations to the state,
//         and also add newly fused load ops to 'dstLoopOps' to be considered
//         as fusion dst load ops in another iteration.
//      *) Remove old src loop nest and its associated state.
//
// The steps of the input-reuse fusion algorithm are as follows:
//
// *) Initialize 'worklist' with node ids from the dependence graph.
// *) For each 'dstNode' in the worklist:
//   *) Find a candidate sibling node 'sibNode' to fuse with 'dstNode' which
//      loads from the same memref, but which has no dependence paths to/from.
//   *) Get a computation slice of 'sibLoopNest', which adjusts its loop
//      bounds to be functions of 'dstLoopNest' IVs and symbols.
//   *) Fuse the 'sibLoopNest' computation slice into the 'dstLoopNest',
//      at a loop depth determined by the cost model in 'isFusionProfitable'.
//      This function also checks that the memref write region of 'sibLoopNest',
//      is preserved in the fused loop nest.
//   *) Update graph state to reflect the fusion of 'sibNode' into 'dstNode'.
//
// Given a graph where top-level operations are vertices in the set 'V' and
// edges in the set 'E' are dependences between vertices, this algorithm
// takes O(V) time for initialization, and has runtime O(V + E).
//
// This greedy algorithm is not 'maximal' due to the current restriction of
// fusing along single producer consumer edges, but there is a TODO: to fix
// this.
//
// TODO: Experiment with other fusion policies.
struct GreedyFusion {
public:
  // The data dependence graph to traverse during fusion.
  MemRefDependenceGraph *mdg;
  // Worklist of graph nodes visited during the fusion pass.
  SmallVector<unsigned, 8> worklist;
  // Parameter for local buffer size threshold.
  unsigned localBufSizeThreshold;
  // Parameter for fast memory space.
  std::optional<unsigned> fastMemorySpace;
  // If true, ignore any additional (redundant) computation tolerance threshold
  // that would have prevented fusion.
  bool maximalFusion;
  // The amount of additional computation that is tolerated while fusing
  // pair-wise as a fraction of the total computation.
  double computeToleranceThreshold;

  using Node = MemRefDependenceGraph::Node;

  GreedyFusion(MemRefDependenceGraph *mdg, unsigned localBufSizeThreshold,
               std::optional<unsigned> fastMemorySpace, bool maximalFusion,
               double computeToleranceThreshold)
      : mdg(mdg), localBufSizeThreshold(localBufSizeThreshold),
        fastMemorySpace(fastMemorySpace), maximalFusion(maximalFusion),
        computeToleranceThreshold(computeToleranceThreshold) {}

  /// Initializes 'worklist' with nodes from 'mdg'.
  void init() {
    // TODO: Add a priority queue for prioritizing nodes by different
    // metrics (e.g. arithmetic intensity/flops-to-bytes ratio).
    worklist.clear();
    for (auto &idAndNode : mdg->nodes) {
      const Node &node = idAndNode.second;
      worklist.push_back(node.id);
    }
  }
  /// Run only sibling fusion on the `mdg`.
  void runSiblingFusionOnly() {
    fuseSiblingNodes();
    eraseUnusedMemRefAllocations();
  }

  /// Run only producer/consumer fusion on the `mdg`.
  void runProducerConsumerFusionOnly() {
    fuseProducerConsumerNodes(
        /*maxSrcUserCount=*/std::numeric_limits<unsigned>::max());
    eraseUnusedMemRefAllocations();
  }

  // Run the GreedyFusion pass.
  // *) First pass through the nodes fuses single-use producer nodes into their
  //    unique consumer.
  // *) Second pass fuses sibling nodes which share no dependence edges.
  // *) Third pass fuses any remaining producer nodes into their users.
  void runGreedyFusion() {
    // TODO: Run this repeatedly until a fixed-point is reached.
    fuseProducerConsumerNodes(/*maxSrcUserCount=*/1);
    fuseSiblingNodes();
    fuseProducerConsumerNodes(
        /*maxSrcUserCount=*/std::numeric_limits<unsigned>::max());
    eraseUnusedMemRefAllocations();
  }

  /// Returns true if a private memref can be created for `memref` given
  /// the fusion scenario reflected by the other arguments.
  bool canCreatePrivateMemRef(Value memref,
                              const DenseSet<Value> &srcEscapingMemRefs,
                              unsigned producerId, unsigned consumerId,
                              bool removeSrcNode) {
    // We can't generate private memrefs if their size can't be computed.
    if (!getMemRefIntOrFloatEltSizeInBytes(cast<MemRefType>(memref.getType())))
      return false;
    const Node *consumerNode = mdg->getNode(consumerId);
    // If `memref` is an escaping one, do not create a private memref
    // for the below scenarios, since doing so will leave the escaping
    // memref unmodified as all the writes originally meant for the
    // escaping memref would be performed on the private memref:
    // 1. The source is to be removed after fusion,
    // OR
    // 2. The destination writes to `memref`.
    if (srcEscapingMemRefs.count(memref) > 0 &&
        (removeSrcNode || consumerNode->getStoreOpCount(memref) > 0))
      return false;

    // Don't create a private memref if 'srcNode' has in edges on
    // 'memref' or 'dstNode' has out edges on 'memref'.
    if (mdg->getIncomingMemRefAccesses(producerId, memref) > 0 ||
        mdg->getOutEdgeCount(consumerId, memref) > 0)
      return false;

    // If 'srcNode' will be removed but it has out edges on 'memref' to
    // nodes other than 'dstNode', we have to preserve dependences and
    // cannot create a private memref.
    if (removeSrcNode &&
        any_of(mdg->outEdges[producerId], [&](const auto &edge) {
          return edge.value == memref && edge.id != consumerId;
        }))
      return false;

    return true;
  }

  /// Perform fusions with node `dstId` as the destination of fusion, with
  /// No fusion is performed when producers with a user count greater than
  /// `maxSrcUserCount` for any of the memrefs involved.
  void performFusionsIntoDest(unsigned dstId, unsigned maxSrcUserCount) {
    LDBG() << "Evaluating dst loop " << dstId;
    // Skip if this node was removed (fused into another node).
    if (mdg->nodes.count(dstId) == 0)
      return;
    // Get 'dstNode' into which to attempt fusion.
    auto *dstNode = mdg->getNode(dstId);
    // Skip if 'dstNode' is not a loop nest.
    if (!isa<AffineForOp>(dstNode->op))
      return;
    // Skip if 'dstNode' is a loop nest returning values.
    // TODO: support loop nests that return values.
    if (dstNode->op->getNumResults() > 0)
      return;

    LDBG() << "Evaluating dst loop " << dstId;

    // Sink sequential loops in 'dstNode' (and thus raise parallel loops)
    // while preserving relative order. This can increase the maximum loop
    // depth at which we can fuse a slice of a producer loop nest into a
    // consumer loop nest.
    sinkSequentialLoops(dstNode);
    auto dstAffineForOp = cast<AffineForOp>(dstNode->op);

    // Try to fuse 'dstNode' with candidate producer loops until a fixed point
    // is reached. Fusing two loops may expose new fusion opportunities.
    bool dstNodeChanged;
    do {
      // Gather src loop candidates for 'dstNode' and visit them in "quasi"
      // reverse program order to minimize the number of iterations needed to
      // reach the fixed point. Note that this is a best effort approach since
      // 'getProducerCandidates' does not always guarantee that program order
      // in 'srcIdCandidates'.
      dstNodeChanged = false;
      SmallVector<unsigned, 16> srcIdCandidates;
      getProducerCandidates(dstId, *mdg, srcIdCandidates);

      for (unsigned srcId : llvm::reverse(srcIdCandidates)) {
        // Get 'srcNode' from which to attempt fusion into 'dstNode'.
        auto *srcNode = mdg->getNode(srcId);
        auto srcAffineForOp = cast<AffineForOp>(srcNode->op);

        LDBG() << "Trying to fuse producer loop nest " << srcId
               << " with consumer loop nest " << dstId;
        LDBG() << "Compute tolerance threshold: " << computeToleranceThreshold;
        LDBG() << "Producer loop nest:";
        LDBG() << *srcNode->op << " and consumer loop nest:";
        LDBG() << *dstNode->op;

        LDBG() << "Evaluating src loop " << srcId << " for dst loop " << dstId;

        // Skip if 'srcNode' is a loop nest returning values.
        // TODO: support loop nests that return values.
        if (isa<AffineForOp>(srcNode->op) && srcNode->op->getNumResults() > 0)
          continue;

        DenseSet<Value> producerConsumerMemrefs;
        gatherProducerConsumerMemrefs(srcId, dstId, *mdg,
                                      producerConsumerMemrefs);

        // DR-DIVERGE (phase barrier): an op with unknown side effects
        // between the two nests (a call — e.g. polybench_timer_start)
        // delimits a program phase.  Hoisting the src computation across it
        // is legal whenever the touched memrefs don't escape, but it
        // dissolves the phase structure: measured on atax at O0, fusing the
        // A-init nest (before the timer call) into the kernel (after it)
        // moved a full array initialization into the timed region — process
        // cycles identical, reported kernel time 4x.  Such reordering is
        // never the intent of loop fusion; keep nests on their own side of
        // any opaque call.
        if (dr_fusion::sideEffectingOpBetween(srcNode->op, dstNode->op)) {
          LDBG() << "Skipping fusion: opaque side-effecting op between the "
                    "nests (phase barrier)";
          continue;
        }

        // DR-DIVERGE (interleaving guard): fusing relocates the src
        // computation inside the dst loop, interleaving it with the dst
        // body across iterations.  If the dst nest STORES to any memref the
        // src nest accesses, later src-slice iterations read or write data
        // the dst body already modified — an interleaving the upstream
        // slice legality analysis demonstrably mishandles.  Observed on
        // PolyBench lu's init: the `B[r][s] += A[r][t]*A[s][t]`
        // accumulation (src, reads A) fused into the `A[r][s] = B[r][s]`
        // copy nest (dst, writes A) makes row r's accumulation read rows
        // < r of A that were already overwritten — the factorization then
        // diverges to inf.  Skip such candidates.
        {
          DenseSet<Value> srcAccessed;
          for (Operation *ld : srcNode->loads)
            srcAccessed.insert(cast<AffineReadOpInterface>(ld).getMemRef());
          for (Operation *st : srcNode->stores)
            srcAccessed.insert(cast<AffineWriteOpInterface>(st).getMemRef());
          bool dstClobbersSrcData =
              llvm::any_of(dstNode->stores, [&](Operation *st) {
                return srcAccessed.count(
                           cast<AffineWriteOpInterface>(st).getMemRef()) > 0;
              });
          if (dstClobbersSrcData) {
            LDBG() << "Skipping fusion: dst nest stores to a memref the src "
                      "nest accesses; interleaving would change the values "
                      "the relocated src computation observes";
            continue;
          }
        }

        // Skip if 'srcNode' out edge count on any memref is greater than
        // 'maxSrcUserCount'.
        if (any_of(producerConsumerMemrefs, [&](Value memref) {
              return mdg->getOutEdgeCount(srcNode->id, memref) >
                     maxSrcUserCount;
            }))
          continue;

        // Gather memrefs in 'srcNode' that are written and escape out of the
        // block (e.g., memref block arguments, returned memrefs,
        // memrefs passed to function calls, etc.).
        DenseSet<Value> srcEscapingMemRefs;
        gatherEscapingMemrefs(srcNode->id, *mdg, srcEscapingMemRefs);

        // Compute an operation list insertion point for the fused loop
        // nest which preserves dependences.
        Operation *fusedLoopInsPoint =
            mdg->getFusedLoopNestInsertionPoint(srcNode->id, dstNode->id);
        if (fusedLoopInsPoint == nullptr)
          continue;

        // It's possible this fusion is at an inner depth (i.e., there are
        // common surrounding affine loops for the source and destination for
        // ops). We need to get this number because the call to canFuseLoops
        // needs to be passed the absolute depth. The max legal depth and the
        // depths we try below are however *relative* and as such don't include
        // the common depth.
        SmallVector<AffineForOp, 4> surroundingLoops;
        getAffineForIVs(*dstAffineForOp, &surroundingLoops);
        unsigned numSurroundingLoops = surroundingLoops.size();

        // Compute the innermost common loop depth for dstNode
        // producer-consumer loads/stores.
        SmallVector<Operation *, 2> dstMemrefOps;
        for (Operation *op : dstNode->loads)
          if (producerConsumerMemrefs.count(
                  cast<AffineReadOpInterface>(op).getMemRef()) > 0)
            dstMemrefOps.push_back(op);
        for (Operation *op : dstNode->stores)
          if (producerConsumerMemrefs.count(
                  cast<AffineWriteOpInterface>(op).getMemRef()))
            dstMemrefOps.push_back(op);
        if (dstMemrefOps.empty())
          continue;
        unsigned dstLoopDepthTest =
            getInnermostCommonLoopDepth(dstMemrefOps) - numSurroundingLoops;

        // Check the feasibility of fusing src loop nest into dst loop nest
        // at loop depths in range [1, dstLoopDepthTest].
        unsigned maxLegalFusionDepth = 0;
        SmallVector<ComputationSliceState, 8> depthSliceUnions;
        depthSliceUnions.resize(dstLoopDepthTest);
        FusionStrategy strategy(FusionStrategy::ProducerConsumer);
        for (unsigned i = 1; i <= dstLoopDepthTest; ++i) {
          FusionResult result =
              affine::canFuseLoops(srcAffineForOp, dstAffineForOp,
                                   /*dstLoopDepth=*/i + numSurroundingLoops,
                                   &depthSliceUnions[i - 1], strategy);
          if (result.value == FusionResult::Success) {
            maxLegalFusionDepth = i;
            LDBG() << "Found valid slice for depth: " << i;
          }
        }

        if (maxLegalFusionDepth == 0) {
          LDBG() << "Can't fuse: fusion is not legal at any depth";
          continue;
        }

        LDBG() << "Max legal depth for fusion: " << maxLegalFusionDepth;

        double computeToleranceThresholdToUse = computeToleranceThreshold;

        // Cyclic dependences in the source nest may be violated when performing
        // slicing-based fusion. They aren't actually violated in cases where no
        // redundant execution of the source happens (1:1 pointwise dep on the
        // producer-consumer memref access for example). Check this and allow
        // fusion accordingly.
        if (hasCyclicDependence(srcAffineForOp)) {
          LDBG() << "Source nest has a cyclic dependence.";
          // Maximal fusion does not check for compute tolerance threshold; so
          // perform the maximal fusion only when the redundanation computation
          // is zero.
          if (maximalFusion) {
            auto srcForOp = cast<AffineForOp>(srcNode->op);
            auto dstForOp = cast<AffineForOp>(dstNode->op);
            int64_t sliceCost;
            int64_t fusedLoopNestComputeCost;
            auto fraction = getAdditionalComputeFraction(
                srcForOp, dstForOp, maxLegalFusionDepth, depthSliceUnions,
                sliceCost, fusedLoopNestComputeCost);
            if (!fraction || fraction > 0) {
              LDBG() << "Can't perform maximal fusion with a cyclic dependence "
                     << "and non-zero additional compute.";
              return;
            }
          } else {
            // Set redundant computation tolerance to zero regardless of what
            // the user specified. Without this, fusion would be invalid.
            LDBG() << "Setting compute tolerance to zero since "
                   << "source has a cylic dependence.";
            computeToleranceThresholdToUse = 0;
          }
        }

        // Check if fusion would be profitable. We skip profitability analysis
        // for maximal fusion since we already know the maximal legal depth to
        // fuse.
        unsigned bestDstLoopDepth = maxLegalFusionDepth;
        if (!maximalFusion) {
          // Retrieve producer stores from the src loop.
          SmallVector<Operation *, 2> producerStores;
          for (Operation *op : srcNode->stores)
            if (producerConsumerMemrefs.count(
                    cast<AffineWriteOpInterface>(op).getMemRef()))
              producerStores.push_back(op);

          assert(!producerStores.empty() && "Expected producer store");
          if (!isFusionProfitable(srcAffineForOp, producerStores,
                                  dstAffineForOp, depthSliceUnions,
                                  maxLegalFusionDepth, &bestDstLoopDepth,
                                  computeToleranceThresholdToUse)) {
            continue;
          }
        }

        assert(bestDstLoopDepth > 0 && "Unexpected loop fusion depth");
        ComputationSliceState &bestSlice =
            depthSliceUnions[bestDstLoopDepth - 1];
        assert(!bestSlice.isEmpty() && "Missing slice union for depth");

        // Determine if 'srcId' can be removed after fusion, taking into
        // account remaining dependences, escaping memrefs and the fusion
        // insertion point.
        bool removeSrcNode = canRemoveSrcNodeAfterFusion(
            srcId, dstId, bestSlice, fusedLoopInsPoint, srcEscapingMemRefs,
            *mdg);

        // DR-DIVERGE (re-execution guard): when the src nest survives
        // fusion, the inserted slice RE-EXECUTES the src computation after
        // the original nest already ran.  That is only sound if the slice
        // reads the same input values the original run saw — false whenever
        // the src updates a memref in place (reads a memref it also
        // writes): the re-execution then reads already-updated data.
        // Example: PolyBench fdtd-2d's `ex[i][j] -= ...` nest fused into
        // the hz nest recomputes ex from the post-update ex, and hz
        // consumes garbage that compounds across time steps.  Skip such
        // candidates outright.
        if (!removeSrcNode) {
          DenseSet<Value> srcWrites;
          for (Operation *st : srcNode->stores)
            srcWrites.insert(cast<AffineWriteOpInterface>(st).getMemRef());
          bool srcReadsItsOwnWrites =
              llvm::any_of(srcNode->loads, [&](Operation *ld) {
                return srcWrites.count(
                           cast<AffineReadOpInterface>(ld).getMemRef()) > 0;
              });
          if (srcReadsItsOwnWrites) {
            LDBG() << "Skipping fusion: src nest would survive fusion but "
                      "updates a memref in place; slice re-execution would "
                      "read already-updated data";
            continue;
          }
        }

        DenseSet<Value> privateMemrefs;
        for (Value memref : producerConsumerMemrefs) {
          if (canCreatePrivateMemRef(memref, srcEscapingMemRefs, srcId, dstId,
                                     removeSrcNode)) {
            // Create a private version of this memref.
            LDBG() << "Creating private memref for " << memref;
            // Create a private version of this memref.
            privateMemrefs.insert(memref);
          }
        }

        // Fuse computation slice of 'srcLoopNest' into 'dstLoopNest'.
        fuseLoops(srcAffineForOp, dstAffineForOp, bestSlice);
        dstNodeChanged = true;

        LDBG() << "Fused src loop " << srcId << " into dst loop " << dstId
               << " at depth " << bestDstLoopDepth << ":";
        LDBG() << dstAffineForOp;

        // Move 'dstAffineForOp' before 'insertPointInst' if needed.
        if (fusedLoopInsPoint != dstAffineForOp)
          dstAffineForOp->moveBefore(fusedLoopInsPoint);

        // Update edges between 'srcNode' and 'dstNode'.
        mdg->updateEdges(srcNode->id, dstNode->id, privateMemrefs,
                         removeSrcNode);

        // Create private memrefs.
        if (!privateMemrefs.empty()) {
          // Note the block into which fusion was performed. This can be used to
          // place `alloc`s that create private memrefs.
          Block *sliceInsertionBlock = bestSlice.insertPoint->getBlock();

          // Gather stores for all the private-to-be memrefs.
          DenseMap<Value, SmallVector<Operation *, 4>> privateMemRefToStores;
          dstAffineForOp.walk([&](AffineWriteOpInterface storeOp) {
            Value storeMemRef = storeOp.getMemRef();
            if (privateMemrefs.count(storeMemRef) > 0)
              privateMemRefToStores[storeMemRef].push_back(storeOp);
          });

          // Replace original memrefs with private memrefs. Note that all the
          // loads and stores on these memrefs will be replaced with a new
          // loads and stores. Any reference to the original ones becomes
          // invalid after this point.
          for (auto &memrefToStoresPair : privateMemRefToStores) {
            ArrayRef<Operation *> storesForMemref = memrefToStoresPair.second;
            Value newMemRef = createPrivateMemRef(
                dstAffineForOp, storesForMemref, bestDstLoopDepth,
                fastMemorySpace, sliceInsertionBlock, localBufSizeThreshold);
            if (!newMemRef)
              continue;
            // Create new node in dependence graph for 'newMemRef' alloc op.
            unsigned newMemRefNodeId = mdg->addNode(newMemRef.getDefiningOp());
            // Add edge from 'newMemRef' node to dstNode.
            mdg->addEdge(newMemRefNodeId, dstId, newMemRef);
          }
          // One or more entries for 'newMemRef' alloc op are inserted into
          // the DenseMap mdg->nodes. Since an insertion may cause DenseMap to
          // reallocate, update dstNode.
          dstNode = mdg->getNode(dstId);
        }

        // Collect dst loop stats after memref privatization transformation.
        LoopNestStateCollector dstLoopCollector;
        dstLoopCollector.collect(dstAffineForOp);

        // Clear and add back loads and stores.
        mdg->clearNodeLoadAndStores(dstNode->id);
        mdg->addToNode(
            dstId, dstLoopCollector.loadOpInsts, dstLoopCollector.storeOpInsts,
            dstLoopCollector.memrefLoads, dstLoopCollector.memrefStores,
            dstLoopCollector.memrefFrees);

        if (removeSrcNode) {
          LDBG() << "Removing src loop " << srcId << " after fusion";
          // srcNode is no longer valid after it is removed from mdg.
          srcAffineForOp.erase();
          mdg->removeNode(srcId);
          srcNode = nullptr;
        }
      }
    } while (dstNodeChanged);
  }

  /// Visit each node in the graph, and for each node, attempt to fuse it with
  /// producer-consumer candidates. No fusion is performed when producers with a
  /// user count greater than `maxSrcUserCount` for any of the memrefs involved
  /// are encountered.
  void fuseProducerConsumerNodes(unsigned maxSrcUserCount) {
    LDBG() << "--- Producer/Consumer Fusion ---";
    init();
    while (!worklist.empty()) {
      unsigned dstId = worklist.back();
      worklist.pop_back();
      performFusionsIntoDest(dstId, maxSrcUserCount);
    }
  }

  // Visits each node in the graph, and for each node, attempts to fuse it with
  // its sibling nodes (nodes which share a parent, but no dependence edges).
  void fuseSiblingNodes() {
    LDBG() << "--- Sibling Fusion ---";
    init();
    while (!worklist.empty()) {
      unsigned dstId = worklist.back();
      worklist.pop_back();

      // Skip if this node was removed (fused into another node).
      if (mdg->nodes.count(dstId) == 0)
        continue;
      // Get 'dstNode' into which to attempt fusion.
      auto *dstNode = mdg->getNode(dstId);
      // Skip if 'dstNode' is not a loop nest.
      if (!isa<AffineForOp>(dstNode->op))
        continue;
      // Attempt to fuse 'dstNode' with its sibling nodes in the graph.
      fuseWithSiblingNodes(dstNode);
    }
  }

  // Attempt to fuse 'dstNode' with sibling nodes in the graph.
  void fuseWithSiblingNodes(Node *dstNode) {
    DenseSet<unsigned> visitedSibNodeIds;
    std::pair<unsigned, Value> idAndMemref;
    auto dstAffineForOp = cast<AffineForOp>(dstNode->op);

    while (findSiblingNodeToFuse(dstNode, &visitedSibNodeIds, &idAndMemref)) {
      unsigned sibId = idAndMemref.first;
      Value memref = idAndMemref.second;
      // TODO: Check that 'sibStoreOpInst' post-dominates all other
      // stores to the same memref in 'sibNode' loop nest.
      auto *sibNode = mdg->getNode(sibId);
      // DR-DIVERGE (phase barrier): see performFusionsIntoDest.
      if (dr_fusion::sideEffectingOpBetween(sibNode->op, dstNode->op)) {
        LDBG() << "Skipping sibling fusion: opaque side-effecting op "
                  "between the nests (phase barrier)";
        continue;
      }
      // Compute an operation list insertion point for the fused loop
      // nest which preserves dependences.
      assert(sibNode->op->getBlock() == dstNode->op->getBlock());
      Operation *insertPointInst =
          sibNode->op->isBeforeInBlock(dstNode->op)
              ? mdg->getFusedLoopNestInsertionPoint(sibNode->id, dstNode->id)
              : mdg->getFusedLoopNestInsertionPoint(dstNode->id, sibNode->id);
      if (insertPointInst == nullptr)
        continue;

      // Check if fusion would be profitable and at what depth.

      // Get unique 'sibNode' load op to 'memref'.
      SmallVector<Operation *, 2> sibLoadOpInsts;
      sibNode->getLoadOpsForMemref(memref, &sibLoadOpInsts);
      // Currently findSiblingNodeToFuse searches for siblings with one load.
      Operation *sibLoadOpInst = llvm::getSingleElement(sibLoadOpInsts);

      // Gather 'dstNode' load ops to 'memref'.
      SmallVector<Operation *, 2> dstLoadOpInsts;
      dstNode->getLoadOpsForMemref(memref, &dstLoadOpInsts);

      // It's possible this fusion is at an inner depth (i.e., there are common
      // surrounding affine loops for the source and destination for ops). We
      // need to get this number because the call to canFuseLoops needs to be
      // passed the absolute depth. The max legal depth and the depths we try
      // below are however *relative* and as such don't include the common
      // depth.
      SmallVector<AffineForOp, 4> surroundingLoops;
      getAffineForIVs(*dstAffineForOp, &surroundingLoops);
      unsigned numSurroundingLoops = surroundingLoops.size();
      SmallVector<AffineForOp, 4> dstLoopIVs;
      getAffineForIVs(*dstLoadOpInsts[0], &dstLoopIVs);
      unsigned dstLoopDepthTest = dstLoopIVs.size() - numSurroundingLoops;
      auto sibAffineForOp = cast<AffineForOp>(sibNode->op);

      // Compute loop depth and slice union for fusion.
      SmallVector<ComputationSliceState, 8> depthSliceUnions;
      depthSliceUnions.resize(dstLoopDepthTest);
      unsigned maxLegalFusionDepth = 0;
      FusionStrategy strategy(memref);
      for (unsigned i = 1; i <= dstLoopDepthTest; ++i) {
        FusionResult result =
            affine::canFuseLoops(sibAffineForOp, dstAffineForOp,
                                 /*dstLoopDepth=*/i + numSurroundingLoops,
                                 &depthSliceUnions[i - 1], strategy);

        if (result.value == FusionResult::Success)
          maxLegalFusionDepth = i;
      }

      LDBG() << "Max legal depth for fusion: " << maxLegalFusionDepth;

      // Skip if fusion is not feasible at any loop depths.
      if (maxLegalFusionDepth == 0)
        continue;

      double computeToleranceThresholdToUse = computeToleranceThreshold;

      // Cyclic dependences in the source nest may be violated when performing
      // slicing-based fusion. They aren't actually violated in cases where no
      // redundant execution of the source happens (1:1 pointwise dep on the
      // producer-consumer memref access for example). Check this and allow
      // fusion accordingly.
      if (hasCyclicDependence(sibAffineForOp)) {
        LDBG() << "Source nest has a cyclic dependence.";
        // Maximal fusion does not check for compute tolerance threshold; so
        // perform the maximal fusion only when the redundanation computation is
        // zero.
        if (maximalFusion) {
          auto dstForOp = cast<AffineForOp>(dstNode->op);
          int64_t sliceCost;
          int64_t fusedLoopNestComputeCost;
          auto fraction = getAdditionalComputeFraction(
              sibAffineForOp, dstForOp, maxLegalFusionDepth, depthSliceUnions,
              sliceCost, fusedLoopNestComputeCost);
          if (!fraction || fraction > 0) {
            LDBG() << "Can't perform maximal fusion with a cyclic dependence "
                   << "and non-zero additional compute.";
            return;
          }
        } else {
          // Set redundant computation tolerance to zero regardless of what the
          // user specified. Without this, fusion would be invalid.
          LDBG() << "Setting compute tolerance to zero since "
                 << "source has a cyclic dependence.";
          computeToleranceThresholdToUse = 0.0;
        }
      }

      unsigned bestDstLoopDepth = maxLegalFusionDepth;
      if (!maximalFusion) {
        // Check if fusion would be profitable. For sibling fusion, the sibling
        // load op is treated as the src "store" op for fusion profitability
        // purposes. The footprint of the load in the slice relative to the
        // unfused source's determines reuse.
        if (!isFusionProfitable(sibAffineForOp, sibLoadOpInst, dstAffineForOp,
                                depthSliceUnions, maxLegalFusionDepth,
                                &bestDstLoopDepth,
                                computeToleranceThresholdToUse))
          continue;
      }

      assert(bestDstLoopDepth > 0 && "Unexpected loop fusion depth");

      const ComputationSliceState &bestSlice =
          depthSliceUnions[bestDstLoopDepth - 1];
      assert(!bestSlice.isEmpty() &&
             "Fusion depth has no computed slice union");

      // Do not perform sibling fusion if it isn't maximal. We always remove the
      // sibling node and as such fusion shouldn't be performed if a part of the
      // slice is used in the destination.
      auto isMaximal = bestSlice.isMaximal();
      if (!isMaximal.value_or(false)) {
        LDBG() << "Slice isn't maximal; not performing sibling fusion.";
        continue;
      }

      // Check if source loop is being inserted in the innermost
      // destination loop. Based on this, the fused loop may be optimized
      // further inside `fuseLoops`.
      bool isInnermostInsertion = (bestDstLoopDepth == dstLoopDepthTest);
      // Fuse computation slice of 'sibLoopNest' into 'dstLoopNest'.
      affine::fuseLoops(sibAffineForOp, dstAffineForOp, bestSlice,
                        isInnermostInsertion);

      auto dstForInst = cast<AffineForOp>(dstNode->op);
      // Update operation position of fused loop nest (if needed).
      if (insertPointInst != dstForInst)
        dstForInst->moveBefore(insertPointInst);

      LDBG() << "Fused sibling nest " << sibId << " into destination nest "
             << dstNode->id << " at depth " << bestDstLoopDepth << ":";
      LDBG() << dstAffineForOp;

      // Update data dependence graph state post fusion.
      updateStateAfterSiblingFusion(sibNode, dstNode);

      // Remove old sibling loop nest.
      // Get op before we invalidate the MDG node.
      Operation *op = sibNode->op;
      mdg->removeNode(sibNode->id);
      op->erase();
    }
  }

  // Searches block argument uses and the graph from 'dstNode' looking for a
  // fusion candidate sibling node which shares no dependences with 'dstNode'
  // but which loads from the same memref. Returns true and sets
  // 'idAndMemrefToFuse' on success. Returns false otherwise.
  bool findSiblingNodeToFuse(Node *dstNode,
                             DenseSet<unsigned> *visitedSibNodeIds,
                             std::pair<unsigned, Value> *idAndMemrefToFuse) {
    // Returns true if 'sibNode' can be fused with 'dstNode' for input reuse
    // on 'memref'.
    auto canFuseWithSibNode = [&](Node *sibNode, Value memref) {
      // Skip if 'outEdge' is not a read-after-write dependence.
      // TODO: Remove restrict to single load op restriction.
      if (sibNode->getLoadOpCount(memref) != 1)
        return false;
      // Skip if there exists a path of dependent edges between
      // 'sibNode' and 'dstNode'.
      if (mdg->hasDependencePath(sibNode->id, dstNode->id) ||
          mdg->hasDependencePath(dstNode->id, sibNode->id))
        return false;
      // Skip sib node if it loads to (and stores from) the same memref on
      // which it also has an input dependence edge.
      DenseSet<Value> loadAndStoreMemrefSet;
      sibNode->getLoadAndStoreMemrefSet(&loadAndStoreMemrefSet);
      if (llvm::any_of(loadAndStoreMemrefSet, [=](Value memref) {
            return mdg->getIncomingMemRefAccesses(sibNode->id, memref) > 0;
          }))
        return false;

      // Check that all stores are to the same memref if any.
      DenseSet<Value> storeMemrefs;
      for (auto *storeOpInst : sibNode->stores) {
        storeMemrefs.insert(
            cast<AffineWriteOpInterface>(storeOpInst).getMemRef());
      }
      return storeMemrefs.size() <= 1;
    };

    // Search for siblings which load the same memref block argument.
    Block *block = dstNode->op->getBlock();
    for (unsigned i = 0, e = block->getNumArguments(); i != e; ++i) {
      for (Operation *user : block->getArgument(i).getUsers()) {
        auto loadOp = dyn_cast<AffineReadOpInterface>(user);
        if (!loadOp)
          continue;
        // Gather loops surrounding 'use'.
        SmallVector<AffineForOp, 4> loops;
        getAffineForIVs(*user, &loops);
        // Skip 'use' if it is not within a loop nest.
        // Find the surrounding affine.for nested immediately within the
        // block.
        auto *it = llvm::find_if(loops, [&](AffineForOp loop) {
          return loop->getBlock() == &mdg->block;
        });
        // Skip 'use' if it is not within a loop nest in `block`.
        if (it == loops.end())
          continue;
        Node *sibNode = mdg->getForOpNode(*it);
        assert(sibNode != nullptr);
        // Skip 'use' if it not a sibling to 'dstNode'.
        if (sibNode->id == dstNode->id)
          continue;
        // Skip 'use' if it has been visited.
        if (visitedSibNodeIds->count(sibNode->id) > 0)
          continue;
        // Skip 'use' if it does not load from the same memref as 'dstNode'.
        auto memref = loadOp.getMemRef();
        if (dstNode->getLoadOpCount(memref) == 0)
          continue;
        // Check if 'sibNode/dstNode' can be input-reuse fused on 'memref'.
        if (canFuseWithSibNode(sibNode, memref)) {
          visitedSibNodeIds->insert(sibNode->id);
          idAndMemrefToFuse->first = sibNode->id;
          idAndMemrefToFuse->second = memref;
          return true;
        }
      }
    }

    // Search for siblings by following edges through an intermediate src node.
    // Collect candidate 'dstNode' input edges in 'inEdges'.
    SmallVector<MemRefDependenceGraph::Edge, 2> inEdges;
    mdg->forEachMemRefInputEdge(
        dstNode->id, [&](MemRefDependenceGraph::Edge inEdge) {
          // Add 'inEdge' if it is a read-after-write dependence or an edge
          // from a memref defining op (e.g. view-like op or alloc op).
          if (dstNode->getLoadOpCount(inEdge.value) > 0 &&
              (mdg->getNode(inEdge.id)->getStoreOpCount(inEdge.value) > 0 ||
               inEdge.value.getDefiningOp() == mdg->getNode(inEdge.id)->op))
            inEdges.push_back(inEdge);
        });

    // Search for sibling nodes to fuse by visiting output edges from each input
    // edge in 'inEdges'.
    for (auto &inEdge : inEdges) {
      // Collect candidate output edges from each node 'inEdge.id' in 'inEdges'.
      SmallVector<MemRefDependenceGraph::Edge, 2> outEdges;
      mdg->forEachMemRefOutputEdge(
          inEdge.id, [&](MemRefDependenceGraph::Edge outEdge) {
            unsigned sibNodeId = outEdge.id;
            if (visitedSibNodeIds->count(sibNodeId) > 0)
              return;
            // Skip output edge if not a sibling using the same memref.
            if (outEdge.id == dstNode->id || outEdge.value != inEdge.value)
              return;
            auto *sibNode = mdg->getNode(sibNodeId);
            if (!isa<AffineForOp>(sibNode->op))
              return;
            // Check if 'sibNode/dstNode' can be input-reuse fused on 'memref'.
            if (canFuseWithSibNode(sibNode, outEdge.value)) {
              // Add candidate 'outEdge' to sibling node.
              outEdges.push_back(outEdge);
            }
          });

      // Add first candidate if any were returned.
      if (!outEdges.empty()) {
        visitedSibNodeIds->insert(outEdges[0].id);
        idAndMemrefToFuse->first = outEdges[0].id;
        idAndMemrefToFuse->second = outEdges[0].value;
        return true;
      }
    }
    return false;
  }

  /// Update data dependence graph state to reflect sibling fusion of 'sibNode'
  /// into 'dstNode'.
  void updateStateAfterSiblingFusion(Node *sibNode, Node *dstNode) {
    // Update 'sibNode' and 'dstNode' input/output edges to reflect fusion.
    mdg->updateEdges(sibNode->id, dstNode->id);

    // Collect dst loop stats after memref privatization transformation.
    auto dstForInst = cast<AffineForOp>(dstNode->op);
    LoopNestStateCollector dstLoopCollector;
    dstLoopCollector.collect(dstForInst);
    // Clear and add back loads and stores
    mdg->clearNodeLoadAndStores(dstNode->id);
    mdg->addToNode(dstNode->id, dstLoopCollector.loadOpInsts,
                   dstLoopCollector.storeOpInsts, dstLoopCollector.memrefLoads,
                   dstLoopCollector.memrefStores, dstLoopCollector.memrefFrees);
  }

  // Clean up any allocs with no users.
  void eraseUnusedMemRefAllocations() {
    for (auto &pair : mdg->memrefEdgeCount) {
      if (pair.second > 0)
        continue;
      auto memref = pair.first;
      // Skip if there exist other uses (return operation or function calls).
      if (!memref.use_empty())
        continue;
      // Use list expected to match the dep graph info.
      auto *op = memref.getDefiningOp();
      if (isa_and_nonnull<memref::AllocOp>(op))
        op->erase();
    }
  }
};

} // namespace

/// Run fusion on `block`.
void DrAffineLoopFusionPass::runOnBlock(Block *block) {
  MemRefDependenceGraph g(*block);
  if (!g.init()) {
    LDBG() << "MDG init failed";
    return;
  }

  std::optional<unsigned> fastMemorySpaceOpt;
  if (fastMemorySpace.hasValue())
    fastMemorySpaceOpt = fastMemorySpace;
  unsigned localBufSizeThresholdBytes = localBufSizeThreshold * 1024;
  GreedyFusion fusion(&g, localBufSizeThresholdBytes, fastMemorySpaceOpt,
                      maximalFusion, computeToleranceThreshold);

  if (affineFusionMode == FusionMode::ProducerConsumer)
    fusion.runProducerConsumerFusionOnly();
  else if (affineFusionMode == FusionMode::Sibling)
    fusion.runSiblingFusionOnly();
  else
    fusion.runGreedyFusion();
}

void DrAffineLoopFusionPass::runOnOperation() {
  // DR-DIVERGE: build the unified-cost-model active context for the duration
  // of this pass invocation.  Precedence: JSON file > CLI options > handler
  // defaults.  Cleared on exit so we never leak past the pass.
  dr_fusion::UnifiedConfig config;
  config.useUnifiedCostModel = useUnifiedCostModel;
  config.emitRationale = emitRationale;

  drcompiler::CpuCostModel cm =
      cpuCostModelFile.empty()
          ? drcompiler::CpuCostModel::getDefault()
          : drcompiler::CpuCostModel::loadFromFile(cpuCostModelFile);
  const auto &archJson = cm.archParams();
  const auto &regsJson = cm.registerParams();

  std::string handlerName;
  if (!drArchHandler.empty())
    handlerName = drArchHandler;
  else if (archJson.handler)
    handlerName = *archJson.handler;
  else if (archJson.triplet)
    handlerName = drcompiler::ArchHandler::pickHandlerForTriple(
                      llvm::Triple(*archJson.triplet))
                      .str();
  else
    handlerName = "generic";

  config.archHandler = drcompiler::ArchHandler::create(handlerName);
  config.archParams = config.archHandler->defaultParams();
  config.regParams = config.archHandler->defaultRegisters();
  if (archJson.triplet)
    config.archParams.triple = llvm::Triple(*archJson.triplet);
  if (archJson.vectorWidthBits)
    config.archParams.vectorWidthBits = *archJson.vectorWidthBits;
  if (archJson.alphaMem)
    config.archParams.alphaMem = *archJson.alphaMem;
  if (archJson.betaReg)
    config.archParams.betaReg = *archJson.betaReg;
  if (archJson.gammaAlu)
    config.archParams.gammaAlu = *archJson.gammaAlu;
  if (regsJson.gpBudget)
    config.regParams.gpBudget = *regsJson.gpBudget;
  if (regsJson.fpBudget)
    config.regParams.fpBudget = *regsJson.fpBudget;
  if (regsJson.vecBudget)
    config.regParams.vecBudget = *regsJson.vecBudget;
  if (regsJson.predBudget)
    config.regParams.predBudget = *regsJson.predBudget;
  if (regsJson.spillReloadCycles)
    config.regParams.spillReloadCycles = *regsJson.spillReloadCycles;
  if (regsJson.spillStoreCycles)
    config.regParams.spillStoreCycles = *regsJson.spillStoreCycles;
  if (!drSpillStrategy.empty())
    config.spillStrategy = drcompiler::parseSpillStrategy(drSpillStrategy);
  else if (archJson.spillStrategy)
    config.spillStrategy = drcompiler::parseSpillStrategy(*archJson.spillStrategy);

  // DR-DIVERGE: pull cache hierarchy from the JSON `cache` block when
  // present.  Leaves Phase-1 defaults otherwise (`UnifiedConfig::cache`).
  {
    const auto &cj = cm.cacheParams();
    if (cj.l1Size)
      config.cache.l1Size = *cj.l1Size;
    if (cj.l2Size)
      config.cache.l2Size = *cj.l2Size;
    if (cj.l3Size)
      config.cache.l3Size = *cj.l3Size;
    if (cj.l1Latency)
      config.cache.l1Latency = *cj.l1Latency;
    if (cj.l2Latency)
      config.cache.l2Latency = *cj.l2Latency;
    if (cj.l3Latency)
      config.cache.l3Latency = *cj.l3Latency;
    if (cj.memLatency)
      config.cache.memLatency = *cj.memLatency;
  }

  dr_fusion::gActive = &config;
  struct ActiveResetter {
    ~ActiveResetter() { dr_fusion::gActive = nullptr; }
  } resetter;

  // Call fusion on every op that has at least two affine.for nests (in post
  // order).
  getOperation()->walk([&](Operation *op) {
    for (Region &region : op->getRegions()) {
      for (Block &block : region.getBlocks()) {
        auto affineFors = block.getOps<AffineForOp>();
        if (!affineFors.empty() && !llvm::hasSingleElement(affineFors))
          runOnBlock(&block);
      }
    }
  });
}

// DR-DIVERGE: ctor renamed and unparameterised wrapper added so callers
// can construct via tablegen options or via the parameterised ctor.
std::unique_ptr<Pass> mlir::createDrAffineLoopFusionPass() {
  return std::make_unique<DrAffineLoopFusionPass>();
}

std::unique_ptr<Pass> mlir::createDrAffineLoopFusionPass(
    unsigned fastMemorySpace, uint64_t localBufSizeThreshold,
    bool maximalFusion, enum affine::FusionMode affineFusionMode) {
  return std::make_unique<DrAffineLoopFusionPass>(
      fastMemorySpace, localBufSizeThreshold, maximalFusion, affineFusionMode);
}
