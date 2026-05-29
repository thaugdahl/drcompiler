//===- LoopTiling.cpp --- Loop tiling pass ------------------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to tile affine loop nests.
//
//===----------------------------------------------------------------------===//

#if __has_include("mlir/Dialect/Affine/Passes.h")
#include "mlir/Dialect/Affine/Passes.h"
#else
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#endif

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <optional>

// DR-DIVERGE: pull in our cost-model primitives + ensure our tablegen Base
// is found in `mlir::` rather than `mlir::affine::`.
#include "drcompiler/Transforms/AffineLoopTile.h"
#include "drcompiler/Analysis/ArchHandler.h"
#include "drcompiler/Analysis/RegisterPressureAnalysis.h"
#include "drcompiler/Analysis/SpillStrategy.h"
#include "drcompiler/Transforms/CpuCostModel.h"
#include "drcompiler/Transforms/DataRecomputation/CacheCostModel.h"

namespace mlir {
using affine::FusionMode;  // unused here but kept for parity with Fusion fork
#define GEN_PASS_DEF_DRAFFINELOOPTILEPASS
#include "drcompiler/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;

#define DEBUG_TYPE "dr-affine-loop-tile"

namespace {

// DR-DIVERGE: pass class renamed; we keep upstream's --affine-loop-tile
// available alongside our fork.
/// A pass to perform loop tiling on all suitable loop nests of a func op.
struct DrAffineLoopTilePass
    : public impl::DrAffineLoopTilePassBase<DrAffineLoopTilePass> {
  DrAffineLoopTilePass() = default;
  explicit DrAffineLoopTilePass(uint64_t cacheSizeBytes,
                                 bool avoidMaxMinBounds = true)
      : avoidMaxMinBounds(avoidMaxMinBounds) {
    this->cacheSizeInKiB = cacheSizeBytes / 1024;
  }

  void runOnOperation() override;
  void getTileSizes(ArrayRef<AffineForOp> band,
                    SmallVectorImpl<unsigned> *tileSizes);

  // Default tile size if nothing is provided.
  constexpr static unsigned kDefaultTileSize = 4;

  // If true, tile sizes are set to avoid max/min in bounds if possible.
  bool avoidMaxMinBounds = true;
};

} // namespace

/// Get bands of loops that are valid to tile from the top-level of `f`.
static void
getTopLevelTileableBands(func::FuncOp f,
                         std::vector<SmallVector<AffineForOp, 6>> &bands) {
  // Get maximal perfect nest of 'affine.for' ops starting from root
  // (inclusive).
  for (AffineForOp forOp : f.getOps<AffineForOp>()) {
    SmallVector<AffineForOp, 6> band;
    getPerfectlyNestedLoops(band, forOp);
    if (isTilingValid(band))
      bands.push_back(band);
  }
}

// DR-DIVERGE: ctor renamed to match drcompiler conventions.
std::unique_ptr<Pass> mlir::createDrAffineLoopTilePass() {
  return std::make_unique<DrAffineLoopTilePass>();
}

/// Reduces each tile size to the largest divisor of the corresponding trip
/// count (if the trip count is known).
static void adjustToDivisorsOfTripCounts(ArrayRef<AffineForOp> band,
                                         SmallVectorImpl<unsigned> *tileSizes) {
  assert(band.size() == tileSizes->size() && "invalid tile size count");
  for (unsigned i = 0, e = band.size(); i < e; i++) {
    unsigned &tSizeAdjusted = (*tileSizes)[i];
    std::optional<uint64_t> mayConst = getConstantTripCount(band[i]);
    if (!mayConst)
      continue;
    // Adjust the tile size to largest factor of the trip count less than
    // tSize.
    uint64_t constTripCount = *mayConst;
    if (constTripCount > 1 && tSizeAdjusted > constTripCount / 2)
      tSizeAdjusted = constTripCount / 2;
    while (constTripCount % tSizeAdjusted != 0)
      tSizeAdjusted--;
  }
}

// Returns tile sizes to use. Checks CL options; if none are specified, sets it
// based on a simple model that looks at the memory footprint and determines
// tile sizes assuming identity accesses / 1:1 tile size proportional footprint
// along each of the dimensions being tiled.
// TODO: evolve this model. Tile size determination is a large area
// to play with in general.
void DrAffineLoopTilePass::getTileSizes(ArrayRef<AffineForOp> band,
                              SmallVectorImpl<unsigned> *tileSizes) {
  if (band.empty())
    return;

  // Use command-line tileSize for all loops if specified.
  if (tileSize) {
    tileSizes->assign(band.size(), tileSize);
    return;
  }

  // Use supplied tile sizes and fill them with default tile size if it's short.
  if (!this->tileSizes.empty()) {
    tileSizes->assign(this->tileSizes.begin(), this->tileSizes.end());
    tileSizes->resize(band.size(), kDefaultTileSize);
    return;
  }
  tileSizes->resize(band.size());

  // If the cache size is zero, set the minimum valid tile size. No good reason
  // to pick another specific size over this.
  if (cacheSizeInKiB == 0) {
    llvm::fill(*tileSizes, 1);
    return;
  }

  // Obtain memory footprint and set tile sizes so that a tile fits in
  // the cache size. This is an approximation with the assumption that the
  // footprint increases with the tile size linearly in that dimension (i.e.,
  // assumes one-to-one access function).
  std::optional<int64_t> fp = getMemoryFootprintBytes(band[0], 0);
  if (!fp) {
    // Fill with default tile sizes if footprint is unknown.
    llvm::fill(*tileSizes, DrAffineLoopTilePass::kDefaultTileSize);
    if (avoidMaxMinBounds)
      adjustToDivisorsOfTripCounts(band, tileSizes);
    // The first loop in the band.
    AffineForOp rootForOp = band[0];
    (void)rootForOp;
    LLVM_DEBUG(
        rootForOp.emitWarning("memory footprint unknown: using default tile "
                              "sizes adjusted to trip count divisors"));
    return;
  }

  // Check how many times larger the cache size is when compared to footprint.
  uint64_t cacheSizeBytes = cacheSizeInKiB * 1024;
  uint64_t excessFactor = llvm::divideCeil(*fp, cacheSizeBytes);
  if (excessFactor <= 1) {
    // No need of any tiling - set tile size to 1.
    llvm::fill(*tileSizes, 1);
    return;
  }

  // DR-DIVERGE: unified cost-model-driven tile-size selection.  When the
  // unified path is active, do a small grid search over uniform candidate
  // tile sizes, scoring each via arch.combineCosts(mem, reg, alu).  Falls
  // back to upstream's nth-root heuristic otherwise.
  unsigned tSize = 0;
  if (useUnifiedCostModel) {
    static constexpr unsigned kCandidates[] = {2, 4, 8, 16, 32, 64};
    const uint64_t baseAlu = 4; // crude per-iter ALU
    uint64_t bestTotal = std::numeric_limits<uint64_t>::max();
    unsigned bestTile = DrAffineLoopTilePass::kDefaultTileSize;

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
    auto handler = drcompiler::ArchHandler::create(handlerName);
    auto ap = handler->defaultParams();
    auto rp = handler->defaultRegisters();
    if (archJson.vectorWidthBits)
      ap.vectorWidthBits = *archJson.vectorWidthBits;
    if (regsJson.gpBudget)
      rp.gpBudget = *regsJson.gpBudget;
    if (regsJson.fpBudget)
      rp.fpBudget = *regsJson.fpBudget;
    if (regsJson.vecBudget)
      rp.vecBudget = *regsJson.vecBudget;
    if (regsJson.predBudget)
      rp.predBudget = *regsJson.predBudget;
    if (regsJson.spillReloadCycles)
      rp.spillReloadCycles = *regsJson.spillReloadCycles;
    if (regsJson.spillStoreCycles)
      rp.spillStoreCycles = *regsJson.spillStoreCycles;
    drcompiler::SpillStrategy sp = drcompiler::SpillStrategy::ExcessHot;
    if (!drSpillStrategy.empty())
      sp = drcompiler::parseSpillStrategy(drSpillStrategy);
    else if (archJson.spillStrategy)
      sp = drcompiler::parseSpillStrategy(*archJson.spillStrategy);

    drcompiler::PressureResult innerPressure;
    {
      drcompiler::PressureQuery pq;
      pq.params = rp;
      pq.strategy = sp;
      pq.tripCount = 1;
      auto inner = band.back();
      innerPressure =
          drcompiler::RegisterPressureAnalysis::analyzeRegionStatic(
              inner.getRegion(), *handler, ap, pq);
    }

    // DR-DIVERGE: cache hierarchy reused across the grid + untiled baseline.
    dr::CacheParams cache{
        static_cast<unsigned>(cacheSizeInKiB * 1024u / 4u), // L1 ~ 1/4 cache
        static_cast<unsigned>(cacheSizeInKiB * 1024u),
        0u, 4u, 12u, 40u, 200u, 64u};
    // Total memory cycles = (lines touched) * (per-access latency).  See
    // matching fix in the fusion fork's bytesToMemCycles helper.
    auto bytesToMemCycles = [&](int64_t bytes) -> uint64_t {
      if (bytes <= 0)
        return 0;
      unsigned perAccess = dr::estimateLoadLatency(bytes, cache);
      uint64_t lineSize = std::max<uint64_t>(cache.cacheLineSize, 1u);
      uint64_t lines =
          (static_cast<uint64_t>(bytes) + lineSize - 1) / lineSize;
      return lines * static_cast<uint64_t>(perAccess);
    };

    // DR-DIVERGE: untiled baseline.  If the best tiled candidate doesn't
    // improve on running the band as-is, fall back to tile=1.  Plan §16
    // mirror of the fusion fork's unfused-baseline rejection.
    uint64_t totalTripCount = 1;
    bool tripsKnown = true;
    for (AffineForOp f : band) {
      auto tc = getConstantTripCount(f);
      if (!tc) { tripsKnown = false; break; }
      totalTripCount *= *tc;
    }
    uint64_t untiledTotal = std::numeric_limits<uint64_t>::max();
    if (tripsKnown) {
      uint64_t untiledMem = bytesToMemCycles(static_cast<int64_t>(*fp));
      uint64_t untiledAlu = totalTripCount * baseAlu;
      uint64_t untiledReg = innerPressure.totalSpillCycles * totalTripCount;
      untiledTotal = handler->combineCosts(
          static_cast<unsigned>(untiledMem),
          static_cast<unsigned>(untiledReg),
          static_cast<unsigned>(untiledAlu), ap);
    }

    for (unsigned candidate : kCandidates) {
      uint64_t tileVolume = 1;
      for (unsigned i = 0; i < band.size(); ++i)
        tileVolume *= candidate;
      uint64_t footprintBytes = static_cast<uint64_t>(*fp) *
                                 (uint64_t)candidate * (uint64_t)candidate /
                                 std::max<uint64_t>(excessFactor, 1);
      uint64_t memCycles = bytesToMemCycles(static_cast<int64_t>(footprintBytes));
      uint64_t aluCycles = tileVolume * baseAlu;
      uint64_t regCycles = innerPressure.totalSpillCycles * tileVolume;
      unsigned total = handler->combineCosts(
          static_cast<unsigned>(memCycles), static_cast<unsigned>(regCycles),
          static_cast<unsigned>(aluCycles), ap);
      if (total < bestTotal) {
        bestTotal = total;
        bestTile = candidate;
      }
    }

    // DR-DIVERGE: untiled baseline rejection.  When the best tiled total
    // isn't better than running the band without tiling, set tSize=1 so the
    // emitted code degenerates to a no-op tile transformation.
    if (tripsKnown && bestTotal >= untiledTotal) {
      if (emitRationale) {
        std::string buf;
        llvm::raw_string_ostream os(buf);
        os << "tile-rationale: REJECT best_total=" << bestTotal
           << " >= untiled_total=" << untiledTotal;
        band.front()->emitRemark(buf);
      }
      tSize = 1;
    } else {
      if (emitRationale) {
        std::string buf;
        llvm::raw_string_ostream os(buf);
        os << "tile-rationale: TILE size=" << bestTile
           << " best_total=" << bestTotal
           << " untiled_total=" << untiledTotal;
        band.front()->emitRemark(buf);
      }
      tSize = bestTile;
    }
  } else {
    // Upstream: nth root of excess factor.
    tSize = static_cast<unsigned>(
        floorl(std::pow(excessFactor, 1.0 / band.size())));
  }
  // We'll keep a running product to determine the last tile size better.
  unsigned cumulProductOfTileSizes = 1;
  for (unsigned i = 0, e = band.size(); i < e; i++) {
    if (i < e - 1)
      (*tileSizes)[i] = tSize;
    else
      // Set last tile size to cover the balance.
      (*tileSizes)[i] = std::max(
          1U, static_cast<unsigned>(excessFactor / cumulProductOfTileSizes));
    cumulProductOfTileSizes *= (*tileSizes)[i];
  }
  if (avoidMaxMinBounds)
    adjustToDivisorsOfTripCounts(band, tileSizes);
}

void DrAffineLoopTilePass::runOnOperation() {
  // Bands of loops to tile.
  std::vector<SmallVector<AffineForOp, 6>> bands;
  getTopLevelTileableBands(getOperation(), bands);

  // Tile each band.
  for (auto &band : bands) {
    // Set up tile sizes; fill missing tile sizes at the end with default tile
    // size or tileSize if one was provided.
    SmallVector<unsigned, 6> tileSizes;
    getTileSizes(band, &tileSizes);
    if (llvm::DebugFlag) {
      auto diag = band[0].emitRemark("using tile sizes [");
      llvm::interleaveComma(tileSizes, llvm::dbgs());
      diag << "]\n";
    }
    SmallVector<AffineForOp, 6> tiledNest;
    if (failed(tilePerfectlyNested(band, tileSizes, &tiledNest))) {
      // An empty band always succeeds.
      assert(!band.empty() && "guaranteed to succeed on empty bands");
      LLVM_DEBUG(band.front()->emitRemark("loop tiling failed!\n"));
      continue;
    }

    // Separate full and partial tiles.
    if (separate) {
      auto intraTileLoops =
          MutableArrayRef<AffineForOp>(tiledNest).drop_front(band.size());
      if (failed(separateFullTiles(intraTileLoops))) {
        assert(!intraTileLoops.empty() &&
               "guaranteed to succeed on empty bands");
        LLVM_DEBUG(intraTileLoops.front()->emitRemark(
            "separation post tiling failed!"));
      }
    }
  }
}
