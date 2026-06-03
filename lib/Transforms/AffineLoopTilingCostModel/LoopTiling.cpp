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
    // DR-DIVERGE: tile-size selection driven by the unified cache<->register
    // cost model.  Candidates are ranked by the TILE-SENSITIVE cost only
    // (memory traffic, which falls as the tile grows because each datum is
    // reused ~tile times before eviction, traded against register pressure,
    // which rises with the tile's c^2 accumulator block).  See the scoring
    // block below.  The previous version ranked by the full combined total,
    // which (a) was dominated by a tile-INVARIANT ALU term that drowned the
    // tile signal, and (b) cast a uint64 traffic estimate to 32-bit, which
    // overflowed to ~0 at realistic sizes — so it degenerated to the
    // smallest tile regardless of weights or problem size.
    llvm::SmallVector<unsigned, 12> candidates{2u,  4u,  6u,  8u,  12u,
                                               16u, 24u, 32u, 48u, 64u};
    uint64_t bestTotal = std::numeric_limits<uint64_t>::max();
    uint64_t untiledTotal = std::numeric_limits<uint64_t>::max();
    unsigned bestTile = 1;

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

    // DR-DIVERGE (reuse + register fix): cache hierarchy from the JSON model
    // (falling back to the cache-size option), used to map a tile's working
    // set to its latency tier.
    const auto &cj = cm.cacheParams();
    auto orElse = [](std::optional<unsigned> v, unsigned dflt) -> unsigned {
      return v ? *v : dflt;
    };
    dr::CacheParams cache{
        orElse(cj.l1Size, static_cast<unsigned>(cacheSizeInKiB * 1024u / 4u)),
        orElse(cj.l2Size, static_cast<unsigned>(cacheSizeInKiB * 1024u)),
        orElse(cj.l3Size, 0u),
        orElse(cj.l1Latency, 4u), orElse(cj.l2Latency, 12u),
        orElse(cj.l3Latency, 40u), orElse(cj.memLatency, 200u), 64u};
    (void)innerPressure; // tile pressure is modeled analytically below

    uint64_t totalTripCount = 1;
    bool tripsKnown = true;
    for (AffineForOp f : band) {
      auto tc = getConstantTripCount(f);
      if (!tc) { tripsKnown = false; break; }
      totalTripCount *= *tc;
    }

    // Tile-sensitive cost model, evaluated in double to avoid the 32-bit
    // overflow that previously zeroed the score at scale.  We deliberately
    // exclude the tile-INVARIANT ALU term (~totalTripCount): it dominates the
    // absolute total and would drown the only signal that distinguishes tile
    // sizes.  The two tile-sensitive components are memory traffic (falls with
    // reuse as the tile grows) and register-block spill (rises with c^2).
    double numIter = static_cast<double>(totalTripCount);
    unsigned d = std::max<unsigned>(band.size(), 1u);
    // Linear extent per dimension assuming a roughly cubical iteration space.
    double nExtent =
        tripsKnown ? std::pow(numIter, 1.0 / static_cast<double>(d)) : 0.0;
    // Register capacity in scalar values = vector registers * lanes/register.
    unsigned elemBits = 64u; // f64 working assumption
    double lanes =
        static_cast<double>(std::max(1u, ap.vectorWidthBits / elemBits));
    double regCap = std::max(1.0, static_cast<double>(rp.vecBudget) * lanes);
    double spillCyc = static_cast<double>(std::max(1u, rp.spillReloadCycles));

    auto tileCost = [&](double c) -> double {
      if (!tripsKnown || nExtent <= 0.0 || c < 1.0)
        return std::numeric_limits<double>::infinity();
      // Per-tile data footprint: a c-by-c block of each distinct array
      // (2D-array assumption, matching upstream's identity-access model),
      // derived from the whole-nest footprint fp ~ (#arrays) * nExtent^2.
      double perTileFP =
          static_cast<double>(*fp) * (c * c) / (nExtent * nExtent);
      if (perTileFP < 1.0)
        perTileFP = 1.0;
      unsigned lat = dr::estimateLoadLatency(
          static_cast<int64_t>(std::min(perTileFP, 9.0e18)), cache);
      // Each datum is reused ~c times within a tile before eviction, so
      // distinct fetches from the bottleneck level scale as numIter / c.
      double memCycles = (numIter / c) * static_cast<double>(lat);
      // The inner tile keeps ~c^2 accumulators live; the excess over the
      // register file spills on the fraction of iterations that touch it.
      // This is what caps the useful tile size below the cache-filling size.
      double liveVals = c * c;
      double regCycles = liveVals > regCap
                             ? numIter * (1.0 - regCap / liveVals) * spillCyc
                             : 0.0;
      return ap.alphaMem * memCycles + ap.betaReg * regCycles;
    };

    auto clampU64 = [](double x) -> uint64_t {
      if (!(x < 1.8e19))
        return std::numeric_limits<uint64_t>::max();
      if (x < 0.0)
        return 0u;
      return static_cast<uint64_t>(x);
    };

    double untiledCost = tileCost(1.0);
    double bestCost = untiledCost;
    bestTile = 1;
    for (unsigned candidate : candidates) {
      double tc = tileCost(static_cast<double>(candidate));
      if (tc < bestCost) {
        bestCost = tc;
        bestTile = candidate;
      }
    }
    bestTotal = clampU64(bestCost);
    untiledTotal = clampU64(untiledCost);

    if (bestTile <= 1) {
      if (emitRationale)
        band.front()->emitRemark(
            "tile-rationale: REJECT (untiled best) untiled_cost=" +
            std::to_string(untiledTotal));
      tSize = 1;
    } else {
      if (emitRationale) {
        std::string buf;
        llvm::raw_string_ostream os(buf);
        os << "tile-rationale: TILE size=" << bestTile
           << " tile_cost=" << bestTotal << " untiled_cost=" << untiledTotal
           << " (nExtent=" << static_cast<uint64_t>(nExtent)
           << " regCap=" << static_cast<uint64_t>(regCap) << ")";
        band.front()->emitRemark(buf);
      }
      tSize = bestTile;
    }
  } else {
    // Upstream: nth root of excess factor.
    tSize = static_cast<unsigned>(
        floorl(std::pow(excessFactor, 1.0 / band.size())));
  }
  // DR-DIVERGE: when the unified cost model chose the tile size, emit it
  // UNIFORMLY across the band.  Upstream sets the last dimension to
  // `excessFactor / product(other dims)` to "cover the balance", but that
  // overrides our chosen size on the innermost loop with a large, unrelated
  // value — for GEMM it leaves the k-loop essentially untiled, which both
  // blows the register block we sized for and measures far slower than the
  // uniform tiling the cost model actually scored.
  unsigned cumulProductOfTileSizes = 1;
  for (unsigned i = 0, e = band.size(); i < e; i++) {
    if (useUnifiedCostModel || i < e - 1)
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
