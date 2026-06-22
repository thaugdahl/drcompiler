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
#include "drcompiler/Analysis/CpuCostModel.h"
#include "drcompiler/Analysis/MachineModel.h"
#include "drcompiler/Analysis/ReuseAnalysis.h"
#include "drcompiler/Transforms/AffineLoopTile.h"

namespace mlir {
using affine::FusionMode; // unused here but kept for parity with Fusion fork
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
  // Returns false when the band should not be tiled at all (v2 REJECT path);
  // v1 always tiled, even with size 1, which paid loop-structure overhead on
  // bands with nothing to reuse.
  bool getTileSizes(ArrayRef<AffineForOp> band,
                    SmallVectorImpl<unsigned> *tileSizes);
  // v2 reuse-driven model (default): gate on evicted temporal reuse, then a
  // per-dimension grid search minimizing inter-tile memory traffic.
  bool getTileSizesV2(ArrayRef<AffineForOp> band,
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
bool DrAffineLoopTilePass::getTileSizes(ArrayRef<AffineForOp> band,
                                        SmallVectorImpl<unsigned> *tileSizes) {
  if (band.empty())
    return false;

  // Use command-line tileSize for all loops if specified.
  if (tileSize) {
    tileSizes->assign(band.size(), tileSize);
    return true;
  }

  // Use supplied tile sizes and fill them with default tile size if it's short.
  if (!this->tileSizes.empty()) {
    tileSizes->assign(this->tileSizes.begin(), this->tileSizes.end());
    tileSizes->resize(band.size(), kDefaultTileSize);
    return true;
  }
  tileSizes->resize(band.size());

  // DR-DIVERGE: the unified path is now the reuse-driven v2 model.
  if (useUnifiedCostModel)
    return getTileSizesV2(band, tileSizes);

  // If the cache size is zero, set the minimum valid tile size. No good reason
  // to pick another specific size over this.
  if (cacheSizeInKiB == 0) {
    llvm::fill(*tileSizes, 1);
    return true;
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
    return true;
  }

  // Check how many times larger the cache size is when compared to footprint.
  uint64_t cacheSizeBytes = cacheSizeInKiB * 1024;
  uint64_t excessFactor = llvm::divideCeil(*fp, cacheSizeBytes);
  if (excessFactor <= 1) {
    // No need of any tiling - set tile size to 1.
    llvm::fill(*tileSizes, 1);
    return true;
  }

  // Upstream: nth root of excess factor, last dimension covers the balance.
  unsigned tSize =
      static_cast<unsigned>(floorl(std::pow(excessFactor, 1.0 / band.size())));
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
  return true;
}

// DR-DIVERGE: v2 reuse-driven tile-size selection.
//
// Decision structure:
//  1. analyzeBandReuse — constant-coefficient per-reference reuse model.
//     Out-of-model bands (imperfect below the band, vector ops, symbolic
//     subscripts, unknown trips) are REJECTED: if we cannot see the reuse we
//     do not restructure the loops.  This is what protects streaming and
//     stencil kernels that v1 tiled for pure overhead (atax reg-block-tile
//     0.55x at EXTRALARGE).
//  2. Gate: some band loop must carry non-degenerate temporal reuse whose
//     reuse distance overflows the target cache share.  Otherwise every
//     datum is either register-held, cache-resident, or never reused —
//     tiling cannot convert any miss to a hit.  REJECT.
//  3. Per-dimension grid search (untiled is a candidate per dim) minimizing
//     inter-tile traffic  sum_ref (writeFactor * tiles * refFootprint(T))
//     subject to footprint(T) <= target.  This replaces v1's uniform-c
//     GEMM-shaped formula (reuse ~c for every kernel, c^2 accumulators).
bool DrAffineLoopTilePass::getTileSizesV2(
    ArrayRef<AffineForOp> band, SmallVectorImpl<unsigned> *tileSizes) {
  auto rationale = [&](const std::string &msg) {
    if (emitRationale)
      band.front()->emitRemark("tile-rationale: " + msg);
  };

  // Upper-bound trips accepted: triangular bands (covariance/syrk
  // j = i..M) tile via min/max bounds; over-estimated footprints only err
  // toward tiling, and the traffic ranking is shape-relative.
  auto infoOr = drcompiler::reuse::analyzeBandReuse(
      band, /*walkRoot=*/nullptr, /*acceptTripUpperBounds=*/true);
  if (failed(infoOr)) {
    rationale("REJECT reason=out-of-model (non-constant trips, imperfect "
              "below band, or non-affine references)");
    return false;
  }
  drcompiler::reuse::BandReuseInfo &info = *infoOr;
  unsigned d = band.size();

  // Target capacity: half the private L2 from the cost-model JSON (same
  // convention as MemoryFission's l2-occupancy-pct default), falling back to
  // the cache-size option.
  drcompiler::CpuCostModel cm =
      cpuCostModelFile.empty()
          ? drcompiler::CpuCostModel::getDefault()
          : drcompiler::CpuCostModel::loadFromFile(cpuCostModelFile);
  int64_t l2Bytes = cm.cacheParams().l2Size
                        ? static_cast<int64_t>(*cm.cacheParams().l2Size)
                        : static_cast<int64_t>(cacheSizeInKiB) * 1024;
  int64_t target = std::max<int64_t>(l2Bytes / 2, 4096);
  // Cache line for the spatial-reuse gate (>1 line => worth tiling), from the
  // JSON if present (CROSSCUTTING.md P0: line size no longer hardcoded in the
  // reuse analysis); 64 B default reproduces the prior constant.
  int64_t cacheLine = cm.cacheParams().cacheLine
                          ? static_cast<int64_t>(*cm.cacheParams().cacheLine)
                          : 64;

  // Gate on evicted temporal reuse.
  bool anyEvicted = false;
  for (unsigned l = 0; l < d && !anyEvicted; ++l)
    anyEvicted = info.loopCarriesEvictedReuse(l, target, cacheLine);
  if (!anyEvicted) {
    rationale("REJECT reason=no-evicted-reuse footprint=" +
              std::to_string(info.footprintBytes(info.tripCounts)) +
              " target=" + std::to_string(target));
    return false;
  }

  // Per-dimension candidates: modest powers plus "untiled" (= trip count).
  // Dims with only an upper-bound trip (triangular) cannot express
  // "untiled" as a tile size; they always pick from the fixed candidates.
  static constexpr uint64_t kCands[] = {16, 24, 32, 48, 64, 96, 128};
  SmallVector<SmallVector<uint64_t, 8>, 6> cands(d);
  for (unsigned l = 0; l < d; ++l) {
    for (uint64_t c : kCands)
      if (c < info.tripCounts[l])
        cands[l].push_back(c);
    if (info.tripIsExact[l] || cands[l].empty())
      cands[l].push_back(info.tripCounts[l]); // untiled (or tiny trip)
  }

  // Exhaustive odometer walk (PolyBench bands are depth <= 4; 8^4 max).
  SmallVector<unsigned, 6> idx(d, 0);
  SmallVector<uint64_t, 6> cur(d), best;
  double bestTraffic = std::numeric_limits<double>::infinity();
  double bestTiles = std::numeric_limits<double>::infinity();
  for (;;) {
    for (unsigned l = 0; l < d; ++l)
      cur[l] = cands[l][idx[l]];
    if (info.footprintBytes(cur) <= target) {
      double tiles = 1.0;
      for (unsigned l = 0; l < d; ++l)
        tiles *= std::ceil(static_cast<double>(info.tripCounts[l]) /
                           static_cast<double>(cur[l]));
      double traffic = 0.0;
      for (unsigned r = 0, e = info.refs.size(); r < e; ++r)
        traffic += (info.refs[r].isWrite ? 2.0 : 1.0) * tiles *
                   static_cast<double>(info.refFootprintBytes(r, cur));
      // Prefer lower traffic; tie-break on fewer tiles (less loop-structure
      // overhead, which also prefers untiled dims).
      if (traffic < bestTraffic * (1.0 - 1e-9) ||
          (traffic < bestTraffic * (1.0 + 1e-9) && tiles < bestTiles)) {
        bestTraffic = traffic;
        bestTiles = tiles;
        best.assign(cur.begin(), cur.end());
      }
    }
    // Odometer increment.
    unsigned l = 0;
    for (; l < d; ++l) {
      if (++idx[l] < cands[l].size())
        break;
      idx[l] = 0;
    }
    if (l == d)
      break;
  }

  if (best.empty()) {
    rationale("REJECT reason=no-feasible-tile target=" +
              std::to_string(target));
    return false;
  }
  bool allUntiled = true;
  for (unsigned l = 0; l < d; ++l)
    allUntiled &= info.tripIsExact[l] && best[l] == info.tripCounts[l];
  if (allUntiled) {
    rationale("REJECT reason=untiled-optimal");
    return false;
  }

  for (unsigned l = 0; l < d; ++l) {
    uint64_t t = best[l];
    // Snap tiled dims to trip-count divisors to avoid min/max bounds (the
    // shared helper would also halve untiled dims, so do it here).
    // Upper-bound-trip dims keep their raw size: their bounds are min/max
    // either way.
    if (avoidMaxMinBounds && info.tripIsExact[l] && t < info.tripCounts[l])
      while (info.tripCounts[l] % t != 0)
        --t;
    (*tileSizes)[l] = static_cast<unsigned>(t);
  }

  if (emitRationale) {
    std::string buf;
    llvm::raw_string_ostream os(buf);
    os << "TILE sizes=[";
    llvm::interleaveComma(*tileSizes, os);
    os << "] footprint=" << info.footprintBytes(best) << " target=" << target
       << " traffic=" << static_cast<uint64_t>(bestTraffic);
    rationale(buf);
  }
  return true;
}

void DrAffineLoopTilePass::runOnOperation() {
  // Working-set gate threshold (bytes; 0 = off).  Prefer the MachineModel's
  // shared LLC (single source of truth, Zen4 default or cpu-cost-model-file) so
  // the gate is not a hard-coded constant; fall back to the explicit llc-gate.
  uint64_t gateBytes = 0;
  if (llcGateFromModel)
    gateBytes = static_cast<uint64_t>(
        drcompiler::MachineModel::fromJson(cpuCostModelFile).l3Size);
  else if (llcGateInKiB > 0)
    gateBytes = llcGateInKiB * 1024;

  // Bands of loops to tile.
  std::vector<SmallVector<AffineForOp, 6>> bands;
  getTopLevelTileableBands(getOperation(), bands);

  // Tile each band.
  for (auto &band : bands) {
    // Working-set gate (opt-in): only tile a band whose FULL footprint exceeds
    // the LLC -- i.e. the untiled set spills last-level cache and the kernel is
    // bandwidth-bound.  Below the gate the set is already cache-resident, so
    // tiling only adds loop/peel overhead (POLYBENCH_PARALLEL_FINDINGS.md #3:
    // measured, N=1024 24MB<32MB tiling HURTS, N=2048 96MB>32MB tiling
    // +38/107%).
    if (gateBytes > 0) {
      std::optional<int64_t> fp = getMemoryFootprintBytes(band[0], 0);
      if (fp && static_cast<uint64_t>(*fp) <= gateBytes) {
        if (emitRationale)
          band[0].emitRemark("tile-rationale: SKIP reason=fits-llc footprint=" +
                             std::to_string(*fp) + " llc-gate=" +
                             std::to_string(gateBytes));
        continue;
      }
    }
    // Set up tile sizes; fill missing tile sizes at the end with default tile
    // size or tileSize if one was provided.
    SmallVector<unsigned, 6> tileSizes;
    // DR-DIVERGE: a REJECT from the cost model now leaves the band entirely
    // untouched.  v1 tiled with size 1, which still restructured the loops
    // and paid min/max-bound overhead for zero reuse benefit.
    if (!getTileSizes(band, &tileSizes))
      continue;
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
