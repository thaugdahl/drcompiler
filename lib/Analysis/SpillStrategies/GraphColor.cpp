//===- GraphColor.cpp - Chaitin-Briggs-style spill estimation --------------===//
//
// Per register class:
//   1. Collect all SSA values of that class observed in the trace.
//   2. Build interference graph: two values interfere iff they are
//      simultaneously live at some program point.
//   3. Simplify: repeatedly remove vertices with degree < budget (these can
//      always be coloured); push onto the stack.
//   4. If a vertex has degree >= budget, choose a spill candidate (Briggs
//      heuristic: highest degree / spill-cost ratio; spill cost here is the
//      number of points the value is live across — a longer live range costs
//      more to spill).  Mark it spilled; remove; continue.
//   5. After all vertices are processed, the number of marked vertices is
//      the number of spills per class.
//
// Cost: each spilled value contributes (spill_reload * live_range_length)
// cycles plus a one-time store (the live-range model approximates one
// reload per program point the value is live across).
//
// Complexity: O(V^2) per class for the graph build; V is bounded by the
// number of distinct values of that class in the region.  Acceptable for
// the loop-body-sized regions we analyse.
//
//===----------------------------------------------------------------------===//

#include "../PressureTrace.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>

namespace drcompiler {

namespace {

struct ClassGraph {
  // Dense vertex ids 0..N-1.  Vertex i corresponds to values[i].
  llvm::SmallVector<mlir::Value, 16> values;
  // Live-range length: number of trace points the value is live across.
  llvm::SmallVector<unsigned, 16> liveLen;
  // Adjacency as bitsets via DenseSet.  Symmetric.
  llvm::SmallVector<llvm::DenseSet<unsigned>, 16> adj;
};

ClassGraph buildClassGraph(const PressureTrace &trace, RegClass cls) {
  ClassGraph g;
  llvm::DenseMap<mlir::Value, unsigned> idOf;

  for (const PressurePoint &pt : trace.points) {
    // Collect ids for values of the target class at this point.
    llvm::SmallVector<unsigned, 8> hereIds;
    for (mlir::Value v : pt.liveSet) {
      auto it = trace.classification.find(v);
      if (it == trace.classification.end())
        continue;
      if (it->second.cls != cls)
        continue;
      auto [insertion, inserted] = idOf.try_emplace(v, g.values.size());
      if (inserted) {
        g.values.push_back(v);
        g.liveLen.push_back(0);
        g.adj.emplace_back();
      }
      hereIds.push_back(insertion->second);
    }

    // Tick live length and add interference edges among present ids.
    for (unsigned id : hereIds)
      ++g.liveLen[id];
    for (unsigned i = 0; i < hereIds.size(); ++i) {
      for (unsigned j = i + 1; j < hereIds.size(); ++j) {
        unsigned a = hereIds[i], b = hereIds[j];
        g.adj[a].insert(b);
        g.adj[b].insert(a);
      }
    }
  }

  return g;
}

/// Returns the number of vertices marked as spills.  Each spill also
/// contributes its live-range length back via `outSpillRangeSum`.
unsigned colorAndCountSpills(ClassGraph &g, unsigned budget,
                             uint64_t &outSpillRangeSum) {
  outSpillRangeSum = 0;
  if (budget == 0)
    return 0;
  unsigned n = g.values.size();
  if (n == 0)
    return 0;

  llvm::SmallVector<unsigned, 16> degree(n);
  llvm::SmallVector<bool, 16> removed(n, false);
  llvm::SmallVector<bool, 16> spilled(n, false);
  for (unsigned i = 0; i < n; ++i)
    degree[i] = g.adj[i].size();

  // Iteratively simplify or pick a spill.
  unsigned active = n;
  while (active > 0) {
    int pick = -1;
    // Prefer a simplifiable vertex (degree < budget).
    for (unsigned i = 0; i < n; ++i) {
      if (removed[i])
        continue;
      if (degree[i] < budget) {
        pick = static_cast<int>(i);
        break;
      }
    }
    if (pick < 0) {
      // Briggs spill heuristic: pick vertex maximising degree / liveLen.
      double bestScore = -1.0;
      for (unsigned i = 0; i < n; ++i) {
        if (removed[i])
          continue;
        double len = static_cast<double>(std::max<unsigned>(g.liveLen[i], 1));
        double score = static_cast<double>(degree[i]) / len;
        if (score > bestScore) {
          bestScore = score;
          pick = static_cast<int>(i);
        }
      }
      spilled[pick] = true;
      outSpillRangeSum += g.liveLen[pick];
    }

    // Remove vertex `pick`.
    removed[pick] = true;
    --active;
    for (unsigned nb : g.adj[pick]) {
      if (!removed[nb] && degree[nb] > 0)
        --degree[nb];
    }
  }

  return static_cast<unsigned>(
      std::count(spilled.begin(), spilled.end(), true));
}

} // namespace

uint64_t aggregateGraphColor(const PressureTrace &trace,
                             const RegisterParams &params,
                             const ArchHandler &arch,
                             const ArchParams &archParams) {
  uint64_t total = 0;
  for (RegClass cls : allRegClasses()) {
    ClassGraph g = buildClassGraph(trace, cls);
    if (g.values.empty())
      continue;
    unsigned budget = params.budgetFor(cls);
    uint64_t rangeSum = 0;
    unsigned spills = colorAndCountSpills(g, budget, rangeSum);
    if (spills == 0)
      continue;
    unsigned perSpill =
        arch.tuneSpillCost(params.spillReloadCycles, cls, archParams);
    // One store + (range-length) reloads per spilled value.  rangeSum already
    // accumulates reload events across spilled vertices.
    total += uint64_t(spills) * params.spillStoreCycles +
             rangeSum * uint64_t(perSpill);
  }
  return total;
}

} // namespace drcompiler
