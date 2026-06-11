//===- MachineModel.cpp - One machine, one description --------------------===//

#include "drcompiler/Analysis/MachineModel.h"
#include "drcompiler/Analysis/CpuCostModel.h"

using namespace drcompiler;

MachineModel MachineModel::fromJson(llvm::StringRef path) {
  MachineModel mm; // built-in defaults

  if (path.empty())
    return mm;

  // Reuse the existing JSON parser — there is exactly one cost-model file and
  // one parser for it.  Absent fields stay at their built-in defaults.
  CpuCostModel cm = CpuCostModel::loadFromFile(path);
  const CpuCacheJsonParams &c = cm.cacheParams();

  if (c.l1Size)
    mm.l1Size = static_cast<int64_t>(*c.l1Size);
  if (c.l2Size)
    mm.l2Size = static_cast<int64_t>(*c.l2Size);
  if (c.l3Size)
    mm.l3Size = static_cast<int64_t>(*c.l3Size);
  if (c.l1Latency)
    mm.l1Lat = *c.l1Latency;
  if (c.l2Latency)
    mm.l2Lat = *c.l2Latency;
  if (c.l3Latency)
    mm.l3Lat = *c.l3Latency;
  if (c.memLatency)
    mm.memLat = *c.memLatency;
  if (c.pageSize)
    mm.pageSize = static_cast<int64_t>(*c.pageSize);
  if (c.l2TlbEntries)
    mm.l2TlbEntries = static_cast<int64_t>(*c.l2TlbEntries);
  if (c.cacheLine)
    mm.cacheLine = static_cast<int64_t>(*c.cacheLine);
  if (c.llcSharers)
    mm.llcSharers = *c.llcSharers;

  return mm;
}
