//===- CpuCostModel.cpp - JSON-backed operation cycle cost table ----------===//
//
// Maps MLIR operation names to estimated ALU cycle costs.  The table can be
// loaded from a JSON file (--cpu-cost-model-file) or falls back to built-in
// defaults that match the original hardcoded cost tables.
//
//===----------------------------------------------------------------------===//

#include "drcompiler/Transforms/CpuCostModel.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#define DRDBG() llvm::dbgs() << "DRCOMP-COST: "

using namespace drcompiler;

// ---------- built-in defaults (matches gen_cpu_cost_model.py "generic") -----

void CpuCostModel::populateDefaults() {
  defaultCost = 5;

  // Free
  table["arith.constant"] = 0;

  // Cheap ALU
  for (llvm::StringRef name :
       {"arith.addi", "arith.addf", "arith.subi", "arith.subf", "arith.xori",
        "arith.andi", "arith.ori", "arith.shrsi", "arith.shrui", "arith.shli",
        "arith.select", "arith.cmpi", "arith.cmpf"})
    table[name] = 1;

  // Multiply
  table["arith.muli"] = 3;
  table["arith.mulf"] = 3;

  // Division / remainder
  for (llvm::StringRef name :
       {"arith.divsi", "arith.divui", "arith.divf", "arith.remsi",
        "arith.remui", "arith.remf"})
    table[name] = 15;

  // Transcendental
  for (llvm::StringRef name :
       {"math.sqrt", "math.exp", "math.log", "math.sin", "math.cos",
        "math.tanh", "math.powf"})
    table[name] = 20;

  // Conversions
  for (llvm::StringRef name :
       {"arith.sitofp", "arith.fptosi", "arith.extsi", "arith.extui",
        "arith.trunci", "arith.index_cast", "arith.bitcast"})
    table[name] = 1;
}

// ---------- public API ------------------------------------------------------

CpuCostModel CpuCostModel::getDefault() {
  CpuCostModel m;
  m.populateDefaults();
  return m;
}

CpuCostModel CpuCostModel::loadFromFile(llvm::StringRef path) {
  CpuCostModel m;
  m.populateDefaults(); // start with defaults as fallback

  auto bufOrErr = llvm::MemoryBuffer::getFile(path);
  if (!bufOrErr) {
    llvm::errs() << "drcompiler warning: could not open CPU cost model file '"
                 << path << "': " << bufOrErr.getError().message()
                 << "; using built-in defaults\n";
    return m;
  }

  auto json = llvm::json::parse(bufOrErr.get()->getBuffer());
  if (!json) {
    llvm::errs() << "drcompiler warning: failed to parse CPU cost model '"
                 << path << "': " << llvm::toString(json.takeError())
                 << "; using built-in defaults\n";
    return m;
  }

  auto *root = json->getAsObject();
  if (!root) {
    llvm::errs() << "drcompiler warning: CPU cost model '" << path
                 << "' is not a JSON object; using built-in defaults\n";
    return m;
  }

  // Read default_cost.
  if (auto dc = root->getInteger("default_cost")) {
    if (*dc < 0) {
      llvm::errs() << "drcompiler warning: negative default_cost in '"
                   << path << "'; using built-in default\n";
    } else {
      m.defaultCost = static_cast<unsigned>(*dc);
    }
  }

  // Read ops table — overwrites any defaults for ops that appear.
  if (auto *ops = root->getObject("ops")) {
    for (auto &kv : *ops) {
      if (auto cost = kv.second.getAsInteger()) {
        if (*cost < 0) {
          llvm::errs() << "drcompiler warning: negative cost for '"
                       << kv.first << "' in '" << path << "'; skipping\n";
          continue;
        }
        m.table[kv.first] = static_cast<unsigned>(*cost);
      } else {
        llvm::errs() << "drcompiler warning: non-integer cost for '"
                     << kv.first << "' in '" << path << "'; skipping\n";
      }
    }
  } else if (root->get("ops")) {
    llvm::errs() << "drcompiler warning: 'ops' in '" << path
                 << "' is not a JSON object; ignoring\n";
  }

  // Read optional cache hierarchy parameters.
  if (auto *cache = root->getObject("cache")) {
    auto readUnsigned = [&](llvm::StringRef key,
                            std::optional<unsigned> &dst) {
      if (auto v = cache->getInteger(key)) {
        if (*v < 0) {
          llvm::errs() << "drcompiler warning: negative " << key << " in '"
                       << path << "'; skipping\n";
          return;
        }
        dst = static_cast<unsigned>(*v);
      } else if (cache->get(key)) {
        llvm::errs() << "drcompiler warning: non-integer " << key << " in '"
                     << path << "'; skipping\n";
      }
    };
    readUnsigned("l1_size", m.cache.l1Size);
    readUnsigned("l2_size", m.cache.l2Size);
    readUnsigned("l3_size", m.cache.l3Size);
    readUnsigned("l1_latency", m.cache.l1Latency);
    readUnsigned("l2_latency", m.cache.l2Latency);
    readUnsigned("l3_latency", m.cache.l3Latency);
    readUnsigned("mem_latency", m.cache.memLatency);
  } else if (root->get("cache")) {
    llvm::errs() << "drcompiler warning: 'cache' in '" << path
                 << "' is not a JSON object; ignoring\n";
  }

  m.fromFile = true;
  DRDBG() << "Loaded CPU cost model from '" << path << "' ("
          << m.table.size() << " ops)\n";
  return m;
}

unsigned CpuCostModel::opCost(mlir::Operation *op) const {
  llvm::StringRef name = op->getName().getStringRef();
  auto it = table.find(name);
  if (it != table.end())
    return it->second;
  return defaultCost;
}
