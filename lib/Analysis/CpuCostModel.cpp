//===- CpuCostModel.cpp - JSON-backed operation cycle cost table ----------===//
//
// Maps MLIR operation names to estimated ALU cycle costs.  The table can be
// loaded from a JSON file (--cpu-cost-model-file) or falls back to built-in
// defaults that match the original hardcoded cost tables.
//
//===----------------------------------------------------------------------===//

#include "drcompiler/Analysis/CpuCostModel.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "drcomp-cost"

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
    readUnsigned("page_size", m.cache.pageSize);
    readUnsigned("l2_tlb_entries", m.cache.l2TlbEntries);
    readUnsigned("cache_line", m.cache.cacheLine);
    readUnsigned("llc_sharers", m.cache.llcSharers);
  } else if (root->get("cache")) {
    llvm::errs() << "drcompiler warning: 'cache' in '" << path
                 << "' is not a JSON object; ignoring\n";
  }

  // Read optional arch block.
  if (auto *archObj = root->getObject("arch")) {
    auto readString = [&](llvm::StringRef key,
                          std::optional<std::string> &dst) {
      if (auto v = archObj->getString(key)) {
        dst = v->str();
      } else if (archObj->get(key)) {
        llvm::errs() << "drcompiler warning: non-string arch." << key
                     << " in '" << path << "'; skipping\n";
      }
    };
    auto readUnsigned = [&](llvm::StringRef key,
                            std::optional<unsigned> &dst) {
      if (auto v = archObj->getInteger(key)) {
        if (*v < 0) {
          llvm::errs() << "drcompiler warning: negative arch." << key
                       << " in '" << path << "'; skipping\n";
          return;
        }
        dst = static_cast<unsigned>(*v);
      } else if (archObj->get(key)) {
        llvm::errs() << "drcompiler warning: non-integer arch." << key
                     << " in '" << path << "'; skipping\n";
      }
    };
    auto readNumber = [&](llvm::StringRef key, std::optional<double> &dst) {
      if (auto v = archObj->getNumber(key))
        dst = *v;
      else if (archObj->get(key))
        llvm::errs() << "drcompiler warning: non-numeric arch." << key << " in '"
                     << path << "'; skipping\n";
    };
    readString("triplet", m.arch.triplet);
    readString("handler", m.arch.handler);
    readUnsigned("vector_width_bits", m.arch.vectorWidthBits);
    readString("spill_strategy", m.arch.spillStrategy);
    // Vector-execution model (WP-G1).
    readUnsigned("vector_bits_native", m.arch.vectorBitsNative);
    readUnsigned("vector_bits_arch", m.arch.vectorBitsArch);
    readUnsigned("vec_reg_budget", m.arch.vecRegBudget);
    readNumber("avx512_freq_throttle", m.arch.avx512FreqThrottle);
    readUnsigned("fma_units", m.arch.fmaUnits);
    readNumber("llc_tile_leniency", m.arch.llcTileLeniency);

    if (auto *weights = archObj->getObject("weights")) {
      auto readWeight = [&](llvm::StringRef key,
                            std::optional<double> &dst) {
        if (auto v = weights->getNumber(key))
          dst = *v;
        else if (weights->get(key))
          llvm::errs() << "drcompiler warning: non-numeric arch.weights."
                       << key << " in '" << path << "'; skipping\n";
      };
      readWeight("alpha_mem", m.arch.alphaMem);
      readWeight("beta_reg", m.arch.betaReg);
      readWeight("gamma_alu", m.arch.gammaAlu);
    }
  } else if (root->get("arch")) {
    llvm::errs() << "drcompiler warning: 'arch' in '" << path
                 << "' is not a JSON object; ignoring\n";
  }

  // Read optional registers block.
  if (auto *regs = root->getObject("registers")) {
    auto readUnsigned = [&](llvm::StringRef key,
                            std::optional<unsigned> &dst) {
      if (auto v = regs->getInteger(key)) {
        if (*v < 0) {
          llvm::errs() << "drcompiler warning: negative registers." << key
                       << " in '" << path << "'; skipping\n";
          return;
        }
        dst = static_cast<unsigned>(*v);
      } else if (regs->get(key)) {
        llvm::errs() << "drcompiler warning: non-integer registers." << key
                     << " in '" << path << "'; skipping\n";
      }
    };
    readUnsigned("gp_budget", m.registers.gpBudget);
    readUnsigned("fp_budget", m.registers.fpBudget);
    readUnsigned("vec_budget", m.registers.vecBudget);
    readUnsigned("pred_budget", m.registers.predBudget);
    readUnsigned("spill_reload_cycles", m.registers.spillReloadCycles);
    readUnsigned("spill_store_cycles", m.registers.spillStoreCycles);
  } else if (root->get("registers")) {
    llvm::errs() << "drcompiler warning: 'registers' in '" << path
                 << "' is not a JSON object; ignoring\n";
  }

  // Read optional thread block (CROSSCUTTING.md III: parallel-execution model).
  if (auto *th = root->getObject("thread")) {
    auto readUnsigned = [&](llvm::StringRef key, std::optional<unsigned> &dst) {
      if (auto v = th->getInteger(key)) {
        if (*v < 0) {
          llvm::errs() << "drcompiler warning: negative thread." << key << " in '"
                       << path << "'; skipping\n";
          return;
        }
        dst = static_cast<unsigned>(*v);
      } else if (th->get(key))
        llvm::errs() << "drcompiler warning: non-integer thread." << key << " in '"
                     << path << "'; skipping\n";
    };
    auto readNumber = [&](llvm::StringRef key, std::optional<double> &dst) {
      if (auto v = th->getNumber(key))
        dst = *v;
      else if (th->get(key))
        llvm::errs() << "drcompiler warning: non-numeric thread." << key << " in '"
                     << path << "'; skipping\n";
    };
    auto readBool = [&](llvm::StringRef key, std::optional<bool> &dst) {
      if (auto v = th->getBoolean(key))
        dst = *v;
      else if (th->get(key))
        llvm::errs() << "drcompiler warning: non-boolean thread." << key << " in '"
                     << path << "'; skipping\n";
    };
    readUnsigned("active_threads", m.thread.activeThreads);
    readUnsigned("smt_per_core", m.thread.smtPerCore);
    readNumber("dram_bytes_per_cycle", m.thread.dramBytesPerCycle);
    readNumber("llc_bytes_per_cycle", m.thread.llcBytesPerCycle);
    readBool("l1_shared", m.thread.l1Shared);
    readBool("l2_shared", m.thread.l2Shared);
    readBool("l3_shared", m.thread.l3Shared);
    readBool("exclusive", m.thread.exclusive);
  } else if (root->get("thread")) {
    llvm::errs() << "drcompiler warning: 'thread' in '" << path
                 << "' is not a JSON object; ignoring\n";
  }

  m.fromFile = true;
  LLVM_DEBUG(llvm::dbgs() << "DRCOMP-COST: Loaded CPU cost model from '"
                          << path << "' (" << m.table.size() << " ops)\n");
  return m;
}

unsigned CpuCostModel::opCost(mlir::Operation *op) const {
  llvm::StringRef name = op->getName().getStringRef();
  auto it = table.find(name);
  if (it != table.end())
    return it->second;
  return defaultCost;
}
