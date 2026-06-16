#ifndef DRCOMPILER_ANALYSIS_CPUCOSTMODEL_H
#define DRCOMPILER_ANALYSIS_CPUCOSTMODEL_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>

namespace drcompiler {

/// Optional cache hierarchy parameters parsed from a cost-model JSON's
/// `cache` object.  Each field is std::nullopt unless present in the file.
/// Consumers should fall back to CLI flags / built-in defaults for any
/// field left unset.
struct CpuCacheJsonParams {
  std::optional<unsigned> l1Size;
  std::optional<unsigned> l2Size;
  std::optional<unsigned> l3Size;
  std::optional<unsigned> l1Latency;
  std::optional<unsigned> l2Latency;
  std::optional<unsigned> l3Latency;
  std::optional<unsigned> memLatency;
  std::optional<unsigned> pageSize;      // NEW (v4): TLB-reach modeling
  std::optional<unsigned> l2TlbEntries;  // NEW (v4): TLB-reach modeling
  std::optional<unsigned> cacheLine;     // NEW (v4): unify w/ MachineModel
  std::optional<unsigned> llcSharers;    // NEW (v4): unify w/ MachineModel
};

/// Optional architecture-block parameters parsed from a cost-model JSON's
/// `arch` object.  Each field is std::nullopt unless present in the file.
struct CpuArchJsonParams {
  std::optional<std::string> triplet;
  std::optional<std::string> handler;
  std::optional<unsigned> vectorWidthBits;
  std::optional<std::string> spillStrategy;
  std::optional<double> alphaMem;
  std::optional<double> betaReg;
  std::optional<double> gammaAlu;
  // Vector-execution model (WP-G1: portable register-block VL selection).
  // Present only when the JSON's `arch` block sets them; absence keeps the
  // built-in (Zen4) MachineModel defaults and the static vl option default.
  std::optional<unsigned> vectorBitsNative; // throughput-effective FP datapath
  std::optional<unsigned> vectorBitsArch;   // widest ISA-encodable vector
  std::optional<unsigned> vecRegBudget;     // vector regs usable for accumulators
  std::optional<double> avx512FreqThrottle; // Intel AVX-512 license downclock
  // Compute-roofline arm (WP-T2: transformer GEMM kernel-kind dispatch).  FP FMA
  // issue pipes; absence keeps the compute arm inert (BW-only, as today).
  std::optional<unsigned> fmaUnits;
};

/// Optional thread / parallel-execution parameters parsed from a cost-model
/// JSON's `thread` object (CROSSCUTTING.md III).  Absence keeps the 1-thread,
/// bandwidth-disabled MachineModel defaults (byte-identical).
struct CpuThreadJsonParams {
  std::optional<unsigned> activeThreads;
  std::optional<unsigned> smtPerCore;
  std::optional<double> dramBytesPerCycle;
  std::optional<double> llcBytesPerCycle;
  std::optional<bool> l1Shared;
  std::optional<bool> l2Shared;
  std::optional<bool> l3Shared;
  std::optional<bool> exclusive; // workload mode (III.4a)
};

/// Optional register-block parameters parsed from a cost-model JSON's
/// `registers` object.  Each field is std::nullopt unless present.
struct CpuRegisterJsonParams {
  std::optional<unsigned> gpBudget;
  std::optional<unsigned> fpBudget;
  std::optional<unsigned> vecBudget;
  std::optional<unsigned> predBudget;
  std::optional<unsigned> spillReloadCycles;
  std::optional<unsigned> spillStoreCycles;
};

/// A table mapping MLIR operation names (e.g. "arith.addi") to estimated
/// cycle costs.  Loaded from a JSON file or populated with built-in defaults.
class CpuCostModel {
public:
  /// Load a cost model from a JSON file.  On success, returns the model.
  /// On failure, logs a warning and returns a model with built-in defaults.
  static CpuCostModel loadFromFile(llvm::StringRef path);

  /// Return a model with the built-in default cost table.
  static CpuCostModel getDefault();

  /// Look up the cycle cost of a single operation.
  unsigned opCost(mlir::Operation *op) const;

  /// Threshold queries used by passes (MemoryFission today; eligible
  /// for tuning per cost-model JSON in the future).
  ///
  /// `minChainCost`    — sum of recompute costs (cycles) below which a
  ///                     candidate chain is not worth materialising into a
  ///                     buffer. 15 ≈ one int division on x86.
  /// `minConsumerCost` — minimum opCost of a single follow-on user that
  ///                     justifies extending an expensive-op tip across
  ///                     that user (e.g. sqrt → div).
  unsigned minChainCost() const { return 15; }
  unsigned minConsumerCost() const { return 10; }

  /// Whether this model was loaded from a file (vs. built-in defaults).
  bool isFromFile() const { return fromFile; }

  /// Optional cache hierarchy parameters parsed from the JSON file.  Fields
  /// are nullopt unless they were present and well-formed.
  const CpuCacheJsonParams &cacheParams() const { return cache; }

  /// Optional `arch` block parameters parsed from the JSON file.
  const CpuArchJsonParams &archParams() const { return arch; }

  /// Optional `registers` block parameters parsed from the JSON file.
  const CpuRegisterJsonParams &registerParams() const { return registers; }

  /// Optional `thread` block parameters parsed from the JSON file.
  const CpuThreadJsonParams &threadParams() const { return thread; }

private:
  llvm::StringMap<unsigned> table;
  unsigned defaultCost = 5;
  bool fromFile = false;
  CpuCacheJsonParams cache;
  CpuArchJsonParams arch;
  CpuRegisterJsonParams registers;
  CpuThreadJsonParams thread;

  void populateDefaults();
};

} // namespace drcompiler

#endif // DRCOMPILER_ANALYSIS_CPUCOSTMODEL_H
