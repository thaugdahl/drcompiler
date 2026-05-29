//===- ArchHandler.h - Target-specific cost-model adapter ------------------===//
//
// An ArchHandler abstracts target-specific decisions for the unified cost
// model:
//   - MLIR type -> register class + count
//   - per-arch spill cost tuning
//   - per-arch constraint vetoes
//   - cycle-weight combination
//
// One handler per (triplet, vector width, feature set).  A 'generic' fallback
// always exists.  Concrete handlers (X86_64_AVX2, X86_64_AVX512, ARM_Neon,
// ...) live under `lib/Analysis/ArchHandlers/`.
//
//===----------------------------------------------------------------------===//

#ifndef DRCOMPILER_ANALYSIS_ARCHHANDLER_H
#define DRCOMPILER_ANALYSIS_ARCHHANDLER_H

#include "drcompiler/Analysis/RegisterClass.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/TargetParser/Triple.h"

#include <memory>
#include <optional>
#include <string>

namespace drcompiler {

/// Architecture-level tuning parameters consumed by the cost combiner and
/// per-arch handlers.  All fields are JSON-overridable.
struct ArchParams {
  llvm::Triple triple;
  unsigned vectorWidthBits = 128;

  // Weights for the unified cost combiner.
  double alphaMem = 1.0;
  double betaReg = 1.0;
  double gammaAlu = 1.0;
};

struct PressureResult;

class ArchHandler {
public:
  virtual ~ArchHandler() = default;

  /// Canonical short name used in JSON dispatch.
  virtual llvm::StringRef name() const = 0;

  /// Default ArchParams for this handler.  Callers (typically the JSON
  /// loader) may override fields.
  virtual ArchParams defaultParams() const = 0;

  /// Default RegisterParams for this handler.
  virtual RegisterParams defaultRegisters() const = 0;

  /// Map an MLIR type to register class + count for this arch.
  virtual RegClassRequirement classifyType(mlir::Type ty,
                                           const ArchParams &p) const = 0;

  /// Adjust an estimated spill cost (in cycles) for this arch / class.
  /// The default identity implementation is appropriate for most archs.
  virtual unsigned tuneSpillCost(unsigned baseCycles, RegClass cls,
                                 const ArchParams &p) const {
    (void)cls;
    (void)p;
    return baseCycles;
  }

  /// Optional arch-specific veto: returns a non-empty diagnostic string when
  /// the candidate pressure violates some hard constraint that should
  /// short-circuit the cost model (e.g. AVX-512 mask spills are unsupported
  /// in some configurations).  Default: never veto.
  virtual std::optional<std::string>
  checkConstraint(const PressureResult &pressure,
                  const RegisterParams &regs) const {
    (void)pressure;
    (void)regs;
    return std::nullopt;
  }

  /// Combine per-aspect cycle estimates into a single scalar cost.  Default
  /// is weighted-sum; per-arch handlers may override (e.g. AVX-512 mask
  /// spills could get a multiplier).
  virtual unsigned combineCosts(unsigned memCycles, unsigned regCycles,
                                unsigned aluCycles,
                                const ArchParams &p) const {
    double total = p.alphaMem * memCycles + p.betaReg * regCycles +
                   p.gammaAlu * aluCycles;
    if (total < 0.0)
      total = 0.0;
    return static_cast<unsigned>(total);
  }

  /// Factory: look up a registered handler by canonical name.  Returns the
  /// generic handler as fallback when `name` is empty or unknown; the
  /// caller may emit a diagnostic in the unknown case using
  /// `isKnown(name)`.
  static std::unique_ptr<ArchHandler> create(llvm::StringRef name);

  /// True iff `name` matches a registered handler.
  static bool isKnown(llvm::StringRef name);

  /// Convenience: pick a handler from an llvm::Triple when no explicit
  /// handler name is provided.  Used by --print-arch-handler when JSON
  /// omits "arch.handler".
  static llvm::StringRef pickHandlerForTriple(const llvm::Triple &t);
};

} // namespace drcompiler

#endif // DRCOMPILER_ANALYSIS_ARCHHANDLER_H
