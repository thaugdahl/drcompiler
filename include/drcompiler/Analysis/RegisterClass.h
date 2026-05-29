//===- RegisterClass.h - Register class taxonomy --------------------------===//
//
// Distinct register file partitions tracked by the pressure model.
//
//   GP   : general-purpose integer / pointer registers
//   FP   : scalar floating-point registers (x87 / SSE-scalar / NEON-FP)
//   Vec  : packed-vector registers (SSE/AVX/AVX-512 / NEON-V / SVE)
//   Pred : predicate / mask registers (AVX-512 k0..k7, SVE p0..p15)
//
// FP is kept distinct from Vec on architectures where the budgets, spill
// costs, or mov-encodings differ enough to matter; per-arch handlers may
// fold FP into Vec by classifying scalar floats into the Vec class.
//
//===----------------------------------------------------------------------===//

#ifndef DRCOMPILER_ANALYSIS_REGISTERCLASS_H
#define DRCOMPILER_ANALYSIS_REGISTERCLASS_H

#include "llvm/ADT/StringRef.h"

#include <array>
#include <cstdint>

namespace drcompiler {

enum class RegClass : uint8_t {
  GP = 0,
  FP = 1,
  Vec = 2,
  Pred = 3,
};

inline constexpr unsigned kNumRegClasses = 4;

inline constexpr std::array<RegClass, kNumRegClasses> allRegClasses() {
  return {RegClass::GP, RegClass::FP, RegClass::Vec, RegClass::Pred};
}

inline llvm::StringRef regClassName(RegClass cls) {
  switch (cls) {
  case RegClass::GP:
    return "gp";
  case RegClass::FP:
    return "fp";
  case RegClass::Vec:
    return "vec";
  case RegClass::Pred:
    return "pred";
  }
  return "unknown";
}

/// Result of classifying an SSA value's type into register-file
/// requirements. `count` covers the case where one MLIR value needs more
/// than one physical register (e.g. `vector<16xf32>` on AVX2 with 256-bit
/// regs → 2 vec regs).
struct RegClassRequirement {
  RegClass cls = RegClass::GP;
  unsigned count = 1;
};

/// Per-architecture register file budgets.  Zero means "no such class on
/// this target" (e.g. Pred is 0 outside AVX-512 / SVE).
struct RegisterParams {
  unsigned gpBudget = 16;
  unsigned fpBudget = 16;
  unsigned vecBudget = 16;
  unsigned predBudget = 0;

  unsigned spillReloadCycles = 5;
  unsigned spillStoreCycles = 1;

  unsigned budgetFor(RegClass cls) const {
    switch (cls) {
    case RegClass::GP:
      return gpBudget;
    case RegClass::FP:
      return fpBudget;
    case RegClass::Vec:
      return vecBudget;
    case RegClass::Pred:
      return predBudget;
    }
    return 0;
  }
};

} // namespace drcompiler

#endif // DRCOMPILER_ANALYSIS_REGISTERCLASS_H
