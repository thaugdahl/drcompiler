//===- ARM_Neon.cpp - AArch64 with NEON (no SVE) ----------------------------===//
//
// 31 GP regs (x0..x30; x31 is SP/ZR), 32 NEON Q-regs (128-bit), no predicates.
// AArch64 has a dedicated FP register file aliased with NEON V-regs.
//
//===----------------------------------------------------------------------===//

#include "drcompiler/Analysis/ArchHandler.h"
#include "mlir/IR/BuiltinTypes.h"

#include <algorithm>

namespace drcompiler {

namespace {

class ARM_NeonHandler : public ArchHandler {
public:
  llvm::StringRef name() const override { return "arm-neon"; }

  ArchParams defaultParams() const override {
    ArchParams p;
    p.triple = llvm::Triple("aarch64-unknown-linux-gnu");
    p.vectorWidthBits = 128;
    // Neoverse N1/N2-class front end: 4-wide decode/rename.
    p.issueWidth = 4;
    return p;
  }

  RegisterParams defaultRegisters() const override {
    RegisterParams r;
    r.gpBudget = 31;
    r.fpBudget = 32;
    r.vecBudget = 32;
    r.predBudget = 0;
    r.spillReloadCycles = 5;
    r.spillStoreCycles = 1;
    return r;
  }

  RegClassRequirement classifyType(mlir::Type ty,
                                   const ArchParams &p) const override {
    if (auto vec = llvm::dyn_cast<mlir::VectorType>(ty)) {
      unsigned elemBits = bitsOf(vec.getElementType());
      uint64_t total = static_cast<uint64_t>(elemBits) * vec.getNumElements();
      unsigned width = std::max<unsigned>(p.vectorWidthBits, 1);
      unsigned count = static_cast<unsigned>((total + width - 1) / width);
      return {RegClass::Vec, std::max(count, 1u)};
    }
    if (llvm::isa<mlir::FloatType>(ty))
      return {RegClass::FP, 1};
    if (auto it = llvm::dyn_cast<mlir::IntegerType>(ty)) {
      if (it.getWidth() == 1)
        return {RegClass::Pred, 1};
      return {RegClass::GP, 1};
    }
    return {RegClass::GP, 1};
  }

private:
  static unsigned bitsOf(mlir::Type elem) {
    if (auto i = llvm::dyn_cast<mlir::IntegerType>(elem))
      return i.getWidth();
    if (auto f = llvm::dyn_cast<mlir::FloatType>(elem))
      return f.getWidth();
    return 32;
  }
};

} // namespace

std::unique_ptr<ArchHandler> makeARM_NeonHandler() {
  return std::make_unique<ARM_NeonHandler>();
}

} // namespace drcompiler
