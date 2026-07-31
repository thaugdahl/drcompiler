//===- X86_64_AVX512.cpp - x86-64 with AVX-512 ------------------------------===//
//
// 16 GP regs, 32 ZMM regs (512-bit), 8 k-mask regs.
// Mask spills are notably expensive: kmov k,m / kmov m,k go through the
// general path and are not as cheap as vector loads/stores.
//
//===----------------------------------------------------------------------===//

#include "drcompiler/Analysis/ArchHandler.h"
#include "mlir/IR/BuiltinTypes.h"

#include <algorithm>

namespace drcompiler {

namespace {

class X86_64_AVX512Handler : public ArchHandler {
public:
  llvm::StringRef name() const override { return "x86-64-avx512"; }

  ArchParams defaultParams() const override {
    ArchParams p;
    p.triple = llvm::Triple("x86_64-unknown-linux-gnu");
    p.vectorWidthBits = 512;
    // 4 ops/cycle — see X86_64_AVX2.cpp: the validated x86 value, kept so this
    // parameterization is byte-identical on every machine the paper reports.
    p.issueWidth = 4;
    return p;
  }

  RegisterParams defaultRegisters() const override {
    RegisterParams r;
    r.gpBudget = 16;
    r.fpBudget = 32; // scalar FP can use any of 32 zmm
    r.vecBudget = 32;
    r.predBudget = 8;
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

  unsigned tuneSpillCost(unsigned baseCycles, RegClass cls,
                         const ArchParams &p) const override {
    (void)p;
    // Mask spills go through GPR scratch — model as 1.5x the base spill cost.
    if (cls == RegClass::Pred)
      return baseCycles + baseCycles / 2;
    return baseCycles;
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

std::unique_ptr<ArchHandler> makeX86_64_AVX512Handler() {
  return std::make_unique<X86_64_AVX512Handler>();
}

} // namespace drcompiler
