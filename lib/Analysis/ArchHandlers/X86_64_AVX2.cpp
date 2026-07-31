//===- X86_64_AVX2.cpp - x86-64 with AVX2 -----------------------------------===//
//
// 16 GP regs, 16 YMM regs (256-bit), no mask regs.
// FP scalars use the low lane of XMM/YMM but are counted in the separate FP
// class (per the FP-class decision); for spill cost purposes, the FP class
// shares a load-store path with Vec on this target.
//
//===----------------------------------------------------------------------===//

#include "drcompiler/Analysis/ArchHandler.h"
#include "mlir/IR/BuiltinTypes.h"

#include <algorithm>

namespace drcompiler {

namespace {

class X86_64_AVX2Handler : public ArchHandler {
public:
  llvm::StringRef name() const override { return "x86-64-avx2"; }

  ArchParams defaultParams() const override {
    ArchParams p;
    p.triple = llvm::Triple("x86_64-unknown-linux-gnu");
    p.vectorWidthBits = 256;
    // 4 ops/cycle: the value every reported x86 result was produced with.
    // Wider dispatch is plausible (Zen4 dispatches 6 macro-ops) but unmeasured
    // here, and raising it re-prices every keep-vs-recompute decision — that
    // needs its own validation run, not a silent default bump.
    p.issueWidth = 4;
    return p;
  }

  RegisterParams defaultRegisters() const override {
    RegisterParams r;
    r.gpBudget = 16;
    r.fpBudget = 16; // shares xmm/ymm physical file
    r.vecBudget = 16;
    r.predBudget = 0;
    r.spillReloadCycles = 5; // L1-resident spill slot
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

std::unique_ptr<ArchHandler> makeX86_64_AVX2Handler() {
  return std::make_unique<X86_64_AVX2Handler>();
}

} // namespace drcompiler
