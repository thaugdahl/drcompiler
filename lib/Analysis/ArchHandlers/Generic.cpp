//===- Generic.cpp - Architecture-agnostic fallback handler ----------------===//
//
// A conservative, target-independent ArchHandler. Used when:
//   - costs.json omits "arch.handler"
//   - the JSON names a handler we don't recognise
//   - the user asks for it explicitly (handler == "generic")
//
// Classification rules:
//   - vector<...>  -> Vec, count = ceil(bitWidth / vectorWidthBits)
//   - f16/f32/f64  -> FP,  count = 1   (separate from Vec per the FP-class decision)
//   - i1           -> Pred, count = 1
//   - integer/index/memref/pointer -> GP, count = 1
//
//===----------------------------------------------------------------------===//

#include "drcompiler/Analysis/ArchHandler.h"

#include "mlir/IR/BuiltinTypes.h"

#include <algorithm>

namespace drcompiler {

namespace {

class GenericArchHandler : public ArchHandler {
public:
  llvm::StringRef name() const override { return "generic"; }

  ArchParams defaultParams() const override {
    ArchParams p;
    p.vectorWidthBits = 128;
    return p;
  }

  RegisterParams defaultRegisters() const override {
    RegisterParams r;
    r.gpBudget = 16;
    r.fpBudget = 16;
    r.vecBudget = 16;
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

std::unique_ptr<ArchHandler> makeGenericArchHandler() {
  return std::make_unique<GenericArchHandler>();
}

} // namespace drcompiler
