//===- ARM_SVE.cpp - AArch64 with SVE / SVE2 --------------------------------===//
//
// 31 GP regs, 32 scalable Z-regs, 16 predicate regs (P0..P15, of which only
// P0..P7 are governing predicates for most instructions).
//
// The hard part of SVE for a compile-time cost model is that the vector length
// is NOT known when we make the decision: an implementation may pick any
// multiple of 128 bits up to 2048, and the same binary runs on all of them.
// This handler therefore models the ARCHITECTURAL FLOOR (128-bit VL) rather
// than guessing a part:
//
//   * Register PRESSURE is then never under-estimated: a value that needs
//     ceil(bits/128) Z-regs at VL=128 needs no more on a wider part, so a
//     candidate we accept here is still register-legal on a 256/512-bit
//     implementation.
//   * Vector THROUGHPUT is under-estimated on a wide part — a 512-bit
//     implementation retires 4x the work per Z-op.  Modeling that requires
//     either a `vscale`-parametric cost or a JSON that pins the deployment
//     target's VL: set `arch.vector_width_bits` (and, for register blocking,
//     `arch.vector_bits_native` / `_arch`) when the target VL is known.
//
// Not selected by triple inference: SVE is a feature, not an architecture, and
// `aarch64-*` alone does not imply it (see ArchHandlerRegistry::
// pickHandlerForTriple, which keeps `arm-neon` as the AArch64 family default).
// Ask for it explicitly with `arch.handler: "arm-sve"` or
// `--dr-arch-handler=arm-sve`.
//
//===----------------------------------------------------------------------===//

#include "drcompiler/Analysis/ArchHandler.h"
#include "mlir/IR/BuiltinTypes.h"

#include <algorithm>

namespace drcompiler {

namespace {

class ARM_SVEHandler : public ArchHandler {
public:
  llvm::StringRef name() const override { return "arm-sve"; }

  ArchParams defaultParams() const override {
    ArchParams p;
    p.triple = llvm::Triple("aarch64-unknown-linux-gnu");
    // Architectural minimum VL — the only width every SVE implementation is
    // guaranteed to have.  Override from JSON when the deployment VL is known.
    p.vectorWidthBits = 128;
    // Neoverse V1/V2-class front end, kept at the conservative 4 shared with
    // the NEON handler rather than claiming a specific part's rename width.
    p.issueWidth = 4;
    return p;
  }

  RegisterParams defaultRegisters() const override {
    RegisterParams r;
    r.gpBudget = 31; // x0..x30
    r.fpBudget = 32; // scalar FP aliases Z0..Z31's low bits
    r.vecBudget = 32;
    // Unlike NEON, SVE has a real predicate file; masks live there instead of
    // burning a Z-reg, and spilling one is a distinct (cheap) operation.
    r.predBudget = 16;
    r.spillReloadCycles = 5;
    r.spillStoreCycles = 1;
    return r;
  }

  RegClassRequirement classifyType(mlir::Type ty,
                                   const ArchParams &p) const override {
    if (auto vec = llvm::dyn_cast<mlir::VectorType>(ty)) {
      // i1 vectors are masks: they occupy the predicate file, not a Z-reg.
      if (auto elemInt = llvm::dyn_cast<mlir::IntegerType>(vec.getElementType());
          elemInt && elemInt.getWidth() == 1)
        return {RegClass::Pred, 1};
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

std::unique_ptr<ArchHandler> makeARM_SVEHandler() {
  return std::make_unique<ARM_SVEHandler>();
}

} // namespace drcompiler
