//===- AppleMSeries.cpp - Apple M-series (aarch64-darwin) -------------------===//
//
// Apple M1/M2/M3 performance core.  Same architectural register file as any
// AArch64 part (31 GP, 32 128-bit V-regs, no SVE predicates), but two things
// differ from the Neoverse-shaped `arm-neon` handler and both matter to the
// cost model:
//
//   1. A much wider front end (~8-wide decode/rename vs 4 on Neoverse N1/N2),
//      so the THROUGHPUT floor of a wide independent recompute cone is half
//      what the generic superscalar assumption prices it at.  This is the
//      whole reason the issue width became a per-handler field instead of the
//      `kIssueWidth = 4` literal in CacheCostModel.cpp.
//
//   2. A 128-byte cache line (all other supported parts are 64).  That lives
//      in the MEMORY hierarchy, not in ArchParams: pass `cache.cache_line:
//      128` in the cost-model JSON (`--cpu-cost-model-file`) together with the
//      probed L1/L2/SLC sizes and latencies.  The handler deliberately does
//      not smuggle cache geometry in — one machine, one description
//      (MachineModel.h, resolution contract).
//
// AMX / the matrix coprocessor is NOT modeled: it is not reachable from the
// vector IR this compiler emits, so an accelerator-aware handler would price
// a datapath the code generator never targets.
//
//===----------------------------------------------------------------------===//

#include "drcompiler/Analysis/ArchHandler.h"
#include "mlir/IR/BuiltinTypes.h"

#include <algorithm>

namespace drcompiler {

namespace {

class AppleMSeriesHandler : public ArchHandler {
public:
  llvm::StringRef name() const override { return "apple-m-series"; }

  ArchParams defaultParams() const override {
    ArchParams p;
    p.triple = llvm::Triple("arm64-apple-macosx");
    p.vectorWidthBits = 128; // NEON; no SVE on any shipping M-series part
    // ~8-wide decode/rename on the P-cores (M1 "Firestorm" onward).  The
    // recompute cone of a wide independent expression retires in half the
    // slots a 4-wide part needs, so recomputation is correspondingly cheaper.
    p.issueWidth = 8;
    return p;
  }

  RegisterParams defaultRegisters() const override {
    RegisterParams r;
    r.gpBudget = 31; // x0..x30 (x31 is SP/ZR)
    r.fpBudget = 32; // scalar FP aliases the V-regs
    r.vecBudget = 32;
    r.predBudget = 0; // no predicate file without SVE
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

std::unique_ptr<ArchHandler> makeAppleMSeriesHandler() {
  return std::make_unique<AppleMSeriesHandler>();
}

} // namespace drcompiler
