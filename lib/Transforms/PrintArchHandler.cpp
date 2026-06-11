//===- PrintArchHandler.cpp - Phase 1 diagnostic pass ----------------------===//
//
// Loads the cost-model JSON, picks an ArchHandler, and emits a remark
// describing the chosen handler + parameters.  Exit criterion for Phase 1
// of the register-pressure plan: `dr-opt --print-arch-handler` shows the
// correct handler for each test JSON.
//
//===----------------------------------------------------------------------===//

#include "drcompiler/Transforms/PrintArchHandler.h"

#include "drcompiler/Analysis/ArchHandler.h"
#include "drcompiler/Analysis/CpuCostModel.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
#define GEN_PASS_DEF_PRINTARCHHANDLERPASS
#include "drcompiler/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace drcompiler;

namespace {

struct PrintArchHandlerPass
    : public impl::PrintArchHandlerPassBase<PrintArchHandlerPass> {
  using impl::PrintArchHandlerPassBase<
      PrintArchHandlerPass>::PrintArchHandlerPassBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // Load (optional) JSON cost model.
    CpuCostModel cm = cpuCostModelFile.empty()
                          ? CpuCostModel::getDefault()
                          : CpuCostModel::loadFromFile(cpuCostModelFile);
    const CpuArchJsonParams &archJson = cm.archParams();
    const CpuRegisterJsonParams &regsJson = cm.registerParams();

    // Pick handler: explicit override > JSON handler > triple-derived >
    // generic.
    std::string handlerName;
    bool jsonNamedUnknown = false;
    if (!handlerOverride.empty()) {
      handlerName = handlerOverride;
    } else if (archJson.handler) {
      handlerName = *archJson.handler;
      if (!ArchHandler::isKnown(handlerName))
        jsonNamedUnknown = true;
    } else if (archJson.triplet) {
      llvm::Triple t(*archJson.triplet);
      handlerName = ArchHandler::pickHandlerForTriple(t).str();
    } else {
      handlerName = "generic";
    }

    auto handler = ArchHandler::create(handlerName);
    ArchParams params = handler->defaultParams();
    RegisterParams regs = handler->defaultRegisters();

    // Layer JSON overrides on top of handler defaults.
    if (archJson.triplet)
      params.triple = llvm::Triple(*archJson.triplet);
    if (archJson.vectorWidthBits)
      params.vectorWidthBits = *archJson.vectorWidthBits;
    if (archJson.alphaMem)
      params.alphaMem = *archJson.alphaMem;
    if (archJson.betaReg)
      params.betaReg = *archJson.betaReg;
    if (archJson.gammaAlu)
      params.gammaAlu = *archJson.gammaAlu;

    if (regsJson.gpBudget)
      regs.gpBudget = *regsJson.gpBudget;
    if (regsJson.fpBudget)
      regs.fpBudget = *regsJson.fpBudget;
    if (regsJson.vecBudget)
      regs.vecBudget = *regsJson.vecBudget;
    if (regsJson.predBudget)
      regs.predBudget = *regsJson.predBudget;
    if (regsJson.spillReloadCycles)
      regs.spillReloadCycles = *regsJson.spillReloadCycles;
    if (regsJson.spillStoreCycles)
      regs.spillStoreCycles = *regsJson.spillStoreCycles;

    // Emit a single-line remark for FileCheck.
    std::string buf;
    llvm::raw_string_ostream os(buf);
    os << "arch-handler: " << handler->name()
       << " vec_width=" << params.vectorWidthBits
       << " gp=" << regs.gpBudget << " fp=" << regs.fpBudget
       << " vec=" << regs.vecBudget << " pred=" << regs.predBudget
       << " spill_reload=" << regs.spillReloadCycles
       << " spill_store=" << regs.spillStoreCycles
       << " weights=(" << params.alphaMem << "," << params.betaReg << ","
       << params.gammaAlu << ")";
    if (jsonNamedUnknown)
      os << " json_handler_unknown=" << *archJson.handler;
    mod.emitRemark(buf);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createPrintArchHandlerPass() {
  return std::make_unique<PrintArchHandlerPass>();
}
