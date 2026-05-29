//===- PrintRegisterPressure.cpp - Phase 2 diagnostic pass -----------------===//

#include "drcompiler/Transforms/PrintRegisterPressure.h"

#include "drcompiler/Analysis/ArchHandler.h"
#include "drcompiler/Analysis/RegisterPressureAnalysis.h"
#include "drcompiler/Analysis/SpillStrategy.h"
#include "drcompiler/Transforms/CpuCostModel.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
#define GEN_PASS_DEF_PRINTREGISTERPRESSUREPASS
#include "drcompiler/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace drcompiler;

namespace {

/// Resolve the arch handler + ArchParams + RegisterParams from the JSON file
/// (and CLI overrides), with the same precedence as PrintArchHandler.
struct ResolvedConfig {
  std::unique_ptr<ArchHandler> handler;
  ArchParams archParams;
  RegisterParams regParams;
  SpillStrategy strategy = SpillStrategy::ExcessHot;
};

ResolvedConfig resolveConfig(llvm::StringRef jsonPath,
                             llvm::StringRef handlerOverride,
                             llvm::StringRef strategyOverride) {
  CpuCostModel cm = jsonPath.empty() ? CpuCostModel::getDefault()
                                     : CpuCostModel::loadFromFile(jsonPath);
  const CpuArchJsonParams &archJson = cm.archParams();
  const CpuRegisterJsonParams &regsJson = cm.registerParams();

  std::string handlerName;
  if (!handlerOverride.empty())
    handlerName = handlerOverride.str();
  else if (archJson.handler)
    handlerName = *archJson.handler;
  else if (archJson.triplet)
    handlerName =
        ArchHandler::pickHandlerForTriple(llvm::Triple(*archJson.triplet))
            .str();
  else
    handlerName = "generic";

  ResolvedConfig cfg;
  cfg.handler = ArchHandler::create(handlerName);
  cfg.archParams = cfg.handler->defaultParams();
  cfg.regParams = cfg.handler->defaultRegisters();

  if (archJson.triplet)
    cfg.archParams.triple = llvm::Triple(*archJson.triplet);
  if (archJson.vectorWidthBits)
    cfg.archParams.vectorWidthBits = *archJson.vectorWidthBits;
  if (archJson.alphaMem)
    cfg.archParams.alphaMem = *archJson.alphaMem;
  if (archJson.betaReg)
    cfg.archParams.betaReg = *archJson.betaReg;
  if (archJson.gammaAlu)
    cfg.archParams.gammaAlu = *archJson.gammaAlu;

  if (regsJson.gpBudget)
    cfg.regParams.gpBudget = *regsJson.gpBudget;
  if (regsJson.fpBudget)
    cfg.regParams.fpBudget = *regsJson.fpBudget;
  if (regsJson.vecBudget)
    cfg.regParams.vecBudget = *regsJson.vecBudget;
  if (regsJson.predBudget)
    cfg.regParams.predBudget = *regsJson.predBudget;
  if (regsJson.spillReloadCycles)
    cfg.regParams.spillReloadCycles = *regsJson.spillReloadCycles;
  if (regsJson.spillStoreCycles)
    cfg.regParams.spillStoreCycles = *regsJson.spillStoreCycles;

  if (!strategyOverride.empty())
    cfg.strategy = parseSpillStrategy(strategyOverride);
  else if (archJson.spillStrategy)
    cfg.strategy = parseSpillStrategy(*archJson.spillStrategy);

  return cfg;
}

std::string formatSummary(StringRef funcName, const PressureResult &res,
                          SpillStrategy strategy) {
  std::string buf;
  llvm::raw_string_ostream os(buf);
  os << "register-pressure: " << funcName
     << " strategy=" << spillStrategyName(strategy)
     << " peak=(gp=" << res.peakLive[0] << ",fp=" << res.peakLive[1]
     << ",vec=" << res.peakLive[2] << ",pred=" << res.peakLive[3] << ")"
     << " excess=(gp=" << res.totalExcess[0] << ",fp=" << res.totalExcess[1]
     << ",vec=" << res.totalExcess[2] << ",pred=" << res.totalExcess[3] << ")"
     << " spill_cycles=" << res.totalSpillCycles;
  return buf;
}

struct PrintRegisterPressurePass
    : public impl::PrintRegisterPressurePassBase<PrintRegisterPressurePass> {
  using impl::PrintRegisterPressurePassBase<
      PrintRegisterPressurePass>::PrintRegisterPressurePassBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    ResolvedConfig cfg =
        resolveConfig(cpuCostModelFile, handlerOverride, spillStrategyOpt);

    PressureQuery opts;
    opts.params = cfg.regParams;
    opts.strategy = cfg.strategy;
    opts.tripCount = tripCount;
    opts.recordPerOp = showPerOp;

    mod.walk([&](FunctionOpInterface fn) {
      if (fn.empty())
        return;
      RegisterPressureAnalysis analysis(fn.getOperation());
      PressureResult res = analysis.query(fn.getFunctionBody(), *cfg.handler,
                                          cfg.archParams, opts);
      fn->emitRemark(formatSummary(fn.getName(), res, cfg.strategy));
      if (showPerOp) {
        for (auto &kv : res.liveAtOp) {
          std::string buf;
          llvm::raw_string_ostream os(buf);
          os << "register-pressure-op: live=(gp=" << kv.second[0]
             << ",fp=" << kv.second[1] << ",vec=" << kv.second[2]
             << ",pred=" << kv.second[3] << ")";
          kv.first->emitRemark(buf);
        }
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createPrintRegisterPressurePass() {
  return std::make_unique<PrintRegisterPressurePass>();
}
