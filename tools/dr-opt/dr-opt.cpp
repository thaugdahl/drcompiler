//===- dr-opt.cpp - mlir-opt with DataRecomputation pass ------------------===//
//
// A thin wrapper around mlir-opt that registers the DataRecomputation pass
// alongside all upstream MLIR dialects and passes.
//
//===----------------------------------------------------------------------===//

#include "drcompiler/Transforms/Passes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();

  // Register the DataRecomputation pass.
  mlir::registerDRCompPassesPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "DRComp optimizer driver\n", registry));
}
