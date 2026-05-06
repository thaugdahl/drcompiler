//===- ConstantGlobalFold.cpp - Strategy 0 -------------------------------===//
#include "drcompiler/Transforms/DataRecomputation/Strategies/ConstantGlobalFold.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace dr::strategies {

void runConstantGlobalFold(mlir::ModuleOp moduleOp,
                           mlir::SymbolTableCollection &symTab,
                           bool emitDiagnostics, bool summaryEnabled) {
  llvm::SmallVector<mlir::Operation *, 8> toErase;

  moduleOp.walk([&](mlir::Operation *op) {
    mlir::Value memref;
    mlir::ValueRange rawIndices;
    if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(op)) {
      memref = load.getMemref();
      rawIndices = load.getIndices();
    } else if (auto load = mlir::dyn_cast<mlir::affine::AffineLoadOp>(op)) {
      memref = load.getMemref();
      rawIndices = load.getIndices();
    } else {
      return;
    }
    auto gg = memref.getDefiningOp<mlir::memref::GetGlobalOp>();
    if (!gg) return;
    auto *symOp =
        symTab.lookupNearestSymbolFrom(moduleOp, gg.getNameAttr());
    auto globalOp = mlir::dyn_cast_or_null<mlir::memref::GlobalOp>(symOp);
    if (!globalOp || !globalOp.getConstant()) return;
    auto initAttrOpt = globalOp.getInitialValue();
    if (!initAttrOpt) return;
    auto memrefTy = mlir::cast<mlir::MemRefType>(globalOp.getType());
    mlir::Type elemTy = memrefTy.getElementType();
    mlir::Attribute valAttr;
    if (auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(*initAttrOpt)) {
      if (!dense.isSplat()) return;
      if (auto fp = mlir::dyn_cast<mlir::DenseFPElementsAttr>(dense)) {
        valAttr =
            mlir::FloatAttr::get(elemTy, fp.getSplatValue<llvm::APFloat>());
      } else if (auto i =
                     mlir::dyn_cast<mlir::DenseIntElementsAttr>(dense)) {
        valAttr =
            mlir::IntegerAttr::get(elemTy, i.getSplatValue<llvm::APInt>());
      } else {
        return;
      }
    } else {
      return;
    }
    mlir::OpBuilder builder(op);
    auto cst = builder.create<mlir::arith::ConstantOp>(
        op->getLoc(), elemTy, mlir::cast<mlir::TypedAttr>(valAttr));
    op->getResult(0).replaceAllUsesWith(cst);
    if (emitDiagnostics)
      op->emitRemark() << "constant-global-fold: ACCEPT";
    if (summaryEnabled) {
      llvm::errs() << "DRSUM: load ";
      op->getLoc().print(llvm::errs());
      llvm::errs() << ": constant-global-fold ACCEPT\n";
    }
    toErase.push_back(op);
    (void)rawIndices;
  });
  for (mlir::Operation *op : toErase) op->erase();
}

} // namespace dr::strategies
