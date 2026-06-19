//===- LowerKrnlGlobal.cpp - krnl.global -> memref.global (krnl-free) ---===//
//
// onnx-mlir's monolithic `convert-krnl-to-llvm` rejects external parallel
// constructs (omp / scf.parallel), so the whole-kernel SPMD path (par->omp)
// cannot lower through it.  This pass strips the only krnl ops that survive
// into the affine stage so the rest of the kernel can be lowered to LLVM by
// HOST mlir-opt (keeping omp intact, then --convert-openmp-to-llvm):
//
//   "krnl.global"() {name, shape, value} : () -> memref<...>
//       =>  memref.global "private" constant @<name> : memref<...> = <value>
//           %g = memref.get_global @<name> : memref<...>
//   "krnl.entry_point"() ...   =>  erased (the run_main_graph wrapper is
//       replaced by a hand-written C harness calling _mlir_ciface_main_graph).
//
// krnl ops are unregistered here (dr-opt runs with -allow-unregistered-dialect),
// so they are matched by op name.  Diagnostic / opt-in; no effect on modules
// without krnl ops.
//
//===----------------------------------------------------------------------===//

#include "drcompiler/Transforms/LowerKrnlGlobal.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
#define GEN_PASS_DEF_LOWERKRNLGLOBALPASS
#include "drcompiler/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

struct LowerKrnlGlobalPass
    : public impl::LowerKrnlGlobalPassBase<LowerKrnlGlobalPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder modBuilder(module.getBody(), module.getBody()->begin());
    unsigned counter = 0;

    SmallVector<Operation *> globals, entryPoints;
    module.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "krnl.global")
        globals.push_back(op);
      else if (name == "krnl.entry_point")
        entryPoints.push_back(op);
    });

    for (Operation *op : globals) {
      auto memrefTy = dyn_cast<MemRefType>(op->getResult(0).getType());
      Attribute value = op->getAttr("value");
      if (!memrefTy || !value) // not the shape we expect: leave it (will error
        continue;              // downstream, surfacing the unhandled case)

      // A unique, valid symbol name (prefer the krnl 'name' attr).
      std::string sym = "krnlg_" + std::to_string(counter++);
      if (auto nameAttr = op->getAttrOfType<StringAttr>("name"))
        sym = "krnlg_" + nameAttr.getValue().str();

      modBuilder.setInsertionPointToStart(module.getBody());
      modBuilder.create<memref::GlobalOp>(
          op->getLoc(), modBuilder.getStringAttr(sym),
          /*sym_visibility=*/modBuilder.getStringAttr("private"),
          /*type=*/TypeAttr::get(memrefTy),
          /*initial_value=*/value,
          /*constant=*/modBuilder.getUnitAttr(),
          /*alignment=*/IntegerAttr());

      OpBuilder b(op);
      auto get = b.create<memref::GetGlobalOp>(op->getLoc(), memrefTy, sym);
      op->getResult(0).replaceAllUsesWith(get.getResult());
      op->erase();
    }
    for (Operation *op : entryPoints)
      op->erase();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createLowerKrnlGlobalPass() {
  return std::make_unique<LowerKrnlGlobalPass>();
}
