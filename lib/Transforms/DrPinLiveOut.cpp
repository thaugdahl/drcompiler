//===- DrPinLiveOut.cpp - Pin live-out memrefs against DCE ---------------===//
//
// The PolyBench MLIR harness suppresses polybench_prevent_dce (cgeist cannot
// lower the argc/strcmp guard or print_array's fprintf), so kernel outputs are
// never observed and an aggressive backend (clang -O3 -march=native) deletes
// the producing computation.  This pass restores observation INSIDE the IR:
// after the kernel's writes (before any dealloc / the return), it reads every
// element of each live-out memref and feeds it to an opaque external sink
// (@__dr_observe).  Because the sink is opaque, every load — and therefore
// every store that produced it — must stay live.
//
// Run AFTER cgeist has preserved the kernel (i.e. cgeist -O0) and after the
// DataRecomputation / register-block passes, before lowering.  The sink must be
// linked in as an opaque function, e.g.  void __dr_observe(double){}.
//
//===----------------------------------------------------------------------===//

#include "drcompiler/Transforms/DrPinLiveOut.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "dr-pin-liveout"

namespace mlir {
#define GEN_PASS_DEF_DRPINLIVEOUTPASS
#include "drcompiler/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

struct DrPinLiveOutPass
    : public impl::DrPinLiveOutPassBase<DrPinLiveOutPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    Type f64 = Float64Type::get(ctx);

    // Declare the opaque external sink:  func private @__dr_observe(f64)
    const char *kSink = "__dr_observe";
    func::FuncOp sink = module.lookupSymbol<func::FuncOp>(kSink);
    if (!sink) {
      OpBuilder mb(ctx);
      mb.setInsertionPointToStart(module.getBody());
      sink = mb.create<func::FuncOp>(module.getLoc(), kSink,
                                     FunctionType::get(ctx, {f64}, {}));
      sink.setPrivate();
    }

    module.walk([&](func::FuncOp fn) {
      if (fn.isExternal() || fn == sink)
        return;
      pinFunc(fn, sink, f64);
    });
  }

private:
  static bool isFloatMemRef(Value v) {
    auto mt = dyn_cast<MemRefType>(v.getType());
    return mt && (mt.getElementType().isF64() || mt.getElementType().isF32());
  }

  void pinFunc(func::FuncOp fn, func::FuncOp sink, Type f64) {
    if (fn.getBody().empty())
      return;

    // Live-out candidates: memref function arguments + memref.alloc results
    // (f32/f64).  Reading inputs/scratch too is harmless and keeps the code
    // simple; the output is what we must not lose.
    SmallVector<Value> targets;
    for (Value arg : fn.getBody().front().getArguments())
      if (isFloatMemRef(arg))
        targets.push_back(arg);
    fn.walk([&](memref::AllocOp a) {
      if (isFloatMemRef(a.getResult()))
        targets.push_back(a.getResult());
    });
    if (targets.empty())
      return;

    // Insert after all writes but while the buffers are still live: before the
    // first dealloc, else before the first return.
    Operation *ip = nullptr;
    fn.walk([&](memref::DeallocOp d) {
      if (!ip)
        ip = d.getOperation();
    });
    if (!ip)
      fn.walk([&](func::ReturnOp r) {
        if (!ip)
          ip = r.getOperation();
      });
    if (!ip)
      return;

    // Only pin allocs that dominate the insertion point: transform passes
    // (e.g. fusion) may create scratch buffers inside loop bodies, which are
    // dead and out of scope at the pin site.
    DominanceInfo dom(fn);
    OpBuilder b(ip);
    for (Value m : targets) {
      if (Operation *def = m.getDefiningOp())
        if (!dom.properlyDominates(def, ip))
          continue;
      emitPin(b, ip->getLoc(), m, sink, f64);
    }
  }

  void emitPin(OpBuilder &b, Location loc, Value m, func::FuncOp sink,
               Type f64) {
    auto mt = cast<MemRefType>(m.getType());
    int rank = mt.getRank();
    if (rank == 0)
      return; // scalars: not a kernel output worth pinning
    bool isF32 = mt.getElementType().isF32();

    SmallVector<Value> lbs, ubs;
    SmallVector<int64_t> steps;
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    for (int d = 0; d < rank; ++d) {
      lbs.push_back(zero);
      if (mt.isDynamicDim(d))
        ubs.push_back(b.create<memref::DimOp>(loc, m, d));
      else
        ubs.push_back(b.create<arith::ConstantIndexOp>(loc, mt.getDimSize(d)));
      steps.push_back(1);
    }

    affine::buildAffineLoopNest(
        b, loc, lbs, ubs, steps,
        [&](OpBuilder &nb, Location nloc, ValueRange ivs) {
          Value v = nb.create<affine::AffineLoadOp>(nloc, m, ivs);
          if (isF32)
            v = nb.create<arith::ExtFOp>(nloc, f64, v);
          nb.create<func::CallOp>(nloc, sink, ValueRange{v});
        });
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createDrPinLiveOutPass() {
  return std::make_unique<DrPinLiveOutPass>();
}
} // namespace mlir
