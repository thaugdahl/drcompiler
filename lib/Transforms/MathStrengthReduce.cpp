//===- MathStrengthReduce.cpp - powf(x, int) -> multiplies --------------===//
//
// Strength-reduce math.powf(x, C) to repeated arith.mulf when C is a small
// non-negative integer constant (exponentiation by squaring).  Exact for
// integer exponents, so unconditionally safe regardless of fastmath.
//
// GPT-2's tanh-approximation GELU computes its cubic term as powf(x, 3.0) -- a
// full libm transcendental call per element.  Neither clang -O2 nor onnx-mlir
// --O3 strength-reduces it: the math.powf carries no fastmath flag, so the
// backend leaves it a `call powf@plt`.  Two multiplies replace it, removing
// ~4.7M libm calls per openai-gpt inference (measured powf=288 call-sites
// surviving in both codegen and onnx-mlir --O3 disassembly) and making the
// surrounding pointwise loop vectorizable.
//
//   %y = math.powf %x, 3.0   ==>   %t = arith.mulf %x, %x ; %y = arith.mulf %t, %x
//
//===----------------------------------------------------------------------===//

#include "drcompiler/Transforms/MathStrengthReduce.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
#define GEN_PASS_DEF_DRMATHSTRENGTHREDUCEPASS
#include "drcompiler/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

/// x^n for n >= 1 via exponentiation by squaring (n=3 -> x*x then *x = 2 muls).
/// Works for scalar and vector float operands alike.
static Value powByMul(OpBuilder &b, Location loc, Value x, uint64_t n) {
  Value result;       // null until the first set bit of n
  Value base = x;
  while (n > 0) {
    if (n & 1u)
      result = result ? b.create<arith::MulFOp>(loc, result, base).getResult()
                      : base;
    n >>= 1;
    if (n)
      base = b.create<arith::MulFOp>(loc, base, base).getResult();
  }
  return result;
}

struct DrMathStrengthReducePass
    : public impl::DrMathStrengthReducePassBase<DrMathStrengthReducePass> {

  void runOnOperation() override {
    func::FuncOp fn = getOperation();
    if (fn.isExternal())
      return;

    // Collect first; we erase ops as we go.
    SmallVector<math::PowFOp> work;
    fn.walk([&](math::PowFOp op) { work.push_back(op); });

    for (math::PowFOp op : work) {
      llvm::APFloat expVal(0.0);
      if (!matchPattern(op.getRhs(), m_ConstantFloat(&expVal)))
        continue;
      // Integer-valued exponent in [1, maxExponent]?
      bool isExact = false;
      llvm::APSInt intExp(/*BitWidth=*/64, /*isUnsigned=*/false);
      if (expVal.convertToInteger(intExp, llvm::APFloat::rmTowardZero,
                                  &isExact) != llvm::APFloat::opOK ||
          !isExact)
        continue;
      int64_t e = intExp.getSExtValue();
      if (e < 1 || static_cast<uint64_t>(e) > maxExponent)
        continue;

      OpBuilder b(op);
      Value repl = powByMul(b, op.getLoc(), op.getLhs(),
                            static_cast<uint64_t>(e));
      op.replaceAllUsesWith(repl);
      op.erase();
    }

    // Optional: lower the remaining transcendentals (exp/tanh/log/erf/...) to
    // vectorizable polynomial approximations.  This removes the libm call
    // barrier so the surrounding pointwise loop auto-vectorizes -- the
    // openai-gpt GELU/softmax tail is otherwise scalar libm (~1.66x backend gap
    // vs onnx-mlir --O3 EmitObj, which does exactly this natively).
    if (polyApprox) {
      RewritePatternSet patterns(&getContext());
      populateMathPolynomialApproximationPatterns(
          patterns, MathPolynomialApproximationOptions{});
      if (failed(applyPatternsGreedily(fn, std::move(patterns))))
        signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createDrMathStrengthReducePass() {
  return std::make_unique<DrMathStrengthReducePass>();
}
