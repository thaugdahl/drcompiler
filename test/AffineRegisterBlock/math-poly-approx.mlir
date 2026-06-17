// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-math-strength-reduce))' | FileCheck %s --check-prefix=OFF
// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-math-strength-reduce{poly-approx=true}))' | FileCheck %s --check-prefix=ON

// dr-math-strength-reduce poly-approx option: lower math.exp/tanh/... to
// vectorizable polynomial approximations (no libm call barrier), so the
// pointwise GELU/softmax loops auto-vectorize.  Default OFF keeps the
// transcendentals (byte-identical to the committed powf-only behavior); ON
// removes them.  This closes the openai-gpt transcendental backend gap that
// onnx-mlir --O3 EmitObj closes natively.

// OFF-LABEL: func @transcendentals
// OFF:         math.tanh
// OFF:         math.exp
// ON-LABEL: func @transcendentals
// ON-NOT:     math.tanh
// ON-NOT:     math.exp
// ON:         arith.{{(mulf|addf|select|cmpf)}}
func.func @transcendentals(%x: f32) -> (f32, f32) {
  %t = math.tanh %x : f32
  %e = math.exp %x : f32
  return %t, %e : f32, f32
}

// poly-approx composes with the powf strength-reduction (both lower to
// call-free arith in one pass).
// ON-LABEL: func @gelu_cubic
// ON-NOT:     math.powf
// ON-NOT:     math.tanh
func.func @gelu_cubic(%x: f32) -> f32 {
  %c3 = arith.constant 3.0 : f32
  %x3 = math.powf %x, %c3 : f32
  %t  = math.tanh %x3 : f32
  return %t : f32
}
