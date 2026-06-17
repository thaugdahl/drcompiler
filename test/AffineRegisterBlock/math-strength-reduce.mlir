// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-math-strength-reduce))' | FileCheck %s

// dr-math-strength-reduce: math.powf(x, integer-const) -> arith.mulf chain
// (exponentiation by squaring), exact for integer exponents.  GPT-2 GELU's
// cubic term `powf(x, 3.0)` is the motivating case -- a libm call neither
// clang -O2 nor onnx-mlir --O3 strength-reduces.  Non-integer / out-of-range
// exponents must be left as math.powf.

// x^3 = x*x then *x  (the GELU cubic).
// CHECK-LABEL: func @cube
// CHECK-NOT:     math.powf
// CHECK:         %[[X2:.*]] = arith.mulf %arg0, %arg0 : f32
// CHECK:         %[[X3:.*]] = arith.mulf %arg0, %[[X2]] : f32
// CHECK:         return %[[X3]]
func.func @cube(%x: f32) -> f32 {
  %c3 = arith.constant 3.0 : f32
  %y = math.powf %x, %c3 : f32
  return %y : f32
}

// x^2 = x*x (single multiply).
// CHECK-LABEL: func @square
// CHECK-NOT:     math.powf
// CHECK:         arith.mulf %arg0, %arg0 : f32
func.func @square(%x: f32) -> f32 {
  %c2 = arith.constant 2.0 : f32
  %y = math.powf %x, %c2 : f32
  return %y : f32
}

// x^4 = (x*x)*(x*x): exponentiation by squaring -> 2 multiplies, no x^4 chain.
// CHECK-LABEL: func @fourth
// CHECK-NOT:     math.powf
// CHECK:         %[[A:.*]] = arith.mulf %arg0, %arg0
// CHECK:         %[[B:.*]] = arith.mulf %[[A]], %[[A]]
// CHECK:         return %[[B]]
func.func @fourth(%x: f32) -> f32 {
  %c4 = arith.constant 4.0 : f32
  %y = math.powf %x, %c4 : f32
  return %y : f32
}

// f64 also handled.
// CHECK-LABEL: func @cube_f64
// CHECK-NOT:     math.powf
// CHECK:         arith.mulf {{.*}} : f64
func.func @cube_f64(%x: f64) -> f64 {
  %c3 = arith.constant 3.0 : f64
  %y = math.powf %x, %c3 : f64
  return %y : f64
}

// Non-integer exponent: left as math.powf (would need a root).
// CHECK-LABEL: func @sqrt_skip
// CHECK:         math.powf
func.func @sqrt_skip(%x: f32) -> f32 {
  %ch = arith.constant 0.5 : f32
  %y = math.powf %x, %ch : f32
  return %y : f32
}

// Negative exponent: left as math.powf (would need a reciprocal).
// CHECK-LABEL: func @neg_skip
// CHECK:         math.powf
func.func @neg_skip(%x: f32) -> f32 {
  %cm1 = arith.constant -1.0 : f32
  %y = math.powf %x, %cm1 : f32
  return %y : f32
}

// Non-constant exponent: left as math.powf.
// CHECK-LABEL: func @dynamic_skip
// CHECK:         math.powf
func.func @dynamic_skip(%x: f32, %e: f32) -> f32 {
  %y = math.powf %x, %e : f32
  return %y : f32
}
