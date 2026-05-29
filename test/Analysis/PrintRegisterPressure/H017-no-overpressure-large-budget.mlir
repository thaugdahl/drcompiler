// RUN: dr-opt %s --pass-pipeline='builtin.module(print-register-pressure{cpu-cost-model-file=%S/Inputs/avx512.json strategy=sum-excess})' -verify-diagnostics

// 4 fp args under AVX-512 (fp_budget=32) — no overpressure regardless of
// strategy.  SumExcess confirms zero cycles when peak <= budget.
// expected-remark@below {{register-pressure: noover strategy=sum-excess peak=(gp=0,fp=5,vec=0,pred=0) excess=(gp=0,fp=0,vec=0,pred=0) spill_cycles=0}}
func.func @noover(%a: f32, %b: f32, %c: f32, %d: f32) -> f32 {
  %ab = arith.addf %a, %b : f32
  %cd = arith.addf %c, %d : f32
  %r = arith.addf %ab, %cd : f32
  return %r : f32
}
