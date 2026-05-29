// RUN: dr-opt %s --pass-pipeline='builtin.module(print-register-pressure)' -verify-diagnostics

// 8 fp args, three additions form a balanced reduction tree.  At the first
// addf: all 8 args still live (none consumed yet) + the new result.
// expected-remark@below {{register-pressure: redux8 strategy=excess-hot peak=(gp=0,fp=9,vec=0,pred=0) excess=(gp=0,fp=0,vec=0,pred=0) spill_cycles=0}}
func.func @redux8(%a: f32, %b: f32, %c: f32, %d: f32,
                  %e: f32, %f: f32, %g: f32, %h: f32) -> f32 {
  %ab = arith.addf %a, %b : f32
  %cd = arith.addf %c, %d : f32
  %ef = arith.addf %e, %f : f32
  %gh = arith.addf %g, %h : f32
  %abcd = arith.addf %ab, %cd : f32
  %efgh = arith.addf %ef, %gh : f32
  %total = arith.addf %abcd, %efgh : f32
  return %total : f32
}
