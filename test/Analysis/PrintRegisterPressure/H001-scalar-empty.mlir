// RUN: dr-opt %s --pass-pipeline='builtin.module(print-register-pressure)' -verify-diagnostics

// Empty function body: no live values at the terminator's program point.
// expected-remark@below {{register-pressure: empty strategy=excess-hot peak=(gp=0,fp=0,vec=0,pred=0) excess=(gp=0,fp=0,vec=0,pred=0) spill_cycles=0}}
func.func @empty() {
  return
}
