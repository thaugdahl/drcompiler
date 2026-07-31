// RUN: dr-opt %s --pass-pipeline='builtin.module(print-arch-handler{cpu-cost-model-file=%S/costs-unknown.json})' -verify-diagnostics

// Unknown handler name falls back to generic; diagnostic records the unknown name.
// expected-remark@below {{arch-handler: generic vec_width=128 issue=4 gp=16 fp=16 vec=16 pred=0 spill_reload=5 spill_store=1 weights=(1.000000e+00,1.000000e+00,1.000000e+00) json_handler_unknown=powerpc-altivec-fictional}}
module {}
