// RUN: dr-opt %s --pass-pipeline='builtin.module(print-arch-handler{cpu-cost-model-file=%S/costs-generic.json})' -verify-diagnostics

// expected-remark@below {{arch-handler: generic vec_width=128 gp=16 fp=16 vec=16 pred=0 spill_reload=5 spill_store=1 weights=(1.000000e+00,1.000000e+00,1.000000e+00)}}
module {}
