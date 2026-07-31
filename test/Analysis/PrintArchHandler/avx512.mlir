// RUN: dr-opt %s --pass-pipeline='builtin.module(print-arch-handler{cpu-cost-model-file=%S/costs-avx512.json})' -verify-diagnostics

// expected-remark@below {{arch-handler: x86-64-avx512 vec_width=512 issue=4 gp=16 fp=32 vec=32 pred=8 spill_reload=5 spill_store=1 weights=(1.000000e+00,1.000000e+00,1.000000e+00)}}
module {}
