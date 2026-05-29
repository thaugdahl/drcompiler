// RUN: dr-opt %s --pass-pipeline='builtin.module(print-arch-handler{cpu-cost-model-file=%S/costs-neon.json})' -verify-diagnostics

// expected-remark@below {{arch-handler: arm-neon vec_width=128 gp=31 fp=32 vec=32 pred=0 spill_reload=5 spill_store=1 weights=(1.000000e+00,1.000000e+00,1.000000e+00)}}
module {}
