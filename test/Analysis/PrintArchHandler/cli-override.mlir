// RUN: dr-opt %s --pass-pipeline='builtin.module(print-arch-handler{cpu-cost-model-file=%S/costs-avx2.json handler=arm-neon})' -verify-diagnostics

// CLI `handler` option picks arm-neon; JSON's registers + weights still layer on top of arm-neon defaults.
// expected-remark@below {{arch-handler: arm-neon vec_width=256 gp=16 fp=16 vec=16 pred=0 spill_reload=5 spill_store=1 weights=(1.000000e+00,1.500000e+00,5.000000e-01)}}
module {}
