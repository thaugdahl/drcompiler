// RUN: dr-opt %s --pass-pipeline='builtin.module(print-arch-handler{cpu-cost-model-file=%S/costs-apple-m.json})' -verify-diagnostics

// Apple M-series: AArch64 register file (31 GP / 32 V / no predicates) but a
// wide front end -- issue=8 vs 4 everywhere else.  The 128 B cache line is
// memory geometry, not an ArchParam: it comes from the JSON `cache` block.
// expected-remark@below {{arch-handler: apple-m-series vec_width=128 issue=8 gp=31 fp=32 vec=32 pred=0 spill_reload=5 spill_store=1 weights=(1.000000e+00,1.000000e+00,1.000000e+00)}}
module {}
