// RUN: dr-opt %s --pass-pipeline='builtin.module(print-arch-handler{cpu-cost-model-file=%S/costs-sve.json})' -verify-diagnostics

// SVE must be named explicitly (a triple never implies it).  vec_width is the
// architectural FLOOR (128 b), so register pressure is never under-estimated on
// a wider implementation; pred=16 is the P0..P15 predicate file NEON lacks.
// expected-remark@below {{arch-handler: arm-sve vec_width=128 issue=4 gp=31 fp=32 vec=32 pred=16 spill_reload=5 spill_store=1 weights=(1.000000e+00,1.000000e+00,1.000000e+00)}}
module {}
