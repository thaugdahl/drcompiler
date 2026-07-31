// RUN: dr-opt %s --pass-pipeline='builtin.module(print-arch-handler{cpu-cost-model-file=%S/costs-triple-darwin.json})' -verify-diagnostics

// JSON omits "handler" and gives an aarch64-DARWIN triple: pickHandlerForTriple
// routes to apple-m-series rather than the arm-neon family default (compare
// triple-only.mlir, which is aarch64-linux and does get arm-neon).
// expected-remark@below {{arch-handler: apple-m-series vec_width=128 issue=8 gp=31 fp=32 vec=32 pred=0 spill_reload=5 spill_store=1 weights=(1.000000e+00,1.000000e+00,1.000000e+00)}}
module {}
