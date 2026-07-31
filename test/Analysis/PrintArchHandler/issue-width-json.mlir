// RUN: dr-opt %s --pass-pipeline='builtin.module(print-arch-handler{cpu-cost-model-file=%S/costs-issue-width.json})' -verify-diagnostics

// `arch.issue_width` overrides the handler default (apple-m-series defaults to
// 8): the throughput divisor in estimateComputeCost is resolved through the
// same JSON-over-default contract as the rest of the machine description, not
// baked into the pass.
// expected-remark@below {{arch-handler: apple-m-series vec_width=128 issue=6 gp=31 fp=32 vec=32 pred=0 spill_reload=5 spill_store=1 weights=(1.000000e+00,1.000000e+00,1.000000e+00)}}
module {}
