// RUN: dr-opt %s --allow-unregistered-dialect --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-cost-model=true dr-buffer-elim=true dr-summary=true})' 2>&1 | FileCheck %s --check-prefix=SUM

// E018: Very large heap buffer (400KB > L2 default 256KB, no L3 modeled).
// Load latency = memLatency (200 cycles). The capacity penalty plus alloc
// overhead make keepCost enormous → eliminate is trivially cheaper.

// SUM: buffer-elim {{.*}}: FEASIBLE (escape=no-escape{{.*}}keep={{[0-9]+}}, elim=0)
func.func @capacity() {
  %a = memref.alloc() : memref<100000xi32>
  %c = arith.constant 1 : i32
  affine.store %c, %a[0] : memref<100000xi32>
  %v = affine.load %a[0] : memref<100000xi32>
  "use.consume"(%v) : (i32) -> ()
  memref.dealloc %a : memref<100000xi32>
  return
}
