// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics

// A1 (critical-path compute cost). The stored value is a WIDE expression: four
// INDEPENDENT math.sqrt (cost 20 each) reduced by a balanced add tree (cost 1
// each). The naive total-op-count model prices this at 4*20 + 3*1 = 83. The
// ILP-aware model prices it by the dependency critical path
// (load -> sqrt[20] -> add[1] -> add[1] = 22), which is what an out-of-order
// core actually realizes. The throughput floor is ceil(83/4) = 21, so the
// reported compute cost is max(22, 21) = 22.
//
// This pins the critical-path behavior: a regression to the op-sum model would
// report compute=83 and this test would fail.

module {
  func.func @fill(%w: memref<?xf64>, %a: memref<?xf64>, %b: memref<?xf64>,
                  %c: memref<?xf64>, %d: memref<?xf64>, %n: i32) {
    %ni = arith.index_cast %n : i32 to index
    affine.for %i = 0 to %ni {
      %av = affine.load %a[%i] : memref<?xf64>
      %bv = affine.load %b[%i] : memref<?xf64>
      %cv = affine.load %c[%i] : memref<?xf64>
      %dv = affine.load %d[%i] : memref<?xf64>
      %as = math.sqrt %av : f64
      %bs = math.sqrt %bv : f64
      %cs = math.sqrt %cv : f64
      %ds = math.sqrt %dv : f64
      %s1 = arith.addf %as, %bs : f64
      %s2 = arith.addf %cs, %ds : f64
      %v  = arith.addf %s1, %s2 : f64
      affine.store %v, %w[%i] : memref<?xf64>
    }
    return
  }

  func.func @use(%w: memref<?xf64>) -> f64 {
    %c0 = arith.constant 0 : index
    // expected-remark @+2 {{load: SINGLE}}
    // expected-remark @+1 {{interproc-cross: REJECT_PLAN}}
    %x = affine.load %w[%c0] : memref<?xf64>
    return %x : f64
  }

  func.func @run(%a: memref<?xf64>, %b: memref<?xf64>, %c: memref<?xf64>,
                 %d: memref<?xf64>, %n: i32) -> f64 {
    // expected-remark @+1 {{cost-model: RECOMPUTE (compute=22}}
    %buf = memref.alloc() : memref<2048xf64>
    %cast = memref.cast %buf : memref<2048xf64> to memref<?xf64>
    call @fill(%cast, %a, %b, %c, %d, %n)
        : (memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>,
           memref<?xf64>, i32) -> ()
    %r = call @use(%cast) : (memref<?xf64>) -> f64
    return %r : f64
  }
}
