// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics

// G008: Consumer count changes the cost-model decision.
// Buffer exceeds L1 (hits L2, 12cy latency). Computation is add+sqrt (21 cycles).
// keepCost = alu+1+N*loadLat = 21+1+12N = 22+12N
// recomputeCost = N*alu = 21N
// N=2: recompute=42 < keep=46 => RECOMPUTE
// N=3: recompute=63 > keep=58 => KEEP (flip point)

module {
  func.func @consumer_flip(%x: f64) -> f64 {
    %c0 = arith.constant 0 : index
    %cst1 = arith.constant 1.0 : f64

    // 9000 elements x 8 bytes = 72000 bytes > L1 (32KB), fits in L2 (256KB).
    // expected-remark @below {{cost-model: RECOMPUTE}}
    %buf = memref.alloc() : memref<9000xf64>

    affine.for %i = 0 to 9000 {
      %s = arith.addf %x, %cst1 : f64
      %val = math.sqrt %s : f64
      affine.store %val, %buf[%i] : memref<9000xf64>
    }

    // Consumer 1
    // expected-remark @below {{full-remat: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %a = memref.load %buf[%c0] : memref<9000xf64>

    // Consumer 2
    // expected-remark @below {{full-remat: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %b = memref.load %buf[%c0] : memref<9000xf64>

    %result = arith.addf %a, %b : f64
    memref.dealloc %buf : memref<9000xf64>
    return %result : f64
  }
}
