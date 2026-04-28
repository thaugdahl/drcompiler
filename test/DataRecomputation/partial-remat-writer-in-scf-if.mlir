// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-partial-remat dr-test-diagnostics})' -verify-diagnostics

// S2-A: %src has writeCount==1 (one affine.store), but that store sits
// inside an scf.if then-block, so it does NOT dominate the post-loop
// rematerialization point.  The pre-fix writerReachesInsertionPoint check
// answered "yes" purely on block-order grounds (scf.if itself precedes
// %out); the dominance-based check answers "no" because scf.if may not
// execute, so cloning the leaf load could read uninitialized memory.

module {
  func.func @test(%cond: i1) -> f32 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f32
    %seven = arith.constant 7.0 : f32
    %src = memref.alloc() : memref<4xf32>
    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %dst = memref.alloc() : memref<1048576xf32>

    scf.if %cond {
      affine.store %seven, %src[0] : memref<4xf32>
    }

    affine.for %j = 0 to 1048576 {
      // expected-remark @+1 {{load: LEAKED}}
      %v = affine.load %src[0] : memref<4xf32>
      %s = arith.addf %v, %one : f32
      affine.store %s, %dst[%j] : memref<1048576xf32>
    }

    // expected-remark @below {{load: SINGLE}}
    // expected-remark @below {{full-remat: REJECT_UNSAFE}}
    // expected-remark @below {{partial-remat: REJECT_UNSAFE (reason=writer-does-not-dominate)}}
    %out = affine.load %dst[%c0] : memref<1048576xf32>

    memref.dealloc %src : memref<4xf32>
    memref.dealloc %dst : memref<1048576xf32>
    return %out : f32
  }
}
