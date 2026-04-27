// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-partial-remat dr-test-diagnostics})' --allow-unregistered-dialect -verify-diagnostics

// S1-D: a write-once buffer escapes to an external call as !llvm.ptr (via
// polygeist.memref2pointer).  buildRootWriteMap must mark the root
// escapesToCall=true so partial rematerialization refuses to clone leaf
// loads of that root -- the cloned load could read memory mutated through
// the leaked pointer.

module {
  llvm.func @sink(!llvm.ptr)

  func.func @test() -> f32 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f32
    // expected-remark @+1 {{cost-model:}}
    %src = memref.alloc() : memref<4xf32>
    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %dst = memref.alloc() : memref<1048576xf32>

    affine.for %i = 0 to 4 {
      %i32 = arith.index_cast %i : index to i32
      %f = arith.sitofp %i32 : i32 to f32
      affine.store %f, %src[%i] : memref<4xf32>
    }

    affine.for %j = 0 to 1048576 {
      // expected-remark @+1 {{load: SINGLE}}
      %v = affine.load %src[0] : memref<4xf32>
      %s = arith.addf %v, %one : f32
      affine.store %s, %dst[%j] : memref<1048576xf32>
    }

    // expected-remark @+2 {{load: SINGLE}}
    // expected-remark @+1 {{partial-remat: REJECT_UNSAFE (reason=escapes-to-call)}}
    %out = affine.load %dst[%c0] : memref<1048576xf32>

    // %src escapes to an external call as !llvm.ptr after the loads.
    // Without the S1-D fix, buildRootWriteMap ignores this operand
    // because its type is !llvm.ptr, and partial remat would ACCEPT.
    %srcptr = "polygeist.memref2pointer"(%src) : (memref<4xf32>) -> !llvm.ptr
    llvm.call @sink(%srcptr) : (!llvm.ptr) -> ()

    memref.dealloc %src : memref<4xf32>
    memref.dealloc %dst : memref<1048576xf32>
    return %out : f32
  }
}
