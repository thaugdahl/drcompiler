// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// Phase-aware seeding: a global written by writers in disjoint static call
// graphs (no common caller) must not pollute a reader's reaching-stores
// across phase boundaries. With coarse module-wide seeding, the load below
// would classify MULTI (seed would inject @ic_writer's store). With phase
// filtering, only @dyn_writer (which shares caller @dyn_driver with
// @dyn_reader) is seeded, so the load is SINGLE.

module {
  memref.global "private" @g : memref<i32> = uninitialized

  func.func private @ic_writer(%x: i32) {
    %g = memref.get_global @g : memref<i32>
    memref.store %x, %g[] : memref<i32>
    return
  }

  func.func private @dyn_writer(%x: i32) {
    %g = memref.get_global @g : memref<i32>
    %one = arith.constant 1 : i32
    %v = arith.addi %x, %one : i32
    memref.store %v, %g[] : memref<i32>
    return
  }

  func.func private @dyn_reader() -> i32 {
    %g = memref.get_global @g : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  // Public phase entries with no caller in the module.
  func.func @ic_driver(%x: i32) {
    call @ic_writer(%x) : (i32) -> ()
    return
  }

  func.func @dyn_driver(%x: i32) -> i32 {
    call @dyn_writer(%x) : (i32) -> ()
    %r = call @dyn_reader() : () -> i32
    return %r : i32
  }
}
