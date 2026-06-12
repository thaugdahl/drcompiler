// WP-G1: the register-block vector width is derived from the machine's
// vector-execution model (MachineModel::preferredVectorElems) when a cost-model
// JSON describes it, and is otherwise the static `vl` option default (8 == the
// value the model derives for the Zen4 dev host).  An explicit --vl always wins.
//
// Same f32 GEMM, three machines:
//   DEFAULT  no JSON      -> Zen4 native 256-bit datapath -> 8 elems  (ymm)
//   XEON     native 512   -> 512-bit datapath             -> 16 elems (zmm)
//   PIN      Xeon JSON + vl=8 (CLI override wins over the JSON)        -> 8
//
// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-register-block))' \
// RUN:   | FileCheck %s --check-prefix=DEFAULT
// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-register-block{cpu-cost-model-file=%S/Inputs/xeon-vector.json}))' \
// RUN:   | FileCheck %s --check-prefix=XEON
// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-register-block{vl=8 cpu-cost-model-file=%S/Inputs/xeon-vector.json}))' \
// RUN:   | FileCheck %s --check-prefix=PIN
//
//   CLAMP    malformed JSON (native 512 > arch 256) -> native clamped to arch
//            256 -> 8 elems, never a vector wider than the register file.
// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-register-block{cpu-cost-model-file=%S/Inputs/bad-native-gt-arch.json}))' 2>/dev/null \
// RUN:   | FileCheck %s --check-prefix=CLAMP

module {
  func.func @gemm(%A: memref<64x64xf32>, %B: memref<64x64xf32>, %C: memref<64x64xf32>) {
    affine.for %i = 0 to 64 {
      affine.for %j = 0 to 64 {
        affine.for %k = 0 to 64 {
          %a = affine.load %A[%i, %k] : memref<64x64xf32>
          %b = affine.load %B[%k, %j] : memref<64x64xf32>
          %c = affine.load %C[%i, %j] : memref<64x64xf32>
          %p = arith.mulf %a, %b : f32
          %s = arith.addf %c, %p : f32
          affine.store %s, %C[%i, %j] : memref<64x64xf32>
        }
      }
    }
    return
  }
}

// DEFAULT: affine.for %{{.*}} = 0 to 64 step 8
// DEFAULT: affine.vector_load %{{.*}} : memref<64x64xf32>, vector<8xf32>
// DEFAULT-NOT: vector<16xf32>

// XEON: affine.for %{{.*}} = 0 to 64 step 16
// XEON: affine.vector_load %{{.*}} : memref<64x64xf32>, vector<16xf32>
// XEON-NOT: vector<8xf32>

// PIN: affine.vector_load %{{.*}} : memref<64x64xf32>, vector<8xf32>
// PIN-NOT: vector<16xf32>

// CLAMP: affine.vector_load %{{.*}} : memref<64x64xf32>, vector<8xf32>
// CLAMP-NOT: vector<16xf32>
