module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_deriche(%arg0: i32, %arg1: i32, %arg2: f32, %arg3: memref<?x2160xf32>, %arg4: memref<?x2160xf32>, %arg5: memref<?x2160xf32>, %arg6: memref<?x2160xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %cst_1 = arith.constant -2.000000e+00 : f32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg0 : i32 to index
    %2 = llvm.mlir.undef : f32
    %alloca = memref.alloca() : memref<f32>
    affine.store %2, %alloca[] : memref<f32>
    %alloca_3 = memref.alloca() : memref<f32>
    affine.store %2, %alloca_3[] : memref<f32>
    %alloca_4 = memref.alloca() : memref<f32>
    affine.store %2, %alloca_4[] : memref<f32>
    %alloca_5 = memref.alloca() : memref<f32>
    affine.store %2, %alloca_5[] : memref<f32>
    %alloca_6 = memref.alloca() : memref<f32>
    affine.store %2, %alloca_6[] : memref<f32>
    %alloca_7 = memref.alloca() : memref<f32>
    affine.store %2, %alloca_7[] : memref<f32>
    %alloca_8 = memref.alloca() : memref<f32>
    affine.store %2, %alloca_8[] : memref<f32>
    %alloca_9 = memref.alloca() : memref<f32>
    affine.store %2, %alloca_9[] : memref<f32>
    %alloca_10 = memref.alloca() : memref<f32>
    affine.store %2, %alloca_10[] : memref<f32>
    %alloca_11 = memref.alloca() : memref<f32>
    affine.store %2, %alloca_11[] : memref<f32>
    %3 = arith.negf %arg2 : f32
    %4 = math.exp %3 : f32
    %5 = arith.subf %cst, %4 : f32
    %6 = arith.mulf %5, %5 : f32
    %7 = arith.mulf %arg2, %cst_0 : f32
    %8 = arith.mulf %7, %4 : f32
    %9 = arith.addf %8, %cst : f32
    %10 = math.exp %7 : f32
    %11 = arith.subf %9, %10 : f32
    %12 = arith.divf %6, %11 : f32
    %13 = arith.mulf %12, %4 : f32
    %14 = arith.subf %arg2, %cst : f32
    %15 = arith.mulf %13, %14 : f32
    %16 = arith.addf %arg2, %cst : f32
    %17 = arith.mulf %13, %16 : f32
    %18 = arith.negf %12 : f32
    %19 = arith.mulf %arg2, %cst_1 : f32
    %20 = math.exp %19 : f32
    %21 = arith.mulf %18, %20 : f32
    %22 = math.powf %cst_0, %3 : f32
    %23 = arith.negf %20 : f32
    affine.for %arg7 = 0 to %1 {
      affine.store %cst_2, %alloca_9[] : memref<f32>
      affine.store %cst_2, %alloca_8[] : memref<f32>
      affine.store %cst_2, %alloca_11[] : memref<f32>
      affine.for %arg8 = 0 to %0 {
        %24 = affine.load %arg3[%arg7, %arg8] : memref<?x2160xf32>
        %25 = arith.mulf %12, %24 : f32
        %26 = affine.load %alloca_11[] : memref<f32>
        %27 = arith.mulf %15, %26 : f32
        %28 = arith.addf %25, %27 : f32
        %29 = affine.load %alloca_9[] : memref<f32>
        %30 = arith.mulf %22, %29 : f32
        %31 = arith.addf %28, %30 : f32
        %32 = affine.load %alloca_8[] : memref<f32>
        %33 = arith.mulf %23, %32 : f32
        %34 = arith.addf %31, %33 : f32
        affine.store %34, %arg5[%arg7, %arg8] : memref<?x2160xf32>
        %35 = affine.load %arg3[%arg7, %arg8] : memref<?x2160xf32>
        affine.store %35, %alloca_11[] : memref<f32>
        affine.store %29, %alloca_8[] : memref<f32>
        %36 = affine.load %arg5[%arg7, %arg8] : memref<?x2160xf32>
        affine.store %36, %alloca_9[] : memref<f32>
      }
    }
    affine.for %arg7 = 0 to %1 {
      affine.store %cst_2, %alloca_3[] : memref<f32>
      affine.store %cst_2, %alloca[] : memref<f32>
      affine.store %cst_2, %alloca_7[] : memref<f32>
      affine.store %cst_2, %alloca_6[] : memref<f32>
      affine.for %arg8 = 0 to %0 {
        %24 = affine.load %alloca_7[] : memref<f32>
        %25 = arith.mulf %17, %24 : f32
        %26 = affine.load %alloca_6[] : memref<f32>
        %27 = arith.mulf %21, %26 : f32
        %28 = arith.addf %25, %27 : f32
        %29 = affine.load %alloca_3[] : memref<f32>
        %30 = arith.mulf %22, %29 : f32
        %31 = arith.addf %28, %30 : f32
        %32 = affine.load %alloca[] : memref<f32>
        %33 = arith.mulf %23, %32 : f32
        %34 = arith.addf %31, %33 : f32
        affine.store %34, %arg6[%arg7, -%arg8 + symbol(%0) - 1] : memref<?x2160xf32>
        affine.store %24, %alloca_6[] : memref<f32>
        %35 = affine.load %arg3[%arg7, -%arg8 + symbol(%0) - 1] : memref<?x2160xf32>
        affine.store %35, %alloca_7[] : memref<f32>
        affine.store %29, %alloca[] : memref<f32>
        %36 = affine.load %arg6[%arg7, -%arg8 + symbol(%0) - 1] : memref<?x2160xf32>
        affine.store %36, %alloca_3[] : memref<f32>
      }
    }
    affine.for %arg7 = 0 to %1 {
      affine.for %arg8 = 0 to %0 {
        %24 = affine.load %arg5[%arg7, %arg8] : memref<?x2160xf32>
        %25 = affine.load %arg6[%arg7, %arg8] : memref<?x2160xf32>
        %26 = arith.addf %24, %25 : f32
        affine.store %26, %arg4[%arg7, %arg8] : memref<?x2160xf32>
      }
    }
    affine.for %arg7 = 0 to %0 {
      affine.store %cst_2, %alloca_10[] : memref<f32>
      affine.store %cst_2, %alloca_9[] : memref<f32>
      affine.store %cst_2, %alloca_8[] : memref<f32>
      affine.for %arg8 = 0 to %1 {
        %24 = affine.load %arg4[%arg8, %arg7] : memref<?x2160xf32>
        %25 = arith.mulf %12, %24 : f32
        %26 = affine.load %alloca_10[] : memref<f32>
        %27 = arith.mulf %15, %26 : f32
        %28 = arith.addf %25, %27 : f32
        %29 = affine.load %alloca_9[] : memref<f32>
        %30 = arith.mulf %22, %29 : f32
        %31 = arith.addf %28, %30 : f32
        %32 = affine.load %alloca_8[] : memref<f32>
        %33 = arith.mulf %23, %32 : f32
        %34 = arith.addf %31, %33 : f32
        affine.store %34, %arg5[%arg8, %arg7] : memref<?x2160xf32>
        %35 = affine.load %arg4[%arg8, %arg7] : memref<?x2160xf32>
        affine.store %35, %alloca_10[] : memref<f32>
        affine.store %29, %alloca_8[] : memref<f32>
        %36 = affine.load %arg5[%arg8, %arg7] : memref<?x2160xf32>
        affine.store %36, %alloca_9[] : memref<f32>
      }
    }
    affine.for %arg7 = 0 to %0 {
      affine.store %cst_2, %alloca_5[] : memref<f32>
      affine.store %cst_2, %alloca_4[] : memref<f32>
      affine.store %cst_2, %alloca_3[] : memref<f32>
      affine.store %cst_2, %alloca[] : memref<f32>
      affine.for %arg8 = 0 to %1 {
        %24 = affine.load %alloca_5[] : memref<f32>
        %25 = arith.mulf %17, %24 : f32
        %26 = affine.load %alloca_4[] : memref<f32>
        %27 = arith.mulf %21, %26 : f32
        %28 = arith.addf %25, %27 : f32
        %29 = affine.load %alloca_3[] : memref<f32>
        %30 = arith.mulf %22, %29 : f32
        %31 = arith.addf %28, %30 : f32
        %32 = affine.load %alloca[] : memref<f32>
        %33 = arith.mulf %23, %32 : f32
        %34 = arith.addf %31, %33 : f32
        affine.store %34, %arg6[-%arg8 + symbol(%1) - 1, %arg7] : memref<?x2160xf32>
        affine.store %24, %alloca_4[] : memref<f32>
        %35 = affine.load %arg4[-%arg8 + symbol(%1) - 1, %arg7] : memref<?x2160xf32>
        affine.store %35, %alloca_5[] : memref<f32>
        affine.store %29, %alloca[] : memref<f32>
        %36 = affine.load %arg6[-%arg8 + symbol(%1) - 1, %arg7] : memref<?x2160xf32>
        affine.store %36, %alloca_3[] : memref<f32>
      }
    }
    affine.for %arg7 = 0 to %1 {
      affine.for %arg8 = 0 to %0 {
        %24 = affine.load %arg5[%arg7, %arg8] : memref<?x2160xf32>
        %25 = affine.load %arg6[%arg7, %arg8] : memref<?x2160xf32>
        %26 = arith.addf %24, %25 : f32
        affine.store %26, %arg4[%arg7, %arg8] : memref<?x2160xf32>
      }
    }
    return
  }
}
