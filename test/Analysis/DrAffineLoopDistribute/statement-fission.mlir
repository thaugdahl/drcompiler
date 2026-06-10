// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-affine-loop-distribute))' | FileCheck %s

// 2mm-shaped imperfect nest: the init store and the beta-scale statement
// (load-mul-store) inside the j body must each fission into their own
// perfect nest, leaving the matmul as a tileable [i,j,k] band.  The pass
// fissions statements at the j level, then re-visits and splits i.

// CHECK-LABEL: func.func @init_matmul
// CHECK:       affine.for %{{.*}} = 0 to 64 {
// CHECK-NEXT:    affine.for %{{.*}} = 0 to 64 {
// CHECK-NEXT:      affine.store
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK:       affine.for %{{.*}} = 0 to 64 {
// CHECK-NEXT:    affine.for %{{.*}} = 0 to 64 {
// CHECK-NEXT:      affine.for %{{.*}} = 0 to 64 {
func.func @init_matmul(%T: memref<64x64xf64>, %A: memref<64x64xf64>, %B: memref<64x64xf64>) {
  %cst = arith.constant 0.0 : f64
  affine.for %i = 0 to 64 {
    affine.for %j = 0 to 64 {
      affine.store %cst, %T[%i, %j] : memref<64x64xf64>
      affine.for %k = 0 to 64 {
        %a = affine.load %A[%i, %k] : memref<64x64xf64>
        %b = affine.load %B[%k, %j] : memref<64x64xf64>
        %m = arith.mulf %a, %b : f64
        %t = affine.load %T[%i, %j] : memref<64x64xf64>
        %s = arith.addf %t, %m : f64
        affine.store %s, %T[%i, %j] : memref<64x64xf64>
      }
    }
  }
  return
}

// The beta-scale statement is a load-mul-store run: the load and mulf are
// absorbed into the statement unit (dead once it moves), so the unit erases
// cleanly from the matmul copy.

// CHECK-LABEL: func.func @beta_scale_matmul
// CHECK:       affine.for %{{.*}} = 0 to 64 {
// CHECK-NEXT:    affine.for %{{.*}} = 0 to 64 {
// CHECK-NEXT:      affine.load
// CHECK-NEXT:      arith.mulf
// CHECK-NEXT:      affine.store
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK:       affine.for %{{.*}} = 0 to 64 {
// CHECK-NEXT:    affine.for %{{.*}} = 0 to 64 {
// CHECK-NEXT:      affine.for %{{.*}} = 0 to 64 {
func.func @beta_scale_matmul(%D: memref<64x64xf64>, %T: memref<64x64xf64>, %C: memref<64x64xf64>, %beta: f64) {
  affine.for %i = 0 to 64 {
    affine.for %j = 0 to 64 {
      %d0 = affine.load %D[%i, %j] : memref<64x64xf64>
      %ds = arith.mulf %d0, %beta : f64
      affine.store %ds, %D[%i, %j] : memref<64x64xf64>
      affine.for %k = 0 to 64 {
        %t = affine.load %T[%i, %k] : memref<64x64xf64>
        %c = affine.load %C[%k, %j] : memref<64x64xf64>
        %m = arith.mulf %t, %c : f64
        %d = affine.load %D[%i, %j] : memref<64x64xf64>
        %s = arith.addf %d, %m : f64
        affine.store %s, %D[%i, %j] : memref<64x64xf64>
      }
    }
  }
  return
}

// Time loop: the second nest writes A which the first nest reads at the
// next t iteration — dependence carried by t, fission must refuse at every
// level (classic stencil shape; splitting would read not-yet-written values).

// CHECK-LABEL: func.func @timeloop
// CHECK:       affine.for %{{.*}} = 0 to 100 {
// CHECK:         affine.for
// CHECK:         affine.for
// CHECK:       }
// CHECK-NOT:   affine.for
func.func @timeloop(%A: memref<512xf64>, %B: memref<512xf64>) {
  affine.for %t = 0 to 100 {
    affine.for %i = 1 to 511 {
      %l = affine.load %A[%i - 1] : memref<512xf64>
      %r = affine.load %A[%i + 1] : memref<512xf64>
      %s = arith.addf %l, %r : f64
      affine.store %s, %B[%i] : memref<512xf64>
    }
    affine.for %i = 1 to 511 {
      %l = affine.load %B[%i] : memref<512xf64>
      affine.store %l, %A[%i] : memref<512xf64>
    }
  }
  return
}
