// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// F002: 64-op linear add chain. Compile-time scaling test.

module {
  func.func @chain_depth_64(%arg0: i32) -> i32 {
    %buf = memref.alloc() : memref<i32>
    %c1 = arith.constant 1 : i32
    %v1 = arith.addi %arg0, %c1 : i32
    %v2 = arith.addi %v1, %c1 : i32
    %v3 = arith.addi %v2, %c1 : i32
    %v4 = arith.addi %v3, %c1 : i32
    %v5 = arith.addi %v4, %c1 : i32
    %v6 = arith.addi %v5, %c1 : i32
    %v7 = arith.addi %v6, %c1 : i32
    %v8 = arith.addi %v7, %c1 : i32
    %v9 = arith.addi %v8, %c1 : i32
    %v10 = arith.addi %v9, %c1 : i32
    %v11 = arith.addi %v10, %c1 : i32
    %v12 = arith.addi %v11, %c1 : i32
    %v13 = arith.addi %v12, %c1 : i32
    %v14 = arith.addi %v13, %c1 : i32
    %v15 = arith.addi %v14, %c1 : i32
    %v16 = arith.addi %v15, %c1 : i32
    %v17 = arith.addi %v16, %c1 : i32
    %v18 = arith.addi %v17, %c1 : i32
    %v19 = arith.addi %v18, %c1 : i32
    %v20 = arith.addi %v19, %c1 : i32
    %v21 = arith.addi %v20, %c1 : i32
    %v22 = arith.addi %v21, %c1 : i32
    %v23 = arith.addi %v22, %c1 : i32
    %v24 = arith.addi %v23, %c1 : i32
    %v25 = arith.addi %v24, %c1 : i32
    %v26 = arith.addi %v25, %c1 : i32
    %v27 = arith.addi %v26, %c1 : i32
    %v28 = arith.addi %v27, %c1 : i32
    %v29 = arith.addi %v28, %c1 : i32
    %v30 = arith.addi %v29, %c1 : i32
    %v31 = arith.addi %v30, %c1 : i32
    %v32 = arith.addi %v31, %c1 : i32
    %v33 = arith.addi %v32, %c1 : i32
    %v34 = arith.addi %v33, %c1 : i32
    %v35 = arith.addi %v34, %c1 : i32
    %v36 = arith.addi %v35, %c1 : i32
    %v37 = arith.addi %v36, %c1 : i32
    %v38 = arith.addi %v37, %c1 : i32
    %v39 = arith.addi %v38, %c1 : i32
    %v40 = arith.addi %v39, %c1 : i32
    %v41 = arith.addi %v40, %c1 : i32
    %v42 = arith.addi %v41, %c1 : i32
    %v43 = arith.addi %v42, %c1 : i32
    %v44 = arith.addi %v43, %c1 : i32
    %v45 = arith.addi %v44, %c1 : i32
    %v46 = arith.addi %v45, %c1 : i32
    %v47 = arith.addi %v46, %c1 : i32
    %v48 = arith.addi %v47, %c1 : i32
    %v49 = arith.addi %v48, %c1 : i32
    %v50 = arith.addi %v49, %c1 : i32
    %v51 = arith.addi %v50, %c1 : i32
    %v52 = arith.addi %v51, %c1 : i32
    %v53 = arith.addi %v52, %c1 : i32
    %v54 = arith.addi %v53, %c1 : i32
    %v55 = arith.addi %v54, %c1 : i32
    %v56 = arith.addi %v55, %c1 : i32
    %v57 = arith.addi %v56, %c1 : i32
    %v58 = arith.addi %v57, %c1 : i32
    %v59 = arith.addi %v58, %c1 : i32
    %v60 = arith.addi %v59, %c1 : i32
    %v61 = arith.addi %v60, %c1 : i32
    %v62 = arith.addi %v61, %c1 : i32
    %v63 = arith.addi %v62, %c1 : i32
    %v64 = arith.addi %v63, %c1 : i32
    memref.store %v64, %buf[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %out = memref.load %buf[] : memref<i32>
    memref.dealloc %buf : memref<i32>
    return %out : i32
  }
}
