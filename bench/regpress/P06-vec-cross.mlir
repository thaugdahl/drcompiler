// vec<8xf32> args: 4 inputs, 10 cross-products kept live before stores.
func.func @p06(%A: memref<10xvector<8xf32>>, %a: vector<8xf32>, %b: vector<8xf32>,
                %c: vector<8xf32>, %d: vector<8xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %c8 = arith.constant 8 : index
  %c9 = arith.constant 9 : index
  %v0 = arith.mulf %a, %b : vector<8xf32>
  %v1 = arith.mulf %a, %c : vector<8xf32>
  %v2 = arith.mulf %a, %d : vector<8xf32>
  %v3 = arith.mulf %b, %c : vector<8xf32>
  %v4 = arith.mulf %b, %d : vector<8xf32>
  %v5 = arith.mulf %c, %d : vector<8xf32>
  %v6 = arith.mulf %v0, %v3 : vector<8xf32>
  %v7 = arith.mulf %v1, %v4 : vector<8xf32>
  %v8 = arith.mulf %v2, %v5 : vector<8xf32>
  %v9 = arith.mulf %v6, %v7 : vector<8xf32>
  memref.store %v0, %A[%c0] : memref<10xvector<8xf32>>
  memref.store %v1, %A[%c1] : memref<10xvector<8xf32>>
  memref.store %v2, %A[%c2] : memref<10xvector<8xf32>>
  memref.store %v3, %A[%c3] : memref<10xvector<8xf32>>
  memref.store %v4, %A[%c4] : memref<10xvector<8xf32>>
  memref.store %v5, %A[%c5] : memref<10xvector<8xf32>>
  memref.store %v6, %A[%c6] : memref<10xvector<8xf32>>
  memref.store %v7, %A[%c7] : memref<10xvector<8xf32>>
  memref.store %v8, %A[%c8] : memref<10xvector<8xf32>>
  memref.store %v9, %A[%c9] : memref<10xvector<8xf32>>
  return
}
