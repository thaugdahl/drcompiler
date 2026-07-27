// DR_PIPELINE: builtin.module(data-recomputation{dr-recompute=true dr-cost-model=true dr-partial-remat=true})
// R14: Producer fills 16MB buffer (DRAM). Consumer reads it.
// Producer: buf[i] = weights[i%16] * 2.0 + 1.0
// weights is 16 elements (128B, L1-resident).
// Partial remat: replace consumer's DRAM load with recomputed value.
// Expected: DRAM load (~200cy) → L1 load + 2 ALU ops (~6cy).

module {
  func.func @main() -> i32 {
    %cst = arith.constant 0.0 : f64
    %c2 = arith.constant 2.0 : f64
    %c1 = arith.constant 1.0 : f64

    %weights = memref.alloc() : memref<16xf64>
    affine.for %w = 0 to 16 {
      %wi = arith.index_cast %w : index to i32
      %wf = arith.sitofp %wi : i32 to f64
      %wval = arith.addf %wf, %c1 : f64
      affine.store %wval, %weights[%w] : memref<16xf64>
    }

    // 2M elements = 16MB f64 → DRAM
    %buf = memref.alloc() : memref<2097152xf64>

    %result = affine.for %rep = 0 to 20 iter_args(%outer = %cst) -> (f64) {
      // Producer: buf[i] = weights[i%16] * 2.0 + 1.0
      affine.for %i = 0 to 2097152 {
        %wi = affine.apply affine_map<(d0) -> (d0 mod 16)>(%i)
        %wv = affine.load %weights[%wi] : memref<16xf64>
        %m = arith.mulf %wv, %c2 : f64
        %val = arith.addf %m, %c1 : f64
        affine.store %val, %buf[%i] : memref<2097152xf64>
      }
      // Consumer: sum of buf
      %inner = affine.for %j = 0 to 2097152 iter_args(%acc = %cst) -> (f64) {
        %v = affine.load %buf[%j] : memref<2097152xf64>
        %s = arith.addf %acc, %v : f64
        affine.yield %s : f64
      }
      %next = arith.addf %outer, %inner : f64
      affine.yield %next : f64
    }

    memref.dealloc %weights : memref<16xf64>
    memref.dealloc %buf : memref<2097152xf64>
    %ret = arith.fptosi %result : f64 to i32
    return %ret : i32
  }
}
