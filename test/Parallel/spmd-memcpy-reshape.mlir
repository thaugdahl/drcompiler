// lower-krnl-global lowers krnl.memcpy.  The openai-gpt multi-head attention
// reshape/transpose bands are krnl.memcpy with an affine.apply flat offset; the
// old lowering emitted a flat scf.for { memref.load; memref.store } over 1-D
// reinterpret_cast views -- an opaque, linearized body that made par-spmd-perband
// bail the band to par.critical (the SPMD serial floor, PARALLEL_SPMD_SPEC.md
// §11.20).  The multi-D lowering delinearizes the flat offset against the memref
// strides into a per-dimension affine copy, so the dependence test proves the
// shard axis owner-disjoint and the band shards into a par.forall.

// RUN: dr-opt %s -allow-unregistered-dialect --pass-pipeline='builtin.module(lower-krnl-global)' | FileCheck %s
// RUN: dr-opt %s -allow-unregistered-dialect --pass-pipeline='builtin.module(lower-krnl-global,dr-par-bubbles{par-spmd-perband})' | FileCheck %s --check-prefix=SHARD

// The head reshape/transpose: dst[b][h][s][i] = src[b][s][h][i].  Offsets are
// affine.apply of the band IVs with coefficients = the destination strides
// (dst 1x12x128x64 strides [98304,8192,64,1]; src 1x128x12x64 strides
// [98304,768,64,1]); the c64 copy fills the contiguous innermost dim.
#dOff = affine_map<(d0, d1, d2) -> (d0 * 98304 + d1 * 64 + d2 * 8192)>
#sOff = affine_map<(d0, d1, d2) -> (d0 * 98304 + d1 * 768 + d2 * 64)>

// Multi-D affine copy on the ORIGINAL memrefs (no 1-D reinterpret_cast, no
// scf.for): src[b,s,h,i] -> dst[b,h,s,i] (the s<->h transpose).
// CHECK-LABEL: func.func @head_reshape
// CHECK-NOT:     memref.reinterpret_cast
// CHECK:         affine.for %[[I:.*]] = 0 to 64
// CHECK:           %[[V:.*]] = affine.load %{{.*}}[%[[B:.*]], %[[S:.*]], %[[H:.*]], %[[I]]] : memref<1x128x12x64xf32>
// CHECK:           affine.store %[[V]], %{{.*}}[%[[B]], %[[H]], %[[S]], %[[I]]] : memref<1x12x128x64xf32>
// CHECK-NOT:     scf.for

// The band shards on the seq axis (extent 128) into a par.forall -- no critical.
// SHARD-LABEL: func.func @head_reshape
// SHARD:         par.region {
// SHARD:           par.forall([0], [128], [1]) {
// SHARD-NOT:       par.critical
func.func @head_reshape(%dst: memref<1x12x128x64xf32>, %src: memref<1x128x12x64xf32>) {
  %c64 = arith.constant 64 : i64
  affine.for %b = 0 to 1 {
    affine.for %s = 0 to 128 {
      affine.for %h = 0 to 12 {
        %do = affine.apply #dOff(%b, %s, %h)
        %so = affine.apply #sOff(%b, %s, %h)
        "krnl.memcpy"(%dst, %src, %c64, %do, %so) : (memref<1x12x128x64xf32>, memref<1x128x12x64xf32>, i64, index, index) -> ()
      }
    }
  }
  return
}
