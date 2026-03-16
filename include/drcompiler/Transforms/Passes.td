#ifndef DRCOMPILER_TRANSFORMS_PASSES_TD
#define DRCOMPILER_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def DataRecomputationPass : Pass<"data-recomputation", "mlir::ModuleOp">
{
  let summary = "Applies data recomputation where likely to be profitable";

  let description = [{
    Load-store provenance analysis: for each memref.load (and llvm.load),
    determines which store operations could have been the last writer.

    Loads are classified into four categories:
      SINGLE  - exactly one known store (recomputation candidate)
      MULTI   - multiple possible stores
      LEAKED  - provenance includes an external/unknown write (nullptr)
      KILLED  - all reaching stores were killed (empty provenance set)

    Both memref and LLVM dialect ops are supported:
      memref.alloc / llvm.call @malloc  → allocation roots
      memref.store / llvm.store         → tracked stores
      memref.load  / llvm.load          → tracked loads
    LLVM loads/stores are tracked with flat (no index) coverage, so they
    are more likely to be classified as MULTI.

    When --dr-recompute is set, SINGLE loads are replaced by rematerializing
    the stored value at the load site (memref loads only).
  }];

  let options = [
    Option<"drTestDiagnostics", "dr-test-diagnostics", "bool", "false",
           "Enables test output">,
    Option<"drDotFile", "dr-dot-file", "std::string", /*default=*/"",
           "Output file path for load-provenance DOT graph">,
    Option<"drRecompute", "dr-recompute", "bool", "false",
           "Replace SINGLE loads with the reaching store's value">,
    Option<"drCostModel", "dr-cost-model", "bool", "false",
           "Gate recomputation on a cache-aware cost model">,
    Option<"drL1Size", "dr-l1-size", "unsigned", "32768",
           "L1 data cache size in bytes">,
    Option<"drL2Size", "dr-l2-size", "unsigned", "262144",
           "L2 cache size in bytes">,
    Option<"drL1Latency", "dr-l1-latency", "unsigned", "4",
           "L1 cache hit latency in cycles">,
    Option<"drL2Latency", "dr-l2-latency", "unsigned", "12",
           "L2 cache hit latency in cycles">,
    Option<"drL3Latency", "dr-l3-latency", "unsigned", "40",
           "L3 cache hit latency in cycles">,
    Option<"drMemLatency", "dr-mem-latency", "unsigned", "200",
           "DRAM access latency in cycles">
  ];

  let dependentDialects = [
    "memref::MemRefDialect", "vector::VectorDialect",
    "affine::AffineDialect", "ptr::PtrDialect",
    "LLVM::LLVMDialect"
  ];

  let constructor = "mlir::createDataRecomputationPass()";
}

def RaiseMallocToMemRefPass
    : Pass<"raise-malloc-to-memref", "mlir::ModuleOp">
{
  let summary = "Raise malloc/free + LLVM load/store/GEP to memref ops";

  let description = [{
    Converts patterns produced by Polygeist cgeist for malloc-based allocations
    back to memref dialect operations.  This is a pre-pass for
    DataRecomputation.

    Canonical input pattern (per malloc site):
      %raw = call @malloc(%bytes) : (i64) -> memref<?xi8>
      %ptr = "polygeist.memref2pointer"(%raw) : (memref<?xi8>) -> !llvm.ptr
      %gep = llvm.getelementptr %ptr[%idx] : (!llvm.ptr, ...) -> !llvm.ptr, T
      llvm.store %val, %gep : T, !llvm.ptr
      %v   = llvm.load %gep : !llvm.ptr -> T
      call @free(%ptr)

    Target output:
      %alloc = memref.alloc(%count) : memref<?xT>
      memref.store %val, %alloc[%idx] : memref<?xT>
      %v = memref.load %alloc[%idx] : memref<?xT>
      memref.dealloc %alloc : memref<?xT>

    Malloc sites whose pointer escapes (passed to a call other than @free,
    stored to memory, cast to integer) are left untouched.
  }];

  let dependentDialects = [
    "memref::MemRefDialect", "arith::ArithDialect",
    "LLVM::LLVMDialect"
  ];

  let constructor = "mlir::createRaiseMallocToMemRefPass()";
}

def MemoryFissionPass
    : Pass<"memory-fission", "mlir::ModuleOp">
{
  let summary = "Cache-aware loop fission: materialize redundant computation into buffers";

  let description = [{
    Identifies expensive computation subtrees that are duplicated across
    multiple loops (typically produced by cgeist -O2 loop fusion) and, when
    the cost model indicates it is profitable, hoists the shared computation
    into a separate producer loop backed by a materialization buffer.

    This is the inverse of loop fusion: it introduces memory to avoid
    redundant computation when the buffer fits in cache and the ALU cost
    exceeds the load cost.

    The pass fingerprints computation chains within affine.for loop bodies
    to detect structural duplicates reading from the same source memref.
  }];

  let options = [
    Option<"l1Size", "l1-size", "unsigned", "32768",
           "L1 data cache size in bytes">,
    Option<"l2Size", "l2-size", "unsigned", "262144",
           "L2 cache size in bytes">,
    Option<"l1Latency", "l1-latency", "unsigned", "4",
           "L1 cache hit latency in cycles">,
    Option<"l2Latency", "l2-latency", "unsigned", "12",
           "L2 cache hit latency in cycles">,
    Option<"l3Latency", "l3-latency", "unsigned", "40",
           "L3 cache hit latency in cycles">,
    Option<"testDiagnostics", "test-diagnostics", "bool", "false",
           "Emit remarks for test diagnostics">
  ];

  let dependentDialects = [
    "memref::MemRefDialect", "affine::AffineDialect",
    "arith::ArithDialect", "math::MathDialect"
  ];

  let constructor = "mlir::createMemoryFissionPass()";
}

#endif
