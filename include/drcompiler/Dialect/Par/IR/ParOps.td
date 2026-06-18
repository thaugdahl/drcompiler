//===- ParOps.td - Runtime-agnostic parallel dialect (M2) --------------===//
//
// The `par` dialect is the contract between the bubble-widening front-end and
// the parallel-runtime back-ends (PARALLEL_PAR_DIALECT_SPEC.md).  M2 defines
// the core ops materialized by `dr-par-bubbles{par-materialize}`:
//   par.region  - a maximal parallel region (one worker team)
//   par.forall  - a constant rectangular iteration space distributed to workers
//   par.yield   - region/forall terminator
// barrier / redistribute / critical / reduce land with M3/M5.
//
//===----------------------------------------------------------------------===//

#ifndef DRCOMPILER_DIALECT_PAR_IR_PAROPS_TD
#define DRCOMPILER_DIALECT_PAR_IR_PAROPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

def Par_Dialect : Dialect {
  let name = "par";
  let summary = "Runtime-agnostic parallel regions (drcompiler)";
  let description = [{
    Names *what* is parallel and *how a constant iteration space maps to
    workers*, never a concrete runtime.  Lowered to scf (sequential reference,
    M2), and later to OpenMP / libdrpar (Phase B).
  }];
  let cppNamespace = "::mlir::par";
}

class Par_Op<string mnemonic, list<Trait> traits = []>
    : Op<Par_Dialect, mnemonic, traits>;

def Par_RegionOp : Par_Op<"region",
    [SingleBlock, NoRegionArguments, RecursiveMemoryEffects]> {
  let summary = "A maximal region executed concurrently by a worker team";
  let description = [{
    The worker team is established at region entry and torn down at exit; nested
    `par.forall`s share the team (one fork/join per region).
  }];
  let regions = (region SizedRegion<1>:$region);
  let assemblyFormat = "$region attr-dict";
}

def Par_ForallOp : Par_Op<"forall",
    [SingleBlock, RecursiveMemoryEffects]> {
  let summary = "Distribute a constant rectangular iteration space across workers";
  let description = [{
    `lowerBounds`, `upperBounds` and `steps` are equal-length constant arrays,
    one entry per distributed dimension; the body block carries one `index`
    induction variable per dimension.  Iterations are partitioned across the
    team (block by default); no order is guaranteed across workers.  (Dynamic /
    affine bounds are a follow-up; M2 is constant-bound.)
  }];
  let arguments = (ins DenseI64ArrayAttr:$lowerBounds,
                       DenseI64ArrayAttr:$upperBounds,
                       DenseI64ArrayAttr:$steps);
  let regions = (region SizedRegion<1>:$region);
  let assemblyFormat =
      "`(` $lowerBounds `,` $upperBounds `,` $steps `)` $region attr-dict";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    unsigned getNumDims() { return getLowerBounds().size(); }
    ::mlir::Block::BlockArgListType getInductionVars() {
      return getBody()->getArguments();
    }
  }];
}

def Par_BarrierOp : Par_Op<"barrier", [HasParent<"RegionOp">]> {
  let summary = "Team barrier: all workers synchronize here";
  let description = [{
    Separates two `par.forall`s in one region when the second consumes data the
    first produces across iterations (an offset producer->consumer dependence):
    keep both parallel, synchronize between them.
  }];
  let assemblyFormat = "attr-dict";
}

def Par_CriticalOp : Par_Op<"critical",
    [SingleBlock, NoRegionArguments, RecursiveMemoryEffects,
     ParentOneOf<["RegionOp", "ForallOp"]>]> {
  let summary = "A sub-region executed by a single worker, in order";
  let description = [{
    Target for peeled boundary slabs (the conflicting sub-iteration-space of an
    affine-overlap peel).  Executed by exactly one worker while the team waits.
  }];
  let regions = (region SizedRegion<1>:$region);
  let assemblyFormat = "$region attr-dict";
}

def Par_RedistributeOp : Par_Op<"redistribute", [HasParent<"RegionOp">]> {
  let summary = "Re-block a memref's worker<->data mapping between sub-regions";
  let description = [{
    Transitory region between two `par.forall`s whose worker->data mappings
    disagree (e.g. a row-parallel producer feeding a column-parallel consumer).
    `from`/`to` name the source and target distributions.
  }];
  let arguments = (ins AnyMemRef:$value, StrAttr:$from, StrAttr:$to);
  let assemblyFormat = "$value `:` type($value) `from` $from `to` $to attr-dict";
}

def Par_YieldOp : Par_Op<"yield",
    [Pure, ReturnLike, Terminator,
     ParentOneOf<["RegionOp", "ForallOp", "CriticalOp"]>]> {
  let summary = "Terminator for par.region / par.forall / par.critical bodies";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
}

#endif // DRCOMPILER_DIALECT_PAR_IR_PAROPS_TD
