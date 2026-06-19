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
    [SingleBlock, RecursiveMemoryEffects, AttrSizedOperandSegments]> {
  let summary = "Distribute a constant rectangular iteration space across workers";
  let description = [{
    `lowerBounds`, `upperBounds` and `steps` are equal-length constant arrays,
    one entry per distributed dimension; the body block carries one `index`
    induction variable per dimension.  Iterations are partitioned across the
    team (block by default); no order is guaranteed across workers.  A dim whose
    `upperBounds` entry is the kDynamic sentinel takes its bound from the
    matching `dynamicUpperBounds` operand (runtime extent).

    When `initVals` (and matching `results`) are present the body's terminator
    is a `par.reduce` contributing one value per result through an associative
    combinator (cross-worker reduction over the distributed space, scf.parallel
    style); otherwise the terminator is `par.yield`.
  }];
  let arguments = (ins DenseI64ArrayAttr:$lowerBounds,
                       DenseI64ArrayAttr:$upperBounds,
                       DenseI64ArrayAttr:$steps,
                       Variadic<Index>:$dynamicUpperBounds,
                       Variadic<AnyType>:$initVals);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);
  let assemblyFormat =
      "`(` $lowerBounds `,` $upperBounds `,` $steps `)` "
      "(`dyn` `(` $dynamicUpperBounds^ `:` type($dynamicUpperBounds) `)`)? "
      "(`init` `(` $initVals^ `:` type($initVals) `)`)? "
      "$region (`->` type($results)^)? attr-dict";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    unsigned getNumDims() { return getLowerBounds().size(); }
    ::mlir::Block::BlockArgListType getInductionVars() {
      return getBody()->getArguments();
    }
    /// A dim's upper bound is dynamic when its static entry is the kDynamic
    /// sentinel; its value is then the matching $dynamicUpperBounds operand.
    bool isDynamicUpperBound(unsigned d) {
      return ::mlir::ShapedType::isDynamic(getUpperBounds()[d]);
    }
    ::mlir::Value getDynamicUpperBound(unsigned d) {
      unsigned k = 0;
      for (unsigned i = 0; i < d; ++i)
        if (::mlir::ShapedType::isDynamic(getUpperBounds()[i])) ++k;
      return getDynamicUpperBounds()[k];
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

def Par_ReduceOp : Par_Op<"reduce",
    [Terminator, HasParent<"ForallOp">]> {
  let summary = "Associative cross-worker reduction contribution (kind-based)";
  let description = [{
    Terminator of a reducing `par.forall`: each iteration contributes one value
    per enclosing-forall result through an associative+commutative combinator
    named by `kinds` (one per contribution): 0=addf 1=mulf 2=maxnumf 3=minnumf
    4=addi 5=muli 6=maxsi 7=minsi.  Workers reduce their slice into a private
    partial; a final tree-combine produces the forall result (under
    `fastmath<reassoc>` for FP).  Kind-based (not a combinator region) to match
    the back-end `__drpar_reduce_*(perThread, n, out, kind)` ABI.
  }];
  let arguments = (ins Variadic<AnyType>:$contributions, DenseI64ArrayAttr:$kinds);
  let assemblyFormat =
      "`(` $contributions `:` type($contributions) `)` `kinds` $kinds attr-dict";
  let hasVerifier = 1;
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
