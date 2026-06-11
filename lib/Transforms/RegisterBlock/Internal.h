//===- Internal.h - shared declarations for the register-block pass -------===//
//
// WP2 (COSTMODEL_V4_SPEC §3): the affine-register-block pass was a single
// ~2000-line god-file.  It is split into three translation units that share
// this internal header (all in namespace drcompiler::rb):
//   * AffineRegisterBlock.cpp -- analysis helpers + the IR-restructuring
//     transforms (canonicalize / interchange / symm-raise / distribute) + the
//     pass driver;
//   * RegisterBlock/Peel.cpp    -- the three triangular peels;
//   * RegisterBlock/Vectorize.cpp -- family detection + the two explicit
//     vector-dialect micro-kernels.
// Only the cross-file symbols are declared here; each file keeps its own
// helpers `static`.
//
//===----------------------------------------------------------------------===//

#ifndef DRCOMPILER_TRANSFORMS_REGISTERBLOCK_INTERNAL_H
#define DRCOMPILER_TRANSFORMS_REGISTERBLOCK_INTERNAL_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"

namespace drcompiler {
namespace rb {

using mlir::AffineMap;
using mlir::Block;
using mlir::IRMapping;
using mlir::IRRewriter;
using mlir::Location;
using mlir::LogicalResult;
using mlir::Operation;
using mlir::Value;
using mlir::affine::AffineForOp;
using mlir::affine::AffineLoadOp;
using mlir::affine::AffineStoreOp;

/// A register-blockable accumulator: a same-address load/store pair in an
/// innermost reduction loop whose stored value depends on the load.
struct Acc {
  AffineLoadOp load;
  AffineStoreOp store;
  Value memref;
  AffineMap map;
  llvm::SmallVector<Value> operands;
  Value storedVal;
  Location loc;
};

enum class RBFamily { Broadcast, Dot };

/// Cross-stage certification (set by the in-place peel on a MAIN it proved
/// row-disjoint) read by collectAccumulators / findAccPair to skip the
/// syntactic alias guard.
extern const char kAccNoAliasAttr[];

// ---- analysis helpers (defined in AffineRegisterBlock.cpp) ----
AffineForOp onlyChildFor(AffineForOp forOp);
bool isInnermost(AffineForOp loop);
llvm::SmallVector<Acc> collectAccumulators(AffineForOp kLoop);
bool findAccPair(AffineForOp loop, AffineStoreOp &outStore,
                 AffineLoadOp &outLoad);
bool addrDependsOnIV(AffineStoreOp store, Value iv);
bool dependsOn(Value root, Value def, Block *loopBody);
bool accumulatorAliasesInput(Block *body, Value accMemref);
Value hoistOperand(Value v, AffineForOp loop, IRRewriter &rewriter,
                   IRMapping &map);

// ---- family detection + vectorizers (defined in Vectorize.cpp) ----
bool innermostStrideOne(AffineLoadOp load, Value iv);
RBFamily detectFamily(AffineForOp red, AffineForOp sIn, int &nMul);
LogicalResult vectorizeBroadcastBand(AffineForOp red, AffineForOp sIn,
                                     unsigned VL, IRRewriter &rewriter);
LogicalResult vectorizeDotBand(AffineForOp red, unsigned VL,
                               IRRewriter &rewriter);

// ---- triangular peels (defined in Peel.cpp) ----
bool peelTriangularNest(AffineForOp sOut, unsigned mr, unsigned kTileTarget,
                        int64_t effLLC, IRRewriter &rewriter);
bool peelTriangularReduction(AffineForOp sOut, unsigned mr,
                             IRRewriter &rewriter);
bool peelInPlaceTriangularInnermost(AffineForOp sOut, unsigned mr,
                                    IRRewriter &rewriter);

} // namespace rb
} // namespace drcompiler

#endif // DRCOMPILER_TRANSFORMS_REGISTERBLOCK_INTERNAL_H
