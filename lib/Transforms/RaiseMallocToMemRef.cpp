//===- RaiseMallocToMemRef.cpp - Raise malloc/GEP/llvm.store to memref ----===//
//
// Converts Polygeist cgeist's malloc-based allocation patterns back to memref
// dialect operations so that the DataRecomputation pass can analyze them.
//
// Canonical pattern from cgeist:
//   %raw = call @malloc(%bytes)                     : (i64) -> memref<?xi8>
//   %ptr = "polygeist.memref2pointer"(%raw)         : (memref<?xi8>) -> !llvm.ptr
//   %gep = llvm.getelementptr %ptr[%idx]            : ... -> !llvm.ptr, Telem
//   llvm.store %val, %gep                           : Telem, !llvm.ptr
//   %v   = llvm.load %gep                           : !llvm.ptr -> Telem
//   call @free(%ptr)                                : (!llvm.ptr) -> ...
//
//===----------------------------------------------------------------------===//

#include "drcompiler/Transforms/RaiseMallocToMemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CallInterfaces.h"

#define DEBUG_TYPE "raise-malloc-to-memref"
#define RMDBG() llvm::dbgs() << "RAISE: "

namespace mlir {
#define GEN_PASS_DEF_RAISEMALLOCTOMEMREFPASS
#include "drcompiler/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

// ===== Malloc Group Identification =====

/// Information about one malloc site and its uses.
struct MallocGroup {
  Operation *mallocCall;       // func.call @malloc(...)
  Operation *memref2ptrOp;     // "polygeist.memref2pointer"
  Value basePtr;               // !llvm.ptr result of memref2pointer
  Type elementType;            // Inferred from GEP element types
  Value elementCount;          // Dynamic count (index type), or nullptr
  int64_t staticCount = -1;    // >= 0 if statically known
  Operation *freeCall;         // func.call @free(...), or nullptr
  SmallVector<LLVM::GEPOp> gepOps;
  SmallVector<LLVM::StoreOp> directStores;  // stores to base ptr (idx 0)
  SmallVector<LLVM::LoadOp> directLoads;    // loads from base ptr (idx 0)
  bool valid = true;
};

/// Check if an operation is "polygeist.memref2pointer" (unregistered dialect).
static bool isMemref2Pointer(Operation *op) {
  return op->getName().getStringRef() == "polygeist.memref2pointer";
}

/// Try to decompose a malloc byte-size argument into (elementCount, sizeof).
/// Pattern: arith.muli(arith.extsi(count), sizeof_const) or just a constant.
static bool decomposeByteSize(Value byteSize, unsigned elemBytes,
                              Value &outCount, int64_t &outStaticCount) {
  outCount = nullptr;
  outStaticCount = -1;

  // Case 1: constant size
  if (auto constOp = byteSize.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
      int64_t bytes = intAttr.getInt();
      if (elemBytes > 0 && bytes % elemBytes == 0) {
        outStaticCount = bytes / elemBytes;
        return true;
      }
    }
    return false;
  }

  // Case 2: arith.muli(something, sizeof_const) or arith.muli(sizeof_const, something)
  auto muliOp = byteSize.getDefiningOp<arith::MulIOp>();
  if (!muliOp)
    return false;

  // Try both operand orderings
  for (int i = 0; i < 2; i++) {
    Value maybeConst = muliOp.getOperand(i);
    Value maybeCount = muliOp.getOperand(1 - i);

    auto constOp = maybeConst.getDefiningOp<arith::ConstantOp>();
    if (!constOp)
      continue;
    auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue());
    if (!intAttr || static_cast<unsigned>(intAttr.getInt()) != elemBytes)
      continue;

    // Found sizeof. Now unwrap arith.extsi if present.
    if (auto extOp = maybeCount.getDefiningOp<arith::ExtSIOp>())
      maybeCount = extOp.getIn();

    // Check if count is a constant
    if (auto countConst = maybeCount.getDefiningOp<arith::ConstantOp>()) {
      if (auto ci = dyn_cast<IntegerAttr>(countConst.getValue())) {
        outStaticCount = ci.getInt();
        return true;
      }
    }

    // Dynamic count
    outCount = maybeCount;
    return true;
  }

  return false;
}

/// Get the byte size of an LLVM/builtin integer or float type.
static unsigned getTypeSizeInBytes(Type type) {
  if ( ! type.isIntOrFloat() ) return 0;
  // For non-byte-aligned types
  return (type.getIntOrFloatBitWidth() + 7) / 8;
}

/// Analyze all users of basePtr and populate the MallocGroup.
static void analyzePtrUses(MallocGroup &group) {
  Type inferredElemType = nullptr;

  for (OpOperand &use : group.basePtr.getUses()) {
    Operation *user = use.getOwner();

    // GEP use
    if (auto gepOp = dyn_cast<LLVM::GEPOp>(user)) {
      group.gepOps.push_back(gepOp);

      // Infer element type from GEP
      Type gepElemType = gepOp.getElemType();
      if (!inferredElemType) {
        inferredElemType = gepElemType;
      } else if (inferredElemType != gepElemType) {
        // Type inconsistency — bail out
        group.valid = false;
        return;
      }
      // Check GEP result uses: only store/load allowed
      for (OpOperand &gepUse : gepOp.getResult().getUses()) {
        Operation *gepUser = gepUse.getOwner();
        if (!isa<LLVM::StoreOp>(gepUser) && !isa<LLVM::LoadOp>(gepUser)) {
          group.valid = false;
          return;
        }
      }
      continue;
    }

    // Direct store to base ptr (index 0)
    if (auto storeOp = dyn_cast<LLVM::StoreOp>(user)) {
      if (storeOp.getAddr() == group.basePtr) {
        group.directStores.push_back(storeOp);
        Type valType = storeOp.getValue().getType();
        if (!inferredElemType)
          inferredElemType = valType;
        else if (inferredElemType != valType) {
          group.valid = false;
          return;
        }
        continue;
      }
    }

    // Direct load from base ptr (index 0)
    if (auto loadOp = dyn_cast<LLVM::LoadOp>(user)) {
      if (loadOp.getAddr() == group.basePtr) {
        group.directLoads.push_back(loadOp);
        Type valType = loadOp.getResult().getType();
        if (!inferredElemType)
          inferredElemType = valType;
        else if (inferredElemType != valType) {
          group.valid = false;
          return;
        }
        continue;
      }
    }

    // call @free: record but don't bail
    if (auto callOp = dyn_cast<CallOpInterface>(user)) {
      auto callableRef = callOp.getCallableForCallee();
      if (auto sym = dyn_cast<SymbolRefAttr>(callableRef)) {
        if (sym.getRootReference().getValue() == "free") {
          group.freeCall = user;
          continue;
        }
      }
    }

    // Any other use: pointer escapes — bail
    group.valid = false;
    return;
  }

  group.elementType = inferredElemType;
  if (!inferredElemType) {
    group.valid = false;
    return;
  }

  // Decompose byte size
  unsigned elemBytes = getTypeSizeInBytes(inferredElemType);
  if (elemBytes == 0) {
    group.valid = false;
    return;
  }

  Value mallocSize = group.mallocCall->getOperand(0);
  if (!decomposeByteSize(mallocSize, elemBytes,
                         group.elementCount, group.staticCount)) {
    group.valid = false;
  }
}

/// Unwrap an index: if it came from arith.index_cast(index -> iN), return
/// the original index value. Otherwise insert a new index_cast.
static Value toIndexType(Value val, OpBuilder &builder, Location loc) {
  if (isa<IndexType>(val.getType()))
    return val;

  // Unwrap: arith.index_cast(index -> i32) → get the index
  if (auto castOp = val.getDefiningOp<arith::IndexCastOp>()) {
    if (isa<IndexType>(castOp.getIn().getType()))
      return castOp.getIn();
  }

  return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), val);
}

// ===== Pass =====

class RaiseMallocToMemRefPass final
    : public impl::RaiseMallocToMemRefPassBase<RaiseMallocToMemRefPass> {
public:
  using RaiseMallocToMemRefPassBase::RaiseMallocToMemRefPassBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    SymbolTableCollection symTables;

    // Phase 1: Identify malloc groups.
    SmallVector<MallocGroup> groups;

    moduleOp.walk([&](CallOpInterface callOp) {
      auto callableRef = callOp.getCallableForCallee();
      auto sym = dyn_cast<SymbolRefAttr>(callableRef);
      if (!sym || sym.getRootReference().getValue() != "malloc")
        return;

      Operation *op = callOp.getOperation();
      if (op->getNumResults() != 1)
        return;

      Value mallocResult = op->getResult(0);
      if (!isa<MemRefType>(mallocResult.getType()))
        return;

      // Find the single polygeist.memref2pointer user
      Operation *m2pOp = nullptr;
      for (OpOperand &use : mallocResult.getUses()) {
        if (isMemref2Pointer(use.getOwner())) {
          if (m2pOp) return;  // multiple memref2pointer — skip
          m2pOp = use.getOwner();
        }
      }
      if (!m2pOp || m2pOp->getNumResults() != 1)
        return;

      MallocGroup group;
      group.mallocCall = op;
      group.memref2ptrOp = m2pOp;
      group.basePtr = m2pOp->getResult(0);
      analyzePtrUses(group);

      if (group.valid)
        groups.push_back(std::move(group));
    });

    // Phase 2 & 3: Rewrite each group.
    for (MallocGroup &group : groups) {
      OpBuilder builder(group.mallocCall);
      Location loc = group.mallocCall->getLoc();

      // Create memref.alloc
      MemRefType memrefType;
      Value alloc;
      if (group.staticCount >= 0) {
        memrefType =
            MemRefType::get({group.staticCount}, group.elementType);
        alloc = builder.create<memref::AllocOp>(loc, memrefType);
      } else {
        memrefType =
            MemRefType::get({ShapedType::kDynamic}, group.elementType);
        Value countIdx =
            toIndexType(group.elementCount, builder, loc);
        alloc = builder.create<memref::AllocOp>(loc, memrefType,
                                                ValueRange{countIdx});
      }

      Value c0;  // lazily created constant 0 : index

      auto getZeroIndex = [&]() -> Value {
        if (!c0) {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointAfterValue(alloc);
          c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
        }
        return c0;
      };

      // Rewrite GEP + store/load
      for (LLVM::GEPOp gepOp : group.gepOps) {
        // GEP has dynamic indices — the first dynamic index is the array index.
        // cgeist pattern: llvm.getelementptr %ptr[%idx] : (!llvm.ptr, iN) -> !llvm.ptr, T
        Value gepIdx;
        auto dynIndices = gepOp.getDynamicIndices();
        if (!dynIndices.empty()) {
          gepIdx = dynIndices[0];
        } else {
          // All-constant GEP — extract from attributes
          // For our pattern this shouldn't happen, but handle gracefully
          gepIdx = nullptr;
        }

        for (OpOperand &use :
             llvm::make_early_inc_range(gepOp.getResult().getUses())) {
          Operation *user = use.getOwner();
          builder.setInsertionPoint(user);

          Value idx = gepIdx ? toIndexType(gepIdx, builder, user->getLoc())
                             : getZeroIndex();

          if (auto storeOp = dyn_cast<LLVM::StoreOp>(user)) {
            builder.create<memref::StoreOp>(
                user->getLoc(), storeOp.getValue(), alloc, ValueRange{idx});
            user->erase();
          } else if (auto loadOp = dyn_cast<LLVM::LoadOp>(user)) {
            auto newLoad = builder.create<memref::LoadOp>(
                user->getLoc(), alloc, ValueRange{idx});
            loadOp.getResult().replaceAllUsesWith(newLoad.getResult());
            user->erase();
          }
        }
        // Erase the now-dead GEP
        if (gepOp->use_empty())
          gepOp->erase();
      }

      // Rewrite direct stores (index 0)
      for (LLVM::StoreOp storeOp : group.directStores) {
        builder.setInsertionPoint(storeOp);
        builder.create<memref::StoreOp>(
            storeOp->getLoc(), storeOp.getValue(), alloc,
            ValueRange{getZeroIndex()});
        storeOp->erase();
      }

      // Rewrite direct loads (index 0)
      for (LLVM::LoadOp loadOp : group.directLoads) {
        builder.setInsertionPoint(loadOp);
        auto newLoad = builder.create<memref::LoadOp>(
            loadOp->getLoc(), alloc, ValueRange{getZeroIndex()});
        loadOp.getResult().replaceAllUsesWith(newLoad.getResult());
        loadOp->erase();
      }

      // Rewrite free → dealloc
      if (group.freeCall) {
        builder.setInsertionPoint(group.freeCall);
        builder.create<memref::DeallocOp>(group.freeCall->getLoc(), alloc);
        group.freeCall->erase();
      }

      // Erase memref2pointer and malloc (now dead)
      group.memref2ptrOp->erase();
      group.mallocCall->erase();
    }

    // Phase 4: Lower any remaining polygeist.memref2pointer ops that were NOT
    // part of a malloc group.  These arise from non-malloc memrefs (e.g.,
    // function arguments, allocas) being cast to raw pointers for I/O or
    // pointer arithmetic.
    //
    // Lowering:
    //   %ptr = "polygeist.memref2pointer"(%m) : (memref<...xT>) -> !llvm.ptr
    // becomes:
    //   %idx = memref.extract_aligned_pointer_as_index %m : memref<...xT> -> index
    //   %int = arith.index_cast %idx : index to i64
    //   %ptr = llvm.inttoptr %int : i64 to !llvm.ptr
    SmallVector<Operation *> residualM2P;
    moduleOp.walk([&](Operation *op) {
      if (isMemref2Pointer(op))
        residualM2P.push_back(op);
    });

    for (Operation *op : residualM2P) {
      OpBuilder builder(op);
      Location loc = op->getLoc();
      Value memrefVal = op->getOperand(0);

      Value ptr;
      if (isa<LLVM::LLVMPointerType>(memrefVal.getType())) {
        // Input is already !llvm.ptr (e.g., after struct memref → !llvm.ptr
        // preprocessing).  The memref2pointer is an identity cast.
        ptr = memrefVal;
      } else {
        Value idx = builder.create<memref::ExtractAlignedPointerAsIndexOp>(
            loc, memrefVal);
        Value i64Val = builder.create<arith::IndexCastOp>(
            loc, builder.getI64Type(), idx);
        ptr = builder.create<LLVM::IntToPtrOp>(
            loc, LLVM::LLVMPointerType::get(op->getContext()), i64Val);
      }

      op->getResult(0).replaceAllUsesWith(ptr);
      op->erase();
    }
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createRaiseMallocToMemRefPass() {
  return std::make_unique<RaiseMallocToMemRefPass>();
}
} // namespace mlir
