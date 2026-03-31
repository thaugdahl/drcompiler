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
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
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

/// Check if an operation is "polygeist.pointer2memref" (unregistered dialect).
static bool isPointer2Memref(Operation *op) {
  return op->getName().getStringRef() == "polygeist.pointer2memref";
}

/// Check if an operation is "polygeist.subindex" (unregistered dialect).
static bool isSubIndex(Operation *op) {
  return op->getName().getStringRef() == "polygeist.subindex";
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

/// Convert an index-typed value to i64 for use with LLVM GEP ops.
/// If the value is already i64, return it directly; if it came from
/// arith.index_cast(i64 -> index), unwrap.
static Value toI64Type(Value val, OpBuilder &builder, Location loc) {
  if (val.getType().isInteger(64))
    return val;

  // Unwrap: arith.index_cast(i64 -> index) → get the i64
  if (auto castOp = val.getDefiningOp<arith::IndexCastOp>()) {
    if (castOp.getIn().getType().isInteger(64))
      return castOp.getIn();
  }

  return builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), val);
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

    // Phase 5: Lower remaining polygeist.pointer2memref ops.
    //
    // These arise when cgeist converts C pointer arguments or globals to
    // memref types:
    //   %m = "polygeist.pointer2memref"(%ptr) : (!llvm.ptr) -> memref<?xT>
    //
    // Use patterns handled:
    //   (a) ptr → memref → memref2pointer: identity chain, fold away
    //   (b) ptr → memref → memref.load/store: lower to llvm.getelementptr
    //       + llvm.load/store, bypassing the memref layer entirely
    //   (c) ptr → memref → affine.load/store: same as (b) for rank-1
    //       identity-map cases
    //   (d) ptr → memref → passed to function: need memref descriptor —
    //       leave the op (file falls back to clang)
    SmallVector<Operation *> residualP2M;
    moduleOp.walk([&](Operation *op) {
      if (isPointer2Memref(op))
        residualP2M.push_back(op);
    });

    for (Operation *op : residualP2M) {
      Value ptrVal = op->getOperand(0);
      Value memrefResult = op->getResult(0);

      // Identity cast: !llvm.ptr → !llvm.ptr
      if (memrefResult.getType() == ptrVal.getType()) {
        memrefResult.replaceAllUsesWith(ptrVal);
        op->erase();
        continue;
      }

      auto memrefTy = dyn_cast<MemRefType>(memrefResult.getType());
      if (!memrefTy) continue;

      // Determine the element type for GEP addressing.
      Type elemTy = memrefTy.getElementType();
      auto llvmPtrTy = LLVM::LLVMPointerType::get(op->getContext());

      // Try to lower every use.  If any use cannot be handled, we must
      // leave the pointer2memref in place (file falls back to clang).
      bool allHandled = true;

      for (OpOperand &use :
           llvm::make_early_inc_range(memrefResult.getUses())) {
        Operation *user = use.getOwner();

        // (a) memref2pointer: fold the chain.
        if (isMemref2Pointer(user)) {
          user->getResult(0).replaceAllUsesWith(ptrVal);
          user->erase();
          continue;
        }

        // (b) memref.load: lower to llvm.getelementptr + llvm.load.
        if (auto loadOp = dyn_cast<memref::LoadOp>(user)) {
          if (loadOp.getMemRef() != memrefResult) {
            allHandled = false;
            continue;
          }
          auto indices = loadOp.getIndices();
          OpBuilder builder(loadOp);
          Location loc = loadOp.getLoc();

          Value addr = ptrVal;
          if (!indices.empty()) {
            // Linearize: for rank-1 (the common case from C pointers),
            // the single index is the GEP offset.  For higher ranks we
            // would need stride computation — bail out for now.
            if (indices.size() != 1) {
              allHandled = false;
              continue;
            }
            Value idx = toI64Type(indices[0], builder, loc);
            addr = builder.create<LLVM::GEPOp>(
                loc, llvmPtrTy, elemTy, ptrVal, ValueRange{idx});
          }
          auto newLoad = builder.create<LLVM::LoadOp>(
              loc, loadOp.getResult().getType(), addr);
          loadOp.getResult().replaceAllUsesWith(newLoad.getResult());
          loadOp->erase();
          continue;
        }

        // (b) memref.store: lower to llvm.getelementptr + llvm.store.
        if (auto storeOp = dyn_cast<memref::StoreOp>(user)) {
          if (storeOp.getMemRef() != memrefResult) {
            allHandled = false;
            continue;
          }
          auto indices = storeOp.getIndices();
          OpBuilder builder(storeOp);
          Location loc = storeOp.getLoc();

          Value addr = ptrVal;
          if (!indices.empty()) {
            if (indices.size() != 1) {
              allHandled = false;
              continue;
            }
            Value idx = toI64Type(indices[0], builder, loc);
            addr = builder.create<LLVM::GEPOp>(
                loc, llvmPtrTy, elemTy, ptrVal, ValueRange{idx});
          }
          builder.create<LLVM::StoreOp>(loc, storeOp.getValue(), addr);
          storeOp->erase();
          continue;
        }

        // (c) affine.load: lower for rank-1 identity-map cases.
        if (auto affLoad = dyn_cast<affine::AffineLoadOp>(user)) {
          if (affLoad.getMemRef() != memrefResult) {
            allHandled = false;
            continue;
          }
          AffineMap map = affLoad.getAffineMap();
          auto mapOperands = affLoad.getMapOperands();
          // Only handle rank-1 with identity or single-result maps.
          if (map.getNumResults() != 1) {
            allHandled = false;
            continue;
          }
          OpBuilder builder(affLoad);
          Location loc = affLoad.getLoc();

          // Materialize the affine expression as arith ops to get an
          // index-typed offset, then cast to i64 for GEP.
          Value offset = builder.create<affine::AffineApplyOp>(
              loc, map, mapOperands);
          Value offsetI64 = toI64Type(offset, builder, loc);
          Value addr = builder.create<LLVM::GEPOp>(
              loc, llvmPtrTy, elemTy, ptrVal, ValueRange{offsetI64});
          auto newLoad = builder.create<LLVM::LoadOp>(
              loc, affLoad.getResult().getType(), addr);
          affLoad.getResult().replaceAllUsesWith(newLoad.getResult());
          affLoad->erase();
          continue;
        }

        // (c) affine.store: lower for rank-1 identity-map cases.
        if (auto affStore = dyn_cast<affine::AffineStoreOp>(user)) {
          if (affStore.getMemRef() != memrefResult) {
            allHandled = false;
            continue;
          }
          AffineMap map = affStore.getAffineMap();
          auto mapOperands = affStore.getMapOperands();
          if (map.getNumResults() != 1) {
            allHandled = false;
            continue;
          }
          OpBuilder builder(affStore);
          Location loc = affStore.getLoc();

          Value offset = builder.create<affine::AffineApplyOp>(
              loc, map, mapOperands);
          Value offsetI64 = toI64Type(offset, builder, loc);
          Value addr = builder.create<LLVM::GEPOp>(
              loc, llvmPtrTy, elemTy, ptrVal, ValueRange{offsetI64});
          builder.create<LLVM::StoreOp>(loc, affStore.getValue(), addr);
          affStore->erase();
          continue;
        }

        // (e) memref.extract_aligned_pointer_as_index: this arises when
        // Phase 4 lowered a memref2pointer that consumed the pointer2memref
        // result.  The semantics is "get the aligned pointer from the
        // memref descriptor as an index".  For a pointer2memref-created
        // memref, the pointer IS the aligned pointer, so we can replace
        // with llvm.ptrtoint of the original pointer.
        if (auto extractOp =
                dyn_cast<memref::ExtractAlignedPointerAsIndexOp>(user)) {
          OpBuilder builder(extractOp);
          Location loc = extractOp.getLoc();
          Value i64Val = builder.create<LLVM::PtrToIntOp>(
              loc, builder.getI64Type(), ptrVal);
          Value idxVal = builder.create<arith::IndexCastOp>(
              loc, builder.getIndexType(), i64Val);
          extractOp.getResult().replaceAllUsesWith(idxVal);
          extractOp->erase();
          continue;
        }

        // Unhandled use — cannot lower this pointer2memref.
        allHandled = false;
      }

      if (allHandled && memrefResult.use_empty())
        op->erase();
    }

    // Phase 6: Lower polygeist.subindex ops.
    //
    // subindex(%memref, %idx) : (memref<?xT>, index) -> memref<?xT>
    //
    // This is pointer arithmetic: result points to memref + idx * sizeof(T).
    //
    // Two lowering strategies depending on the use pattern:
    //
    //   (a) memref2pointer uses: extract aligned pointer, GEP by offset,
    //       fold downstream memref2pointer ops to the GEP'd pointer.
    //
    //   (b) Non-pointer uses (func.call args, scf.yield, etc.): for the
    //       common rank-1 i8 case, lower to memref.view which creates a
    //       new memref with identity layout starting at byte offset %idx.
    //       This is a standard MLIR op that finalize-memref-to-llvm can
    //       lower, and it preserves the memref<?xi8> type that callers
    //       expect.
    SmallVector<Operation *> residualSI;
    moduleOp.walk([&](Operation *op) {
      if (isSubIndex(op))
        residualSI.push_back(op);
    });

    for (Operation *op : residualSI) {
      OpBuilder builder(op);
      Location loc = op->getLoc();
      Value memrefVal = op->getOperand(0);
      Value idxVal = op->getOperand(1);
      Value siResult = op->getResult(0);

      auto srcType = dyn_cast<MemRefType>(memrefVal.getType());
      auto dstType = dyn_cast<MemRefType>(siResult.getType());
      if (!srcType || !dstType)
        continue;

      // --- Strategy (a): handle memref2pointer uses via pointer GEP ---
      //
      // Check if there are any memref2pointer uses before creating the
      // GEP chain (avoid emitting dead code if there are none).
      bool hasM2PUses = false;
      for (OpOperand &use : siResult.getUses()) {
        if (isMemref2Pointer(use.getOwner())) {
          hasM2PUses = true;
          break;
        }
      }

      if (hasM2PUses) {
        Value baseIdx =
            builder.create<memref::ExtractAlignedPointerAsIndexOp>(
                loc, memrefVal);
        Value i64Base = builder.create<arith::IndexCastOp>(
            loc, builder.getI64Type(), baseIdx);
        Value basePtr = builder.create<LLVM::IntToPtrOp>(
            loc, LLVM::LLVMPointerType::get(op->getContext()), i64Base);

        // GEP: base + idx * sizeof(element)
        Value gepPtr = builder.create<LLVM::GEPOp>(
            loc, LLVM::LLVMPointerType::get(op->getContext()),
            dstType.getElementType(), basePtr, ValueRange{idxVal});

        // Fold memref2pointer uses to the GEP'd pointer.
        for (OpOperand &use :
             llvm::make_early_inc_range(siResult.getUses())) {
          Operation *user = use.getOwner();
          if (isMemref2Pointer(user)) {
            user->getResult(0).replaceAllUsesWith(gepPtr);
            user->erase();
          }
        }
      }

      // --- Strategy (b): handle remaining non-pointer uses ---
      //
      // For rank-1 memref<?xi8> with identity layout, use memref.view to
      // create a new memref starting at byte offset %idx.  memref.view
      // shifts the aligned pointer in the descriptor and produces identity
      // layout (offset 0) — exactly the memref<?xi8> type that downstream
      // func.call, scf.yield, etc. expect.
      //
      // For i8 elements, element offset == byte offset, so %idx can be
      // used directly as the byte_shift operand of memref.view.
      if (!siResult.use_empty() &&
          srcType.getRank() == 1 && dstType.getRank() == 1 &&
          srcType.getElementType().isInteger(8) &&
          dstType.getElementType().isInteger(8) &&
          srcType.getLayout().isIdentity() &&
          dstType.getLayout().isIdentity()) {

        // Compute remaining size: dim(%memref, 0) - %idx
        Value c0dim = builder.create<arith::ConstantIndexOp>(loc, 0);
        Value dim = builder.create<memref::DimOp>(loc, memrefVal, c0dim);
        Value remaining = builder.create<arith::SubIOp>(loc, dim, idxVal);

        // memref.view %src[%byte_shift][%remaining_size]
        //   : memref<?xi8> to memref<?xi8>
        auto viewOp = builder.create<memref::ViewOp>(
            loc, dstType, memrefVal, idxVal, ValueRange{remaining});

        siResult.replaceAllUsesWith(viewOp.getResult());
      }

      // Erase the subindex only if all uses have been handled.
      if (siResult.use_empty())
        op->erase();
      // else: leave the op — file will fall back to clang at mlir-opt.
    }
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createRaiseMallocToMemRefPass() {
  return std::make_unique<RaiseMallocToMemRefPass>();
}
} // namespace mlir
