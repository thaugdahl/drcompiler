add_mlir_library(DRCompTransforms
  DataRecomputation.cpp
  MemoryFission.cpp
  RaiseMallocToMemRef.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/drcompiler/Transforms

  DEPENDS
    DRCompPassesIncGen

  LINK_LIBS PUBLIC
    MLIRAnalysis
    MLIRFunctionInterfaces
    MLIRLoopLikeInterface
    MLIRMemorySlotInterfaces
    MLIRPass
    MLIRRuntimeVerifiableOpInterface
    MLIRSCFDialect
    MLIRAffineDialect
    MLIRArithDialect
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRLLVMDialect
    MLIRPtrDialect
    MLIRVectorDialect
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRTransformUtils
    MLIRIR
)
