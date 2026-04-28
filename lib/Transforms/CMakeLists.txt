add_mlir_library(DRCompTransforms
  CpuCostModel.cpp
  DataRecomputation.cpp
  DataRecomputation/CacheCostModel.cpp
  DataRecomputation/RematKernel.cpp
  DataRecomputation/Strategies/ConstantGlobalFold.cpp
  DataRecomputation/Strategies/CrossFnOrdered.cpp
  DataRecomputation/Strategies/DirectForward.cpp
  DataRecomputation/Strategies/FullRemat.cpp
  DataRecomputation/Strategies/InterprocRemat.cpp
  DataRecomputation/Strategies/PartialRemat.cpp
  MemoryFission.cpp
  RaiseMallocToMemRef.cpp
  Utils/OpDispatchUtils.cpp
  Utils/MemrefBaseAnalysis.cpp

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
