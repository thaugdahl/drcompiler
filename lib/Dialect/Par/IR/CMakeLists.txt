add_mlir_library(DRCompParDialect
  ParOps.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/drcompiler/Dialect/Par

  DEPENDS
    MLIRParOpsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRControlFlowInterfaces
    MLIRSideEffectInterfaces
    MLIRSupport
)
