add_mlir_library(DRCompAnalysis
  ArchHandlerRegistry.cpp
  CpuCostModel.cpp
  MachineModel.cpp
  ParAliasOracle.cpp
  ArchHandlers/Generic.cpp
  ArchHandlers/X86_64_AVX2.cpp
  ArchHandlers/X86_64_AVX512.cpp
  ArchHandlers/ARM_Neon.cpp
  ArchHandlers/ARM_SVE.cpp
  ArchHandlers/AppleMSeries.cpp
  RegisterPressureAnalysis.cpp
  ReuseAnalysis.cpp
  SpillStrategy.cpp
  SpillStrategies/ExcessHot.cpp
  SpillStrategies/SumExcess.cpp
  SpillStrategies/GraphColor.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/drcompiler/Analysis

  LINK_LIBS PUBLIC
    MLIRAffineAnalysis
    MLIRAffineDialect
    MLIRAffineUtils
    MLIRAnalysis
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRSideEffectInterfaces
    MLIRViewLikeInterface
    MLIRIR
    MLIRSupport
)
