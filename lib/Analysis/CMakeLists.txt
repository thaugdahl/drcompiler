add_mlir_library(DRCompAnalysis
  ArchHandlerRegistry.cpp
  ArchHandlers/Generic.cpp
  ArchHandlers/X86_64_AVX2.cpp
  ArchHandlers/X86_64_AVX512.cpp
  ArchHandlers/ARM_Neon.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/drcompiler/Analysis

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
)
