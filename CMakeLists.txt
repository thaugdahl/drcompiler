cmake_minimum_required(VERSION 3.20)
project(drcompiler LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#===----------------------------------------------------------------------===//
# LLVM / MLIR / Clang
#===----------------------------------------------------------------------===//

# Allow the user to point at a specific LLVM install tree.
# Usage: cmake -DLLVM_INSTALL_DIR=/path/to/llvm-install ..
set(LLVM_INSTALL_DIR "" CACHE PATH "Root of an LLVM+MLIR+Clang install tree")
if(LLVM_INSTALL_DIR)
  list(PREPEND CMAKE_PREFIX_PATH
    "${LLVM_INSTALL_DIR}/lib/cmake/llvm"
    "${LLVM_INSTALL_DIR}/lib/cmake/mlir"
    "${LLVM_INSTALL_DIR}/lib/cmake/clang")
endif()

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

message(STATUS "Using LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "  LLVM CMake dir : ${LLVM_CMAKE_DIR}")
message(STATUS "  MLIR CMake dir : ${MLIR_CMAKE_DIR}")
message(STATUS "  Clang CMake dir: ${CLANG_CMAKE_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})

# Project-local include trees (source and build for tablegen outputs).
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# Match LLVM/MLIR build flags.
if(NOT LLVM_ENABLE_RTTI)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()
if(NOT LLVM_ENABLE_EH)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions")
endif()

#===----------------------------------------------------------------------===//
# Polygeist cgeist frontend (C/C++ → MLIR)
#
# cgeist is built separately against Polygeist's bundled LLVM 18 and used
# purely as a CLI tool (textual MLIR interface).  See scripts/build-cgeist.sh.
#===----------------------------------------------------------------------===//

find_program(CGEIST_EXECUTABLE cgeist
  HINTS ENV CGEIST_DIR
  PATH_SUFFIXES bin)

if(CGEIST_EXECUTABLE)
  message(STATUS "Found cgeist: ${CGEIST_EXECUTABLE}")
else()
  message(WARNING
    "cgeist not found – drcc wrapper will not be fully functional.\n"
    "Build Polygeist separately (see scripts/build-cgeist.sh) and ensure "
    "cgeist is on PATH or set CGEIST_DIR.")
endif()

#===----------------------------------------------------------------------===//
# Sub-directories
#===----------------------------------------------------------------------===//

add_subdirectory(include/drcompiler/Transforms)
add_subdirectory(lib/Transforms)
add_subdirectory(tools)

#===----------------------------------------------------------------------===//
# drcc wrapper script generation
#===----------------------------------------------------------------------===//

set(DROPT_EXECUTABLE "$<TARGET_FILE:dr-opt>")
set(MLIR_TRANSLATE_EXECUTABLE "${LLVM_TOOLS_BINARY_DIR}/mlir-translate")
set(CLANG_EXECUTABLE "${LLVM_TOOLS_BINARY_DIR}/clang")
set(MLIR_OPT_EXECUTABLE "${LLVM_TOOLS_BINARY_DIR}/mlir-opt")

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/tools/drcc/drcc.sh.in
  ${CMAKE_CURRENT_BINARY_DIR}/drcc.sh.configured
  @ONLY
)

# Use a generator expression for dr-opt path (resolved at build time).
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/drcc
  COMMAND ${CMAKE_COMMAND}
    -DINPUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/drcc.sh.configured
    -DOUTPUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/drcc
    -DDROPT=$<TARGET_FILE:dr-opt>
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/GenerateDrcc.cmake
  DEPENDS dr-opt ${CMAKE_CURRENT_BINARY_DIR}/drcc.sh.configured
  COMMENT "Generating drcc wrapper script"
)
add_custom_target(drcc-wrapper ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/drcc)
