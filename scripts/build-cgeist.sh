#!/usr/bin/env bash
#===----------------------------------------------------------------------===//
# Build Polygeist's cgeist against its bundled LLVM 18.
#
# LLVM 18 requires GCC ≤ 14 / libstdc++ ≤ 14 (breaks with GCC 15+).
# On modern distros (Arch, Fedora 42+), use the Docker method instead:
#
#   docker build -f docker/cgeist-builder.Dockerfile -o polygeist-install .
#
# This script is for systems with a compatible toolchain (Ubuntu ≤ 24.04).
#
# Usage:
#   ./scripts/build-cgeist.sh [POLYGEIST_SRC] [INSTALL_PREFIX]
#
# Defaults:
#   POLYGEIST_SRC  = ./polygeist  (cloned if missing)
#   INSTALL_PREFIX = ./polygeist-install
#===----------------------------------------------------------------------===//
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

POLYGEIST_SRC="${1:-${PROJECT_DIR}/polygeist}"
INSTALL_PREFIX="${2:-${PROJECT_DIR}/polygeist-install}"
NPROC="${NPROC:-$(nproc)}"

echo "=== Polygeist/cgeist build ==="
echo "  Source:  ${POLYGEIST_SRC}"
echo "  Install: ${INSTALL_PREFIX}"
echo "  Jobs:    ${NPROC}"

# --- Clone Polygeist if needed ---
if [[ ! -d "${POLYGEIST_SRC}" ]]; then
  echo "--- Cloning Polygeist ---"
  git clone --depth 1 --recurse-submodules --shallow-submodules \
    https://github.com/llvm/Polygeist.git "${POLYGEIST_SRC}"
fi

LLVM_SRC="${POLYGEIST_SRC}/llvm-project"
if [[ ! -d "${LLVM_SRC}/llvm" ]]; then
  echo "--- Initializing llvm-project submodule ---"
  (cd "${POLYGEIST_SRC}" && git submodule update --init --depth 1 llvm-project)
fi

# --- Step 1: Build LLVM 18 + MLIR + Clang ---
LLVM_BUILD="${POLYGEIST_SRC}/llvm-build"
echo "--- Building LLVM/MLIR/Clang (this takes a while the first time) ---"
cmake -G Ninja -S "${LLVM_SRC}/llvm" -B "${LLVM_BUILD}" \
  -DLLVM_ENABLE_PROJECTS="clang;mlir" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_USE_LINKER=lld
cmake --build "${LLVM_BUILD}" -j "${NPROC}"

# --- Step 2: Build Polygeist/cgeist against the LLVM we just built ---
POLYGEIST_BUILD="${POLYGEIST_SRC}/build"
echo "--- Building cgeist ---"
cmake -G Ninja -S "${POLYGEIST_SRC}" -B "${POLYGEIST_BUILD}" \
  -DMLIR_DIR="${LLVM_BUILD}/lib/cmake/mlir" \
  -DCLANG_DIR="${LLVM_BUILD}/lib/cmake/clang" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}"
cmake --build "${POLYGEIST_BUILD}" --target cgeist -j "${NPROC}"

# --- Install just the cgeist binary ---
mkdir -p "${INSTALL_PREFIX}/bin"
cp "${POLYGEIST_BUILD}/tools/cgeist/cgeist" "${INSTALL_PREFIX}/bin/cgeist"

echo "=== Done ==="
echo "cgeist installed to: ${INSTALL_PREFIX}/bin/cgeist"
echo ""
echo "Add to PATH:  export PATH=\"${INSTALL_PREFIX}/bin:\$PATH\""
