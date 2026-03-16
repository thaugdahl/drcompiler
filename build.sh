#!/usr/bin/env bash
#===----------------------------------------------------------------------===//
# Helper script to configure and build drcompiler.
#
# Usage:
#   ./build.sh [configure|build|rebuild|clean] [options]
#
# Options:
#   --llvm-dir DIR          Path to LLVM+MLIR+Clang install tree
#   --polygeist-dir DIR     Path to pre-installed Polygeist
#   --build-polygeist       Fetch and build Polygeist automatically
#   --build-type TYPE       CMAKE_BUILD_TYPE (default: Release)
#   --jobs N                Parallel jobs (default: nproc)
#===----------------------------------------------------------------------===//
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# --- Defaults ---
LLVM_DIR="${LLVM_DIR:-/home/tor/Dev/marco/install/llvm-project}"
POLYGEIST_DIR="${POLYGEIST_DIR:-}"
BUILD_POLYGEIST=OFF
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc)}"
ACTION="${1:-build}"

shift || true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --llvm-dir)       LLVM_DIR="$2";       shift 2 ;;
    --polygeist-dir)  POLYGEIST_DIR="$2";   shift 2 ;;
    --build-polygeist) BUILD_POLYGEIST=ON;  shift ;;
    --build-type)     BUILD_TYPE="$2";      shift 2 ;;
    --jobs)           JOBS="$2";            shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

configure() {
  mkdir -p "$BUILD_DIR"
  cmake -G Ninja -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DLLVM_INSTALL_DIR="$LLVM_DIR" \
    ${POLYGEIST_DIR:+-DPOLYGEIST_INSTALL_DIR="$POLYGEIST_DIR"} \
    -DDRCOMP_BUILD_POLYGEIST="$BUILD_POLYGEIST" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
}

build() {
  if [[ ! -f "$BUILD_DIR/build.ninja" ]]; then
    configure
  fi
  cmake --build "$BUILD_DIR" -j "$JOBS"
}

case "$ACTION" in
  configure)
    configure
    ;;
  build)
    build
    ;;
  rebuild)
    configure
    build
    ;;
  clean)
    rm -rf "$BUILD_DIR"
    echo "Cleaned $BUILD_DIR"
    ;;
  *)
    echo "Usage: $0 [configure|build|rebuild|clean] [options]"
    exit 1
    ;;
esac
