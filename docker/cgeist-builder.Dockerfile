# Builds cgeist (Polygeist C/C++ → MLIR frontend) against LLVM 18.
# Ubuntu 22.04 ships GCC 11 / libstdc++ 11, which LLVM 18 builds cleanly with.
#
# Usage:
#   docker build -f docker/cgeist-builder.Dockerfile -o polygeist-install .
#
# Result: polygeist-install/bin/cgeist

FROM ubuntu:22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake ninja-build git ca-certificates \
      g++ python3 lld \
    && rm -rf /var/lib/apt/lists/*

# Clone Polygeist (includes llvm-project submodule pinned to LLVM 18)
RUN git clone --depth 1 --recurse-submodules --shallow-submodules \
      https://github.com/llvm/Polygeist.git /src/polygeist

# Step 1: Build LLVM 18 + MLIR + Clang
RUN cmake -G Ninja -S /src/polygeist/llvm-project/llvm -B /build/llvm \
      -DLLVM_ENABLE_PROJECTS="clang;mlir" \
      -DLLVM_TARGETS_TO_BUILD="host" \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_USE_LINKER=lld \
    && cmake --build /build/llvm -j"$(nproc)"

# Step 2: Build cgeist
RUN cmake -G Ninja -S /src/polygeist -B /build/polygeist \
      -DMLIR_DIR=/build/llvm/lib/cmake/mlir \
      -DCLANG_DIR=/build/llvm/lib/cmake/clang \
      -DLLVM_TARGETS_TO_BUILD="host" \
      -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /build/polygeist --target cgeist -j"$(nproc)"

# Export just the binary
FROM scratch AS export
COPY --from=builder /build/polygeist/bin/cgeist /bin/cgeist
