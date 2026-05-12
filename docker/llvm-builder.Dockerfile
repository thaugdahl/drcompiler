# ==========================================================================
# drcc-llvm-builder: builds upstream LLVM 22 (clang + MLIR), installed
# to /opt/llvm.
#
# Build for host arch (default x86_64):
#   docker buildx build -f docker/llvm-builder.Dockerfile \
#     -t drcc-llvm-builder:x86_64 --load .
#
# Build for aarch64 (cross-compiled natively on amd64 host):
#   docker buildx build --platform linux/arm64 \
#     -f docker/llvm-builder.Dockerfile -t drcc-llvm-builder:aarch64 --load .
#
# Override pinned tag:
#   docker buildx build ... --build-arg LLVM_TAG=llvmorg-22.1.2 ...
#
# Used by base.Dockerfile via COPY --from=drcc-llvm-builder:<arch>.
# ==========================================================================

# Stage 1: Build host tblgen tools natively (always amd64).
# Required as build-time executables when cross-compiling for arm64.
# Cached independently — only rebuilds when LLVM_TAG changes.
FROM --platform=$BUILDPLATFORM ubuntu:24.04 AS host-tools
ARG LLVM_TAG=llvmorg-22.1.1

RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake ninja-build git ca-certificates g++ python3 lld \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${LLVM_TAG}" \
      https://github.com/llvm/llvm-project.git /src/llvm

RUN cmake -G Ninja -S /src/llvm/llvm -B /build/llvm-host \
      -DLLVM_ENABLE_PROJECTS="clang;mlir" \
      -DLLVM_TARGETS_TO_BUILD="host" \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_USE_LINKER=lld \
    && cmake --build /build/llvm-host -j"$(nproc)" \
         --target llvm-tblgen clang-tblgen mlir-tblgen

# Stage 2: Full LLVM build — native amd64 or cross-compiled for arm64.
# Always runs on amd64; TARGETARCH (auto-injected by buildx) selects path.
FROM --platform=$BUILDPLATFORM ubuntu:24.04 AS builder
ARG LLVM_TAG=llvmorg-22.1.1
ARG TARGETARCH=amd64

RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake ninja-build git ca-certificates g++ python3 lld \
    && if [ "$TARGETARCH" = "arm64" ]; then \
         apt-get install -y --no-install-recommends \
           gcc-aarch64-linux-gnu g++-aarch64-linux-gnu binutils-aarch64-linux-gnu; \
       fi \
    && rm -rf /var/lib/apt/lists/*

# Reuse source tree from host-tools (avoids a second git clone)
COPY --from=host-tools /src/llvm /src/llvm

# Host tblgen binaries — used by cross-compile path, ignored for native
COPY --from=host-tools \
      /build/llvm-host/bin/llvm-tblgen \
      /build/llvm-host/bin/clang-tblgen \
      /build/llvm-host/bin/mlir-tblgen \
      /opt/host-tools/bin/

RUN if [ "$TARGETARCH" = "arm64" ]; then \
      printf '%s\n' \
        'set(CMAKE_SYSTEM_NAME Linux)' \
        'set(CMAKE_SYSTEM_PROCESSOR aarch64)' \
        'set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)' \
        'set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)' \
        'set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)' \
        'set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)' \
        'set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)' \
        > /opt/aarch64-toolchain.cmake; \
    fi

RUN if [ "$TARGETARCH" = "arm64" ]; then \
      cmake -G Ninja -S /src/llvm/llvm -B /build/llvm \
        -DLLVM_ENABLE_PROJECTS="clang;mlir" \
        -DLLVM_TARGETS_TO_BUILD="AArch64" \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_ENABLE_RTTI=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/llvm \
        -DCMAKE_TOOLCHAIN_FILE=/opt/aarch64-toolchain.cmake \
        -DLLVM_TABLEGEN=/opt/host-tools/bin/llvm-tblgen \
        -DCLANG_TABLEGEN=/opt/host-tools/bin/clang-tblgen \
        -DLLVM_NATIVE_TOOL_DIR=/opt/host-tools/bin \
        -DLLVM_HOST_TRIPLE=aarch64-linux-gnu \
        -DLLVM_DEFAULT_TARGET_TRIPLE=aarch64-linux-gnu \
        -DLLVM_INSTALL_UTILS=ON \
      && cmake --build /build/llvm -j"$(nproc)" \
      && cmake --install /build/llvm; \
    else \
      cmake -G Ninja -S /src/llvm/llvm -B /build/llvm \
        -DLLVM_ENABLE_PROJECTS="clang;mlir" \
        -DLLVM_TARGETS_TO_BUILD="host" \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_ENABLE_RTTI=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/llvm \
        -DLLVM_USE_LINKER=lld \
        -DLLVM_INSTALL_UTILS=ON \
      && cmake --build /build/llvm -j"$(nproc)" \
      && cmake --install /build/llvm; \
    fi
