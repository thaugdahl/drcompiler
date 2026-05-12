# ==========================================================================
# drcc-cgeist-builder: builds cgeist (Polygeist C/C++ → MLIR frontend)
# against Polygeist's bundled LLVM 18.
#
# Build for host arch (default x86_64):
#   docker buildx build -f docker/cgeist-builder.Dockerfile \
#     -t drcc-cgeist-builder:x86_64 --load .
#
# Build for aarch64 (cross-compiled natively on amd64 host):
#   docker buildx build --platform linux/arm64 \
#     -f docker/cgeist-builder.Dockerfile -t drcc-cgeist-builder:aarch64 --load .
#
# Used by base.Dockerfile via COPY --from=drcc-cgeist-builder:<arch>.
# ==========================================================================

# Stage 1: Build LLVM 18 host tblgen tools natively (always amd64).
# Required as build-time executables when cross-compiling for arm64.
FROM --platform=$BUILDPLATFORM ubuntu:24.04 AS host-tools

RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake ninja-build git ca-certificates g++ python3 lld \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --recurse-submodules --shallow-submodules \
      https://github.com/llvm/Polygeist.git /src/polygeist

RUN cmake -G Ninja -S /src/polygeist/llvm-project/llvm -B /build/llvm18-host \
      -DLLVM_ENABLE_PROJECTS="clang;mlir" \
      -DLLVM_TARGETS_TO_BUILD="host" \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_USE_LINKER=lld \
    && cmake --build /build/llvm18-host -j"$(nproc)" \
         --target llvm-tblgen clang-tblgen mlir-tblgen

# Stage 2: Full LLVM 18 + Polygeist build.
# Always runs on amd64; TARGETARCH selects native vs cross-compile path.
FROM --platform=$BUILDPLATFORM ubuntu:24.04 AS builder
ARG TARGETARCH=amd64

RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake ninja-build git ca-certificates g++ python3 lld \
    && if [ "$TARGETARCH" = "arm64" ]; then \
         apt-get install -y --no-install-recommends \
           gcc-aarch64-linux-gnu g++-aarch64-linux-gnu binutils-aarch64-linux-gnu; \
       fi \
    && rm -rf /var/lib/apt/lists/*

# Reuse Polygeist source tree (includes bundled llvm-project)
COPY --from=host-tools /src/polygeist /src/polygeist

# Host tblgen binaries — used by cross-compile path, ignored for native
COPY --from=host-tools \
      /build/llvm18-host/bin/llvm-tblgen \
      /build/llvm18-host/bin/clang-tblgen \
      /build/llvm18-host/bin/mlir-tblgen \
      /opt/host-tools18/bin/

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
      cmake -G Ninja -S /src/polygeist/llvm-project/llvm -B /build/llvm18 \
        -DLLVM_ENABLE_PROJECTS="clang;mlir" \
        -DLLVM_TARGETS_TO_BUILD="AArch64" \
        -DLLVM_DEFAULT_TARGET_TRIPLE="aarch64-linux-gnu" \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_TOOLCHAIN_FILE=/opt/aarch64-toolchain.cmake \
        -DLLVM_TABLEGEN=/opt/host-tools18/bin/llvm-tblgen \
        -DCLANG_TABLEGEN=/opt/host-tools18/bin/clang-tblgen \
        -DLLVM_NATIVE_TOOL_DIR=/opt/host-tools18/bin \
      && cmake --build /build/llvm18 -j"$(nproc)"; \
    else \
      cmake -G Ninja -S /src/polygeist/llvm-project/llvm -B /build/llvm18 \
        -DLLVM_ENABLE_PROJECTS="clang;mlir" \
        -DLLVM_TARGETS_TO_BUILD="host" \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_USE_LINKER=lld \
      && cmake --build /build/llvm18 -j"$(nproc)"; \
    fi

RUN if [ "$TARGETARCH" = "arm64" ]; then \
      cmake -G Ninja -S /src/polygeist -B /build/polygeist \
        -DMLIR_DIR=/build/llvm18/lib/cmake/mlir \
        -DCLANG_DIR=/build/llvm18/lib/cmake/clang \
        -DLLVM_TARGETS_TO_BUILD="AArch64" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_TOOLCHAIN_FILE=/opt/aarch64-toolchain.cmake \
        -DLLVM_NATIVE_TOOL_DIR=/opt/host-tools18/bin \
      && cmake --build /build/polygeist --target cgeist -j"$(nproc)"; \
    else \
      cmake -G Ninja -S /src/polygeist -B /build/polygeist \
        -DMLIR_DIR=/build/llvm18/lib/cmake/mlir \
        -DCLANG_DIR=/build/llvm18/lib/cmake/clang \
        -DLLVM_TARGETS_TO_BUILD="host" \
        -DCMAKE_BUILD_TYPE=Release \
      && cmake --build /build/polygeist --target cgeist -j"$(nproc)"; \
    fi
