ARG ARCH_TAG=x86_64

# Target-arch LLVM: cmake configs, headers, libs for linking onnx-mlir.
FROM --platform=$BUILDPLATFORM drcc-llvm-builder:${ARCH_TAG} AS target-llvm

# Host-arch LLVM: native mlir-tblgen needed as a build tool during cmake.
# Always x86_64 because we only ever cross-compile from an amd64 host.
FROM --platform=$BUILDPLATFORM drcc-llvm-builder:x86_64 AS host-llvm

# Builder: cross-compile onnx-mlir on the amd64 host. No QEMU needed.
FROM --platform=$BUILDPLATFORM ubuntu:24.04 AS builder
# TARGETARCH must be passed explicitly by build.sh (--build-arg TARGETARCH=arm64)
# because Docker sets it to the stage platform ($BUILDPLATFORM), not --platform.
ARG TARGETARCH=amd64
ARG MLIR_DIR=/opt/llvm/lib/cmake/mlir
ARG ABSL_TAG=20240722.0

RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential ca-certificates cmake ninja-build \
      python3 python3-dev python3-pip \
    && if [ "$TARGETARCH" = "arm64" ]; then \
         apt-get install -y --no-install-recommends \
           gcc-aarch64-linux-gnu g++-aarch64-linux-gnu binutils-aarch64-linux-gnu; \
       fi \
    && pip3 install --break-system-packages onnx \
    && rm -rf /var/lib/apt/lists/*

# Target-arch LLVM (headers + libs for linking)
COPY --from=target-llvm /opt/llvm /opt/llvm
# Overwrite mlir-tblgen with native (host-arch) binary so cmake can execute
# it during the tblgen code-generation step. Same pattern as drcc.Dockerfile.
COPY --from=host-llvm /opt/llvm/bin/mlir-tblgen /opt/llvm/bin/mlir-tblgen

# Create aarch64 cross-compile toolchain file
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

# Build abseil-cpp for the target arch.
# Ubuntu 24.04 ships absl 20230125 which is missing targets like
# absl::log_internal_check_op. Build a newer version from source.
RUN git clone --depth 1 --branch ${ABSL_TAG} \
      https://github.com/abseil/abseil-cpp /tmp/absl && \
    if [ "$TARGETARCH" = "arm64" ]; then \
      cmake -G Ninja -S /tmp/absl -B /tmp/absl-build \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=OFF \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DCMAKE_TOOLCHAIN_FILE=/opt/aarch64-toolchain.cmake; \
    else \
      cmake -G Ninja -S /tmp/absl -B /tmp/absl-build \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=OFF \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON; \
    fi && \
    cmake --build /tmp/absl-build && \
    cmake --install /tmp/absl-build && \
    rm -rf /tmp/absl /tmp/absl-build

# Pull ONNX-MLIR source code and create build dir
RUN git clone --recursive https://github.com/onnx/onnx-mlir.git /build/onnx-mlir && \
    mkdir -p /build/onnx-mlir/build

# Patch MLIR API renames not present in llvmorg-22.1.1:
#   writeUnownedBlob  -> writeOwnedBlob  (DialectBytecodeWriter, added in c83ebf19)
#   getNumThreadsVarsMutable -> getNumThreadsMutable  (omp::ParallelOp rename)
RUN grep -rl writeUnownedBlob /build/onnx-mlir | xargs -r \
      sed -i 's/writeUnownedBlob/writeOwnedBlob/g' && \
    grep -rl getNumThreadsVarsMutable /build/onnx-mlir | xargs -r \
      sed -i 's/getNumThreadsVarsMutable/getNumThreadsMutable/g'

# Build ONNX-MLIR for the target arch.
RUN if [ "$TARGETARCH" = "arm64" ]; then \
      cmake -G Ninja -S /build/onnx-mlir -B /build/onnx-mlir/build \
        -DMLIR_DIR=${MLIR_DIR} \
        -DLLVM_DIR=/opt/llvm/lib/cmake/llvm \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_EXTERNAL_LIT=/opt/llvm/bin/llvm-lit \
        -DCMAKE_TOOLCHAIN_FILE=/opt/aarch64-toolchain.cmake; \
    else \
      cmake -G Ninja -S /build/onnx-mlir -B /build/onnx-mlir/build \
        -DMLIR_DIR=${MLIR_DIR} \
        -DLLVM_DIR=/opt/llvm/lib/cmake/llvm \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_EXTERNAL_LIT=/opt/llvm/bin/llvm-lit; \
    fi && \
    cmake --build /build/onnx-mlir/build -j"$(nproc)"

WORKDIR /build/onnx-mlir/build
