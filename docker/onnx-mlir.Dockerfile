FROM drcc-base AS base

ARG MLIR_DIR=/opt/llvm/lib/cmake/mlir

RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential ca-certificates cmake python3 python3-dev python3-pip && \
    pip3 install --break-system-packages onnx

# Ubuntu 24.04 ships absl 20230125 which is missing targets like
# absl::log_internal_check_op. Build a newer version from source.
ARG ABSL_TAG=20240722.0
RUN git clone --depth 1 --branch ${ABSL_TAG} \
      https://github.com/abseil/abseil-cpp /tmp/absl && \
    cmake -G Ninja -S /tmp/absl -B /tmp/absl-build \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=OFF \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
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

# Build ONNX-MLIR
RUN cmake -G Ninja -S /build/onnx-mlir -B /build/onnx-mlir/build \
       -DMLIR_DIR=${MLIR_DIR} \
       -DLLVM_DIR=/opt/llvm/lib/cmake/llvm \
       -DCMAKE_BUILD_TYPE=Release \
       -DLLVM_EXTERNAL_LIT=/opt/llvm/bin/llvm-lit && \
     cmake --build /build/onnx-mlir/build -j"$(nproc)"

WORKDIR /build/onnx-mlir/build
