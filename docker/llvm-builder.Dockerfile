# ==========================================================================
# drcc-llvm-builder: builds upstream LLVM 22 (clang + MLIR), installed
# to /opt/llvm.
#
# Build and cache this image independently of cgeist:
#   docker build -f docker/llvm-builder.Dockerfile -t drcc-llvm-builder .
#
# Override the pinned tag:
#   docker build --build-arg LLVM_TAG=llvmorg-22.1.2 \
#                -f docker/llvm-builder.Dockerfile -t drcc-llvm-builder .
#
# Used by base.Dockerfile via COPY --from=drcc-llvm-builder.
# ==========================================================================

FROM ubuntu:24.04

ARG LLVM_TAG=llvmorg-22.1.1

RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake ninja-build git ca-certificates g++ python3 lld \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${LLVM_TAG}" \
      https://github.com/llvm/llvm-project.git /src/llvm

RUN cmake -G Ninja -S /src/llvm/llvm -B /build/llvm \
      -DLLVM_ENABLE_PROJECTS="clang;mlir" \
      -DLLVM_TARGETS_TO_BUILD="host" \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DLLVM_ENABLE_RTTI=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/llvm \
      -DLLVM_USE_LINKER=lld \
      -DLLVM_INSTALL_UTILS=ON \
    && cmake --build /build/llvm -j"$(nproc)" \
    && cmake --install /build/llvm
