# ==========================================================================
# drcc-base: LLVM 22 + cgeist base image.
#
# Build once, reuse as the base for drcc. Rebuilding drcompiler doesn't
# require recompiling LLVM.
#
#   docker build -f docker/base.Dockerfile -t drcc-base .
# ==========================================================================

ARG LLVM_VERSION=22
ARG LLVM_TAG=llvmorg-22.1.1

# --------------------------------------------------------------------------
# Stage 1: Build cgeist against Polygeist's bundled LLVM 18
# --------------------------------------------------------------------------
FROM ubuntu:24.04 AS cgeist-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake ninja-build git ca-certificates g++ python3 lld \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --recurse-submodules --shallow-submodules \
      https://github.com/llvm/Polygeist.git /src/polygeist

RUN cmake -G Ninja -S /src/polygeist/llvm-project/llvm -B /build/llvm18 \
      -DLLVM_ENABLE_PROJECTS="clang;mlir" \
      -DLLVM_TARGETS_TO_BUILD="host" \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_USE_LINKER=lld \
    && cmake --build /build/llvm18 -j"$(nproc)"

RUN cmake -G Ninja -S /src/polygeist -B /build/polygeist \
      -DMLIR_DIR=/build/llvm18/lib/cmake/mlir \
      -DCLANG_DIR=/build/llvm18/lib/cmake/clang \
      -DLLVM_TARGETS_TO_BUILD="host" \
      -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /build/polygeist --target cgeist -j"$(nproc)"

# --------------------------------------------------------------------------
# Stage 2: Build upstream LLVM (clang + MLIR), install to /opt/llvm
# --------------------------------------------------------------------------
FROM ubuntu:24.04 AS llvm-builder

ARG LLVM_TAG

RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake ninja-build git ca-certificates g++ python3 lld \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${LLVM_TAG}" \
      https://github.com/llvm/llvm-project.git /src/llvm

RUN cmake -G Ninja -S /src/llvm/llvm -B /build/llvm \
      -DLLVM_ENABLE_PROJECTS="clang;mlir" \
      -DLLVM_TARGETS_TO_BUILD="host" \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/llvm \
      -DLLVM_USE_LINKER=lld \
    && cmake --build /build/llvm -j"$(nproc)" \
    && cmake --install /build/llvm

# --------------------------------------------------------------------------
# Stage 3: Base image with LLVM 22 install + cgeist + build tools
# --------------------------------------------------------------------------
FROM ubuntu:24.04

ARG LLVM_VERSION

RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake ninja-build g++ python3 \
      libc6-dev libstdc++6 zlib1g libzstd1 \
    && rm -rf /var/lib/apt/lists/*

# Full LLVM 22 install (needed to compile dr-opt)
COPY --from=llvm-builder /opt/llvm /opt/llvm
ENV PATH="/opt/llvm/bin:${PATH}"

# cgeist + clang 18 resource headers
COPY --from=cgeist-builder /build/polygeist/bin/cgeist /usr/local/bin/cgeist
COPY --from=cgeist-builder /build/llvm18/lib/clang/18/include \
                           /usr/local/lib/clang/18/include
