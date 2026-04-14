# ==========================================================================
# drcc-cgeist-builder: builds cgeist (Polygeist C/C++ → MLIR frontend)
# against Polygeist's bundled LLVM 18.
#
# Build and cache this image independently of LLVM 22:
#   docker build -f docker/cgeist-builder.Dockerfile -t drcc-cgeist-builder .
#
# Used by base.Dockerfile via COPY --from=drcc-cgeist-builder.
# ==========================================================================

FROM ubuntu:24.04

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
