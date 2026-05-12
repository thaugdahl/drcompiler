# ==========================================================================
# drcc: Data Recomputation compiler pipeline.
#
# Use build.sh which wires up --build-context for drcc-base:
#
#   docker/build.sh --arch x86_64   # or aarch64
#
# Usage:
#   docker run --rm -v "$PWD":/work drcc:aarch64 -c /work/foo.c -o /work/foo.o
# ==========================================================================

ARG ARCH_TAG=x86_64

# Target-arch LLVM: headers, libs, and cmake configs for the output binary.
FROM --platform=$BUILDPLATFORM drcc-llvm-builder:${ARCH_TAG} AS target-llvm

# Host-arch LLVM: native mlir-tblgen needed as a build tool during cmake.
# Always x86_64 because we only ever cross-compile from an amd64 host.
FROM --platform=$BUILDPLATFORM drcc-llvm-builder:x86_64 AS host-llvm

# Builder stage: cross-compile dr-opt on the build host (amd64).
# Same pattern as llvm-builder/cgeist-builder — no QEMU needed.
FROM --platform=$BUILDPLATFORM ubuntu:24.04 AS dr-opt-builder
# TARGETARCH must be passed explicitly by build.sh (--build-arg TARGETARCH=arm64)
# because Docker sets it to the stage platform ($BUILDPLATFORM) not the global --platform.
ARG TARGETARCH=amd64

RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake ninja-build g++ \
    && if [ "$TARGETARCH" = "arm64" ]; then \
         apt-get install -y --no-install-recommends \
           gcc-aarch64-linux-gnu g++-aarch64-linux-gnu binutils-aarch64-linux-gnu; \
       fi \
    && rm -rf /var/lib/apt/lists/*

# Target-arch LLVM (for headers + libs when linking dr-opt)
COPY --from=target-llvm /opt/llvm /opt/llvm
# Overwrite mlir-tblgen with the native (host-arch) binary.
# MLIRConfig.cmake hardcodes the path /opt/llvm/bin/mlir-tblgen; overwriting
# ensures cmake can execute it on the build host during the tblgen step.
# The arm64 LLVM libraries in /opt/llvm/lib/ are untouched.
COPY --from=host-llvm /opt/llvm/bin/mlir-tblgen /opt/llvm/bin/mlir-tblgen

COPY . /src/drcompiler

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
      cmake -G Ninja -S /src/drcompiler -B /build/drcompiler \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_INSTALL_DIR=/opt/llvm \
        -DCMAKE_TOOLCHAIN_FILE=/opt/aarch64-toolchain.cmake; \
    else \
      cmake -G Ninja -S /src/drcompiler -B /build/drcompiler \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_INSTALL_DIR=/opt/llvm; \
    fi \
    && cmake --build /build/drcompiler -j"$(nproc)" \
    && cp /build/drcompiler/tools/dr-opt/dr-opt /usr/local/bin/dr-opt

# Generate the drcc wrapper in the builder stage (amd64) so the assembly
# stage needs no RUN steps and avoids exec format errors without QEMU.
COPY tools/drcc/drcc.sh.in /tmp/drcc.sh.in
RUN sed \
      -e 's|@CGEIST_EXECUTABLE@|/usr/local/bin/cgeist|g' \
      -e 's|%%DROPT_EXECUTABLE%%|/usr/local/bin/dr-opt|g' \
      -e 's|@MLIR_TRANSLATE_EXECUTABLE@|/opt/llvm/bin/mlir-translate|g' \
      -e 's|@CLANG_EXECUTABLE@|/opt/llvm/bin/clang|g' \
      -e 's|@MLIR_OPT_EXECUTABLE@|/opt/llvm/bin/mlir-opt|g' \
      /tmp/drcc.sh.in > /usr/local/bin/drcc \
    && chmod +x /usr/local/bin/drcc

# Stage 2: Assembly image — arm64 ubuntu + all pipeline binaries.
# Only COPY steps here: no RUN needed, no QEMU required.
FROM drcc-base:${ARCH_TAG}

COPY --from=dr-opt-builder /usr/local/bin/dr-opt /usr/local/bin/dr-opt
COPY --from=dr-opt-builder /usr/local/bin/drcc /usr/local/bin/drcc

# Struct-memref text preprocessor (called by drcc before dr-opt)
COPY tools/drcc/rewrite-struct-memrefs.py /usr/local/bin/rewrite-struct-memrefs.py

# Probing scripts for generating hardware-specific cost models.
COPY scripts/gen_cpu_cost_model.py scripts/cache_latency_bench.c \
     scripts/probe-cost-model.sh \
     /usr/local/share/drcompiler/scripts/

ENTRYPOINT ["drcc"]
