# ==========================================================================
# drcc-lean: Minimal runtime image for the drcc compiler pipeline.
#
# Copies the compiler executables (cgeist, dr-opt, mlir-opt, mlir-translate,
# clang) plus the cost-model probe toolchain (llvm-mca + probe scripts) from
# the full drcc image — no LLVM headers, no cmake, no build toolchain.
# Significantly smaller than drcc:aarch64.
#
# Must be built after drcc:${ARCH_TAG} exists. Use build.sh:
#   docker/build.sh --arch aarch64
#
# Or standalone (requires drcc:aarch64 already built):
#   docker build --platform linux/arm64 \
#     --build-arg ARCH_TAG=aarch64 \
#     -f docker/drcc-lean.Dockerfile -t drcc-lean:aarch64 .
#
# Usage identical to drcc:
#   docker run --rm -v "$PWD":/work drcc-lean:aarch64 -c /work/foo.c -o /work/foo.o
# ==========================================================================

ARG ARCH_TAG=x86_64
FROM drcc:${ARCH_TAG} AS full

FROM ubuntu:24.04

# python3       — rewrite-struct-memrefs.py (called by drcc before dr-opt)
# libc6-dev     — system C headers that cgeist needs to process C source files,
#                 plus C start files (crt1.o etc.) that clang needs when linking
# gcc           — provides crtbeginS.o (needed when clang links PIE executables)
#                 and a GCC installation for cgeist to query the host target triple
# binutils      — the 'ld' linker, called by clang for the final link step
# libstdc++6    — C++ runtime shared lib (cgeist, dr-opt, mlir-* are C++ binaries)
# zlib1g        — needed by LLVM tools for section compression
# libzstd1      — needed by LLVM tools for ZSTD compression
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 \
      libc6-dev \
      gcc \
      binutils \
      libstdc++6 \
      zlib1g \
      libzstd1 \
    && rm -rf /var/lib/apt/lists/*

# cgeist: Polygeist C/C++ → MLIR frontend (LLVM 18-based)
COPY --from=full /usr/local/bin/cgeist /usr/local/bin/cgeist
# Clang 18 compiler-builtin headers (stddef.h, stdint.h, etc.) that cgeist
# needs to process C files. Auto-detected by drcc via dirname(cgeist)/../lib/clang/18.
COPY --from=full /usr/local/lib/clang/18/include /usr/local/lib/clang/18/include

# dr-opt: mlir-opt wrapper with the DataRecomputation + MemoryFission passes
COPY --from=full /usr/local/bin/dr-opt /usr/local/bin/dr-opt

# LLVM 22 tools — copied to the same paths the drcc wrapper script expects
COPY --from=full /opt/llvm/bin/mlir-opt       /opt/llvm/bin/mlir-opt
COPY --from=full /opt/llvm/bin/mlir-translate /opt/llvm/bin/mlir-translate
COPY --from=full /opt/llvm/bin/clang          /opt/llvm/bin/clang
# llvm-mca — ALU latency probing for the CPU cost model (probe-cost-model.sh).
# Optional at runtime (the probe falls back to generic ALU costs without it),
# but copied so the lean image reproduces the full image's probe quality.
COPY --from=full /opt/llvm/bin/llvm-mca       /opt/llvm/bin/llvm-mca
# LLVM 22 compiler-builtin headers (stddef.h, stdint.h, etc.) needed when
# clang compiles C source files directly (e.g. polybench kernels).
COPY --from=full /opt/llvm/lib/clang          /opt/llvm/lib/clang

# drcc wrapper script + struct-memref rewriter
COPY --from=full /usr/local/bin/drcc                      /usr/local/bin/drcc
COPY --from=full /usr/local/bin/rewrite-struct-memrefs.py /usr/local/bin/rewrite-struct-memrefs.py

# Cost-model probe scripts (gen_cpu_cost_model.py, cache_latency_bench.c,
# probe-cost-model.sh) — lets the lean image generate hardware-specific cost
# models via probe-cost-model-docker.sh, same as the full drcc image.
COPY --from=full /usr/local/share/drcompiler/scripts/ /usr/local/share/drcompiler/scripts/

ENTRYPOINT ["drcc"]
