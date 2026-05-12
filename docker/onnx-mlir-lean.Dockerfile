# ==========================================================================
# onnx-mlir-lean: Minimal runtime image for ONNX-MLIR compilation on Pi.
#
# Keeps only what the benchmark pipeline needs:
#   onnx-mlir       — ONNX → Krnl/Affine MLIR frontend (steps 1, baseline EmitObj)
#   onnx-mlir-opt   — Krnl pass runner (steps 2, 4)
#   opt / llc       — LLVM bitcode optimizer + backend (called by onnx-mlir --EmitObj)
#   libcruntime.a   — static inference runtime, linked into benchmark harness
#   OnnxMlirRuntime.h + onnx-mlir/Runtime/ — harness API headers
#   python3 + onnx  — model shape inference for harness generation
#   gcc             — harness compilation and linking
#
# Must be built after onnx-mlir:${ARCH_TAG}:
#   docker build --platform linux/arm64 \
#     --build-arg ARCH_TAG=aarch64 \
#     -f docker/onnx-mlir-lean.Dockerfile -t onnx-mlir-lean:aarch64 .
# ==========================================================================

ARG ARCH_TAG=x86_64
FROM onnx-mlir:${ARCH_TAG} AS full

FROM ubuntu:24.04

# python3       — harness generation (shape inference via onnx package)
# gcc           — compile and link C benchmark harness against libcruntime.a
# libc6-dev     — C headers needed when compiling harness
# libstdc++6    — C++ runtime (libcruntime.a has C++ internals)
# zlib1g        — LLVM compression dependency in onnx-mlir binary
# libzstd1      — LLVM ZSTD dependency in onnx-mlir binary
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip \
      gcc \
      libc6-dev \
      libstdc++-13-dev \
      zlib1g \
      libzstd1 \
    && pip3 install --break-system-packages onnx \
    && rm -rf /var/lib/apt/lists/*

# ONNX-MLIR compiler binaries — kept at original build paths so existing
# scripts that hardcode ONNX_MLIR_BIN/ONNX_MLIR_OPT_BIN continue to work.
COPY --from=full \
  /build/onnx-mlir/build/Release/bin/onnx-mlir \
  /build/onnx-mlir/build/Release/bin/onnx-mlir
COPY --from=full \
  /build/onnx-mlir/build/Release/bin/onnx-mlir-opt \
  /build/onnx-mlir/build/Release/bin/onnx-mlir-opt

# LLVM tools required by onnx-mlir --EmitObj:
#   opt — optimizes LLVM bitcode (step 3/5 of EmitObj pipeline)
#   llc — compiles bitcode to object file (step 4/5 of EmitObj pipeline)
COPY --from=full /opt/llvm/bin/opt /opt/llvm/bin/opt
COPY --from=full /opt/llvm/bin/llc /opt/llvm/bin/llc

# Inference runtime — libcruntime.a and headers kept at original paths so
# harness link commands (-I/build/onnx-mlir/include, libcruntime.a) work unchanged.
COPY --from=full \
  /build/onnx-mlir/build/Release/lib/libcruntime.a \
  /build/onnx-mlir/build/Release/lib/libcruntime.a
COPY --from=full \
  /build/onnx-mlir/include/ \
  /build/onnx-mlir/include/

ENTRYPOINT ["/build/onnx-mlir/build/Release/bin/onnx-mlir"]
