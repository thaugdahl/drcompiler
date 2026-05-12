# ==========================================================================
# drcc-base: LLVM 22 + cgeist base image.
#
# Assembles pre-built artifacts from the two builder images.
# Use build.sh to build with the correct arch tag:
#
#   docker/build.sh --arch x86_64   # or aarch64
# ==========================================================================

ARG ARCH_TAG=x86_64

# Builder images always carry build-host binaries (cross-compiled or native).
# $BUILDPLATFORM = host arch; tells Docker not to reject them for arch mismatch.
FROM --platform=$BUILDPLATFORM drcc-llvm-builder:${ARCH_TAG} AS llvm-builder
FROM --platform=$BUILDPLATFORM drcc-cgeist-builder:${ARCH_TAG} AS cgeist-builder

# Final image: respects --platform passed to docker buildx build
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake ninja-build g++ python3 \
      libc6-dev libstdc++6 zlib1g libzstd1 \
      lmbench \
    && rm -rf /var/lib/apt/lists/*

# Full LLVM 22 install (needed to compile dr-opt)
COPY --from=llvm-builder /opt/llvm /opt/llvm
ENV PATH="/opt/llvm/bin:${PATH}"

# cgeist + clang 18 resource headers
COPY --from=cgeist-builder /build/polygeist/bin/cgeist /usr/local/bin/cgeist
COPY --from=cgeist-builder /build/llvm18/lib/clang/18/include \
                           /usr/local/lib/clang/18/include
