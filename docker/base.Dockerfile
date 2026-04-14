# ==========================================================================
# drcc-base: LLVM 22 + cgeist base image.
#
# Assembles pre-built artifacts from the two builder images.
# Build the builders first (each caches independently):
#
#   docker build -f docker/cgeist-builder.Dockerfile -t drcc-cgeist-builder .
#   docker build -f docker/llvm-builder.Dockerfile   -t drcc-llvm-builder .
#   docker build -f docker/base.Dockerfile            -t drcc-base .
#
# Then build drcc on top:
#   docker build -f docker/drcc.Dockerfile -t drcc .
# ==========================================================================

FROM drcc-llvm-builder AS llvm-builder
FROM drcc-cgeist-builder AS cgeist-builder

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
