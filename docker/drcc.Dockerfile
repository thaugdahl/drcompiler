# ==========================================================================
# drcc: Data Recomputation compiler pipeline.
#
# Requires drcc-base (build it first if you haven't):
#   docker build -f docker/base.Dockerfile -t drcc-base .
#
# Then build drcc (fast — only compiles dr-opt):
#   docker build -f docker/drcc.Dockerfile -t drcc .
#
# Usage:
#   docker run --rm -v "$PWD":/work drcc -c /work/foo.c -o /work/foo.o
# ==========================================================================

FROM drcc-base

# lmbench for cache latency probing (baked into drcc-base on next full rebuild).
RUN apt-get update && apt-get install -y --no-install-recommends lmbench \
    && rm -rf /var/lib/apt/lists/*

COPY . /src/drcompiler

RUN cmake -G Ninja -S /src/drcompiler -B /build/drcompiler \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_INSTALL_DIR=/opt/llvm \
    && cmake --build /build/drcompiler -j"$(nproc)" \
    && cp /build/drcompiler/tools/dr-opt/dr-opt /usr/local/bin/dr-opt \
    && rm -rf /build/drcompiler /src/drcompiler

# drcc wrapper script (pre-configured with absolute paths)
COPY tools/drcc/drcc.sh.in /tmp/drcc.sh.in
RUN sed \
      -e 's|@CGEIST_EXECUTABLE@|/usr/local/bin/cgeist|g' \
      -e 's|%%DROPT_EXECUTABLE%%|/usr/local/bin/dr-opt|g' \
      -e 's|@MLIR_TRANSLATE_EXECUTABLE@|/opt/llvm/bin/mlir-translate|g' \
      -e 's|@CLANG_EXECUTABLE@|/opt/llvm/bin/clang|g' \
      -e 's|@MLIR_OPT_EXECUTABLE@|/opt/llvm/bin/mlir-opt|g' \
      /tmp/drcc.sh.in > /usr/local/bin/drcc \
    && chmod +x /usr/local/bin/drcc \
    && rm /tmp/drcc.sh.in

# Struct-memref text preprocessor (called by drcc before dr-opt)
COPY tools/drcc/rewrite-struct-memrefs.py /usr/local/bin/rewrite-struct-memrefs.py

# Probing scripts for generating hardware-specific cost models.
COPY scripts/gen_cpu_cost_model.py scripts/cache_latency_bench.c \
     scripts/probe-cost-model.sh \
     /usr/local/share/drcompiler/scripts/

ENTRYPOINT ["drcc"]
