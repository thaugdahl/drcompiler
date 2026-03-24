# ==========================================================================
# drcc-spec: SPEC CPU 2017 benchmarking image.
#
# Layers:
#   drcc-base (base.Dockerfile)  — LLVM 22 + cgeist
#   drcc      (drcc.Dockerfile)  — dr-opt + drcc wrapper
#   drcc-spec (this file)        — scripts, config, instrumentation tools
#
# Build:
#   docker build -f docker/base.Dockerfile -t drcc-base .
#   docker build -f docker/drcc.Dockerfile -t drcc .
#   docker build -f docker/spec.Dockerfile -t drcc-spec .
#
# Run (SPEC2017 mounted from host — not baked in for licensing):
#   docker run --rm \
#     -v /path/to/SPEC2017:/spec2017 \
#     -v ./results:/results \
#     drcc-spec 505.mcf_r 519.lbm_r
#
# For cache hit-rate metrics (perf stat), add --privileged:
#   docker run --rm --privileged \
#     -v /path/to/SPEC2017:/spec2017 \
#     -v ./results:/results \
#     drcc-spec --size ref 505.mcf_r
#
# Interactive shell:
#   docker run --rm -it --entrypoint bash \
#     -v /path/to/SPEC2017:/spec2017 drcc-spec
# ==========================================================================

FROM drcc

# Override drcc's ENTRYPOINT (we need shell access, not the compiler).
ENTRYPOINT ["/opt/drcc/scripts/run-spec.sh"]

# Default: tier 1 benchmarks (clean C, near-certain to compile).
# Override with: docker run drcc-spec --tier 2    (adds C++ benchmarks)
#                docker run drcc-spec --tier all   (everything plausible)
#                docker run drcc-spec 505.mcf_r    (specific benchmark)
CMD ["--tier", "1"]

# GNU time (-v for peak RSS), binutils (size(1) for code size),
# rsync (to populate SPEC volume from read-only source mount).
RUN apt-get update && apt-get install -y --no-install-recommends \
      time \
      binutils \
      rsync \
    && rm -rf /var/lib/apt/lists/*

# SPEC config for drcc.
COPY config/drcc.cfg /opt/drcc/config/drcc.cfg

# Orchestration and instrumentation scripts.
COPY scripts/run-spec.sh          /opt/drcc/scripts/run-spec.sh
COPY scripts/instrumented-run.sh  /opt/drcc/scripts/instrumented-run.sh
RUN chmod +x /opt/drcc/scripts/run-spec.sh /opt/drcc/scripts/instrumented-run.sh

# Ensure drcc is on PATH (already true from parent image, but be explicit).
ENV PATH="/usr/local/bin:/opt/llvm/bin:${PATH}"

# Mount points:
#   /spec2017-src  — host SPEC installation (read-only)
#   /spec2017      — installed SPEC copy (Docker named volume, writable)
#   /results       — benchmark output
VOLUME ["/spec2017-src", "/spec2017", "/results"]
