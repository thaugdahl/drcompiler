#!/usr/bin/env bash
# Build context is the repo root so drcc.Dockerfile can COPY from tools/ and scripts/.
# Run from docker/ or from onnx-mlir/ — the cd below handles either.
set -euo pipefail

ARCH=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --arch) ARCH="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2
       echo "Usage: $0 [--arch x86_64|aarch64]" >&2
       exit 1 ;;
  esac
done

# Default to native arch
if [[ -z "$ARCH" ]]; then
  case "$(uname -m)" in
    x86_64)        ARCH="x86_64" ;;
    aarch64|arm64) ARCH="aarch64" ;;
    *) echo "Unrecognised native arch: $(uname -m)" >&2; exit 1 ;;
  esac
fi

case "$ARCH" in
  x86_64)  DOCKER_ARCH="amd64" ;;
  aarch64) DOCKER_ARCH="arm64" ;;
  *) echo "Unknown --arch: $ARCH  (use x86_64 or aarch64)" >&2; exit 1 ;;
esac

PLATFORM="linux/${DOCKER_ARCH}"

cd "$(dirname "$0")/.."

echo "==> Building for ${ARCH} (${PLATFORM})"

# Builder images: all internal stages run on the build host (amd64) regardless
# of target. TARGETARCH is passed explicitly because we are not using --platform.
# This avoids buildx's registry-first metadata resolution for local images.
docker build \
  --build-arg TARGETARCH="${DOCKER_ARCH}" \
  -f docker/llvm-builder.Dockerfile \
  -t "drcc-llvm-builder:${ARCH}" .

docker build \
  --build-arg TARGETARCH="${DOCKER_ARCH}" \
  -f docker/cgeist-builder.Dockerfile \
  -t "drcc-cgeist-builder:${ARCH}" .

# Assembly images: --platform selects the target arch for the final ubuntu layer.
# These reference the above builder images from the local daemon (docker build
# checks local store first for unregistered image names).
docker build \
  --platform "${PLATFORM}" \
  --build-arg ARCH_TAG="${ARCH}" \
  -f docker/base.Dockerfile \
  -t "drcc-base:${ARCH}" .

docker build \
  --platform "${PLATFORM}" \
  --build-arg ARCH_TAG="${ARCH}" \
  --build-arg TARGETARCH="${DOCKER_ARCH}" \
  -f docker/drcc.Dockerfile \
  -t "drcc:${ARCH}" .

# onnx-mlir takes hours to build and requires arm64 Python3 dev headers for
# cross-compilation (complex multiarch apt setup). Skip if a correct-arch image
# already exists. To force a full rebuild, delete the image first and re-run,
# or build natively on an arm64 host.
_existing_onnx_arch=$(docker image inspect "onnx-mlir:${ARCH}" --format "{{.Architecture}}" 2>/dev/null || true)
if [ "${_existing_onnx_arch}" = "${DOCKER_ARCH}" ]; then
  echo "==> onnx-mlir:${ARCH} already exists as ${DOCKER_ARCH} — skipping"
else
  docker build \
    --platform "${PLATFORM}" \
    --build-arg ARCH_TAG="${ARCH}" \
    --build-arg TARGETARCH="${DOCKER_ARCH}" \
    -f docker/onnx-mlir.Dockerfile \
    -t "onnx-mlir:${ARCH}" .
fi

# Lean images: built with the default docker driver, which resolves
# docker-image:// contexts from the local daemon directly (no registry).
# The multiarch builder's docker-container driver can't load arm64 images from
# the local daemon, so we use the default driver + binfmt for the RUN steps.
if [ "${ARCH}" = "aarch64" ]; then
  echo "==> Registering arm64 binfmt for lean image apt-get steps..."
  docker run --privileged --rm tonistiigi/binfmt --install arm64 >/dev/null 2>&1 || true
fi

docker build \
  --platform "${PLATFORM}" \
  --build-arg ARCH_TAG="${ARCH}" \
  --build-context "drcc:${ARCH}=docker-image://drcc:${ARCH}" \
  -f docker/drcc-lean.Dockerfile \
  -t "drcc-lean:${ARCH}" .

docker build \
  --platform "${PLATFORM}" \
  --build-arg ARCH_TAG="${ARCH}" \
  --build-context "onnx-mlir:${ARCH}=docker-image://onnx-mlir:${ARCH}" \
  -f docker/onnx-mlir-lean.Dockerfile \
  -t "onnx-mlir-lean:${ARCH}" .

echo "==> Done. Images tagged :${ARCH}"
