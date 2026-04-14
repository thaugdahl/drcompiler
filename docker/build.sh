#!/usr/bin/bash
# Build context is the repo root (onnx-mlir/) so drcc.Dockerfile can COPY
# source files from tools/ and scripts/.  Run this script from docker/ or
# from onnx-mlir/ — the cd below makes the context path work either way.
cd "$(dirname "$0")/.."

# Each of these can be rebuilt independently and cached
docker build -f docker/cgeist-builder.Dockerfile -t drcc-cgeist-builder .
docker build -f docker/llvm-builder.Dockerfile   -t drcc-llvm-builder .

# Assembly — fast, just COPYs from the two builder images
docker build -f docker/base.Dockerfile -t drcc-base .

# Then as before
docker build -f docker/drcc.Dockerfile      -t drcc .
docker build -f docker/onnx-mlir.Dockerfile -t onnx-mlir .
