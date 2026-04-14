# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is drcompiler

An out-of-tree MLIR compiler for cache-aware data recomputation analysis. It classifies `memref.load` ops by store provenance (SINGLE/MULTI/LEAKED/KILLED) and can replace single-provenance loads with the stored value, eliminating redundant memory traffic. Built for evaluating optimizations on SPEC CPU 2017.

Uses **upstream llvm/llvm-project** (LLVM 22), NOT the marco-compiler fork.

## Build

```bash
cmake -G Ninja -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_INSTALL_DIR=/path/to/llvm22-install

ninja -C build
```

Requires: CMake >= 3.20, Ninja, LLVM 22 install with MLIR+Clang. cgeist (Polygeist) must be on PATH or pointed to via `CGEIST_DIR` env var.

Build outputs: `build/tools/dr-opt/dr-opt` (optimizer) and `build/drcc` (compiler wrapper script).

## Testing

Tests use LLVM lit (ShTest format). All test files are `.mlir`.

```bash
# Run all tests
ninja -C build check-drcompiler

# Run a single test
llvm-lit test/DataRecomputation/scalar-store-kills-prior.mlir

# Run tests in a subdirectory
llvm-lit test/DataRecomputation/
```

Tests use `// RUN:` lines that invoke `dr-opt` with pass pipelines and verify output via `FileCheck` or `-verify-diagnostics`. The diagnostic testing pattern:
```mlir
// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics
// expected-remark @below {{load: SINGLE}}
```

## Compilation pipeline

```
source.c → cgeist (LLVM 18) → MLIR (Affine/SCF/MemRef)
  → sed DLTI fixup (vector<Nxi32> → vector<Nxi64>)
  → dr-opt (raise-malloc-to-memref, data-recomputation, memory-fission)
  → mlir-opt (lower affine → scf → cf → llvm, finalize-memref-to-llvm)
  → mlir-translate --mlir-to-llvmir
  → clang -O2 backend → .o
```

cgeist (LLVM 18) and dr-opt (LLVM 22) are decoupled via textual `.mlir` files — no shared libraries. The DLTI sed fixup bridges the serialization format change between LLVM 18 and 22.

`drcc` (`tools/drcc/drcc.sh.in`) orchestrates this pipeline as a drop-in `cc`/`c++` for SPEC2017. Tool paths are substituted by CMake at configure time.

## Architecture

### Three passes (registered in `include/drcompiler/Transforms/Passes.td`)

1. **`raise-malloc-to-memref`** (`lib/Transforms/RaiseMallocToMemRef.cpp`) — Converts Polygeist's malloc/free + LLVM load/store/GEP back to memref ops.

2. **`data-recomputation`** (`lib/Transforms/DataRecomputation.cpp`, ~1700 LOC) — The core pass. Performs interprocedural load-store provenance analysis, classifying every `memref.load`. Key options:
   - `dr-recompute`: replace SINGLE-provenance loads with stored value
   - `dr-cost-model`: gate recomputation on cache hierarchy cost model
   - `dr-partial-remat`: allow cloning non-chainable leaf loads as partial rematerialization (requires `dr-cost-model`; per-load decision gated by a strict `alu+leaf < loadLat` check)
   - `dr-partial-max-leaves`: cap on partial-leaf loads per remat (default 4)
   - `dr-test-diagnostics`: emit classification remarks for testing
   - `dr-dot-file=<path>`: emit GraphViz provenance graph
   - Cache params: `dr-l1-size`, `dr-l2-size`, `dr-l1-latency`, `dr-l2-latency`, `dr-l3-latency`, `dr-mem-latency`

3. **`memory-fission`** (`lib/Transforms/MemoryFission.cpp`) — Inverse of loop fusion. Splits fused loops when materializing to buffers is cheaper (cache-aware cost model). Supports nested loop fission.

### Key headers

- `include/drcompiler/Transforms/DataRecomputation/AnalysisState.h` — Core type aliases: `StoreMap`, `LoadProvenanceMap`, `EnrichedCallGraph`, `AnalysisContext`
- `include/drcompiler/Transforms/DataRecomputationIndexing.h` — `PointSet` type for concrete index coverage tracking

### dr-opt tool

`tools/dr-opt/dr-opt.cpp` — A thin `mlir-opt` wrapper that registers all upstream MLIR passes + the 3 custom passes.

## Environment variables

- `LLVM_INSTALL_DIR` — CMake: path to LLVM 22 install
- `CGEIST_DIR` — CMake/runtime: path to cgeist binary
- `DR_PASS_FLAGS` — Runtime: override the pass pipeline used by drcc (default: `--pass-pipeline=builtin.module(raise-malloc-to-memref,data-recomputation{dr-recompute=true})`)

## Benchmarking

`scripts/benchmark.sh <source.c> [iterations]` — Compares 5 pipeline configurations (baseline, greedy recompute, cost-model gated, memory-fission, clang-only reference) and reports median wall-clock time.
