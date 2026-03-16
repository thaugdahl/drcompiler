# drcompiler

An out-of-tree MLIR compiler that implements **data recomputation analysis**
for C/C++ programs. It classifies every `memref.load` by its store provenance
(SINGLE, MULTI, LEAKED, KILLED) and can optionally replace single-provenance
loads with the stored value directly, eliminating redundant memory traffic.

Built for evaluating data recomputation optimizations on **SPEC CPU 2017**.

## Pipeline

```
                cgeist               dr-opt              mlir-translate       clang
  source.c  ──────────►  MLIR  ──────────────►  MLIR  ──────────────────►  LLVM IR  ────►  .o
             (LLVM 18)   Affine/SCF/MemRef      (optimized)                 (LLVM 22)
```

The `drcc` wrapper script orchestrates this pipeline as a drop-in `cc`
replacement for SPEC2017's build system. Each component is described below.

## Components

### dr-opt

A thin `mlir-opt` wrapper that registers the DataRecomputation pass alongside
all upstream MLIR passes and dialects. This is the core of the project — it
links against **LLVM 22** and operates on standard MLIR dialects (Affine, SCF,
MemRef, Arith, Func).

```bash
# Analyze load provenance (diagnostics mode)
dr-opt --data-recomputation --dr-test-diagnostics input.mlir

# Apply recomputation transformation
dr-opt --data-recomputation --dr-recompute input.mlir -o output.mlir

# Emit a GraphViz DOT graph of load-store provenance
dr-opt --data-recomputation --dr-dot-file=provenance.dot input.mlir
```

### cgeist (Polygeist)

Polygeist's C/C++ frontend that lowers source code to high-level MLIR with
Affine loop nests and MemRef memory operations — the representation needed for
polyhedral analysis. This is the only tool that can raise C semantics to
Affine/SCF/MemRef (as opposed to ClangIR, which lowers to its own CIR dialect
and then straight to LLVM).

cgeist is built separately against **Polygeist's bundled LLVM 18**. It
communicates with the rest of the pipeline via textual `.mlir` files — no
shared libraries or linking.

### drcc

A Bash wrapper script that chains cgeist, dr-opt, mlir-translate, and clang
into a single compiler invocation. Acts as a drop-in replacement for `cc`/`c++`
in SPEC2017 config files:

```cfg
CC  = /path/to/build/drcc
CXX = /path/to/build/drcc
```

Supports standard compiler flags: `-c`, `-S`, `-E`, `-o`, `-l`, `-L`, `-Wl,`,
multi-file compilation, and link-only invocations.

### DLTI fixup

The DLTI (Data Layout Type Interface) dialect changed its serialization format
between LLVM 18 and 22. cgeist emits data layout entries as
`vector<Nxi32>` while LLVM 22 expects `vector<Nxi64>`. The `drcc` wrapper
applies an in-place `sed` fixup on cgeist's output before passing it to dr-opt.
This is a cosmetic format change — the layout values themselves are identical.

## Building

### Prerequisites

- CMake >= 3.20, Ninja
- An LLVM install with MLIR and Clang (upstream [llvm/llvm-project](https://github.com/llvm/llvm-project), tested with 22.1.x)
- Docker (for building cgeist on systems with GCC >= 15)

### 1. Build cgeist

LLVM 18 does not compile with GCC 15+ or libc++ 19+. On modern systems (Arch,
Fedora 42+), build cgeist inside Docker:

```bash
docker build -f docker/cgeist-builder.Dockerfile -o polygeist-install .
```

This uses Ubuntu 22.04 (GCC 11) internally and exports a single binary to
`polygeist-install/bin/cgeist`. On systems with GCC <= 14 (Ubuntu <= 24.04),
you can build natively instead:

```bash
./scripts/build-cgeist.sh
```

### 2. Build drcompiler

```bash
export PATH="$PWD/polygeist-install/bin:$PATH"

cmake -G Ninja -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_INSTALL_DIR=/path/to/llvm22-install

ninja -C build
```

CMake will find `cgeist` on PATH (or you can set the `CGEIST_DIR` environment
variable). Outputs:

- `build/tools/dr-opt/dr-opt` — the optimizer
- `build/drcc` — the compiler wrapper

### 3. Verify

```bash
# Test the pass
build/tools/dr-opt/dr-opt \
  --data-recomputation --dr-test-diagnostics \
  test/simple-provenance.mlir

# Test the full pipeline
echo 'int add(int a, int b) { return a + b; }' > /tmp/test.c
build/drcc -c /tmp/test.c -o /tmp/test.o -v
```

## Docker deployment (SPEC2017)

A self-contained Docker image packages the entire pipeline:

```bash
# Build the image (takes a while — builds LLVM 18 + LLVM 22 from source)
docker build -f docker/drcc.Dockerfile -t drcc .

# Compile C to object files
docker run --rm -v "$PWD":/work drcc -c /work/foo.c -o /work/foo.o

# Link on the host
ld.lld foo.o -o foo -lc
```

See `docker/drcc.Dockerfile` for the full multi-stage build. It uses
Ubuntu 24.04 with upstream [llvm/llvm-project](https://github.com/llvm/llvm-project)
(pinned to `llvmorg-22.1.1`). The LLVM version can be changed via build args:

```bash
docker build -f docker/drcc.Dockerfile \
  --build-arg LLVM_TAG=llvmorg-22.1.1 \
  --build-arg LLVM_VERSION=22 \
  -t drcc .
```

For SPEC2017, point the config at `drcc` inside the container or bind-mount
the image's `/usr/local/bin/drcc` into your benchmark runner.

## Project structure

```
drcompiler/
├── include/drcompiler/Transforms/
│   ├── Passes.td                  # Tablegen pass definition
│   ├── Passes.h                   # Pass registration header
│   ├── DataRecomputation.h        # Pass declaration
│   ├── DataRecomputationIndexing.h  # PointSet type for index coverage
│   └── DataRecomputation/
│       ├── AnalysisState.h        # Type aliases (StoreMap, LoadProvenanceMap, ...)
│       └── DotEmitter.h           # GraphViz DOT output
├── lib/Transforms/
│   └── DataRecomputation.cpp      # Pass implementation
├── tools/
│   ├── dr-opt/                    # mlir-opt wrapper with DR pass
│   └── drcc/drcc.sh.in            # Compiler wrapper template
├── docker/
│   └── cgeist-builder.Dockerfile  # Builds cgeist in Ubuntu 22.04
├── scripts/
│   └── build-cgeist.sh            # Native cgeist build script
└── test/
    └── simple-provenance.mlir     # Smoke test
```

## Pass options

| Flag | Description |
|------|-------------|
| `--data-recomputation` | Run the provenance analysis |
| `--dr-recompute` | Replace SINGLE-provenance loads with the stored value |
| `--dr-test-diagnostics` | Emit remarks classifying each load (for testing) |
| `--dr-dot-file=<path>` | Write a GraphViz DOT graph of provenance relationships |

## Load classifications

| Class | Meaning |
|-------|---------|
| SINGLE | Exactly one store can be the last writer |
| MULTI | Multiple stores may be the last writer |
| LEAKED | The memref escapes (passed to a call, etc.) — external writes possible |
| KILLED | No reaching store found |
