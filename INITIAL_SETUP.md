# drcompiler — Initial Setup Notes

## Goal

Build an out-of-tree Polygeist-based C/C++ compiler that integrates the
DataRecomputation MLIR pass for testing against SPEC2017 benchmarks.

The DataRecomputation pass performs load-store provenance analysis on
memref-level MLIR: for each `memref.load`, it determines which `memref.store`
operations could have been the last writer, classifies loads (SINGLE, MULTI,
LEAKED, KILLED), and optionally replaces SINGLE loads by rematerializing the
stored value at the load site.

## What existed before

### DataRecomputation pass (in the marco project)

The pass lived inside the marco compiler project at
`/home/tor/Dev/marco/source/marco/`. The relevant files were:

| File | Role |
|------|------|
| `lib/Transforms/DataRecomputation.cpp` | Pass implementation (~1250 lines) |
| `include/marco/Transforms/DataRecomputation.h` | Pass declaration |
| `include/marco/Transforms/Passes.h` | Pass registration (`GEN_PASS_REGISTRATION`) |
| `include/marco/Transforms/Passes.td` | Tablegen pass definition |
| `include/marco/Transforms/DataRecomputationIndexing.h` | `PointSet` type for index coverage |
| `include/marco/Transforms/DataRecomputation/AnalysisState.h` | All analysis vocabulary types |
| `include/marco/Transforms/DataRecomputation/DotEmitter.h` | GraphViz DOT output (~260 lines) |

The marco build system used `add_mlir_library(MLIRCustomTransforms ...)` and
tablegen via `mlir_tablegen(Passes.h.inc -gen-pass-decls -name CustomPasses)`.

### LLVM/MLIR installations

Two LLVM 22.0.0git trees were present:

1. **`/home/tor/Dev/PhD/DRComp/llvm-install/`** — Built with
   `LLVM_ENABLE_PROJECTS=clang;clang-tools-extra` only. Has LLVM and Clang
   cmake configs but **no MLIR cmake**. Not usable for MLIR out-of-tree builds.

2. **`/home/tor/Dev/marco/install/llvm-project/`** — Full install with LLVM,
   MLIR, and Clang cmake configs under `lib/cmake/{llvm,mlir,clang}`. Includes
   `mlir-opt`, `mlir-tblgen`, `mlir-translate`, and all MLIR libraries. This is
   the install used by the marco project and is the one we depend on.

3. **`/home/tor/Dev/PhD/DRComp/llvm-project/`** — Full LLVM monorepo source
   with a build directory, but the build was configured without MLIR
   (`LLVM_ENABLE_PROJECTS=clang;clang-tools-extra`). The MLIR source is present
   but never built here.

No Polygeist installation was found anywhere on the system.

## What was done

### 1. Project structure

Created `drcompiler/` as a standard MLIR out-of-tree project (modeled after
`mlir/examples/standalone` in the LLVM tree):

```
drcompiler/
├── CMakeLists.txt                              # Top-level: finds LLVM/MLIR/Clang, Polygeist
├── build.sh                                    # Convenience configure/build script
├── cmake/modules/
│   └── GenerateDrcc.cmake                      # Resolves generator expressions for drcc
├── include/drcompiler/Transforms/
│   ├── CMakeLists.txt                          # Tablegen: Passes.td → Passes.h.inc
│   ├── Passes.td                               # Pass tablegen definition
│   ├── Passes.h                                # GEN_PASS_REGISTRATION include
│   ├── DataRecomputation.h                     # Pass declaration
│   ├── DataRecomputationIndexing.h             # PointSet type
│   └── DataRecomputation/
│       ├── AnalysisState.h                     # Vocabulary types (DRPassContext, StoreMap, etc.)
│       └── DotEmitter.h                        # GraphViz DOT emission
├── lib/Transforms/
│   ├── CMakeLists.txt                          # add_mlir_library(DRCompTransforms ...)
│   └── DataRecomputation.cpp                   # Pass implementation
├── tools/
│   ├── CMakeLists.txt
│   ├── dr-opt/
│   │   ├── CMakeLists.txt                      # add_llvm_executable(dr-opt ...)
│   │   └── dr-opt.cpp                          # mlir-opt wrapper with DR pass
│   └── drcc/
│       └── drcc.sh.in                          # SPEC2017 compiler wrapper template
└── test/
    └── simple-provenance.mlir                  # Smoke test
```

### 2. Adapting the pass from marco

All source and header files were copied from the marco tree and adapted:

- **Include path prefix**: Changed `marco/Transforms/` → `drcompiler/Transforms/`
  in every `#include` directive and header guard.

- **Tablegen target name**: Changed from `MLIRCustomTransformsPassIncGen` /
  `-name CustomPasses` to `DRCompPassesIncGen` / `-name DRCompPasses`. This
  affects the generated registration function name
  (`registerDRCompPassesPasses()`).

- **Library name**: Changed from `MLIRCustomTransforms` to `DRCompTransforms`.

- **Vector dialect include**: The original had
  `#include "mlir/Dialect/Vector/IR/VectorDialect.h.inc"` which is the
  tablegen-generated include. Changed to
  `#include "mlir/Dialect/Vector/IR/VectorOps.h"` which is the standard public
  header for LLVM 22.

No logic changes were made to the pass itself — the analysis and transformation
code is identical.

### 3. CMake build system

**Top-level `CMakeLists.txt`**:

- Uses `find_package(LLVM/MLIR/Clang REQUIRED CONFIG)`.
- Accepts `LLVM_INSTALL_DIR` to prepend the cmake prefix path for all three
  packages from a single install root.
- Includes MLIR's `TableGen.cmake`, `AddLLVM.cmake`, `AddMLIR.cmake`,
  `HandleLLVMOptions.cmake`.
- Sets up both source and binary include directories (binary dir is needed for
  the tablegen-generated `Passes.h.inc`).
- Matches LLVM build flags: `-fno-rtti` and `-fno-exceptions` when LLVM was
  built without them.
- Polygeist support via three modes:
  - `POLYGEIST_INSTALL_DIR` — point at pre-built Polygeist.
  - `DRCOMP_BUILD_POLYGEIST=ON` — fetch and build via `ExternalProject_Add`.
  - Fallback: search PATH for `cgeist`.
- Generates the `drcc` wrapper script at build time, substituting tool paths.

**`include/drcompiler/Transforms/CMakeLists.txt`**:

```cmake
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name DRCompPasses)
add_public_tablegen_target(DRCompPassesIncGen)
```

This runs `mlir-tblgen` on `Passes.td` to produce `Passes.h.inc` in the build
directory. The `-name DRCompPasses` flag sets the generated registration
function to `registerDRCompPassesPasses()`.

**`lib/Transforms/CMakeLists.txt`**:

```cmake
add_mlir_library(DRCompTransforms
  DataRecomputation.cpp
  DEPENDS DRCompPassesIncGen
  LINK_LIBS PUBLIC
    MLIRAnalysis MLIRFunctionInterfaces MLIRLoopLikeInterface ...
)
```

The `DEPENDS DRCompPassesIncGen` ensures tablegen runs before compilation. Link
dependencies mirror the original marco build plus the additional dialect
libraries the pass includes (`MLIRAffineDialect`, `MLIRArithDialect`,
`MLIRMemRefDialect`, `MLIRLLVMDialect`, `MLIRPtrDialect`, `MLIRVectorDialect`).

**`tools/dr-opt/CMakeLists.txt`**:

Links against all MLIR dialect, conversion, and extension libs (via the
`MLIR_DIALECT_LIBS`, `MLIR_CONVERSION_LIBS`, `MLIR_EXTENSION_LIBS` global
properties), plus `MLIROptLib`, the three `MLIRRegisterAll*` libraries, and our
`DRCompTransforms`.

Initially this was missing `MLIRRegisterAllDialects`,
`MLIRRegisterAllPasses`, and `MLIRRegisterAllExtensions`, which caused linker
errors for `mlir::registerAllPasses()` etc. Adding those three targets fixed it.

### 4. The dr-opt tool

`dr-opt.cpp` is a thin `mlir-opt` wrapper:

```cpp
mlir::registerAllPasses();
mlir::registerDRCompPassesPasses();  // our pass
mlir::DialectRegistry registry;
mlir::registerAllDialects(registry);
mlir::registerAllExtensions(registry);
return mlir::MlirOptMain(argc, argv, "DRComp optimizer driver\n", registry);
```

This gives a full `mlir-opt` with all upstream passes and dialects, plus the
`--data-recomputation` pass. Pass options are specified via the pipeline string:

```
dr-opt --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true})' input.mlir
```

### 5. The drcc wrapper script

`tools/drcc/drcc.sh.in` is a bash script that acts as a drop-in C/C++ compiler
for SPEC2017. The pipeline for each source file is:

1. **cgeist** (Polygeist): `source.c` → MLIR (affine/scf/memref dialects)
2. **dr-opt**: Run `--data-recomputation --dr-recompute` on the MLIR
3. **mlir-translate**: MLIR → LLVM IR (`--mlir-to-llvmir`)
4. **clang**: LLVM IR → object file (or assembly)

For link-only invocations, it delegates directly to clang. The script parses
standard compiler flags (`-c`, `-S`, `-E`, `-o`, `-l`, `-L`, etc.) and routes
them appropriately. Tool paths are substituted at cmake configure time via
`@VARIABLE@` placeholders.

A two-stage generation process handles the `dr-opt` path (which is a CMake
generator expression, only resolved at build time):
1. `configure_file()` resolves `@CGEIST_EXECUTABLE@`, `@MLIR_TRANSLATE_EXECUTABLE@`, `@CLANG_EXECUTABLE@`.
2. A custom command runs `GenerateDrcc.cmake` to substitute the `dr-opt` path
   from the generator expression.

### 6. Build and verification

Configured with:

```bash
cmake -G Ninja -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_INSTALL_DIR=/home/tor/Dev/marco/install/llvm-project \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

Build produced:
- `build/tools/dr-opt/dr-opt` (~153 MB static binary)
- `build/drcc` (generated wrapper script)

Verified the pass works:

```
$ dr-opt --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics=true})' test/simple-provenance.mlir
```

Output correctly classified:
- `@simple_store_load`: single store → load as **SINGLE**
- `@branched_store`: two stores in if/else → load as **MULTI**

### 7. Issues encountered and resolved

1. **CMake target name collision**: The custom target `drcc` collided with the
   `tools/drcc/` subdirectory in ninja's build graph. Renamed the custom target
   to `drcc-wrapper`.

2. **Missing link libraries**: `dr-opt` initially failed to link with undefined
   references to `mlir::registerAllPasses()`, `mlir::registerAllDialects()`,
   and `mlir::registerAllExtensions()`. Fixed by adding
   `MLIRRegisterAllDialects`, `MLIRRegisterAllPasses`, and
   `MLIRRegisterAllExtensions` to the link targets.

3. **Vector dialect header**: The original code included
   `mlir/Dialect/Vector/IR/VectorDialect.h.inc` (tablegen output). Changed to
   `mlir/Dialect/Vector/IR/VectorOps.h` (public header) for the out-of-tree
   build where we don't regenerate upstream tablegen outputs.

## Remaining work

### Polygeist

No Polygeist installation exists yet. To get the full `drcc` pipeline working:

```bash
git clone https://github.com/llvm/Polygeist.git
cd Polygeist && mkdir build && cd build
cmake .. -GNinja \
  -DMLIR_DIR=/home/tor/Dev/marco/install/llvm-project/lib/cmake/mlir \
  -DCLANG_DIR=/home/tor/Dev/marco/install/llvm-project/lib/cmake/clang
ninja
```

Then reconfigure drcompiler:

```bash
./build.sh rebuild --polygeist-dir /path/to/Polygeist/build
```

Or set `DRCOMP_BUILD_POLYGEIST=ON` to have cmake fetch and build it
automatically (uses `ExternalProject_Add`).

### SPEC2017 integration

In a SPEC2017 config file, set:

```cfg
CC  = /path/to/drcompiler/build/drcc
CXX = /path/to/drcompiler/build/drcc
```

The `drcc` wrapper handles compilation flags, separate compilation (`-c`),
linking, and multi-file builds. Pass-specific options can be tuned via the
`DR_PASS_FLAGS` environment variable (defaults to
`--data-recomputation --dr-recompute`).

### LLVM installation note

The project currently depends on the marco project's LLVM install at
`/home/tor/Dev/marco/install/llvm-project`. To make it self-contained, either:

- Rebuild the local `llvm-project/` with MLIR enabled:
  ```bash
  cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;mlir" \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DCMAKE_INSTALL_PREFIX=../install
  ninja install
  ```
- Or continue using the marco install (both are LLVM 22.0.0git).
