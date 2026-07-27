# ONNX-MLIR / LLVM 22.1.1 Build Fixes

These are the patches required to build onnx-mlir (main, as of 2026-03-12)
against `llvmorg-22.1.1`.

## 1. LLVM must be built with RTTI (`llvm-builder.Dockerfile`)

**Flag:** `-DLLVM_ENABLE_RTTI=ON`

LLVM defaults to `-fno-rtti`. onnx-mlir's bundled stablehlo and rapidcheck both
require C++ typeinfo. stablehlo inherits from LLVM types with virtual functions,
so the compiler generates vtable typeinfo entries pointing to base-class typeinfo
that doesn't exist in an RTTI-off LLVM build.

## 2. Abseil must be built from source (`onnx-mlir.Dockerfile`)

**Version used:** `20240722.0`

Ubuntu 24.04 ships absl `20230125.3`, which is missing targets such as
`absl::log_internal_check_op` that onnx-mlir requires. Build absl 20240722.0
from source and install to `/usr/local` before configuring onnx-mlir.

## 3. `writeUnownedBlob` → `writeOwnedBlob` (`onnx-mlir.Dockerfile`)

**Affected file:** `third_party/stablehlo/stablehlo/dialect/VhloBytecode.cpp`

`mlir::DialectBytecodeWriter::writeUnownedBlob` was added to LLVM main in
commit `c83ebf19` (2026-02-19) to support packed i1 dense element serialisation,
but was not backported to the `release/22.x` branch. The replacement,
`writeOwnedBlob`, has the same signature; the blob data is always live at the
call site so the ownership difference is safe.

```
sed -i 's/writeUnownedBlob/writeOwnedBlob/g'
```

## 4. `getNumThreadsVarsMutable` → `getNumThreadsMutable` (`onnx-mlir.Dockerfile`)

**Affected file:** `src/Transform/ProcessKrnlParallelClause.cpp`

`mlir::omp::ParallelOp::getNumThreadsVarsMutable` was renamed to
`getNumThreadsMutable` in LLVM 22. Applied via sed after clone.

```
sed -i 's/getNumThreadsVarsMutable/getNumThreadsMutable/g'
```

## 5. Install `onnx` Python package (`onnx-mlir.Dockerfile`)

**Package:** `pip install onnx`

The `docs/doc_example` build target runs a Python script that imports `onnx`
to generate a sample `.onnx` model. Without it the build fails at the very end
(target 1226/1235). Install via pip with `--break-system-packages` on Ubuntu 24.04.

## 6. `llvm::scope_exit` → `llvm::make_scope_exit` (`onnx-mlir.Dockerfile`)

**Affected file:** `third_party/stablehlo/stablehlo/transforms/StablehloRefineShapes.cpp`

LLVM 22 removed the directly-constructed `llvm::scope_exit(callable)` class form;
only the factory `llvm::make_scope_exit(callable)` remains. Same RAII object and
single-callable signature, so the rename is a drop-in.

```
error: 'scope_exit' is not a member of 'llvm'; did you mean 'make_scope_exit'?
sed -i 's/llvm::scope_exit/llvm::make_scope_exit/g'
```

## 7. ONNXToLinalg pass-option description: strip embedded quotes (`onnx-mlir.Dockerfile`)

**Affected file:** `src/Conversion/ONNXToLinalg/Passes.td`

The `linalg-ops` pass option's description contains escaped double-quotes in its
example text:

```
"--convert-onnx-to-linalg='linalg-ops=\"MatMul.*\"'. "
```

`mlir-tblgen` from llvmorg-22.1.1 copies the description into the generated
`Passes.h.inc` `cl::desc(...)` C++ string literal **without re-escaping the inner
`"`**. The emitted literal therefore closes early at `linalg-ops="`, leaving
`MatMul.*` parsed as a user-defined-literal suffix:

```
Passes.h.inc:83: error: unable to find string literal operator 'operator""MatMul'
  with 'const char [237]', 'long unsigned int' arguments
```

Fix by removing the inner quotes from the (cosmetic) example string:

```
sed -i 's/\"MatMul\.\*\"/MatMul.*/g'   # applied to src/Conversion/ONNXToLinalg
```

This bug is present at the pinned commit — the original four-patch list above was
never validated through to a complete build.

## 8. Affine passes header path (`onnx-mlir.Dockerfile`)

**Affected file:** `src/Tools/onnx-mlir-opt/RegisterPasses.cpp`

In llvmorg-22.1.1 the Affine passes header lives at
`mlir/Dialect/Affine/Passes.h`; onnx-mlir includes the older
`mlir/Dialect/Affine/Transforms/Passes.h`, which no longer exists:

```
RegisterPasses.cpp:42: fatal error:
  mlir/Dialect/Affine/Transforms/Passes.h: No such file or directory
```

```
sed -i 's#mlir/Dialect/Affine/Transforms/Passes.h#mlir/Dialect/Affine/Passes.h#g'
```

## Pinning

The clone is pinned via `ARG ONNX_MLIR_REF` to
`3db4b49b02e0a3c179177678d90119064d026e6d` — the last commit on 2026-03-12, the
date this patch list was validated against. onnx-mlir `main` is a moving target;
later commits introduce new LLVM-22 API drift these patches don't cover (e.g.
HEAD around 2026-06 fails with a `mlir-tblgen` Pass-option emission mismatch in
`ONNXToLinalg/Passes.h.inc`). Pinning to the validated commit keeps rebuilds
reproducible. When bumping the ref, rebuild and extend the patch list above for
any new renames/mismatches.
