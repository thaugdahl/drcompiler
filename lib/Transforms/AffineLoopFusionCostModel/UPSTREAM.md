# AffineLoopFusionCostModel — upstream tracking

`LoopFusion.cpp` in this directory is a verbatim fork of upstream MLIR's
`mlir/lib/Dialect/Affine/Transforms/LoopFusion.cpp`, plus a localized
replacement of the profitability cost model (`isFusionProfitable`).  The
fork exists so we can drive fusion decisions from the unified cost model
(memory + register pressure + ALU) introduced by
REGISTER_PRESSURE_PLAN.md §7.

## Pinned upstream SHA

```
ce6d22760765e001a404d136c1d4dc1dce497791
```

(captured from `/home/tor/Dev/marco/source/llvm-project` at the time of
the fork — Marco's bundled LLVM 22 tree).

## Re-sync discipline

Periodically diff this directory against the pinned upstream file and
cherry-pick correctness fixes:

```
diff -u /path/to/llvm-project/mlir/lib/Dialect/Affine/Transforms/LoopFusion.cpp \
        lib/Transforms/AffineLoopFusionCostModel/LoopFusion.cpp
```

Our localized changes are intentionally minimal so that re-sync stays
mechanical.  See `// DR-DIVERGE:` markers in `LoopFusion.cpp` for every
deliberate deviation from upstream.

## Renamings vs upstream

| Upstream | Drcompiler fork |
|---|---|
| pass option `--affine-loop-fusion` | `--dr-affine-loop-fusion` |
| pass class `LoopFusion` | `DrAffineLoopFusionPass` |
| `GEN_PASS_DEF_AFFINELOOPFUSION` | `GEN_PASS_DEF_DRAFFINELOOPFUSIONPASS` |
| `DEBUG_TYPE "affine-fusion"` | `DEBUG_TYPE "dr-affine-fusion"` |
| ctor `mlir::affine::createLoopFusionPass()` | `mlir::createDrAffineLoopFusionPass()` |

All other code is verbatim from upstream.
