# AffineLoopTilingCostModel — upstream tracking

`LoopTiling.cpp` is a verbatim fork of upstream MLIR's
`mlir/lib/Dialect/Affine/Transforms/LoopTiling.cpp`, with a localized
replacement of the tile-size selection placeholder (`getTileSizes`).  The
fork exists so we can drive tile-size selection from drcompiler's unified
cost model (memory hierarchy + register pressure + ALU) as described in
REGISTER_PRESSURE_PLAN.md §16.

## Pinned upstream SHA

```
ce6d22760765e001a404d136c1d4dc1dce497791
```

(captured from `/home/tor/Dev/marco/source/llvm-project`.)

## Renamings vs upstream

| Upstream | Drcompiler fork |
|---|---|
| pass option `--affine-loop-tile` | `--dr-affine-loop-tile` |
| pass class `LoopTiling` | `DrAffineLoopTilePass` |
| `GEN_PASS_DEF_AFFINELOOPTILING` | `GEN_PASS_DEF_DRAFFINELOOPTILEPASS` |
| `DEBUG_TYPE "affine-loop-tile"` | `DEBUG_TYPE "dr-affine-loop-tile"` |
| ctor `mlir::affine::createLoopTilingPass()` | `mlir::createDrAffineLoopTilePass()` |

All deliberate divergences are marked `// DR-DIVERGE:` in the source.
