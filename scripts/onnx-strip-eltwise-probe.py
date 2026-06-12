#!/usr/bin/env python3
# onnx-strip-eltwise-probe.py — WP-G4 measure-first probe.
#
# Stub the elementwise nests from a codegen .dr.mlir for a TIMING probe
# (correctness ignored): remove every top-level (4-space-indent) affine.for nest
# that is elementwise -- contains a ReLU (arith.maxnumf/maximumf) OR is a scalar
# compute nest (has mulf/addf/subf, NO iter_args, NO vector). Init (store %cst)
# and the vectorized/promoted contraction nests (iter_args / vector) are kept.
#
# Usage: onnx-strip-eltwise-probe.py <codegen.dr.mlir> <stubbed-out.mlir>
# Rebuild the stubbed .mlir through the bench back-half and diff the median
# against the unstubbed binary; the delta bounds the elementwise-fusion upside.
# Measured on resnet50 (2026-06-12): ~0.012 s (~1.3%) -> NO-GO. See
# ONNX_O3_GAP_RESULTS.md WP-G4.
import sys
src, dst = sys.argv[1], sys.argv[2]
lines = open(src).read().split('\n')
out = []
i = 0
n = len(lines)
removed = 0
kept_nests = 0
while i < n:
    line = lines[i]
    # top-level nest opener: exactly 4 leading spaces then 'affine.for'
    if line.startswith('    affine.for ') and line.rstrip().endswith('{') and (len(line) - len(line.lstrip())) == 4:
        # collect the balanced block until the matching 4-space '    }'
        j = i
        depth = 0
        block = []
        while j < n:
            l = lines[j]
            block.append(l)
            depth += l.count('{') - l.count('}')
            if depth == 0:
                break
            j += 1
        body = '\n'.join(block)
        is_relu = ('arith.maxnumf' in body) or ('arith.maximumf' in body)
        has_compute = ('arith.mulf' in body) or ('arith.addf' in body) or ('arith.subf' in body)
        has_iterargs = 'iter_args' in body
        has_vector = 'vector' in body
        eltwise = is_relu or (has_compute and not has_iterargs and not has_vector)
        if eltwise:
            removed += 1
        else:
            out.extend(block)
            kept_nests += 1
        i = j + 1
        continue
    out.append(line)
    i += 1
open(dst, 'w').write('\n'.join(out))
sys.stderr.write(f"removed {removed} eltwise nests, kept {kept_nests} top-level nests\n")
