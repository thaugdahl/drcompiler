#!/usr/bin/env python3
# Make a fixed-batch ONNX model dynamic-batch so onnx-mlir --shapeInformation can
# set batch=N for throughput SPMD.  The openai-gpt export bakes batch=1 into its
# un-flatten Reshape shape constants ([1,128,12,64] etc.); shapeInformation alone
# leaves them at 1 (onnx-mlir warns "inferred dim 16 != existing dim 1, use
# existing").  This: (1) sets input/output batch dim symbolic, (2) drops baked
# intermediate value_info so shapes re-infer, (3) rewrites Reshape/Expand shape
# constants whose leading dim is 1 (and have no -1 yet) to -1 (dynamic batch).
#
# Usage:  python3 patch-onnx-dynamic-batch.py in.onnx out.onnx [inspect]
# Run inside the onnx-mlir docker image (has python3 + onnx), e.g.
#   docker run --rm --entrypoint python3 -v "$W:$W" onnx-mlir:x86_64 \
#     scripts/patch-onnx-dynamic-batch.py model.onnx model_dynbatch.onnx
#
# Validated: openai-gpt -> batch=16 compiles with 447 `to 16` loops, output
# 16x128x768, and our batch-axis SPMD is BYTE-IDENTICAL to the batch-1 reference
# per sample (PARALLEL_SPMD_SPEC.md §11.22).
import onnx, sys
from onnx import numpy_helper

src, dst = sys.argv[1], sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else 'patch'
m = onnx.load(src)
g = m.graph
inits = {t.name: t for t in g.initializer}
const_out = {}
for n in g.node:
    if n.op_type == 'Constant':
        for a in n.attribute:
            if a.name == 'value':
                const_out[n.output[0]] = a.t

def arr_of(name):
    if name in inits:
        return numpy_helper.to_array(inits[name]), 'init'
    if name in const_out:
        return numpy_helper.to_array(const_out[name]), 'const'
    return None, None

if mode == 'inspect':
    print('inputs:', [(i.name, [(d.dim_value if d.HasField('dim_value') else d.dim_param)
                                for d in i.type.tensor_type.shape.dim]) for i in g.input])
    print('value_info count:', len(g.value_info))
    for n in g.node:
        if n.op_type in ('Reshape', 'Expand', 'Unsqueeze'):
            for ipt in n.input[1:]:
                a, kind = arr_of(ipt)
                if a is not None:
                    print(' ', n.op_type, n.name, kind, a.tolist())
    sys.exit(0)

# 1. input/output batch dim -> symbolic
for io in list(g.input) + list(g.output):
    d = io.type.tensor_type.shape.dim
    if len(d) >= 1:
        d[0].ClearField('dim_value'); d[0].dim_param = 'B'
# 2. drop baked intermediate shapes
del g.value_info[:]
# 3. Reshape/Expand shape constants: leading 1 -> -1
patched = 0
for n in g.node:
    if n.op_type not in ('Reshape', 'Expand'):
        continue
    for ipt in n.input[1:]:
        tgt = inits.get(ipt) or const_out.get(ipt)
        if tgt is None:
            continue
        a = numpy_helper.to_array(tgt).copy()
        if a.ndim == 1 and a.size >= 1 and int(a[0]) == 1 and not (a == -1).any():
            a[0] = -1
            tgt.CopyFrom(numpy_helper.from_array(a, tgt.name))
            patched += 1
print('patched reshape/expand shapes:', patched)
onnx.save(m, dst)
