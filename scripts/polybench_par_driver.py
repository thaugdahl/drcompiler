#!/usr/bin/env python3
# Generate a self-contained timing driver for a polybench kernel .mlir.
# Splices the kernel func verbatim, emits @main that allocs/fills concrete
# memrefs, casts to the kernel's dynamic-leading type, and times K calls.
#
# spec: kernel_mlir_path, scalars (list of (value, 'i32'|'f64'|'f32')),
#       memrefs (list of full-shape int lists, in arg order), elem ('f64'|'f32'), K
import sys, re, json

def main():
    spec = json.load(open(sys.argv[1]))
    src = open(spec['mlir']).read()
    # extract kernel function (from 'func.func @kernel' to its matching closing brace at col 2)
    m = re.search(r'(  func\.func @(kernel_\w+)\(.*?\n  \})', src, re.S)
    kfn = m.group(1); kname = m.group(2)
    sig = re.search(r'func\.func @\w+\((.*?)\)\s*attributes', kfn, re.S).group(1)
    args = [a.strip() for a in re.split(r',(?![^<]*>)', sig)]
    argtypes = [a.split(':',1)[1].strip() for a in args]
    elem = spec['elem']
    zero = '0.0' if elem in ('f64','f32') else '0'

    scalars = spec['scalars']      # list of [value, type]
    memrefs = spec['memrefs']      # list of full int shapes (arg order, only memref args)
    K = spec.get('K', 7)

    out = []
    for mp in re.findall(r'^#\w+ = affine_map<[^\n]*>', src, re.M):
        out.append(mp)
    out.append('func.func private @printMemrefF64(memref<*xf64>)')
    out.append('func.func private @rtclock() -> f64')
    out.append(kfn)
    out.append('func.func @main() {')
    out.append('  %c0 = arith.constant 0 : index')
    out.append('  %c1 = arith.constant 1 : index')

    # scalar SSA values
    callargs = []
    si = 0; mi = 0
    primes = [97, 89, 101, 83, 79, 103, 107]
    allocs = []   # (ssa_concrete, full_shape, dyn_type)
    for idx, t in enumerate(argtypes):
        if t.startswith('memref'):
            shape = memrefs[mi]; mi += 1
            inner = 'x'.join(str(d) for d in shape[1:])
            full = 'x'.join(str(d) for d in shape)
            ctype = f'memref<{full}x{elem}>'
            dyntype = t  # the kernel's declared (dynamic-leading) type
            nm = f'%m{idx}'
            out.append(f'  {nm}c = memref.alloc() : {ctype}')
            # fill
            P = primes[mi % len(primes)]
            ivs = [f'%i{idx}_{d}' for d in range(len(shape))]
            for d,(iv,dim) in enumerate(zip(ivs, shape)):
                out.append(f'  scf.for {iv} = %c0 to {" ".join([f"%d{idx}_{d}"])}_v step %c1 {{'.replace(f'%d{idx}_{d}_v', f'%c{dim}_{idx}_{d}'))
            # need dim constants; declare before loops -> simpler: inline constants
            # (redo: we already opened loops referencing %cDIM; declare them)
            # Build linear index
            out.append(f'    %lin{idx}_0 = arith.constant 0 : index')
            prev = f'%lin{idx}_0'
            for d,(iv,dim) in enumerate(zip(ivs, shape)):
                out.append(f'    %mul{idx}_{d} = arith.muli {prev}, %c{dim}_{idx}_{d} : index')
                out.append(f'    %add{idx}_{d} = arith.addi %mul{idx}_{d}, {iv} : index')
                prev = f'%add{idx}_{d}'
            out.append(f'    %rem{idx} = arith.remui {prev}, %cP{idx} : index')
            out.append(f'    %ri{idx} = arith.index_cast %rem{idx} : index to i64')
            out.append(f'    %rf{idx} = arith.sitofp %ri{idx} : i64 to {elem}')
            out.append(f'    %val{idx}_0 = arith.divf %rf{idx}, %cPf{idx} : {elem}')
            stored = f'%val{idx}_0'
            # optional diagonal dominance (for solvers): boost the diagonal so pivots are safe
            if spec.get('diag') and len(shape) == 2:
                out.append(f'    %boost{idx} = arith.constant 1000000.0 : {elem}')
                out.append(f'    %valb{idx} = arith.addf %val{idx}_0, %boost{idx} : {elem}')
                out.append(f'    %eqd{idx} = arith.cmpi eq, {ivs[0]}, {ivs[1]} : index')
                out.append(f'    %val{idx} = arith.select %eqd{idx}, %valb{idx}, %val{idx}_0 : {elem}')
                stored = f'%val{idx}'
            out.append(f'    memref.store {stored}, {nm}c[{", ".join(ivs)}] : {ctype}')
            for _ in shape:
                out.append('  }')
            out.append(f'  {nm} = memref.cast {nm}c : {ctype} to {dyntype}')
            allocs.append((f'{nm}c', ctype, shape))
            callargs.append(nm)
        else:
            v, vt = scalars[si]; si += 1
            nm = f'%s{idx}'
            if vt == 'i32':
                out.append(f'  {nm} = arith.constant {v} : i32')
            else:
                out.append(f'  {nm} = arith.constant {float(v)} : {vt}')
            callargs.append(nm)

    out_text = '\n'.join(out)
    # declare the dim/prime constants used in fills (collect from text)
    consts = []
    seen = set()
    # dim constants %cDIM_idx_d
    for idx, t in enumerate(argtypes):
        pass
    # easier: regenerate constants by scanning needed names
    needed = set(re.findall(r'%c(\d+)_(\d+)_(\d+)\b', out_text))
    declared = []
    for dim, i_, d_ in sorted(needed):
        declared.append(f'  %c{dim}_{i_}_{d_} = arith.constant {dim} : index')
    # prime constants
    primeneed = set(re.findall(r'%cP(\d+)\b', out_text))
    for idx in sorted(primeneed, key=int):
        P = primes[(int(idx)) % len(primes)]
        # mi for that arg: recompute — just use a fixed prime per idx
        declared.append(f'  %cP{idx} = arith.constant {P} : index')
        declared.append(f'  %cPf{idx} = arith.constant {float(P)} : {elem}')
    # splice declarations right after %c1
    out_text = out_text.replace('  %c1 = arith.constant 1 : index\n',
                                '  %c1 = arith.constant 1 : index\n' + '\n'.join(declared) + '\n', 1)

    # timing loop
    tl = []
    tl.append(f'  %cK = arith.constant {K} : index')
    tl.append(f'  %T = memref.alloc() : memref<{K}xf64>')
    tl.append('  scf.for %r = %c0 to %cK step %c1 {')
    tl.append('    %t0 = func.call @rtclock() : () -> f64')
    tl.append(f'    func.call @{kname}({", ".join(callargs)}) : ({", ".join(argtypes)}) -> ()')
    tl.append('    %t1 = func.call @rtclock() : () -> f64')
    tl.append('    %dt = arith.subf %t1, %t0 : f64')
    tl.append(f'    memref.store %dt, %T[%r] : memref<{K}xf64>')
    tl.append('  }')
    tl.append(f'  %U = memref.cast %T : memref<{K}xf64> to memref<*xf64>')
    tl.append('  func.call @printMemrefF64(%U) : (memref<*xf64>) -> ()')
    # checksum over all memref args (sum of all elements), printed as memref<1xf64>
    tl.append('  %CK = memref.alloc() : memref<1xf64>')
    tl.append(f'  %zero = arith.constant {zero} : {elem}')
    tl.append('  %zd = arith.constant 0.0 : f64')
    tl.append('  memref.store %zd, %CK[%c0] : memref<1xf64>')
    for ci,(cssa,ctype,shape) in enumerate(allocs):
        ivs = [f'%ck{ci}_{d}' for d in range(len(shape))]
        for d,(iv,dim) in enumerate(zip(ivs,shape)):
            tl.append(f'  scf.for {iv} = %c0 to %ckd{ci}_{d} step %c1 {{')
        tl.append(f'    %ev{ci} = memref.load {cssa}[{", ".join(ivs)}] : {ctype}')
        if elem != 'f64':
            tl.append(f'    %evd{ci} = arith.extf %ev{ci} : {elem} to f64')
            ev = f'%evd{ci}'
        else:
            ev = f'%ev{ci}'
        tl.append(f'    %acc{ci} = memref.load %CK[%c0] : memref<1xf64>')
        tl.append(f'    %nacc{ci} = arith.addf %acc{ci}, {ev} : f64')
        tl.append(f'    memref.store %nacc{ci}, %CK[%c0] : memref<1xf64>')
        for _ in shape:
            tl.append('  }')
    tl.append('  %CKU = memref.cast %CK : memref<1xf64> to memref<*xf64>')
    tl.append('  func.call @printMemrefF64(%CKU) : (memref<*xf64>) -> ()')
    tl.append('  return')
    tl.append('}')
    # checksum dim constants
    ckdecl = []
    for ci,(cssa,ctype,shape) in enumerate(allocs):
        for d,dim in enumerate(shape):
            ckdecl.append(f'  %ckd{ci}_{d} = arith.constant {dim} : index')
    tl = ['\n'.join(ckdecl)] + tl
    out_text = out_text.rstrip()
    # the @main currently ends with the last arg block; append timing before final brace
    # remove the trailing 'func.func @main() {' artifacts — instead we built body inline; just append
    print(out_text)
    print('\n'.join(tl))

main()
