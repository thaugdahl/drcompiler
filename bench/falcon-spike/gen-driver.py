#!/usr/bin/env python3
"""Synthesize an MLIR @drv_main driver for a bare PolyBench affine kernel.

The kernels in bench/falcon-spike/results-shapes/ are Polygeist output: one
`func.func @kernel_<name>(...)` carrying `llvm.linkage<external>` and no main,
so they cannot be executed directly. This appends a driver into the same module:

  * memref.alloc every memref argument (dynamic dims -> --max-dim, see below)
  * re-initialize every memref with a deterministic pattern before each rep,
    OUTSIDE the timed region, so repeated calls stay numerically bounded
  * call the kernel `reps` times, bracketing each call with @rtclock
  * print the per-rep times, then print a checksum over every memref argument
    so the JIT cannot dead-code-eliminate the kernel call

Dynamic memref dimensions are NOT recoverable from the signature (PolyBench
covariance passes both data[1400][1200] and cov[1200][1200] as
memref<?x1200xf64>), so every `?` is allocated at the largest constant loop
bound found in the function. Over-allocating is safe and timing-faithful: the
minor (row-stride) dimensions are static, so the access pattern and cache
behaviour over the region the kernel actually touches are unchanged -- only the
number of allocated rows grows.
"""
import argparse
import re
import sys

_CTR = [0]


def fresh(prefix="t"):
    _CTR[0] += 1
    return f"%{prefix}{_CTR[0]}"


def split_top(s):
    """Split on commas not nested inside <>, (), [] or {}."""
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "<([{":
            depth += 1
        elif ch in ">)]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def match_paren(text, open_idx):
    depth = 0
    i = open_idx
    while i < len(text):
        if text[i] in "<([{":
            depth += 1
        elif text[i] in ">)]}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def find_kernel(text, name=None):
    """Return (name, arg-list string, return type or None)."""
    cands = []
    for m in re.finditer(r"func\.func\s+(?:private\s+)?@([A-Za-z_][\w.$]*)\s*\(", text):
        fname = m.group(1)
        close = match_paren(text, m.end() - 1)
        if close == -1:
            continue
        args = text[m.end():close]
        nl = text.find("\n", close)
        rest = text[close + 1:nl if nl != -1 else len(text)]
        is_decl = "{" not in rest
        rm = re.search(r"->\s*([^\s{,]+)", rest)
        ret = rm.group(1) if rm else None
        cands.append((fname, args, ret, is_decl))

    if name:
        for fname, args, ret, _ in cands:
            if fname == name:
                return fname, args, ret
        sys.exit(f"gen-driver: function @{name} not found")

    real = [c for c in cands if not c[3] and c[0] not in ("main", "drv_main")]
    if not real:
        sys.exit("gen-driver: no candidate kernel function found")
    for fname, args, ret, _ in real:
        if fname.startswith("kernel"):
            return fname, args, ret
    return real[0][0], real[0][1], real[0][2]


class MemRef:
    def __init__(self, ty):
        self.ty = ty
        inner = ty[len("memref<"):-1]
        if "," in inner:
            sys.exit(f"gen-driver: memref with layout/memspace unsupported: {ty}")
        parts = inner.split("x")
        self.elem = parts[-1]
        self.dims = parts[:-1]
        self.rank = len(self.dims)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mlir")
    ap.add_argument("--kernel", default=None)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--max-dim", type=int, default=0,
                    help="extent used for '?' dims (default: max constant loop bound)")
    a = ap.parse_args()

    text = open(a.mlir).read()
    kname, argstr, kret = find_kernel(text, a.kernel)

    argtys = []
    for piece in split_top(argstr):
        ty = piece.split(":", 1)[1].strip() if ":" in piece else piece.strip()
        argtys.append(ty)

    # ---- infer extent for dynamic dims ------------------------------------
    maxdim = a.max_dim
    if maxdim <= 0:
        bounds = [int(x) for x in re.findall(r"to\s+(\d+)\b", text)]
        for mm in re.findall(r"affine_map<[^>]*>", text):
            bounds += [int(x) for x in re.findall(r"\b(\d{2,})\b", mm)]
        for ty in argtys:
            if ty.startswith("memref<"):
                bounds += [int(d) for d in MemRef(ty).dims if d.isdigit()]
        maxdim = max(bounds) if bounds else 0
    if maxdim <= 0:
        sys.exit("gen-driver: could not infer a size for dynamic dims; pass --max-dim")

    reps = a.reps
    head, allocs, inits, checks, body = [], [], [], [], []
    names, calltys, memrefs = [], [], []

    head.append("  func.func private @printMemrefF64(memref<*xf64>)")
    head.append("  func.func private @rtclock() -> f64")
    head.append("  func.func @drv_main() {")
    head.append("    %c0 = arith.constant 0 : index")
    head.append("    %c1 = arith.constant 1 : index")
    head.append(f"    %cD = arith.constant {maxdim} : index")
    head.append("    %cMOD = arith.constant 97 : index")
    head.append("    %fMOD = arith.constant 9.700000e+01 : f64")
    head.append("    %fzero = arith.constant 0.000000e+00 : f64")

    for i, ty in enumerate(argtys):
        nm = f"%a{i}"
        names.append(nm)
        calltys.append(ty)
        if ty.startswith("memref<"):
            mr = MemRef(ty)
            ndyn = sum(1 for d in mr.dims if d == "?")
            dynargs = "(" + ", ".join(["%cD"] * ndyn) + ")" if ndyn else "()"
            allocs.append(f"    {nm} = memref.alloc{dynargs} : {ty}")
            memrefs.append((nm, mr))
        elif ty in ("i32", "i64"):
            allocs.append(f"    {nm} = arith.constant {maxdim} : {ty}")
        elif ty in ("f32", "f64"):
            allocs.append(f"    {nm} = arith.constant {maxdim}.0 : {ty}")
        elif ty == "index":
            allocs.append(f"    {nm} = arith.constant {maxdim} : index")
        else:
            sys.exit(f"gen-driver: unsupported argument type {ty!r}")

    def loops(mr, nm, body_fn, base_indent="    "):
        """Nested scf.for over every dim; body_fn(ind, idxs, lin) -> lines."""
        out, idxs, ind = [], [], base_indent
        for d, dim in enumerate(mr.dims):
            iv = fresh("i")
            if dim == "?":
                ext = "%cD"
            else:
                ext = fresh("d")
                out.append(f"{ind}{ext} = arith.constant {dim} : index")
            out.append(f"{ind}scf.for {iv} = %c0 to {ext} step %c1 {{")
            idxs.append((iv, ext))
            ind += "  "
        # linear index = ((i0*D1)+i1)*D2 + i2 ...
        if idxs:
            lin = idxs[0][0]
            for d in range(1, len(idxs)):
                mul = fresh("m")
                out.append(f"{ind}{mul} = arith.muli {lin}, {idxs[d][1]} : index")
                add = fresh("p")
                out.append(f"{ind}{add} = arith.addi {mul}, {idxs[d][0]} : index")
                lin = add
        else:
            lin = "%c0"
        out += body_fn(ind, [iv for iv, _ in idxs], lin)
        for _ in mr.dims:
            ind = ind[:-2]
            out.append(f"{ind}}}")
        return out

    # ---- init: value = ((linear index) mod 97) / 97.0 ---------------------
    for nm, mr in memrefs:
        def init_body(ind, idxs, lin, nm=nm, mr=mr):
            b = []
            idx = "[" + ", ".join(idxs) + "]" if idxs else "[]"
            r = fresh("r")
            b.append(f"{ind}{r} = arith.remui {lin}, %cMOD : index")
            ci = fresh("ci")
            b.append(f"{ind}{ci} = arith.index_cast {r} : index to i64")
            if mr.elem in ("f32", "f64"):
                fv = fresh("f")
                b.append(f"{ind}{fv} = arith.sitofp {ci} : i64 to f64")
                dv = fresh("q")
                b.append(f"{ind}{dv} = arith.divf {fv}, %fMOD : f64")
                v = dv
                if mr.elem == "f32":
                    tv = fresh("tr")
                    b.append(f"{ind}{tv} = arith.truncf {v} : f64 to f32")
                    v = tv
            elif mr.elem == "i32":
                v = fresh("w")
                b.append(f"{ind}{v} = arith.trunci {ci} : i64 to i32")
            elif mr.elem == "i64":
                v = ci
            else:
                sys.exit(f"gen-driver: unsupported element type {mr.elem!r}")
            b.append(f"{ind}memref.store {v}, {nm}{idx} : {mr.ty}")
            return b
        inits += loops(mr, nm, init_body, "      ")

    # ---- checksum over every memref argument ------------------------------
    for nm, mr in memrefs:
        def chk_body(ind, idxs, lin, nm=nm, mr=mr):
            b = []
            idx = "[" + ", ".join(idxs) + "]" if idxs else "[]"
            e = fresh("e")
            b.append(f"{ind}{e} = memref.load {nm}{idx} : {mr.ty}")
            v = e
            if mr.elem == "f32":
                x = fresh("x")
                b.append(f"{ind}{x} = arith.extf {v} : f32 to f64")
                v = x
            elif mr.elem in ("i32", "i64"):
                src = v
                if mr.elem == "i32":
                    w = fresh("sx")
                    b.append(f"{ind}{w} = arith.extsi {src} : i32 to i64")
                    src = w
                x = fresh("x")
                b.append(f"{ind}{x} = arith.sitofp {src} : i64 to f64")
                v = x
            ac = fresh("ac")
            b.append(f"{ind}{ac} = memref.load %CK[%c0] : memref<1xf64>")
            na = fresh("na")
            b.append(f"{ind}{na} = arith.addf {ac}, {v} : f64")
            b.append(f"{ind}memref.store {na}, %CK[%c0] : memref<1xf64>")
            return b
        checks += loops(mr, nm, chk_body, "    ")

    body += allocs
    body.append(f"    %T = memref.alloc() : memref<{reps}xf64>")
    body.append(f"    %cR = arith.constant {reps} : index")
    body.append("    scf.for %rep = %c0 to %cR step %c1 {")
    body += inits
    body.append("      %t0 = func.call @rtclock() : () -> f64")
    ret_ty = kret if kret else "()"
    lhs = f"{fresh('kr')} = " if kret else ""
    body.append(f"      {lhs}func.call @{kname}({', '.join(names)}) : "
                f"({', '.join(calltys)}) -> {ret_ty}")
    body.append("      %t1 = func.call @rtclock() : () -> f64")
    body.append("      %dt = arith.subf %t1, %t0 : f64")
    body.append(f"      memref.store %dt, %T[%rep] : memref<{reps}xf64>")
    body.append("    }")
    body.append("    %CK = memref.alloc() : memref<1xf64>")
    body.append("    memref.store %fzero, %CK[%c0] : memref<1xf64>")
    body += checks
    body.append(f"    %TU = memref.cast %T : memref<{reps}xf64> to memref<*xf64>")
    body.append("    func.call @printMemrefF64(%TU) : (memref<*xf64>) -> ()")
    body.append("    %CKU = memref.cast %CK : memref<1xf64> to memref<*xf64>")
    body.append("    func.call @printMemrefF64(%CKU) : (memref<*xf64>) -> ()")
    body.append("    return")
    body.append("  }")

    driver = "\n".join(head + body) + "\n"

    stripped = text.rstrip()
    idx = stripped.rfind("\n}")
    if idx == -1:
        sys.exit("gen-driver: could not locate module closing brace")
    sys.stdout.write(stripped[:idx + 1] + driver + "}\n")


if __name__ == "__main__":
    main()
