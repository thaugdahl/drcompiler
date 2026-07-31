#!/usr/bin/env python3
"""Rewrite onnx-mlir affine output into Falcon-lazystack-parseable MLIR.

  "krnl.global"() <{name = "N", shape=[...], value = dense<...> : tensor<T>}> : () -> memref<T>
     ->  module-level:  memref.global "private" constant @N : memref<T> = dense_resource<__elided__>
         in-body:       %x = memref.get_global @N : memref<T>
  drop "krnl.entry_point"()
"""
import re, sys

src = open(sys.argv[1]).read()
globals_decls = []
seen = set()

def find_matching(s, i, open_ch, close_ch):
    """i points at open_ch; return index just past matching close."""
    depth = 0
    while i < len(s):
        c = s[i]
        if c == '"':                      # skip string literal
            i += 1
            while i < len(s) and s[i] != '"':
                i += 2 if s[i] == '\\' else 1
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unbalanced")

out = []
pos = 0
pat = re.compile(r'(%\w+) = "krnl\.global"\(\)\s*<')
for m in pat.finditer(src):
    if m.start() < pos:
        continue
    out.append(src[pos:m.start()])
    res = m.group(1)
    brk = find_matching(src, m.end() - 1, '<', '>')       # the <{ ... }> blob
    body = src[m.end():brk - 1]                           # inside <...>
    nm = re.search(r'name\s*=\s*"([^"]+)"', body).group(1)
    tailm = re.compile(r'\s*:\s*\(\)\s*->\s*(memref<[^>]+>)').match(src, brk)
    ty = tailm.group(1)
    if nm not in seen:
        seen.add(nm)
        globals_decls.append(
            f'  memref.global "private" constant @{nm} : {ty} = '
            f'dense_resource<__elided__> {{alignment = 64 : i64}}')
    indent = ' ' * (m.start() - src.rfind('\n', 0, m.start()) - 1)
    out.append(f'{res} = memref.get_global @{nm} : {ty}')
    pos = tailm.end()
out.append(src[pos:])
txt = ''.join(out)

# drop krnl.entry_point (generic form, attr-dict in {} then : () -> ())
txt = re.sub(r'^\s*"krnl\.entry_point"\(\)\s*\{.*?\}\s*:\s*\(\)\s*->\s*\(\)\s*$',
             '', txt, flags=re.M | re.S)

# inject globals right after the top-level `module ... {`
mm = re.search(r'^module[^\n]*\{\s*$', txt, flags=re.M)
ins = mm.end()
txt = txt[:ins] + '\n' + '\n'.join(globals_decls) + txt[ins:]
open(sys.argv[2], 'w').write(txt)
print(f"rewrote {len(seen)} krnl.global -> memref.global; wrote {sys.argv[2]}")
