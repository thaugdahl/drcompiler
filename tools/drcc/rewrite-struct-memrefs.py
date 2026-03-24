#!/usr/bin/env python3
"""Rewrite memref types with !llvm.struct element types to !llvm.ptr.

cgeist (Polygeist, LLVM 18) represents C struct pointers as
memref<?x!llvm.struct<"name", (fields...)>>, but LLVM 22 MLIR forbids
!llvm.struct as a memref element type.  Since struct-typed memrefs are always
immediately cast to !llvm.ptr via polygeist.memref2pointer before any access,
replacing the memref type with !llvm.ptr is semantically correct.

Usage: rewrite-struct-memrefs.py <file.mlir>   (rewrites in-place)
       rewrite-struct-memrefs.py < in.mlir      (stdout)
"""

import sys


def _find_matching_close(text, pos):
    """Given *pos* pointing at '<', return index of the matching '>'.
    Returns -1 if the brackets are unbalanced."""
    depth = 1
    i = pos + 1
    while i < len(text):
        c = text[i]
        if c == "<":
            depth += 1
        elif c == ">":
            depth -= 1
            if depth == 0:
                return i
        elif c == '"':
            # Skip quoted struct names like "struct.node_t"
            i += 1
            while i < len(text) and text[i] != '"':
                i += 1
        i += 1
    return -1


def rewrite_struct_memrefs(text):
    """Replace every ``memref<...x!llvm.struct<...>>`` with ``!llvm.ptr``."""
    out = []
    i = 0

    while i < len(text):
        idx = text.find("memref<", i)
        if idx == -1:
            out.append(text[i:])
            break

        # Find the balanced close of this memref<...>
        open_pos = idx + 6          # index of the '<'
        close_pos = _find_matching_close(text, open_pos)

        if close_pos == -1:
            # Unbalanced — emit verbatim and advance past "memref<"
            out.append(text[i : idx + 7])
            i = idx + 7
            continue

        inner = text[open_pos + 1 : close_pos]

        # Only rewrite if the element type is !llvm.struct<...> AND the
        # leading dimension is dynamic (?).  Static dimensions like
        # memref<1x!llvm.struct<...>> are stack allocations (memref.alloca)
        # and must stay as memref types.
        struct_tag = "!llvm.struct<"
        spos = inner.find(struct_tag)
        if spos != -1:
            prefix = inner[:spos]
            if all(c in "0123456789?x " for c in prefix) and "?" in prefix:
                out.append(text[i:idx])
                out.append("!llvm.ptr")
                i = close_pos + 1
                continue

        # Not a struct memref — emit "memref<" and keep scanning inside.
        out.append(text[i : idx + 7])
        i = idx + 7

    return "".join(out)


def main():
    if len(sys.argv) == 2:
        path = sys.argv[1]
        with open(path, "r") as f:
            text = f.read()
        result = rewrite_struct_memrefs(text)
        with open(path, "w") as f:
            f.write(result)
    elif not sys.stdin.isatty():
        sys.stdout.write(rewrite_struct_memrefs(sys.stdin.read()))
    else:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
