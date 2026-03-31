#!/usr/bin/env python3
"""Rewrite memref types with LLVM dialect element types to !llvm.ptr.

cgeist (Polygeist, LLVM 18) represents C pointers as memref types with
LLVM dialect element types that LLVM 22 MLIR forbids:
  - memref<?x!llvm.struct<"name", (fields...)>>  (struct pointers)
  - memref<?x!llvm.func<ret (args...)>>          (function pointers)
  - memref<?x!llvm.ptr>                          (void*/opaque pointers)

Since these memrefs are always immediately cast to !llvm.ptr via
polygeist.memref2pointer before any access, replacing the memref type
with !llvm.ptr is semantically correct.

Static-dimension cases (memref<1x!llvm.struct<...>>) from stack-allocated
structs (memref.alloca) are rewritten from memref.alloca to llvm.alloca
so that the type is preserved at the LLVM dialect level.

Usage: rewrite-struct-memrefs.py <file.mlir>   (rewrites in-place)
       rewrite-struct-memrefs.py < in.mlir      (stdout)
"""

import re
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


def _find_memref_llvm_type(text, start):
    """Find the next memref<...x!llvm.XXX> starting at *start*.

    Matches any LLVM dialect element type: !llvm.struct<...>, !llvm.func<...>,
    !llvm.ptr, etc.

    Returns (begin, end, dims_prefix, llvm_body) or None.
    - begin/end are indices into text covering the full "memref<...>" span.
    - dims_prefix is the dimension text before !llvm (e.g., "1x", "?x").
    - llvm_body is the full LLVM type (e.g., "!llvm.struct<...>", "!llvm.ptr").
    """
    idx = text.find("memref<", start)
    if idx == -1:
        return None

    open_pos = idx + 6  # index of the '<'
    close_pos = _find_matching_close(text, open_pos)
    if close_pos == -1:
        return None

    inner = text[open_pos + 1 : close_pos]
    llvm_tag = "!llvm."
    spos = inner.find(llvm_tag)
    if spos == -1:
        return None

    prefix = inner[:spos]
    if not all(c in "0123456789?x " for c in prefix):
        return None

    llvm_body = inner[spos:]
    return (idx, close_pos + 1, prefix.strip(), llvm_body)


def rewrite_llvm_alloca(text):
    """Rewrite memref.alloca() with LLVM dialect element types to llvm.alloca.

    Pattern:
        %name = memref.alloca() : memref<Nx!llvm.TYPE>
    Becomes:
        %name__saN = llvm.mlir.constant(N : i64) : i64
        %name = llvm.alloca %name__saN x !llvm.TYPE : (i64) -> !llvm.ptr

    This must run BEFORE the generic memref-to-ptr rewrite so that
    the memref.alloca op is converted rather than just having its type
    replaced (which would be invalid — memref.alloca needs a memref type).
    """
    # Match:  %ssa_name = memref.alloca() : memref<dims x !llvm.XXX>
    # We process line-by-line since alloca is always a single statement.
    lines = text.split("\n")
    out_lines = []
    # Counter for generating unique SSA names when the alloca name is reused
    sa_counter = 0

    for line in lines:
        # Quick check before expensive regex
        if "memref.alloca()" not in line or "!llvm." not in line:
            out_lines.append(line)
            continue

        # Extract the SSA name and the memref type
        # Pattern: <indent>%<name> = memref.alloca() : memref<...>
        m = re.match(
            r"^(\s*)(%\S+)\s*=\s*memref\.alloca\(\)\s*:\s*(memref<.*)",
            line,
        )
        if not m:
            out_lines.append(line)
            continue

        indent = m.group(1)
        ssa_name = m.group(2)
        memref_text = m.group(3)

        # Parse the memref type to extract the LLVM type body
        info = _find_memref_llvm_type(memref_text, 0)
        if info is None:
            out_lines.append(line)
            continue

        _, _, dims_prefix, llvm_body = info

        # Extract the static count from dims_prefix (e.g., "1x" -> 1)
        dims_prefix = dims_prefix.rstrip("x").strip()
        if not dims_prefix:
            dims_prefix = "1"
        try:
            count = int(dims_prefix)
        except ValueError:
            # Dynamic dimension — shouldn't happen with alloca, skip
            out_lines.append(line)
            continue

        # Generate unique SSA name for the constant
        # Strip the leading % from ssa_name to create the constant name
        base_name = ssa_name.lstrip("%")
        const_name = f"%{base_name}__sa{sa_counter}"
        sa_counter += 1

        # Emit:
        #   %name__saN = llvm.mlir.constant(count : i64) : i64
        #   %name = llvm.alloca %name__saN x !llvm.TYPE : (i64) -> !llvm.ptr
        out_lines.append(
            f'{indent}{const_name} = llvm.mlir.constant({count} : i64) : i64'
        )
        out_lines.append(
            f'{indent}{ssa_name} = llvm.alloca {const_name} x {llvm_body}'
            f' : (i64) -> !llvm.ptr'
        )
        continue

    return "\n".join(out_lines)


def rewrite_llvm_dealloc(text):
    """Rewrite memref.dealloc with LLVM-typed memrefs to call @free.

    Pattern:
        memref.dealloc %name : memref<...x!llvm.TYPE>
    Becomes:
        call @free(%name) : (!llvm.ptr) -> ()

    Also injects ``func.func private @free(!llvm.ptr)`` at the module level
    if any deallocs were rewritten.

    Must run BEFORE the generic memref-to-ptr rewrite (same reason as alloca).
    """
    lines = text.split("\n")
    out_lines = []
    rewrote_any = False

    for line in lines:
        if "memref.dealloc" not in line or "!llvm." not in line:
            out_lines.append(line)
            continue

        m = re.match(
            r"^(\s*)memref\.dealloc\s+(%\S+)\s*:\s*(memref<.*)",
            line,
        )
        if not m:
            out_lines.append(line)
            continue

        indent = m.group(1)
        ssa_name = m.group(2)
        memref_text = m.group(3)

        info = _find_memref_llvm_type(memref_text, 0)
        if info is None:
            out_lines.append(line)
            continue

        rewrote_any = True
        out_lines.append(
            f'{indent}func.call @free({ssa_name}) : (!llvm.ptr) -> ()'
        )
        continue

    result = "\n".join(out_lines)

    # Inject @free declaration at the module level if we rewrote any deallocs
    # and @free isn't already declared.
    if rewrote_any and "@free" not in text:
        # Insert before the first func.func in the module body.
        idx = result.find("func.func ")
        if idx != -1:
            result = (
                result[:idx]
                + "func.func private @free(!llvm.ptr)\n  "
                + result[idx:]
            )

    return result


def rewrite_llvm_memrefs(text):
    """Replace every ``memref<...x!llvm.XXX>`` with ``!llvm.ptr``.

    Handles any LLVM dialect element type (!llvm.struct, !llvm.func, !llvm.ptr)
    with both dynamic (``memref<?x...>``) and static (``memref<1x...>``)
    dimensions.  Static-dimension memref.alloca ops must be handled by
    rewrite_llvm_alloca() BEFORE this function runs.
    """
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
            # Unbalanced -- emit verbatim and advance past "memref<"
            out.append(text[i : idx + 7])
            i = idx + 7
            continue

        inner = text[open_pos + 1 : close_pos]

        # Rewrite if the element type is any !llvm.* type, regardless of
        # whether the leading dimension is dynamic (?) or static (N).
        # The alloca case has already been handled by rewrite_llvm_alloca().
        llvm_tag = "!llvm."
        spos = inner.find(llvm_tag)
        if spos != -1:
            prefix = inner[:spos]
            if all(c in "0123456789?x " for c in prefix):
                out.append(text[i:idx])
                out.append("!llvm.ptr")
                i = close_pos + 1
                continue

        # Not an LLVM-typed memref -- emit "memref<" and keep scanning inside.
        out.append(text[i : idx + 7])
        i = idx + 7

    return "".join(out)


def rewrite(text):
    """Apply all LLVM-memref rewrites in the correct order."""
    # Phase 1: Rewrite memref.alloca() with LLVM types to llvm.alloca.
    # Must happen first so the alloca op is converted before Phase 3
    # replaces the memref type with !llvm.ptr (which would leave a broken
    # memref.alloca() : !llvm.ptr).
    text = rewrite_llvm_alloca(text)
    # Phase 2: Rewrite memref.dealloc with LLVM types to llvm.call @free.
    # Must also happen before Phase 3 for the same reason.
    text = rewrite_llvm_dealloc(text)
    # Phase 3: Replace all remaining memref<...x!llvm.XXX> with !llvm.ptr.
    # Run to fixpoint: nested types like memref<?x memref<?x !llvm.struct<...>>>
    # require multiple passes — inner layers get replaced first, producing
    # memref<?x !llvm.ptr> which is caught on the next pass.
    for _ in range(10):  # generous bound; typically 2 passes suffice
        prev = text
        text = rewrite_llvm_memrefs(text)
        if text == prev:
            break
    return text


def main():
    if len(sys.argv) == 2:
        path = sys.argv[1]
        with open(path, "r") as f:
            text = f.read()
        result = rewrite(text)
        with open(path, "w") as f:
            f.write(result)
    elif not sys.stdin.isatty():
        sys.stdout.write(rewrite(sys.stdin.read()))
    else:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
