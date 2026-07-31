#!/usr/bin/env python3
"""Ground-truth reuse distance per band loop, in distinct cache lines.

`BandReuseInfo::loopCarriesEvictedReuse(l, cacheBytes)` is the tiler's
profitability gate.  It compares `reuseDistanceBytes(l)` -- the sum over
references of elemBytes x product-of-extents -- against the cache size.

That sum is in ELEMENTS x elemBytes, with no cache-line rounding, so a
column-strided reference (one f64 per 64 B line) is under-counted 8x.  This
script recomputes the same quantity by enumerating distinct lines touched in
one iteration of loop `l` (outer loops fixed, inner loops full) and reports
where the verdict flips.
"""

LINE = 64
ELEM = 8
L1 = 32768


def verdict(dist, cache=L1):
    return "EVICTED" if dist > cache else "fits"


def show(kernel, loop, model_bytes, true_lines, cache=L1):
    true_bytes = true_lines * LINE
    vm, vt = verdict(model_bytes, cache), verdict(true_bytes, cache)
    flag = "  <-- FLIP" if vm != vt else ""
    print(
        f"{kernel:<12} loop{loop}  model={model_bytes:>10,} B ({vm:>7})   "
        f"true={true_bytes:>10,} B ({vt:>7}){flag}"
    )


print(f"reuse distance per loop, verdict at cacheBytes={L1}\n")

# --- gemm: C 1000x1100, A 1000x1200, B 1200x1100.  band [i, k, j] ------------
# One k-iteration: A[i][k] 1 elem; B[k][j] row of 1100 contiguous f64;
# C[i][j] row of 1100 contiguous f64.  All row-major => lines ~= bytes/64.
g_lines = 1 + (1100 * ELEM + LINE - 1) // LINE * 2
show("gemm", 1, 8 + 8800 + 8800, g_lines)

# --- trmm: A 1000x1000, B 1000x1200.  band [i, j, k] ------------------------
# One j-iteration (i fixed, k full over (i,1000)): A[k][i] is a COLUMN walk --
# 999 distinct rows, each a separate line.  B[k][j] likewise one element per
# row.  B[i][j] is a single element.
t_lines = 999 + 999 + 1
show("trmm", 1, 8000 + 8000 + 8, t_lines)

# --- covariance: data 1400x1200, cov 1200x1200.  band [i, j, k] -------------
# One j-iteration (i fixed, k full 0..1400): data[k][i] and data[k][j] are both
# column walks of 1400 elements, one line each; cov[i][j] one element.
c_lines = 1400 + 1400 + 1
show("covariance", 1, 11200 + 11200 + 8, c_lines)

print()
print("gemm is row-major throughout, so element-granular and line-granular")
print("footprints agree and the verdict matches.  trmm and covariance reuse")
print("column-strided references; the model under-counts them 8x and reports")
print("'fits L1' for a window that really overflows it by 4-5x.")
