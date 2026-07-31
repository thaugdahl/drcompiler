#!/usr/bin/env python3
"""Ground-truth distinct-byte / distinct-cache-line footprints for the
falcon-spike corpus, computed by direct enumeration of the access sets.

Compare against `dr-test-reuse-analysis`'s `footprint=` field, which sums a
per-reference bounding box over references instead of taking their union.

Enumeration is over *cache lines*, not elements, so the numbers are directly
comparable to a compulsory-miss count from an analytical cache model.
"""

LINE = 64
ELEM = 8  # f64


def lines_of(base_id, index_fn, ranges, row_stride):
    """Distinct (array, line) pairs touched by one reference.

    index_fn maps a point in `ranges` to a flat element offset.
    """
    out = set()
    for pt in iter_points(ranges):
        off = index_fn(*pt) * ELEM
        out.add((base_id, off // LINE))
    return out


def iter_points(ranges):
    if not ranges:
        yield ()
        return
    head, *tail = ranges
    for v in head:
        for rest in iter_points(tail):
            yield (v,) + rest


def report(name, refs, model_bytes):
    """refs: list of sets of (array, line)."""
    union = set().union(*refs)
    summed = sum(len(r) for r in refs)
    true_b = len(union) * LINE
    print(
        f"{name:<12} model={model_bytes/1e6:8.2f} MB  "
        f"true={true_b/1e6:8.2f} MB  over={model_bytes/true_b:5.2f}x  "
        f"(sum-of-refs={summed*LINE/1e6:.2f} MB)"
    )
    return true_b


# ---------------------------------------------------------------- jacobi-2d --
# A, B: 1300x1300 f64.  i,j in [1,1299).
N = 1300
rng = range(1, 1299)
A_offsets = [(0, 0), (0, -1), (0, 1), (1, 0), (-1, 0)]
jac_refs = [
    {("A", ((i + di) * N + (j + dj)) * ELEM // LINE) for i in rng for j in rng}
    for di, dj in A_offsets
]
jac_refs.append({("B", (i * N + j) * ELEM // LINE) for i in rng for j in rng})
report("jacobi-2d", jac_refs, 80870592)

# ---------------------------------------------------------------- seidel-2d --
# A: 1200x1200 f64, in-place 9-point.  i,j in [1,1199).
N = 1200
rng = range(1, 1199)
S_offsets = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)]
sei_refs = [
    {("A", ((i + di) * N + (j + dj)) * ELEM // LINE) for i in rng for j in rng}
    for di, dj in S_offsets
]
report("seidel-2d", sei_refs, 103334688)

# ------------------------------------------------------------------ heat-3d --
# A, B: 120x120x120 f64.  i,j,k in [1,119).
N = 120
rng = range(1, 119)
H_offsets = [(0, 0, 0), (-1, 0, 0), (1, 0, 0), (0, -1, 0),
             (0, 1, 0), (0, 0, -1), (0, 0, 1)]
heat_refs = [
    {
        ("A", ((i + di) * N * N + (j + dj) * N + (k + dk)) * ELEM // LINE)
        for i in rng for j in rng for k in rng
    }
    for di, dj, dk in H_offsets
]
heat_refs.append(
    {("B", (i * N * N + j * N + k) * ELEM // LINE)
     for i in rng for j in rng for k in rng}
)
report("heat-3d", heat_refs, 105154048)

# --------------------------------------------------------------------- trmm --
# A: 1000x1000, B: 1000x1200.  k in (i, 1000), j in [0,1200).
M, NN = 1000, 1200
trmm_A = {("A", (k * M + i) * ELEM // LINE)
          for i in range(M) for k in range(i + 1, M)}
trmm_Bk = {("B", (k * NN + j) * ELEM // LINE)
           for i in range(M) for k in range(i + 1, M) for j in (0, NN - 1)}
# B[k][j] and B[i][j] both sweep all of B for the j range; enumerate rows only.
trmm_Bk = {("B", (k * NN + j) * ELEM // LINE)
           for k in range(1, M) for j in range(NN)}
trmm_Bi = {("B", (i * NN + j) * ELEM // LINE)
           for i in range(M - 1) for j in range(NN)}
report("trmm", [trmm_A, trmm_Bk, trmm_Bi], 27200000)

# --------------------------------------------------------------- covariance --
# data: 1400x1200, cov: 1200x1200.  i in [0,1200), j in [i,1200), k in [0,1400)
M, NN = 1200, 1400
cov_di = {("d", (k * M + i) * ELEM // LINE)
          for i in range(M) for k in range(NN)}
cov_dj = {("d", (k * M + j) * ELEM // LINE)
          for j in range(M) for k in range(NN)}
cov_c = {("c", (i * M + j) * ELEM // LINE)
         for i in range(M) for j in range(i, M)}
report("covariance", [cov_di, cov_dj, cov_c], 38400000)

# --------------------------------------------------------------- controls ----
# gemm: C 1000x1100, A 1000x1200, B 1200x1100 -- disjoint arrays, no overlap.
g_C = {("C", x) for x in range(1000 * 1100 * ELEM // LINE)}
g_A = {("A", x) for x in range(1000 * 1200 * ELEM // LINE)}
g_B = {("B", x) for x in range(1200 * 1100 * ELEM // LINE)}
report("gemm", [g_C, g_A, g_B], 28960000)
