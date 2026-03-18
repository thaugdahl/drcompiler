// div_chain.c — 4 loops with duplicated sqrt+div chain (single source memref)
//
// sqrt+div costs ~35 ALU units.  With 4 consumers:
//   recompute = 4*35 = 140  vs  keep = 35 + 1 + 4*4 = 52  →  fission wins.
//
// N=4000 keeps the fission buffer in L1 cache.
// REPS loop is inside kernel() so fission applies to the hot path.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void kernel(double *restrict x, double scale,
            double *restrict a, double *restrict b,
            double *restrict c, double *restrict d,
            int n, int reps) {
  for (int r = 0; r < reps; r++) {
    for (int i = 0; i < n; i++)
      a[i] += sqrt(x[i]) / scale + 1.0;

    for (int i = 0; i < n; i++)
      b[i] += sqrt(x[i]) / scale - 1.0;

    for (int i = 0; i < n; i++)
      c[i] += sqrt(x[i]) / scale * 3.0;

    for (int i = 0; i < n; i++)
      d[i] += sqrt(x[i]) / scale + sqrt(x[i]) / scale;
  }
}

int main(int argc, char **argv) {
  int n = 4000;
  int reps = 60000;
  if (argc > 1) n = atoi(argv[1]);
  if (argc > 2) reps = atoi(argv[2]);

  double *x = (double *)malloc(n * sizeof(double));
  double *a = (double *)calloc(n, sizeof(double));
  double *b = (double *)calloc(n, sizeof(double));
  double *c = (double *)calloc(n, sizeof(double));
  double *d = (double *)calloc(n, sizeof(double));

  srand(42);
  for (int i = 0; i < n; i++)
    x[i] = (double)(rand() % 10000 + 1);

  kernel(x, 3.14, a, b, c, d, n, reps);

  printf("a[0]=%.6f b[0]=%.6f c[0]=%.6f d[0]=%.6f\n", a[0], b[0], c[0], d[0]);

  free(x); free(a); free(b); free(c); free(d);
  return 0;
}
