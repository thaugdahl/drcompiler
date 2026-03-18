// sqrt_consumers.c — 3 loops with duplicated sqrt+div chain
//
// cgeist -O2 preserves the separate loops with the expensive computation
// inline in each.  memory-fission detects the duplicated fingerprint and
// materializes a shared buffer, computing sqrt/div only once.
//
// N=4000 keeps the fission buffer (32KB) in L1 cache so loads are cheap.
// The REPS loop is inside kernel() so fission applies to the hot path
// (fission only anchors on function-argument memrefs, not local allocs).
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void kernel(double *restrict x, double scale,
            double *restrict a, double *restrict b, double *restrict c,
            int n, int reps) {
  for (int r = 0; r < reps; r++) {
    for (int i = 0; i < n; i++)
      a[i] += sqrt(x[i]) / scale + 1.0;

    for (int i = 0; i < n; i++)
      b[i] += sqrt(x[i]) / scale * 2.5;

    for (int i = 0; i < n; i++)
      c[i] += sqrt(x[i]) / scale - 0.5;
  }
}

int main(int argc, char **argv) {
  int n = 4000;
  int reps = 100000;
  if (argc > 1) n = atoi(argv[1]);
  if (argc > 2) reps = atoi(argv[2]);

  double *x = (double *)malloc(n * sizeof(double));
  double *a = (double *)calloc(n, sizeof(double));
  double *b = (double *)calloc(n, sizeof(double));
  double *c = (double *)calloc(n, sizeof(double));

  srand(42);
  for (int i = 0; i < n; i++)
    x[i] = (double)(rand() % 10000 + 1);

  kernel(x, 3.14, a, b, c, n, reps);

  printf("a[0]=%.6f b[0]=%.6f c[0]=%.6f\n", a[0], b[0], c[0]);

  free(x); free(a); free(b); free(c);
  return 0;
}
