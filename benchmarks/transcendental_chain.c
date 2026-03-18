// transcendental_chain.c — 2 loops with duplicated exp*log chain
//
// The exp+log chain costs ~40 ALU units.  With 2 consumers:
//   recompute = 2*40 = 80  vs  keep = 40 + 1 + 2*4 = 49  →  fission wins.
//
// N=4000 keeps the fission buffer in L1 cache.
// REPS loop is inside kernel() so fission applies to the hot path.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void kernel(double *restrict x, double *restrict y,
            double *restrict out1, double *restrict out2,
            int n, int reps) {
  for (int r = 0; r < reps; r++) {
    for (int i = 0; i < n; i++)
      out1[i] += exp(x[i]) * log(y[i]) + 0.5;

    for (int i = 0; i < n; i++)
      out2[i] += exp(x[i]) * log(y[i]) * -0.1;
  }
}

int main(int argc, char **argv) {
  int n = 4000;
  int reps = 60000;
  if (argc > 1) n = atoi(argv[1]);
  if (argc > 2) reps = atoi(argv[2]);

  double *x  = (double *)malloc(n * sizeof(double));
  double *y  = (double *)malloc(n * sizeof(double));
  double *o1 = (double *)calloc(n, sizeof(double));
  double *o2 = (double *)calloc(n, sizeof(double));

  srand(42);
  for (int i = 0; i < n; i++) {
    x[i] = 0.001 * (rand() % 1000 + 1);
    y[i] = (double)(rand() % 10000 + 1);
  }

  kernel(x, y, o1, o2, n, reps);

  printf("o1[0]=%.6f o2[0]=%.6f\n", o1[0], o2[0]);

  free(x); free(y); free(o1); free(o2);
  return 0;
}
