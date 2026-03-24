/*===-- cache_latency_bench.c - Pointer-chasing cache latency benchmark ----===*
 * Measures memory access latency at various working set sizes to identify
 * cache hierarchy latencies (L1/L2/L3/DRAM).
 *
 * Output: one line per working set size: "SIZE_BYTES LATENCY_NS"
 *
 * Build: cc -O2 -o cache_latency_bench cache_latency_bench.c
 * Usage: ./cache_latency_bench [max_size_mb]
 *===----------------------------------------------------------------------===*/

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Each node occupies one cache line (64 bytes). */
#define CACHELINE 64

struct node {
  struct node *next;
  char _pad[CACHELINE - sizeof(struct node *)];
};

/*
 * Sattolo's algorithm: produce a random permutation that is guaranteed to be
 * a single cycle of length n.  Every element is visited exactly once.
 */
static void sattolo(size_t *perm, size_t n) {
  for (size_t i = 0; i < n; i++)
    perm[i] = i;
  for (size_t i = n - 1; i > 0; i--) {
    size_t j = (size_t)rand() % i; /* j in [0, i-1] */
    size_t tmp = perm[i];
    perm[i] = perm[j];
    perm[j] = tmp;
  }
}

static double measure(size_t n_nodes) {
  if (n_nodes < 2)
    n_nodes = 2;

  struct node *arena =
      (struct node *)aligned_alloc(CACHELINE, n_nodes * sizeof(struct node));
  if (!arena) {
    fprintf(stderr, "alloc failed for %zu nodes\n", n_nodes);
    return -1;
  }
  memset(arena, 0, n_nodes * sizeof(struct node));

  size_t *perm = (size_t *)malloc(n_nodes * sizeof(size_t));
  sattolo(perm, n_nodes);

  /* Wire up the linked list according to the permutation. */
  for (size_t i = 0; i < n_nodes; i++)
    arena[perm[i]].next = &arena[perm[(i + 1) % n_nodes]];
  free(perm);

  /* Warmup: traverse the full cycle twice. */
  struct node *p = &arena[0];
  for (size_t i = 0; i < n_nodes * 2; i++)
    p = p->next;

  /* Timed run: enough iterations for stable measurement. */
  size_t iters = 4000000;
  if (iters < n_nodes * 4)
    iters = n_nodes * 4;

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  for (size_t i = 0; i < iters; i++) {
    p = p->next;
    /* Compiler barrier: prevent the load from being optimized away. */
    __asm__ volatile("" : "+r"(p));
  }
  clock_gettime(CLOCK_MONOTONIC, &t1);

  double elapsed_ns =
      (double)(t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
  double lat = elapsed_ns / (double)iters;

  free(arena);
  return lat;
}

int main(int argc, char **argv) {
  size_t max_mb = 512;
  if (argc > 1)
    max_mb = (size_t)atol(argv[1]);

  srand(42);

  /* Test sizes: powers of 2 from 2 KB up to max_mb MB. */
  size_t size = 2048;
  size_t max_bytes = max_mb * 1024 * 1024;
  while (size <= max_bytes) {
    size_t n_nodes = size / sizeof(struct node);
    double lat = measure(n_nodes);
    if (lat >= 0)
      printf("%zu %.3f\n", size, lat);
    fflush(stdout);
    size *= 2;
  }

  return 0;
}
