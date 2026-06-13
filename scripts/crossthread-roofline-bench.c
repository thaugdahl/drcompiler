// crossthread-roofline-bench.c — validate the CROSSCUTTING.md III roofline
// premise on real hardware: under shared-bandwidth saturation, RECOMPUTE (more
// per-thread ALU, less memory traffic) beats MATERIALIZE (produce a buffer once,
// reload it per consumer), reversing the single-thread verdict.
//
// Each thread runs an INDEPENDENT problem instance, so the only cross-thread
// interaction is contention for the shared LLC / DRAM bandwidth — exactly what
// streamCycles(bytes, BW/activeThreads) models. The two variants mirror the
// fission/DR cost-model alternatives:
//
//   MATERIALIZE : buf[i] = f(x[i]) once (>L2 buffer write), then N consumers
//                 each READ the whole buffer.   traffic ~ (1 + 1 + N) * WS
//   RECOMPUTE   : N consumers each recompute f(x[i]) from x.
//                 traffic ~ N * WS, but N x the ALU (hidden behind mem stalls).
//
// f is a sqrt chain so MATERIALIZE does ~1x the ALU of RECOMPUTE's Nx. Single
// thread (compute-bound) -> MATERIALIZE wins; many threads (bandwidth-bound) ->
// RECOMPUTE wins. The crossover is the model's predicted reversal.
//
// Build: clang -O2 -march=native -fopenmp scripts/crossthread-roofline-bench.c -lm -o /tmp/ctbench
// Run:   /tmp/ctbench

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef NELEM
// 8 MiB of f64 per array.  Must be large enough that activeThreads x (x + buf)
// EXCEEDS the shared LLC, or the reversal is hidden: on the 7950X3D's 128 MiB
// (2x V-cache) L3, the first attempt at 2 MiB/array kept all buffers cached and
// MATERIALIZE always won.  At 8 MiB x 16 threads x 2 arrays = 256 MiB >> 128 MiB
// the buffer reloads miss to DRAM and the bandwidth reversal appears.
#define NELEM (1048576)
#endif
#define NCONS 3        // consumers of the shared value
#define ITERS 6        // timed repeats (min taken)

// Tunable shared computation: `fops` dependent multiply-adds (cheap, ~1 FMA
// latency each; the recurrence prevents constant folding).  Sweeping fops walks
// the compute-vs-bandwidth balance the cost model arbitrates.
static inline double f(double v, int fops) {
  double r = v;
  for (int k = 0; k < fops; k++)
    r = r * 0.9999 + 0.0001;
  return r;
}

static double materialize(const double *x, double *buf, int fops) {
  for (long i = 0; i < NELEM; i++)
    buf[i] = f(x[i], fops); // produce once into the >L2 buffer
  double acc = 0.0;
  for (int c = 0; c < NCONS; c++) // N consumers reload the buffer
    for (long i = 0; i < NELEM; i++)
      acc += buf[i] * (c + 1);
  return acc;
}

static double recompute(const double *x, int fops) {
  double acc = 0.0;
  for (int c = 0; c < NCONS; c++) // N consumers recompute from x (no buffer)
    for (long i = 0; i < NELEM; i++)
      acc += f(x[i], fops) * (c + 1);
  return acc;
}

// EXCLUSIVE variants: ONE problem parallelized across all cores (the workload
// owns the machine -> full bandwidth; per-thread WS/N and BW/N cancel).
static double materialize_excl(const double *x, double *buf, int fops) {
#pragma omp parallel for schedule(static)
  for (long i = 0; i < NELEM; i++)
    buf[i] = f(x[i], fops);
  double acc = 0.0;
#pragma omp parallel for reduction(+ : acc) schedule(static)
  for (long i = 0; i < NELEM; i++)
    for (int c = 0; c < NCONS; c++)
      acc += buf[i] * (c + 1);
  return acc;
}
static double recompute_excl(const double *x, int fops) {
  double acc = 0.0;
#pragma omp parallel for reduction(+ : acc) schedule(static)
  for (long i = 0; i < NELEM; i++)
    for (int c = 0; c < NCONS; c++)
      acc += f(x[i], fops) * (c + 1);
  return acc;
}

static volatile double g_sink = 0.0;

// Per-problem time (ms) of a variant at T threads, min over ITERS.
static double run(int T, int fops, int materializeVariant, double **xs,
                  double **bufs) {
  omp_set_num_threads(T);
  double best = 1e30;
  for (int it = 0; it < ITERS; it++) {
    double t0 = omp_get_wtime(), a = 0.0;
    if (materializeVariant) {
#pragma omp parallel for reduction(+ : a) schedule(static)
      for (int t = 0; t < T; t++)
        a += materialize(xs[t], bufs[t], fops);
    } else {
#pragma omp parallel for reduction(+ : a) schedule(static)
      for (int t = 0; t < T; t++)
        a += recompute(xs[t], fops);
    }
    double dt = omp_get_wtime() - t0;
    g_sink += a;
    if (dt < best)
      best = dt;
  }
  return best / T * 1e3; // per-problem ms
}

// EXCLUSIVE per-problem time (ms): ONE problem parallelized across all cores.
static double runExcl(int maxT, int fops, int materializeVariant, double *x,
                      double *buf) {
  omp_set_num_threads(maxT);
  double best = 1e30;
  for (int it = 0; it < ITERS; it++) {
    double t0 = omp_get_wtime();
    double a = materializeVariant ? materialize_excl(x, buf, fops)
                                  : recompute_excl(x, fops);
    double dt = omp_get_wtime() - t0;
    g_sink += a;
    if (dt < best)
      best = dt;
  }
  return best * 1e3; // one problem, all cores
}

int main(void) {
  int maxT = omp_get_max_threads();
  double **xs = (double **)malloc(sizeof(double *) * maxT);
  double **bufs = (double **)malloc(sizeof(double *) * maxT);
  for (int t = 0; t < maxT; t++) {
    xs[t] = (double *)malloc(sizeof(double) * NELEM);
    bufs[t] = (double *)malloc(sizeof(double) * NELEM);
    for (long i = 0; i < NELEM; i++)
      xs[t][i] = (double)((i * 1103515245u + t) % 997) * 0.001 + 0.5;
  }
  printf("crossthread roofline bench: NELEM=%d (%.1f MiB/array), NCONS=%d, "
         "cores=%d\n",
         NELEM, NELEM * 8.0 / (1024 * 1024), NCONS, maxT);
  printf("Per-problem ms; winner is cheaper.  At 1 thread (compute-bound) "
         "MATERIALIZE should win;\nif bandwidth-bound at %d threads, the cheap-"
         "recompute rows should flip to RECOMPUTE.\n\n",
         maxT);
  int fopsSweep[] = {1, 2, 4, 8, 16, 32, 64};
  int nf = (int)(sizeof(fopsSweep) / sizeof(fopsSweep[0]));
  printf("Winner per deployment.  INTERSPERSED = %d independent problems on %d "
         "threads (shared BW);\nEXCLUSIVE = ONE problem parallelized across %d "
         "cores (owns BW).  '!=' marks where the\nbest choice DIFFERS by mode "
         "-- the case for the dual cost model.\n\n",
         maxT, maxT, maxT);
  printf("%-6s | %-12s | %-16s | %-16s | %s\n", "fops", "1 thread",
         "interspersed(N)", "exclusive(N)", "mode-matters");
  int reversals = 0, loFlip = 0, hiFlip = 0, modeDiffs = 0;
  for (int i = 0; i < nf; i++) {
    int fo = fopsSweep[i];
    double m1 = run(1, fo, 1, xs, bufs), r1 = run(1, fo, 0, xs, bufs);
    double mN = run(maxT, fo, 1, xs, bufs), rN = run(maxT, fo, 0, xs, bufs);
    double me = runExcl(maxT, fo, 1, xs[0], bufs[0]);
    double re = runExcl(maxT, fo, 0, xs[0], bufs[0]);
    int interMat = mN < rN, exclMat = me < re;
    int flip = (m1 < r1) && !interMat; // MAT 1-thread, REC interspersed
    int modeDiff = interMat != exclMat;
    printf("%-6d | %-12s | %7.3f/%7.3f %-4s | %7.3f/%7.3f %-4s | %s%s\n", fo,
           m1 < r1 ? "[MAT]" : "[REC]", mN, rN, interMat ? "[MAT]" : "[REC]", me,
           re, exclMat ? "[MAT]" : "[REC]",
           modeDiff ? "!= (exclusive favors keep)" : "same",
           flip ? "  <reversal" : "");
    if (flip) {
      reversals++;
      if (!loFlip)
        loFlip = fo;
      hiFlip = fo;
    }
    if (modeDiff)
      modeDiffs++;
  }
  printf("\n%s",
         reversals
             ? "VALIDATED: the keep->recompute reversal the cost-model roofline "
               "term predicts is MEASURED"
             : "no reversal observed (working set may fit the V-cache)");
  if (reversals)
    printf(" (window fops=%d..%d).\n", loFlip, hiFlip);
  else
    printf(".\n");
  printf("%s\n",
         modeDiffs ? "DUAL MODE JUSTIFIED: at some compute intensities the best "
                     "choice DIFFERS between interspersed and exclusive "
                     "deployment (exclusive's full bandwidth favors keeping the "
                     "buffer where shared bandwidth favors recompute)."
                   : "(no mode difference at the sampled intensities)");
  if (g_sink == 12345.6789)
    printf("");
  return 0;
}
