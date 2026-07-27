/* dispatch_probe — estimate per-execution-unit dispatch width on x86-64.
 *
 * For each instruction class we run a tight loop of *independent* ops (no
 * loop-carried dependency across enough registers to expose all ports). The
 * steady-state throughput in core cycles gives:
 *
 *     dispatch_width(class) = instructions_retired / core_cycles   (= #ports)
 *
 * We also run *dependent* chains (latency) and, crucially, report the
 * realized core frequency under each sustained workload:
 *
 *     GHz = core_cycles / wall_ns
 *
 * The GHz column is the frequency-license signal: on Intel AVX-512 throttles
 * the clock; comparing scalar / AVX2 / AVX-512 FMA in achieved GFLOP/s tells
 * you whether widening pays once the clock penalty is folded in. On AMD Zen4
 * (double-pumped AVX-512, no license) you'll see ~flat GHz — also useful.
 *
 * Core cycles come from perf_event_open (PERF_COUNT_HW_CPU_CYCLES,
 * exclude_kernel=1); works at perf_event_paranoid<=2. Without perf we still
 * report wall-time GFLOP/s but cannot separate frequency from IPC.
 *
 * Build:  make            (or: cc -O2 -march=native -o dispatch_probe dispatch_probe.c)
 * Run:    ./dispatch_probe [--core N] [--target-gops G]
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sched.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>

#if !defined(__x86_64__)
#error "dispatch_probe targets x86-64"
#endif

/* ------------------------------------------------------------------ perf */

static int g_fd_cyc = -1;
static int g_fd_ins = -1;
static int g_have_perf = 0;

static int perf_open(uint64_t config) {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof pe);
    pe.type = PERF_TYPE_HARDWARE;
    pe.size = sizeof pe;
    pe.config = config;
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    return (int)syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
}

static void perf_init(void) {
    g_fd_cyc = perf_open(PERF_COUNT_HW_CPU_CYCLES);
    g_fd_ins = perf_open(PERF_COUNT_HW_INSTRUCTIONS);
    g_have_perf = (g_fd_cyc >= 0);
    if (!g_have_perf)
        fprintf(stderr,
            "warning: perf_event_open failed (core-cycle / GHz columns = n/a).\n"
            "         try: sudo sysctl kernel.perf_event_paranoid=1\n\n");
}

static inline void perf_start(void) {
    if (!g_have_perf) return;
    ioctl(g_fd_cyc, PERF_EVENT_IOC_RESET, 0);
    ioctl(g_fd_cyc, PERF_EVENT_IOC_ENABLE, 0);
    if (g_fd_ins >= 0) { ioctl(g_fd_ins, PERF_EVENT_IOC_RESET, 0); ioctl(g_fd_ins, PERF_EVENT_IOC_ENABLE, 0); }
}
static inline void perf_stop(uint64_t *cyc, uint64_t *ins) {
    *cyc = 0; *ins = 0;
    if (!g_have_perf) return;
    ioctl(g_fd_cyc, PERF_EVENT_IOC_DISABLE, 0);
    if (g_fd_ins >= 0) ioctl(g_fd_ins, PERF_EVENT_IOC_DISABLE, 0);
    if (read(g_fd_cyc, cyc, 8) != 8) *cyc = 0;
    if (g_fd_ins >= 0 && read(g_fd_ins, ins, 8) != 8) *ins = 0;
}

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}

/* ----------------------------------------------------------- kernel macros */

/* 14-way independent fan-out (uses xmm/ymm/zmm 0..13; 14,15 = sources). */
#define L14(M) M(0)M(1)M(2)M(3)M(4)M(5)M(6)M(7)M(8)M(9)M(10)M(11)M(12)M(13)

#define INIT_Z(i) "vmovapd %%zmm15,%%zmm" #i "\n\t"
#define INIT_Y(i) "vmovapd %%ymm15,%%ymm" #i "\n\t"
#define INIT_X(i) "vmovapd %%xmm15,%%xmm" #i "\n\t"

#define FMAZ(i)  "vfmadd231pd %%zmm14,%%zmm15,%%zmm" #i "\n\t"
#define FMAY(i)  "vfmadd231pd %%ymm14,%%ymm15,%%ymm" #i "\n\t"
#define FMAX(i)  "vfmadd231sd %%xmm14,%%xmm15,%%xmm" #i "\n\t"
#define ADDZ(i)  "vaddpd %%zmm15,%%zmm" #i ",%%zmm" #i "\n\t"
#define ADDY(i)  "vaddpd %%ymm15,%%ymm" #i ",%%ymm" #i "\n\t"
#define ADDX(i)  "vaddsd %%xmm15,%%xmm" #i ",%%xmm" #i "\n\t"
#define DIVZ(i)  "vdivpd %%zmm14,%%zmm15,%%zmm" #i "\n\t"
#define SQRTZ(i) "vsqrtpd %%zmm15,%%zmm" #i "\n\t"

#define XMMCLOB \
    "xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7", \
    "xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15"

#define REP8(x)  x x x x x x x x
#define REP32(x) REP8(x) REP8(x) REP8(x) REP8(x)

/* Flat replication of a body B, so one outer branch amortizes over ~1024 ops
 * (branch tax ~0.2% instead of ~12%). Body reuses the same regs -> the
 * dependency chains stay spaced by the fan-out, still port-bound. */
#define X2(B)   B B
#define X4(B)   X2(B) X2(B)
#define X8(B)   X4(B) X4(B)
#define X16(B)  X8(B) X8(B)
#define X32(B)  X16(B) X16(B)
#define X64(B)  X32(B) X32(B)    /* x14-op group = 896 ops */
#define X128(B) X64(B) X64(B)    /* x8-op  group = 1024 ops */

/* Each vector throughput kernel: broadcast 1.0 -> src regs, seed 14 accs,
 * then loop 14 independent ops. 1.0 sources keep results normal (no
 * denormal/NaN slowdown) without ever overflowing within the run. */

#define VEC_TP_KERNEL(NAME, BCAST, SRCCOPY, INITM, BODYM)        \
static void NAME(uint64_t n) {                                   \
    double one = 1.0;                                            \
    __asm__ __volatile__(                                        \
        BCAST                                                    \
        SRCCOPY                                                  \
        L14(INITM)                                               \
        ".p2align 4\n"                                           \
        "1:\n\t"                                                 \
        X64(L14(BODYM))                                          \
        "dec %[n]\n\t"                                           \
        "jnz 1b\n\t"                                             \
        : [n] "+r"(n)                                            \
        : [one] "m"(one)                                         \
        : "cc", XMMCLOB);                                        \
}

VEC_TP_KERNEL(k_fma_zmm, "vbroadcastsd %[one],%%zmm15\n\t", "vmovapd %%zmm15,%%zmm14\n\t", INIT_Z, FMAZ)
VEC_TP_KERNEL(k_fma_ymm, "vbroadcastsd %[one],%%ymm15\n\t", "vmovapd %%ymm15,%%ymm14\n\t", INIT_Y, FMAY)
VEC_TP_KERNEL(k_fma_xmm, "vmovsd %[one],%%xmm15\n\t",       "vmovapd %%xmm15,%%xmm14\n\t", INIT_X, FMAX)
VEC_TP_KERNEL(k_add_zmm, "vbroadcastsd %[one],%%zmm15\n\t", "",                            INIT_Z, ADDZ)
VEC_TP_KERNEL(k_add_ymm, "vbroadcastsd %[one],%%ymm15\n\t", "",                            INIT_Y, ADDY)
VEC_TP_KERNEL(k_add_xmm, "vmovsd %[one],%%xmm15\n\t",       "",                            INIT_X, ADDX)
VEC_TP_KERNEL(k_div_zmm, "vbroadcastsd %[one],%%zmm15\n\t", "vmovapd %%zmm15,%%zmm14\n\t", INIT_Z, DIVZ)
VEC_TP_KERNEL(k_sqrt_zmm,"vbroadcastsd %[one],%%zmm15\n\t", "",                            INIT_Z, SQRTZ)

/* Latency: single dependent chain, heavily unrolled to hide loop overhead. */
#define VEC_LAT_KERNEL(NAME, SETUP, CHAINOP)                     \
static void NAME(uint64_t n) {                                   \
    double one = 1.0;                                            \
    __asm__ __volatile__(                                        \
        SETUP                                                    \
        ".p2align 4\n"                                           \
        "1:\n\t"                                                 \
        REP32(CHAINOP)                                           \
        "dec %[n]\n\t"                                           \
        "jnz 1b\n\t"                                             \
        : [n] "+r"(n)                                            \
        : [one] "m"(one)                                         \
        : "cc", XMMCLOB);                                        \
}
VEC_LAT_KERNEL(k_lat_fma_xmm,
    "vmovsd %[one],%%xmm15\n\tvmovapd %%xmm15,%%xmm14\n\tvmovapd %%xmm15,%%xmm0\n\t",
    "vfmadd231sd %%xmm14,%%xmm15,%%xmm0\n\t")
VEC_LAT_KERNEL(k_lat_fma_zmm,
    "vbroadcastsd %[one],%%zmm15\n\tvmovapd %%zmm15,%%zmm14\n\tvmovapd %%zmm15,%%zmm0\n\t",
    "vfmadd231pd %%zmm14,%%zmm15,%%zmm0\n\t")
VEC_LAT_KERNEL(k_lat_div_zmm,
    "vbroadcastsd %[one],%%zmm15\n\tvmovapd %%zmm15,%%zmm14\n\tvmovapd %%zmm15,%%zmm0\n\t",
    "vdivpd %%zmm14,%%zmm0,%%zmm0\n\t")   /* dst=zmm0/zmm14 -> chain on zmm0 */
VEC_LAT_KERNEL(k_lat_sqrt_zmm,
    "vbroadcastsd %[one],%%zmm15\n\tvmovapd %%zmm15,%%zmm0\n\t",
    "vsqrtpd %%zmm0,%%zmm0\n\t")

/* ------------------------------------------------- integer / memory kernels */

/* 8 independent integer ALU adds (r8..r15), replicated to 1024/iter. */
#define IADD8 "add $1,%%r8\n\tadd $1,%%r9\n\tadd $1,%%r10\n\tadd $1,%%r11\n\t" \
              "add $1,%%r12\n\tadd $1,%%r13\n\tadd $1,%%r14\n\tadd $1,%%r15\n\t"
static void k_int_add(uint64_t n) {
    __asm__ __volatile__(
        "xor %%r8d,%%r8d\n\txor %%r9d,%%r9d\n\txor %%r10d,%%r10d\n\txor %%r11d,%%r11d\n\t"
        "xor %%r12d,%%r12d\n\txor %%r13d,%%r13d\n\txor %%r14d,%%r14d\n\txor %%r15d,%%r15d\n\t"
        ".p2align 4\n1:\n\t"
        X128(IADD8)
        "dec %[n]\n\tjnz 1b\n\t"
        : [n] "+r"(n)
        :
        : "cc","r8","r9","r10","r11","r12","r13","r14","r15");
}

static unsigned char g_buf[4096] __attribute__((aligned(64)));
static void *g_chase;

/* 8 independent L1 loads (within one line -> pure load-port test), x1024/iter. */
#define LOAD8 "mov 0(%[p]),%%r8\n\tmov 8(%[p]),%%r9\n\tmov 16(%[p]),%%r10\n\tmov 24(%[p]),%%r11\n\t" \
              "mov 32(%[p]),%%r12\n\tmov 40(%[p]),%%r13\n\tmov 48(%[p]),%%r14\n\tmov 56(%[p]),%%r15\n\t"
static void k_load_64(uint64_t n) {
    void *p = g_buf;
    __asm__ __volatile__(
        ".p2align 4\n1:\n\t"
        X128(LOAD8)
        "dec %[n]\n\tjnz 1b\n\t"
        : [n] "+r"(n)
        : [p] "r"(p)
        : "cc","r8","r9","r10","r11","r12","r13","r14","r15","memory");
}

/* 8 independent L1 stores, x1024/iter. */
#define STORE8 "mov %%r8,0(%[p])\n\tmov %%r8,8(%[p])\n\tmov %%r8,16(%[p])\n\tmov %%r8,24(%[p])\n\t" \
               "mov %%r8,32(%[p])\n\tmov %%r8,40(%[p])\n\tmov %%r8,48(%[p])\n\tmov %%r8,56(%[p])\n\t"
static void k_store_64(uint64_t n) {
    void *p = g_buf;
    __asm__ __volatile__(
        "xor %%r8d,%%r8d\n\t"
        ".p2align 4\n1:\n\t"
        X128(STORE8)
        "dec %[n]\n\tjnz 1b\n\t"
        : [n] "+r"(n)
        : [p] "r"(p)
        : "cc","r8","memory");
}

/* Dependent L1 load chain -> load-use latency. g_chase holds its own address. */
static void k_lat_load(uint64_t n) {
    void *c = &g_chase;
    __asm__ __volatile__(
        "mov %[c],%%rax\n\t"
        ".p2align 4\n1:\n\t"
        REP8("mov (%%rax),%%rax\n\t")
        "dec %[n]\n\tjnz 1b\n\t"
        : [n] "+r"(n)
        : [c] "r"(c)
        : "cc","rax","memory");
}

/* ------------------------------------------------------------ kernel table */

typedef struct {
    const char *name;
    const char *cls;      /* FPU / IPU / SFU / MPU */
    void (*fn)(uint64_t);
    int    ipi;           /* independent ops per loop iteration */
    double flops;         /* flops per op (0 = non-FP) */
    int    bytes;         /* bytes per op (mem kernels; else 0) */
    int    is_lat;        /* 1 = latency kernel */
} Kern;

/* (Kernel table is built in main, next to its metadata.) */

typedef struct {
    const char *name; double instr_per_cyc; double ghz; double metric; /* GFLOP/s or GB/s */
    double cyc_per_op; int have_cyc; const char *unit; const char *cls;
} Result;

static Result run_kernel(const Kern *k, uint64_t target_ops) {
    uint64_t iters = target_ops / (uint64_t)k->ipi;
    if (iters < 1) iters = 1;

    k->fn(iters / 4 + 1);            /* warmup: settle frequency + caches */

    uint64_t best_cyc = (uint64_t)-1; double best_ns = 0, best_ipc = 0; int have = 0;
    for (int r = 0; r < 2; r++) {
        uint64_t cyc, ins; double t0, t1;
        perf_start();
        t0 = now_ns();
        k->fn(iters);
        t1 = now_ns();
        perf_stop(&cyc, &ins);
        double ipc = (cyc ? (double)(iters * (uint64_t)k->ipi) / (double)cyc : 0);
        if (cyc && cyc < best_cyc) { best_cyc = cyc; best_ns = t1 - t0; best_ipc = ipc; have = 1; }
        if (!cyc) { best_ns = t1 - t0; }   /* perf unavailable: keep last wall time */
    }

    double total_ops = (double)(iters * (uint64_t)k->ipi);
    Result R; memset(&R, 0, sizeof R);
    R.name = k->name; R.cls = k->cls; R.have_cyc = have;
    R.instr_per_cyc = best_ipc;
    R.ghz = (have && best_ns > 0) ? (double)best_cyc / best_ns : 0;
    R.cyc_per_op = (have ? (double)best_cyc / total_ops : 0);
    if (best_ns <= 0) best_ns = 1;
    if (k->bytes) { R.metric = total_ops * k->bytes / best_ns; R.unit = "GB/s"; }
    else          { R.metric = total_ops * k->flops / best_ns; R.unit = "GFLOP/s"; }
    return R;
}

/* ------------------------------------------------------------------- main */

static void cpu_model(char *out, size_t n) {
    out[0] = 0;
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) return;
    char line[512];
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "model name", 10) == 0) {
            char *c = strchr(line, ':');
            if (c) { c += 2; c[strcspn(c, "\n")] = 0; strncpy(out, c, n - 1); out[n-1]=0; }
            break;
        }
    }
    fclose(f);
}

int main(int argc, char **argv) {
    int core = 0;
    double target_gops = 1.0;        /* ~1e9 independent ops per throughput run */
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--core") && i + 1 < argc) core = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--target-gops") && i + 1 < argc) target_gops = atof(argv[++i]);
        else { fprintf(stderr, "usage: %s [--core N] [--target-gops G]\n", argv[0]); return 2; }
    }

    cpu_set_t set; CPU_ZERO(&set); CPU_SET(core, &set);
    if (sched_setaffinity(0, sizeof set, &set) != 0)
        fprintf(stderr, "warning: could not pin to core %d\n", core);

    g_chase = &g_chase;              /* self-referential cell for load-latency chase */
    perf_init();

    char model[256]; cpu_model(model, sizeof model);
    printf("dispatch_probe  cpu=\"%s\"  core=%d  perf=%s\n",
           model[0] ? model : "?", core, g_have_perf ? "on" : "off(wall-only)");
    printf("note: instr/cyc = dispatch width (ports). GHz = realized clock under "
           "sustained load (frequency-license signal).\n\n");

    const uint64_t TP  = (uint64_t)(target_gops * 1e9);
    const uint64_t LAT = (uint64_t)(target_gops * 1e9 / 4);

    /* ipi = independent ops per outer-loop iteration (vector body = 14*64=896,
     * int/mem body = 8*128=1024); the single dec/jnz per iter is negligible. */
    const Kern tp[] = {
        { "fma.f64.scalar","FPU", k_fma_xmm, 896, 2,  0, 0 },
        { "fma.f64.avx2",  "FPU", k_fma_ymm, 896, 8,  0, 0 },
        { "fma.f64.avx512","FPU", k_fma_zmm, 896, 16, 0, 0 },
        { "add.f64.scalar","FPU", k_add_xmm, 896, 1,  0, 0 },
        { "add.f64.avx2",  "FPU", k_add_ymm, 896, 4,  0, 0 },
        { "add.f64.avx512","FPU", k_add_zmm, 896, 8,  0, 0 },
        { "int.add.r64",   "IPU", k_int_add, 1024,0,  0, 0 },
        { "div.f64.avx512","SFU", k_div_zmm, 896, 8,  0, 0 },
        { "sqrt.f64.avx512","SFU",k_sqrt_zmm,896, 8,  0, 0 },
        { "load.r64.L1",   "MPU", k_load_64, 1024,0,  8, 0 },
        { "store.r64.L1",  "MPU", k_store_64,1024,0,  8, 0 },
    };
    const Kern lat[] = {
        { "fma.f64.scalar", "FPU", k_lat_fma_xmm, 32, 2, 0, 1 },
        { "fma.f64.avx512", "FPU", k_lat_fma_zmm, 32, 16,0, 1 },
        { "div.f64.avx512", "SFU", k_lat_div_zmm, 32, 8, 0, 1 },
        { "sqrt.f64.avx512","SFU", k_lat_sqrt_zmm,32, 8, 0, 1 },
        { "load.r64.L1",    "MPU", k_lat_load,     8, 0, 8, 1 },
    };

    printf("== THROUGHPUT (dispatch width) ==\n");
    printf("%-18s %-4s %10s %8s %12s\n", "kernel","cls","instr/cyc","GHz","throughput");
    Result fma_s={0}, fma_v2={0}, fma_v5={0};
    for (size_t i = 0; i < sizeof tp / sizeof tp[0]; i++) {
        Result R = run_kernel(&tp[i], TP);
        if (R.have_cyc)
            printf("%-18s %-4s %10.2f %8.3f %8.1f %s\n",
                   R.name, R.cls, R.instr_per_cyc, R.ghz, R.metric, R.unit);
        else
            printf("%-18s %-4s %10s %8s %8.1f %s\n",
                   R.name, R.cls, "n/a", "n/a", R.metric, R.unit);
        if (!strcmp(R.name,"fma.f64.scalar")) fma_s = R;
        if (!strcmp(R.name,"fma.f64.avx2"))   fma_v2 = R;
        if (!strcmp(R.name,"fma.f64.avx512")) fma_v5 = R;
    }

    printf("\n== LATENCY (dependent chain, cyc/op) ==\n");
    printf("%-18s %-4s %10s %8s\n", "kernel","cls","cyc/op","GHz");
    for (size_t i = 0; i < sizeof lat / sizeof lat[0]; i++) {
        Result R = run_kernel(&lat[i], LAT);
        if (R.have_cyc)
            printf("%-18s %-4s %10.2f %8.3f\n", R.name, R.cls, R.cyc_per_op, R.ghz);
        else
            printf("%-18s %-4s %10s %8s\n", R.name, R.cls, "n/a", "n/a");
    }

    /* --------- vectorize-vs-scalar verdict for FP FMA (the headline) --------- */
    printf("\n== FP FMA: vectorize vs scalar (achieved, freq-folded) ==\n");
    printf("%-10s %12s %8s %10s\n", "width", "GFLOP/s", "GHz", "vs scalar");
    if (fma_s.metric > 0) {
        printf("%-10s %12.1f %8.3f %9.2fx\n", "scalar", fma_s.metric, fma_s.ghz, 1.0);
        if (fma_v2.metric > 0)
            printf("%-10s %12.1f %8.3f %9.2fx\n", "avx2",  fma_v2.metric, fma_v2.ghz, fma_v2.metric/fma_s.metric);
        if (fma_v5.metric > 0)
            printf("%-10s %12.1f %8.3f %9.2fx\n", "avx512",fma_v5.metric, fma_v5.ghz, fma_v5.metric/fma_s.metric);

        const char *best_name = "scalar"; double best_m = fma_s.metric;
        if (fma_v2.metric > best_m) { best_m = fma_v2.metric; best_name = "avx2"; }
        if (fma_v5.metric > best_m) { best_m = fma_v5.metric; best_name = "avx512"; }
        printf("-> best FP-FMA width: %s (%.1f GFLOP/s)\n", best_name, best_m);

        if (fma_v5.have_cyc && fma_v2.have_cyc && fma_v5.ghz > 0 && fma_v2.ghz > 0) {
            double drop = (fma_v2.ghz - fma_v5.ghz) / fma_v2.ghz * 100.0;
            if (drop > 4.0)
                printf("-> AVX-512 clock %.1f%% below AVX2: frequency-license throttle PRESENT "
                       "(weigh lane gain against clock loss).\n", drop);
            else
                printf("-> AVX-512 clock %.1f%% vs AVX2: no meaningful throttle (widen freely).\n", -drop);
        }
    } else {
        printf("(need perf core cycles for the freq verdict; lower perf_event_paranoid)\n");
    }

    return 0;
}
