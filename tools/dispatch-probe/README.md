# dispatch-probe

Microarchitectural probe that estimates the **maximum concurrent dispatch
width** onto each execution-unit class, plus the **realized core frequency
under sustained load**. Built to inform the codegen tradeoff:

> vectorize under a (possibly frequency-licensed) wide AVX path  **vs.**
> issue multiple independent narrower / scalar instructions across ports.

## Method

For every instruction class we run a loop whose body is the independent-op
group (14 vector regs, or 8 GPRs) replicated to ~1024 ops, so the single
`dec`/`jnz` per iteration is ~0.2% overhead instead of ~12% — the loop branch
never pollutes the measured port (matters most for `int.add`, where the branch
would otherwise steal ALU slots; FP kernels are unaffected since `dec`/`jnz`
use different ports). The body stays ~5 KB, well inside L1I. Enough register
fan-out exposes every port; steady-state throughput in **core cycles** gives
the dispatch width:

```
dispatch_width(class) = instructions_retired / core_cycles      (= usable ports)
```

A second set of kernels runs a *dependent* chain → instruction **latency**.

The decisive column is **GHz = core_cycles / wall_ns**: the actual clock the
core sustained while hammering that op mix. On Intel, AVX-512 (and heavy
AVX2) drop into a lower licensed frequency; this probe shows it directly. The
`GFLOP/s` columns are wall-clock — they already fold the frequency in, so the
"best width" verdict is the honest one.

Core cycles come from `perf_event_open(PERF_COUNT_HW_CPU_CYCLES, exclude_kernel=1)`,
which works at `perf_event_paranoid <= 2`. Without perf the tool still reports
wall-clock GFLOP/s but cannot separate IPC from frequency (GHz = n/a).

## Classes probed

| class | meaning              | kernels |
|-------|----------------------|---------|
| FPU   | floating-point ALU   | FMA / ADD — scalar, AVX2 (ymm), AVX-512 (zmm) |
| IPU   | integer ALU          | `add r64` |
| SFU   | "special" (div/sqrt) | `vdivpd`, `vsqrtpd` (zmm) — not fully pipelined |
| MPU   | memory ports         | L1 load throughput, store throughput, load-use latency |

Sources are initialized to `1.0` so results stay normal — no denormal/NaN
slowdown skewing the FP numbers.

## Build & run

```bash
make
./dispatch_probe                 # core 0
./dispatch_probe --core 8        # pin elsewhere (see CCD note)
./dispatch_probe --target-gops 4 # longer runs = steadier frequency
```

No LLVM dependency; standalone C. Not wired into the main `ninja` build — add
`add_subdirectory(dispatch-probe)` to `tools/CMakeLists.txt` only if you want it.

## Reading the output

```
== THROUGHPUT (dispatch width) ==
kernel             cls   instr/cyc      GHz   throughput
fma.f64.avx2       FPU        2.00    4.735     75.7 GFLOP/s   <- 2 FMA ports @256b
fma.f64.avx512     FPU        1.00    5.017     80.2 GFLOP/s   <- 1 zmm/cyc (double-pumped)
int.add.r64        IPU        3.99    5.230      ...           <- 4 ALU ports
load.r64.L1        MPU        2.97    4.829    114.7 GB/s      <- 3 load ports
store.r64.L1       MPU        2.00    4.980     79.5 GB/s      <- 2 store ports

== FP FMA: vectorize vs scalar (achieved, freq-folded) ==
width           GFLOP/s      GHz  vs scalar
scalar             18.9    4.736      1.00x
avx2               75.7    4.735      4.00x
avx512             80.2    5.017      4.24x
-> best FP-FMA width: avx512
-> AVX-512 clock 6.0% vs AVX2: no meaningful throttle (widen freely).
```

### Applying it to the vectorize/scalar decision

- **`instr/cyc` per class = how many independent ops of that kind dispatch per
  cycle.** If scalar FMA is 2/cyc, two independent scalar FMA chains run at the
  same instruction rate as one — the scalar path is only worth it when it lets
  you keep more ports busy than the vector path saturates.
- **Effective scalar compute = `instr/cyc(scalar) × GHz(scalar)`**; vector =
  `instr/cyc(vec) × lanes × GHz(vec)`. The wide path wins unless its realized
  GHz drops enough to erase the lane factor — exactly what the GHz column
  surfaces.
- **Frequency-license throttle present** (Intel AVX-512): the verdict line
  flags it; trust the GFLOP/s, not the lane count.
- **No throttle** (AMD Zen4 here): AVX-512 is double-pumped, so 1 zmm/cyc ==
  2 ymm/cyc == equal FLOP/cyc; widen for code density, not raw compute.

## Notes / caveats

- **Pin the right core.** On heterogeneous parts (e.g. 7950X3D: V-cache CCD0
  clocks lower than CCD1) frequency differs per die — probe both.
- Use a fixed governor (`cpupower frequency-set -g performance`) for repeatable
  GHz; the tool reports whatever the core actually sustained regardless.
- SMT sibling activity steals ports — measure on an otherwise-idle machine.
- x86-64 only (inline asm). Requires AVX-512 for the zmm kernels (`-march=native`).
