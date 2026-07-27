#!/usr/bin/env python3
"""Generate a CPU cost model JSON file for drcompiler.

The file maps MLIR operation names to their estimated cycle costs.
Both passes (data-recomputation and memory-fission) use this table
to weight ALU operations when comparing recompute-vs-keep decisions.

Usage:
    # Static presets (no hardware probing):
    python3 gen_cpu_cost_model.py                    # print to stdout
    python3 gen_cpu_cost_model.py -o costs.json      # write to file
    python3 gen_cpu_cost_model.py --target skylake    # use a preset

    # Hardware probing (uses llvm-mca for ALU costs, lmbench or built-in
    # pointer-chasing benchmark for cache latencies):
    python3 gen_cpu_cost_model.py --probe
    python3 gen_cpu_cost_model.py --probe --llvm-mca /opt/llvm/bin/llvm-mca
    python3 gen_cpu_cost_model.py --probe -o costs.json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

# ---------- presets ----------

# Generic x86-64 (Haswell-ish defaults — matches the original hardcoded table).
GENERIC = {
    "default_cost": 5,
    "ops": {
        # Free
        "arith.constant": 0,

        # Cheap ALU (1 cycle)
        "arith.addi":       1,
        "arith.addf":       1,
        "arith.subi":       1,
        "arith.subf":       1,
        "arith.xori":       1,
        "arith.andi":       1,
        "arith.ori":        1,
        "arith.shrsi":      1,
        "arith.shrui":      1,
        "arith.shli":       1,
        "arith.select":     1,
        "arith.cmpi":       1,
        "arith.cmpf":       1,

        # Multiply (3 cycles)
        "arith.muli":       3,
        "arith.mulf":       3,

        # Division / remainder (15 cycles)
        "arith.divsi":      15,
        "arith.divui":      15,
        "arith.divf":       15,
        "arith.remsi":      15,
        "arith.remui":      15,
        "arith.remf":       15,

        # Transcendental (20 cycles)
        "math.sqrt":        20,
        "math.exp":         20,
        "math.log":         20,
        "math.sin":         20,
        "math.cos":         20,
        "math.tanh":        20,
        "math.powf":        20,

        # Conversions (1 cycle)
        "arith.sitofp":     1,
        "arith.fptosi":     1,
        "arith.extsi":      1,
        "arith.extui":      1,
        "arith.trunci":     1,
        "arith.index_cast": 1,
        "arith.bitcast":    1,
    },
}

# Skylake (slightly adjusted for known μarch costs).
SKYLAKE = dict(GENERIC)
SKYLAKE["ops"] = dict(GENERIC["ops"])
SKYLAKE["ops"].update({
    "arith.muli":  3,
    "arith.mulf":  4,
    "arith.divsi": 26,
    "arith.divui": 26,
    "arith.divf":  11,
    "arith.remsi": 26,
    "arith.remui": 26,
    "arith.remf":  11,
    "math.sqrt":   12,
    "math.exp":    18,
    "math.log":    18,
    "math.sin":    50,
    "math.cos":    50,
    "math.tanh":   40,
    "math.powf":   50,
})

PRESETS = {
    "generic": GENERIC,
    "skylake": SKYLAKE,
}

# ---------- arch + register-file model ----------
#
# Mirrors drcompiler's per-ArchHandler tables (lib/Analysis/ArchHandlers/*.cpp)
# so a probed/preset JSON reproduces the compiler's built-in defaults exactly.
# CpuCostModel.cpp parses these into m.arch (handler/triplet/vector_width_bits/
# spill_strategy/weights) and m.registers (gp/fp/vec/pred budgets + spill
# cycles); without them every arch/register knob falls back to compiler
# defaults and the cache-architecture flags cannot be exercised from JSON.

# handler name -> (vector_width_bits, gp, fp, vec, pred, reload_cy, store_cy)
ARCH_HANDLERS = {
    "generic":       (128, 16, 16, 16, 0, 5, 1),
    "x86-64-avx2":   (256, 16, 16, 16, 0, 5, 1),
    "x86-64-avx512": (512, 16, 32, 32, 8, 5, 1),
    "arm-neon":      (128, 31, 32, 32, 0, 5, 1),
}

# Unified cost-model blend weights — ArchHandler.h defaults (arch-independent).
DEFAULT_WEIGHTS = {"alpha_mem": 1.0, "beta_reg": 1.0, "gamma_alu": 1.0}
# Matches the dr-affine-* pass option default (spill-strategy="excess-hot").
DEFAULT_SPILL_STRATEGY = "excess-hot"


def _host_triplet(cc):
    """Target triple via `cc -dumpmachine`; fall back to a platform guess."""
    try:
        out = subprocess.run([cc, "-dumpmachine"], capture_output=True,
                             text=True, timeout=10)
        t = out.stdout.strip()
        if t:
            return t
    except (OSError, subprocess.SubprocessError):
        pass
    import platform
    return f"{platform.machine() or 'unknown'}-unknown-linux-gnu"


def _cpu_flags():
    """x86/ARM feature tokens from /proc/cpuinfo (empty set if unavailable)."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith(("flags", "Features")):
                    return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def _cpu_vendor_family():
    """(vendor_id, family_int) from /proc/cpuinfo; ('', 0) if unavailable."""
    vendor, family = "", 0
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("vendor_id") and not vendor:
                    vendor = line.split(":", 1)[1].strip()
                elif line.startswith("cpu family") and not family:
                    try:
                        family = int(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                if vendor and family:
                    break
    except OSError:
        pass
    return vendor, family


def native_vector_bits(arch_bits):
    """Throughput-effective FP datapath width, which can be NARROWER than the
    ISA-encodable width (`arch_bits`) when a part double-pumps wide vectors over
    narrower FP pipes.  This is exactly the vector_bits_native the MachineModel
    consumes for register-block VL selection (MachineModel.h): a wider-than-native
    op is cracked at the same FLOP rate, so the model should prefer the native
    width unless the accumulator tile forces it wider.

    * <=256-bit ISA (AVX2/NEON/generic): native == arch (no double-pump).
    * AVX-512 (arch 512):
        - AMD family <= 0x19 (Zen4 and earlier AVX-512): 2x256-bit FP pipes
          execute AVX-512 double-pumped -> native 256.
        - AMD family >= 0x1a (Zen5+): native 512.
        - Intel with AVX-512 (SKX/CLX/ICX/SPR ...): native 512-bit FMA.
        - Unknown AVX-512 vendor: assume native == arch (conservative: no
          double-pump modelled).
    """
    if arch_bits <= 256:
        return arch_bits
    vendor, family = _cpu_vendor_family()
    if vendor == "AuthenticAMD":
        return 256 if family <= 0x19 else 512
    if vendor == "GenuineIntel":
        return 512
    return arch_bits


def detect_handler(cc="cc"):
    """Pick the finest ArchHandler the host supports → (triplet, handler).

    Finer than the compiler's triple-only inference (ArchHandlerRegistry.cpp's
    pickHandlerForTriple always returns the x86_64 family default
    x86-64-avx2) — probing is precisely where avx512-vs-avx2 should be resolved
    from real CPU flags.
    """
    triplet = _host_triplet(cc)
    arch = triplet.split("-", 1)[0]
    if arch in ("aarch64", "arm64"):
        return triplet, "arm-neon"
    if arch in ("x86_64", "amd64"):
        flags = _cpu_flags()
        if "avx512f" in flags:
            return triplet, "x86-64-avx512"
        if "avx2" in flags:
            return triplet, "x86-64-avx2"
    return triplet, "generic"


def arch_registers_for(handler, triplet):
    """Return (arch_dict, registers_dict) for a handler name."""
    vw, gp, fp, vec, pred, reload_cy, store_cy = \
        ARCH_HANDLERS.get(handler, ARCH_HANDLERS["generic"])
    arch = {
        "triplet": triplet,
        "handler": handler,
        "vector_width_bits": vw,
        # vector_bits_arch = widest ISA-encodable vector; vector_bits_native =
        # throughput-effective FP datapath (narrower on double-pumping parts like
        # Zen4).  MachineModel reads THESE (not vector_width_bits) to drive
        # register-block VL and to set hasExplicitVectorModel; emitting them is
        # what makes the probed machine's vector model actually take effect.
        "vector_bits_arch": vw,
        "vector_bits_native": native_vector_bits(vw),
        "spill_strategy": DEFAULT_SPILL_STRATEGY,
        "weights": dict(DEFAULT_WEIGHTS),
    }
    registers = {
        "gp_budget": gp,
        "fp_budget": fp,
        "vec_budget": vec,
        "pred_budget": pred,
        "spill_reload_cycles": reload_cy,
        "spill_store_cycles": store_cy,
    }
    return arch, registers


# Preset name -> (handler, triplet) for the non-probing --target path.
PRESET_ARCH = {
    "generic": ("generic", "x86_64-unknown-linux-gnu"),
    "skylake": ("x86-64-avx2", "x86_64-unknown-linux-gnu"),
}

# ---------- MLIR op → x86-64 assembly for llvm-mca ----------

MLIR_TO_X86 = {
    "arith.constant": None,
    "arith.addi":       "addl %eax, %ebx",
    "arith.addf":       "vaddss %xmm0, %xmm1, %xmm2",
    "arith.subi":       "subl %eax, %ebx",
    "arith.subf":       "vsubss %xmm0, %xmm1, %xmm2",
    "arith.xori":       "xorl %eax, %ebx",
    "arith.andi":       "andl %eax, %ebx",
    "arith.ori":        "orl %eax, %ebx",
    "arith.shrsi":      "sarl %cl, %eax",
    "arith.shrui":      "shrl %cl, %eax",
    "arith.shli":       "shll %cl, %eax",
    "arith.select":     "cmovnel %eax, %ebx",
    "arith.cmpi":       "cmpl %eax, %ebx",
    "arith.cmpf":       "vucomiss %xmm0, %xmm1",
    "arith.muli":       "imull %eax, %ebx",
    "arith.mulf":       "vmulss %xmm0, %xmm1, %xmm2",
    "arith.divsi":      "idivl %ecx",
    "arith.divui":      "divl %ecx",
    "arith.divf":       "vdivss %xmm0, %xmm1, %xmm2",
    "arith.remsi":      "idivl %ecx",
    "arith.remui":      "divl %ecx",
    "arith.remf":       "vdivss %xmm0, %xmm1, %xmm2",
    "math.sqrt":        "vsqrtss %xmm0, %xmm1, %xmm2",
    "math.exp":         None,
    "math.log":         None,
    "math.sin":         None,
    "math.cos":         None,
    "math.tanh":        None,
    "math.powf":        None,
    "arith.sitofp":     "vcvtsi2ssl %eax, %xmm0, %xmm1",
    "arith.fptosi":     "vcvttss2si %xmm0, %eax",
    "arith.extsi":      "movslq %eax, %rax",
    "arith.extui":      "movzbl %al, %eax",
    "arith.trunci":     "movl %eax, %eax",
    "arith.index_cast": "movslq %eax, %rax",
    "arith.bitcast":    "vmovd %xmm0, %eax",
}

# Transcendentals have no single x86 instruction; estimate as multiples of sqrt.
TRANSCENDENTAL_SQRT_MULT = {
    "math.exp":  1.3,
    "math.log":  1.3,
    "math.sin":  3.5,
    "math.cos":  3.5,
    "math.tanh": 2.7,
    "math.powf": 3.5,
}


# ---------- llvm-mca probing ----------

def probe_op_latencies(llvm_mca, mcpu="native"):
    """Run llvm-mca for each MLIR op and return {op_name: latency_cycles}."""
    ops = {}
    sqrt_lat = None

    for mlir_op, asm in MLIR_TO_X86.items():
        if asm is None:
            if mlir_op == "arith.constant":
                ops[mlir_op] = 0
            continue
        lat = _mca_latency(llvm_mca, mcpu, asm)
        if lat is not None:
            ops[mlir_op] = lat
            if mlir_op == "math.sqrt":
                sqrt_lat = lat

    # Estimate transcendentals relative to sqrt.
    if sqrt_lat is not None:
        for op, mult in TRANSCENDENTAL_SQRT_MULT.items():
            if op not in ops:
                ops[op] = max(1, round(sqrt_lat * mult))

    return ops


def _mca_latency(llvm_mca, mcpu, asm):
    """Return the scheduling-model latency for a single instruction."""
    try:
        r = subprocess.run(
            [llvm_mca, f"--mcpu={mcpu}"],
            input=asm, capture_output=True, text=True, timeout=10,
        )
        in_table = False
        for line in r.stdout.splitlines():
            if "[1]" in line and "[2]" in line and "Instructions:" in line:
                in_table = True
                continue
            if in_table and line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        return int(parts[1])
                    except ValueError:
                        pass
                break
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        print(f"warning: llvm-mca failed: {e}", file=sys.stderr)
    return None


# ---------- cache latency probing ----------

def read_cache_sizes():
    """Read cache sizes from sysfs. Returns {level: size_bytes}."""
    result = {}
    base = "/sys/devices/system/cpu/cpu0/cache"
    try:
        for idx in sorted(os.listdir(base)):
            idx_path = os.path.join(base, idx)
            if not os.path.isdir(idx_path):
                continue
            try:
                ctype = _read_sysfs(os.path.join(idx_path, "type"))
                level = int(_read_sysfs(os.path.join(idx_path, "level")))
                size_str = _read_sysfs(os.path.join(idx_path, "size"))
            except (FileNotFoundError, ValueError):
                continue
            if ctype not in ("Data", "Unified"):
                continue
            if size_str.endswith("K"):
                size = int(size_str[:-1]) * 1024
            elif size_str.endswith("M"):
                size = int(size_str[:-1]) * 1024 * 1024
            else:
                size = int(size_str)
            result[level] = size
    except FileNotFoundError:
        pass
    return result


def _read_sysfs(path):
    with open(path) as f:
        return f.read().strip()


def get_cpu_freq_ghz():
    """Get CPU base frequency in GHz."""
    for path in [
        "/sys/devices/system/cpu/cpu0/cpufreq/base_frequency",
        "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq",
    ]:
        try:
            khz = int(_read_sysfs(path))
            return khz / 1e6
        except (FileNotFoundError, ValueError):
            continue
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("cpu MHz"):
                    return float(line.split(":")[1].strip()) / 1000.0
    except (FileNotFoundError, ValueError):
        pass
    return 4.0  # fallback


def probe_cache_latencies(cc="cc"):
    """Measure cache latencies. Returns dict with sizes and latencies in cycles."""
    cache_sizes = read_cache_sizes()
    freq_ghz = get_cpu_freq_ghz()

    measurements = _run_lmbench()
    if not measurements:
        measurements = _run_builtin_bench(cc)
    if not measurements:
        print("warning: no cache measurements, using defaults", file=sys.stderr)
        return {}

    return _classify(measurements, cache_sizes, freq_ghz)


def _find_lat_mem_rd():
    """Find lmbench's lat_mem_rd binary."""
    found = shutil.which("lat_mem_rd")
    if found:
        return found
    # Ubuntu puts lmbench binaries under /usr/lib/lmbench/bin/<arch>/.
    import glob
    for path in glob.glob("/usr/lib/lmbench/bin/*/lat_mem_rd"):
        if os.access(path, os.X_OK):
            return path
    return None


def _run_lmbench():
    """Try lmbench lat_mem_rd. Returns [(size_bytes, latency_ns)]."""
    lat_mem_rd = _find_lat_mem_rd()
    if not lat_mem_rd:
        return []
    try:
        print("Running lmbench lat_mem_rd (this takes ~60s)...", file=sys.stderr)
        r = subprocess.run(
            [lat_mem_rd, "512", "64"],
            capture_output=True, text=True, timeout=300,
        )
        # lmbench writes data to stderr in some builds.
        text = r.stdout or r.stderr
        pairs = _parse_size_lat_pairs(text, size_unit_mb=True)
        # Detect broken timer — common in Docker containers.
        if not pairs:
            return []
        lats = [lat for _, lat in pairs if lat > 0]
        if not lats or max(lats) < 0.01:
            print("warning: lmbench returned all-zero latencies "
                  "(timer issue), falling back", file=sys.stderr)
            return []
        # A real cache hierarchy shows >= 10× between smallest and largest
        # latency.  If not, the data is unreliable.
        if max(lats) < 10 * min(lats):
            print("warning: lmbench shows no cache differentiation "
                  "(likely Docker timer issue), falling back",
                  file=sys.stderr)
            return []
        return pairs
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        print(f"warning: lmbench failed: {e}", file=sys.stderr)
        return []


def _run_builtin_bench(cc):
    """Compile and run the pointer-chasing benchmark."""
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "cache_latency_bench.c")
    if not os.path.exists(src):
        print(f"warning: {src} not found", file=sys.stderr)
        return []
    with tempfile.TemporaryDirectory() as tmpdir:
        binary = os.path.join(tmpdir, "cache_latency_bench")
        try:
            subprocess.run(
                [cc, "-O2", "-o", binary, src],
                check=True, capture_output=True, text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"warning: failed to compile benchmark: {e}", file=sys.stderr)
            return []
        try:
            print("Running pointer-chasing benchmark (this takes ~60s)...",
                  file=sys.stderr)
            r = subprocess.run(
                [binary, "512"],
                capture_output=True, text=True, timeout=300,
            )
            return _parse_size_lat_pairs(r.stdout, size_unit_mb=False)
        except (subprocess.TimeoutExpired, OSError) as e:
            print(f"warning: benchmark failed: {e}", file=sys.stderr)
            return []


def _parse_size_lat_pairs(text, size_unit_mb):
    """Parse 'size latency_ns' lines."""
    pairs = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith('"'):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            size_val = float(parts[0])
            lat_ns = float(parts[1])
            if size_unit_mb:
                size_bytes = int(size_val * 1024 * 1024)
            else:
                size_bytes = int(size_val)
            pairs.append((size_bytes, lat_ns))
        except ValueError:
            continue
    return pairs


def _classify(measurements, cache_sizes, freq_ghz):
    """Extract per-level latencies from measurements and cache sizes."""
    l1 = cache_sizes.get(1, 32768)
    l2 = cache_sizes.get(2, 262144)
    l3 = cache_sizes.get(3, 33554432)

    def _lat_near(target):
        best, best_diff = None, float("inf")
        for sz, lat in measurements:
            d = abs(sz - target)
            if d < best_diff:
                best, best_diff = lat, d
        return best

    def _ns2cy(ns):
        return max(1, round(ns * freq_ghz))

    # Sample well within each level's range.
    l1_ns = _lat_near(l1 // 2)
    l2_ns = _lat_near(l1 * 2)
    l3_ns = _lat_near(min(l2 * 4, l3 // 2))
    mem_ns = _lat_near(l3 * 2)

    # If we couldn't get a measurement past L3, try the largest we have.
    if mem_ns is None and measurements:
        mem_ns = max(measurements, key=lambda p: p[0])[1]

    result = {}
    labels = [
        ("l1_size", l1),
        ("l2_size", l2),
        ("l3_size", l3),
    ]
    for k, v in labels:
        result[k] = v

    latencies = [
        ("l1_latency", l1_ns, f"L1d ({l1 // 1024}K)"),
        ("l2_latency", l2_ns, f"L2 ({l2 // 1024}K)"),
        ("l3_latency", l3_ns, f"L3 ({l3 // 1024}K)"),
        ("mem_latency", mem_ns, "DRAM"),
    ]
    print(f"\nCache hierarchy (freq {freq_ghz:.2f} GHz):", file=sys.stderr)
    for key, ns, label in latencies:
        if ns is not None:
            cy = _ns2cy(ns)
            result[key] = cy
            print(f"  {label:16s}  {ns:7.1f} ns  = {cy:4d} cycles",
                  file=sys.stderr)
        else:
            print(f"  {label:16s}  (no data)", file=sys.stderr)
    result["freq_ghz"] = round(freq_ghz, 2)
    print(file=sys.stderr)
    return result


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a CPU cost model JSON for drcompiler")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output file (default: stdout)")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--target", type=str, default=None,
                       choices=sorted(PRESETS.keys()),
                       help="Target microarchitecture preset")
    group.add_argument("--probe", action="store_true",
                       help="Probe the current hardware using llvm-mca "
                            "and a cache latency benchmark")

    parser.add_argument("--llvm-mca", type=str, default=None,
                        metavar="PATH",
                        help="Path to llvm-mca (default: search PATH)")
    parser.add_argument("--mcpu", type=str, default="native",
                        help="Target CPU for llvm-mca (default: native)")
    parser.add_argument("--cc", type=str, default="cc",
                        help="C compiler for building cache benchmark")
    args = parser.parse_args()

    if args.probe:
        model = _do_probe(args)
    else:
        target = args.target or "generic"
        model = dict(PRESETS[target])  # shallow copy: don't mutate the preset
        # Attach the matching arch/register model so preset JSON is
        # schema-complete (parser expects optional arch + registers blocks).
        handler, triplet = PRESET_ARCH.get(target, PRESET_ARCH["generic"])
        model["arch"], model["registers"] = arch_registers_for(handler, triplet)

    text = json.dumps(model, indent=2) + "\n"

    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(text)


def _do_probe(args):
    """Hardware probing: llvm-mca for ALU costs, benchmark for cache."""
    # Resolve llvm-mca.
    mca = args.llvm_mca or shutil.which("llvm-mca")
    if not mca:
        print("warning: llvm-mca not found, using generic ALU costs",
              file=sys.stderr)
        ops = dict(GENERIC["ops"])
    else:
        print(f"Probing ALU latencies via {mca} (--mcpu={args.mcpu})...",
              file=sys.stderr)
        ops = probe_op_latencies(mca, args.mcpu)
        # Fill in any ops we didn't get from llvm-mca.
        for op, cost in GENERIC["ops"].items():
            if op not in ops:
                ops[op] = cost

    model = {
        "default_cost": GENERIC["default_cost"],
        "ops": dict(sorted(ops.items())),
    }

    cache = probe_cache_latencies(args.cc)
    if cache:
        model["cache"] = cache

    triplet, handler = detect_handler(args.cc)
    print(f"Arch handler: {handler}  ({triplet})", file=sys.stderr)
    model["arch"], model["registers"] = arch_registers_for(handler, triplet)

    return model


if __name__ == "__main__":
    main()
