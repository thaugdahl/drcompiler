#!/usr/bin/env python3
"""Calibrate RegisterPressureAnalysis predictions against LLVM regalloc.

For each MLIR program in the calibration set:

  1. Run `dr-opt --print-register-pressure` to extract our predicted spill
     cycle count (and per-class peak live count).
  2. Lower the program through `mlir-opt` to standard LLVM dialect, then
     `mlir-translate --mlir-to-llvmir` to LLVM IR, then `llc -O3 -stats` to
     get LLVM's spill statistics.
  3. Pair (predicted, measured) values, compute Pearson r per spill strategy
     and report.

Phase 2 exit gate (REGISTER_PRESSURE_PLAN.md §2): Pearson r > 0.85 on the
calibration set, otherwise the approach is suspect and we pivot the thesis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DROPT = os.environ.get("DROPT", "build-marco/tools/dr-opt/dr-opt")
MLIR_OPT = os.environ.get("MLIR_OPT", "mlir-opt")
MLIR_TRANSLATE = os.environ.get("MLIR_TRANSLATE", "mlir-translate")
LLC = os.environ.get("LLC", "llc")

# llc -stats prints lines like "  12 regalloc                - Number of spilled live ranges"
# Used iff llc is stats-enabled (LLVM_FORCE_ENABLE_STATS=ON).
SPILL_RX = re.compile(
    r"^\s*(\d+)\s+regalloc\s+-\s+Number of spilled live ranges\s*$",
    re.MULTILINE,
)
RELOAD_RX = re.compile(
    r"^\s*(\d+)\s+regalloc\s+-\s+Number of reloads inserted\s*$",
    re.MULTILINE,
)
# Asm-side spill detection: mov{,l,q,sd,...} touching N(%rsp) inside the
# function body where N < stack-frame size (so excluding caller's arg slots).
PROLOGUE_RX = re.compile(r"^\s*subq?\s+\$(\d+),\s*%rsp\s*$", re.MULTILINE)
RSP_MEMOP_RX = re.compile(r"(-?\d+)\(%rsp\)")
ASM_FN_END_RX = re.compile(r"^\s*\.cfi_endproc\s*$", re.MULTILINE)
PRED_RX = re.compile(
    r"register-pressure: (?P<fn>\S+) strategy=(?P<strat>\S+) "
    r"peak=\(gp=(?P<gp>\d+),fp=(?P<fp>\d+),vec=(?P<vec>\d+),pred=(?P<pred>\d+)\) "
    r"excess=\([^)]*\) spill_cycles=(?P<cyc>\d+)"
)


@dataclass
class PerFunction:
    fn: str
    pred_peak: Dict[str, int] = field(default_factory=dict)
    pred_spill_cycles: int = 0


@dataclass
class Measurement:
    program: str
    config: str
    predicted_cycles: int
    predicted_peak: Dict[str, int]
    measured_spills: int
    measured_reloads: int


def run(cmd: List[str], stdin: Optional[bytes] = None, env=None) -> bytes:
    proc = subprocess.run(
        cmd, input=stdin, capture_output=True, check=False, env=env
    )
    if proc.returncode != 0:
        sys.stderr.write(
            f"error: {' '.join(cmd)} failed (rc={proc.returncode})\n"
            f"  stderr: {proc.stderr.decode(errors='replace')[:1000]}\n"
        )
        raise SystemExit(1)
    return proc.stdout


def parse_predicted(stderr_text: str) -> List[PerFunction]:
    out: Dict[str, PerFunction] = {}
    for m in PRED_RX.finditer(stderr_text):
        rec = out.setdefault(m["fn"], PerFunction(fn=m["fn"]))
        rec.pred_peak = {
            "gp": int(m["gp"]),
            "fp": int(m["fp"]),
            "vec": int(m["vec"]),
            "pred": int(m["pred"]),
        }
        rec.pred_spill_cycles = int(m["cyc"])
    return list(out.values())


def predict(program: Path, costs_json: Optional[Path], strategy: str,
            trip: int) -> List[PerFunction]:
    pipeline = (
        f"builtin.module(print-register-pressure{{"
        f"strategy={strategy} trip-count={trip}"
        f"{f' cpu-cost-model-file={costs_json}' if costs_json else ''}}})"
    )
    cmd = [DROPT, str(program), f"--pass-pipeline={pipeline}"]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        sys.stderr.write(
            f"predict failed for {program}\nstderr: {proc.stderr.decode(errors='replace')[:1000]}\n"
        )
        raise SystemExit(1)
    return parse_predicted(proc.stderr.decode(errors="replace"))


def lower_to_ll(program: Path, tmp: Path) -> Path:
    """Lower MLIR -> LLVM IR.  Pipeline must match the project's standard
    affine -> scf -> cf -> llvm chain.  Returns the .ll path."""
    # Stage 1: dr-opt with our passes (raise-malloc + data-recomputation off).
    dropt_out = tmp / "stage1.mlir"
    run([DROPT, str(program), "--pass-pipeline=builtin.module(raise-malloc-to-memref)",
         "-o", str(dropt_out)])

    # Stage 2: mlir-opt lower-everything-to-LLVM-dialect.
    mlir_pipeline = (
        "builtin.module("
        "lower-affine,"
        "convert-scf-to-cf,"
        "finalize-memref-to-llvm,"
        "convert-cf-to-llvm,"
        "convert-arith-to-llvm,"
        "convert-vector-to-llvm,"
        "convert-func-to-llvm,"
        "reconcile-unrealized-casts)"
    )
    stage2 = tmp / "stage2.mlir"
    run([MLIR_OPT, str(dropt_out), f"--pass-pipeline={mlir_pipeline}",
         "-o", str(stage2)])

    # Stage 3: mlir-translate -> LLVM IR.
    stage3 = tmp / "stage3.ll"
    run([MLIR_TRANSLATE, "--mlir-to-llvmir", str(stage2), "-o", str(stage3)])
    return stage3


def measure_spills_via_stats(ll: Path, target_triple: str,
                              mcpu: str) -> Tuple[int, int, bool]:
    """Try llc -stats first (requires LLVM_FORCE_ENABLE_STATS).  Returns
    (spilled_live_ranges, reloads_inserted, used_stats)."""
    cmd = [LLC, "-O3", "-stats", "-march=x86-64",
           f"-mtriple={target_triple}", f"-mcpu={mcpu}", str(ll), "-o", os.devnull]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    stats = proc.stderr.decode(errors="replace") + proc.stdout.decode(errors="replace")
    s = SPILL_RX.search(stats)
    r = RELOAD_RX.search(stats)
    if s or r:
        return (int(s.group(1)) if s else 0, int(r.group(1)) if r else 0, True)
    return (0, 0, False)


def measure_spills_via_asm(ll: Path, target_triple: str,
                            mcpu: str) -> Tuple[int, int]:
    """Fallback: lower to asm and count mov-rsp instructions whose offset
    is *inside* the function's stack frame.  Returns (stores, reloads)."""
    cmd = [LLC, "-O3", "-march=x86-64",
           f"-mtriple={target_triple}", f"-mcpu={mcpu}", str(ll), "-o", "-"]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        return (0, 0)
    asm = proc.stdout.decode(errors="replace")
    stores = 0
    reloads = 0
    for fn_chunk in re.split(r"\n\s*\.globl\s+|\n\s*\.type\s+.*?,\s*@function\s*\n",
                              asm):
        m = PROLOGUE_RX.search(fn_chunk)
        if not m:
            continue
        stk = int(m.group(1))
        # Anything above the frame is a caller arg slot; only count
        # 0 <= offset < stk as spill slots.
        for line in fn_chunk.splitlines():
            # Skip the prologue/epilogue line itself.
            if "%rsp" not in line:
                continue
            for off in RSP_MEMOP_RX.findall(line):
                offv = int(off)
                if 0 <= offv < stk:
                    # Determine direction by which operand has the memref.
                    stripped = line.strip()
                    # mov x, off(%rsp)  -> store ; mov off(%rsp), x -> reload
                    # Heuristic: position of '(%rsp)' vs ','.
                    if stripped.find("(%rsp)") < stripped.find(","):
                        reloads += 1
                    else:
                        stores += 1
    return (stores, reloads)


def measure_spills(ll: Path, target_triple: str,
                    mcpu: str) -> Tuple[int, int]:
    s, r, ok = measure_spills_via_stats(ll, target_triple, mcpu)
    if ok:
        return (s, r)
    return measure_spills_via_asm(ll, target_triple, mcpu)


def pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return float("nan")
    return cov / (sx * sy)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--programs", required=True,
                   help="directory containing .mlir calibration programs")
    p.add_argument("--costs", default=None,
                   help="cost-model JSON to use for prediction (omit for built-in)")
    p.add_argument("--strategy", default="excess-hot",
                   choices=["excess-hot", "sum-excess", "graph-color"])
    p.add_argument("--trip-count", type=int, default=128,
                   help="trip count multiplier (ExcessHot)")
    p.add_argument("--triple", default="x86_64-unknown-linux-gnu")
    p.add_argument("--mcpu", default="haswell")
    p.add_argument("--csv", default=None, help="write per-program CSV here")
    p.add_argument("--measure", choices=["spills", "reloads"], default="reloads",
                   help="measurement metric to correlate against predicted cycles")
    args = p.parse_args()

    progs = sorted(Path(args.programs).rglob("*.mlir"))
    if not progs:
        sys.stderr.write(f"no .mlir programs under {args.programs}\n")
        return 1

    measurements: List[Measurement] = []
    skipped = 0

    for prog in progs:
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                ll = lower_to_ll(prog, tmp)
                spills, reloads = measure_spills(ll, args.triple, args.mcpu)

            preds = predict(prog,
                            Path(args.costs) if args.costs else None,
                            args.strategy, args.trip_count)
            total_cycles = sum(r.pred_spill_cycles for r in preds)
            agg_peak = {k: max((r.pred_peak.get(k, 0) for r in preds),
                               default=0)
                         for k in ("gp", "fp", "vec", "pred")}
            measurements.append(Measurement(
                program=prog.name, config=args.strategy,
                predicted_cycles=total_cycles,
                predicted_peak=agg_peak,
                measured_spills=spills, measured_reloads=reloads,
            ))
        except SystemExit:
            sys.stderr.write(f"  skipping {prog} (pipeline failure)\n")
            skipped += 1

    if not measurements:
        sys.stderr.write("no measurements collected\n")
        return 1

    xs = [m.predicted_cycles for m in measurements]
    ys = [m.measured_spills if args.measure == "spills" else m.measured_reloads
          for m in measurements]
    r = pearson([float(x) for x in xs], [float(y) for y in ys])

    print(f"# calibration: {len(measurements)} programs, strategy={args.strategy}, "
          f"measure={args.measure}")
    print(f"# Pearson r = {r:.4f}")
    print(f"# (gate: r > 0.85)  {'PASS' if r > 0.85 else 'FAIL'}")
    if skipped:
        print(f"# skipped {skipped} programs (lowering failure)")
    print()

    if args.csv:
        with open(args.csv, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["program", "strategy", "predicted_cycles",
                        "peak_gp", "peak_fp", "peak_vec", "peak_pred",
                        "measured_spills", "measured_reloads"])
            for m in measurements:
                w.writerow([m.program, m.config, m.predicted_cycles,
                            m.predicted_peak["gp"], m.predicted_peak["fp"],
                            m.predicted_peak["vec"], m.predicted_peak["pred"],
                            m.measured_spills, m.measured_reloads])
        print(f"# CSV written: {args.csv}")

    # Inline mini table.
    print(f"{'program':40} {'pred_cycles':>12} {'spills':>8} {'reloads':>8}")
    for m in measurements:
        print(f"{m.program:40} {m.predicted_cycles:>12} "
              f"{m.measured_spills:>8} {m.measured_reloads:>8}")

    return 0 if not math.isnan(r) and r > 0.85 else 2


if __name__ == "__main__":
    sys.exit(main())
