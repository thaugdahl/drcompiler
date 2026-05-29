#!/usr/bin/env python3
"""Nelder-Mead calibration of the unified cost model's (alpha_mem, beta_reg,
gamma_alu) weights against measured PolyBench runtime.

For each candidate (alpha, beta, gamma):
  1. Patch a base costs.json with the candidate weights.
  2. Run `polybench-bench.sh` against the calibration kernel subset, drcomp
     fusion + tiling enabled.
  3. Sum (or geomean) the measured median times across kernels.
  4. Return that as the objective to minimise.

Nelder-Mead via scipy minimises the objective over the 3-D weight space.
Output: best weights + their per-kernel times.

Per REGISTER_PRESSURE_PLAN.md §8.3, the calibration set must be DISJOINT
from the held-out evaluation set so we don't overfit.  This script accepts
two kernel filters: `--calibration` and `--holdout` (the latter is run with
the best weights at the end so the report shows both numbers).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCIPY_AVAILABLE = True
try:
    from scipy.optimize import minimize
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class CalibrationResult:
    weights: Tuple[float, float, float]
    objective: float
    per_kernel: Dict[str, float]


def patch_costs_json(base: Optional[Path], alpha: float, beta: float,
                     gamma: float, out: Path) -> None:
    """Read `base` (or {}) and write a new JSON with the patched weights."""
    if base and base.exists():
        with open(base) as fp:
            data = json.load(fp)
    else:
        data = {}
    data.setdefault("arch", {})
    data["arch"].setdefault("weights", {})
    data["arch"]["weights"]["alpha_mem"] = alpha
    data["arch"]["weights"]["beta_reg"] = beta
    data["arch"]["weights"]["gamma_alu"] = gamma
    with open(out, "w") as fp:
        json.dump(data, fp, indent=2)


def run_polybench(costs_json: Path, configs: str, dataset: str,
                  iters: int, kernel_filter: str,
                  polybench_dir: Path) -> Dict[str, float]:
    """Run polybench-bench.sh and parse the CSV; return {kernel: median} for
    the *last* config in the list (we calibrate against drcomp-tile-fuse by
    default)."""
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fp:
        csv_path = Path(fp.name)
    try:
        cmd = [
            "bash", str(Path(__file__).parent / "polybench-bench.sh"),
            "--polybench-dir", str(polybench_dir),
            "--iters", str(iters),
            "--dataset", dataset,
            "--configs", configs,
            "--cost-model", str(costs_json),
            "--csv", str(csv_path),
        ]
        if kernel_filter:
            cmd.extend(["--kernel", kernel_filter])
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr.decode(errors="replace")[-2000:])
            return {}
        last_config = configs.split(",")[-1]
        result: Dict[str, float] = {}
        with open(csv_path) as fp:
            reader = csv.DictReader(fp)
            col = f"{last_config}_median"
            for row in reader:
                v = row.get(col, "FAIL")
                if v in ("FAIL", "ERR", ""):
                    continue
                try:
                    result[row["kernel"]] = float(v)
                except ValueError:
                    continue
        return result
    finally:
        csv_path.unlink(missing_ok=True)


def objective_geomean(times: Dict[str, float]) -> float:
    """Geometric mean of per-kernel times.  Returns +inf if no measurements
    (so Nelder-Mead avoids degenerate candidate regions)."""
    if not times:
        return float("inf")
    logs = [math.log(t) for t in times.values() if t > 0]
    if not logs:
        return float("inf")
    return math.exp(sum(logs) / len(logs))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-costs", type=Path, default=None,
                   help="base costs.json to patch (defaults to empty)")
    p.add_argument("--polybench-dir", type=Path, required=True,
                   help="root of PolyBench/C source tree")
    p.add_argument("--configs", default="cgeist-base,drcomp-tile-fuse",
                   help="configs passed to polybench-bench.sh; the LAST one "
                        "is the calibration target")
    p.add_argument("--dataset", default="LARGE",
                   choices=["MINI", "SMALL", "STANDARD", "LARGE", "EXTRALARGE"])
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--calibration", default="",
                   help="kernel-name substring filter for calibration set")
    p.add_argument("--holdout", default="",
                   help="kernel-name substring filter for held-out evaluation")
    p.add_argument("--init", nargs=3, type=float, default=[1.0, 1.0, 1.0],
                   help="initial (alpha, beta, gamma) seeds")
    p.add_argument("--max-iter", type=int, default=30,
                   help="max Nelder-Mead iterations")
    p.add_argument("--csv", type=Path, default=None,
                   help="per-iteration log CSV path")
    args = p.parse_args()

    if not SCIPY_AVAILABLE:
        sys.stderr.write("scipy not available; pip install scipy\n")
        return 1

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as fp:
            csv.writer(fp).writerow(
                ["iter", "alpha_mem", "beta_reg", "gamma_alu", "objective",
                 "n_kernels"]
            )
    iteration = [0]

    def objective(x) -> float:
        alpha, beta, gamma = [max(0.01, v) for v in x]
        with tempfile.TemporaryDirectory() as td:
            costs = Path(td) / "costs.json"
            patch_costs_json(args.base_costs, alpha, beta, gamma, costs)
            per_kernel = run_polybench(costs, args.configs, args.dataset,
                                        args.iters, args.calibration,
                                        args.polybench_dir)
        score = objective_geomean(per_kernel)
        iteration[0] += 1
        sys.stderr.write(
            f"[iter {iteration[0]}] weights=({alpha:.3f}, {beta:.3f}, "
            f"{gamma:.3f}) -> geomean={score:.6f} "
            f"({len(per_kernel)} kernels)\n"
        )
        if args.csv:
            with open(args.csv, "a", newline="") as fp:
                csv.writer(fp).writerow(
                    [iteration[0], alpha, beta, gamma, score, len(per_kernel)]
                )
        return score

    result = minimize(
        objective, args.init, method="Nelder-Mead",
        options={"maxiter": args.max_iter, "xatol": 0.05, "fatol": 1e-4,
                 "disp": True},
    )
    best_alpha, best_beta, best_gamma = [max(0.01, v) for v in result.x]

    print()
    print("=" * 60)
    print(f"Best weights: alpha_mem={best_alpha:.4f}, beta_reg={best_beta:.4f}, "
          f"gamma_alu={best_gamma:.4f}")
    print(f"Calibration geomean: {result.fun:.6f}")
    print("=" * 60)

    if args.holdout:
        print()
        print(f"# Held-out evaluation set (filter={args.holdout!r}):")
        with tempfile.TemporaryDirectory() as td:
            costs = Path(td) / "costs.json"
            patch_costs_json(args.base_costs, best_alpha, best_beta,
                             best_gamma, costs)
            holdout_results = run_polybench(costs, args.configs, args.dataset,
                                             args.iters, args.holdout,
                                             args.polybench_dir)
        if holdout_results:
            print(f"{'kernel':24s} {'median (s)':>12s}")
            for k, v in sorted(holdout_results.items()):
                print(f"{k:24s} {v:12.6f}")
            print(f"\nHoldout geomean: {objective_geomean(holdout_results):.6f}")
        else:
            print("  (no measurements)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
