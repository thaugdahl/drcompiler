#!/usr/bin/env python3
"""Evaluate REGISTER_PRESSURE_PLAN.md §8.2 hypotheses from polybench-bench CSV.

Hypotheses:
  H1 drcomp avoids regressions
       count kernels where upstream-fuse < none (upstream over-fused).
       For those, verify drcomp-fuse matches or beats none.
  H2 drcomp preserves wins
       count kernels where upstream-fuse > none (legitimate fusion wins).
       For those, verify drcomp-fuse matches upstream-fuse.
  H3 predicted spills correlate with measured spills
       requires per-kernel pairs (predicted_cycles, measured_spills) — use
       scripts/calibrate_regpress.py output for this; this script just
       reports if a CSV is supplied.
  H4 weighted-sum beats hard-constraint baseline
       compare drcomp-fuse against an ablation where regCost>0 -> reject.
       Provide the ablation CSV column via --ablation-csv.

Usage:
  ./polybench_summary.py --csv polybench.csv \
      [--regpress-csv calibrate_regpress.csv] \
      [--ablation-csv drcomp-fuse-hard.csv]
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

NOISE_FLOOR_REL = 0.02  # 2% delta = noise on most setups


def load_csv(path: Path) -> Dict[str, Dict[str, float]]:
    """Return {kernel: {config: median_time}}."""
    out: Dict[str, Dict[str, float]] = {}
    with open(path) as fp:
        reader = csv.DictReader(fp)
        median_cols = [c for c in reader.fieldnames if c.endswith("_median")]
        for row in reader:
            kname = row["kernel"]
            out[kname] = {}
            for col in median_cols:
                cfg = col[: -len("_median")]
                v = row[col]
                if v in ("FAIL", "ERR", ""):
                    continue
                try:
                    out[kname][cfg] = float(v)
                except ValueError:
                    pass
    return out


def rel_diff(ref: float, val: float) -> float:
    if ref <= 0:
        return float("nan")
    return (val - ref) / ref


def h1_no_regressions(data: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    """For each kernel where upstream-fuse was a regression vs none,
    label drcomp-fuse as 'recovered' / 'still regressed' / 'matches upstream'.
    """
    out: Dict[str, str] = {}
    for kname, runs in data.items():
        none = runs.get("none")
        upstream = runs.get("upstream-fuse")
        drcomp = runs.get("drcomp-fuse")
        if none is None or upstream is None or drcomp is None:
            continue
        upstream_delta = rel_diff(none, upstream)
        if upstream_delta <= NOISE_FLOOR_REL:
            continue  # upstream did not regress vs none
        drcomp_delta = rel_diff(none, drcomp)
        if drcomp_delta <= NOISE_FLOOR_REL:
            out[kname] = "recovered"
        elif drcomp_delta < upstream_delta:
            out[kname] = "improved"
        else:
            out[kname] = "still-regressed"
    return out


def h2_preserves_wins(data: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    """For each kernel where upstream-fuse was a legit win vs none, label
    drcomp-fuse as 'preserved' / 'lost'."""
    out: Dict[str, str] = {}
    for kname, runs in data.items():
        none = runs.get("none")
        upstream = runs.get("upstream-fuse")
        drcomp = runs.get("drcomp-fuse")
        if none is None or upstream is None or drcomp is None:
            continue
        upstream_delta = rel_diff(none, upstream)
        if upstream_delta >= -NOISE_FLOOR_REL:
            continue  # upstream did not win
        drcomp_delta = rel_diff(none, drcomp)
        if drcomp_delta <= upstream_delta + NOISE_FLOOR_REL:
            out[kname] = "preserved"
        else:
            out[kname] = "lost"
    return out


def h3_pearson(reg_csv: Path) -> Optional[Tuple[float, int]]:
    """Pearson r between predicted_cycles and measured spills, read from
    scripts/calibrate_regpress.py CSV output."""
    xs: List[float] = []
    ys: List[float] = []
    with open(reg_csv) as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            try:
                xs.append(float(row["predicted_cycles"]))
                ys.append(float(row["measured_reloads"] or row["measured_spills"]))
            except (KeyError, ValueError):
                continue
    if len(xs) < 2:
        return None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return (cov / (sx * sy), n)


def h4_compare_ablation(data: Dict[str, Dict[str, float]],
                         ablation_csv: Optional[Path]) -> Dict[str, str]:
    """drcomp-fuse vs a 'hard-constraint' ablation (regCost>0 -> reject).
    Reads the ablation CSV's `_median` for drcomp-fuse-hard."""
    if ablation_csv is None:
        return {}
    ablation = load_csv(ablation_csv)
    out: Dict[str, str] = {}
    for kname, runs in data.items():
        drcomp = runs.get("drcomp-fuse")
        hard = ablation.get(kname, {}).get("drcomp-fuse")
        if drcomp is None or hard is None:
            continue
        delta = rel_diff(hard, drcomp)
        if delta <= -NOISE_FLOOR_REL:
            out[kname] = "weighted-wins"
        elif delta >= NOISE_FLOOR_REL:
            out[kname] = "hard-wins"
        else:
            out[kname] = "tie"
    return out


def fmt_table(title: str, verdicts: Dict[str, str]) -> None:
    print()
    print(f"=== {title} ===")
    if not verdicts:
        print("  (no applicable kernels)")
        return
    counts: Dict[str, int] = {}
    for v in verdicts.values():
        counts[v] = counts.get(v, 0) + 1
    print(f"{'kernel':24s} {'verdict':20s}")
    for k in sorted(verdicts):
        print(f"{k:24s} {verdicts[k]:20s}")
    print("--")
    for verdict, n in sorted(counts.items()):
        print(f"  {verdict:24s} = {n}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True,
                   help="polybench-bench.sh output CSV (with `none`, "
                        "`upstream-fuse`, `drcomp-fuse` columns)")
    p.add_argument("--regpress-csv", type=Path, default=None,
                   help="scripts/calibrate_regpress.py output CSV for H3")
    p.add_argument("--ablation-csv", type=Path, default=None,
                   help="polybench-bench CSV for the hard-constraint ablation "
                        "(H4)")
    args = p.parse_args()

    data = load_csv(args.csv)

    h1 = h1_no_regressions(data)
    fmt_table("H1 — drcomp avoids regressions where upstream over-fused", h1)

    h2 = h2_preserves_wins(data)
    fmt_table("H2 — drcomp preserves upstream's legitimate wins", h2)

    if args.regpress_csv:
        r = h3_pearson(args.regpress_csv)
        print()
        print("=== H3 — Pearson r (predicted vs measured spills) ===")
        if r is None:
            print("  (insufficient data)")
        else:
            corr, n = r
            verdict = "PASS" if corr > 0.85 else "FAIL"
            print(f"  r = {corr:.4f}  (n={n})  gate r>0.85 -> {verdict}")

    h4 = h4_compare_ablation(data, args.ablation_csv)
    fmt_table("H4 — weighted-sum vs hard-constraint ablation", h4)

    return 0


if __name__ == "__main__":
    sys.exit(main())
