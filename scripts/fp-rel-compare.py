#!/usr/bin/env python3
"""Real floating-point output comparator for the verify harness.

Replaces the previous no-op awk gate in polybench-verify.sh (which set ok=1 on
ANY diff marker line and so ALWAYS passed -- a 1e6x error passed).  Extracts
paired numeric tokens from two output dumps and computes the max RELATIVE error;
exits 0 (PASS) iff max_rel_err <= tol, 1 (FAIL) if it exceeds tol, 2 (ERROR) on
token-count mismatch or a non-finite value (NaN/inf -> a real failure, never a
silent pass).

Usage: fp-rel-compare.py REF OUT [tol]   (default tol = 1e-6 relative)
"""
import math
import re
import sys

_NUM = re.compile(
    r'[-+]?(?:\d+\.\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?'
    r'|\d+[eE][-+]?\d+|\d+|inf|nan)', re.IGNORECASE)


def tokens(path):
    out = []
    with open(path, errors="replace") as fp:
        for line in fp:
            for t in _NUM.findall(line):
                try:
                    out.append(float(t))
                except ValueError:
                    pass
    return out


def main():
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-6
    a, b = tokens(sys.argv[1]), tokens(sys.argv[2])
    if not a or len(a) != len(b):
        print(f"ERROR token-count {len(a)} vs {len(b)}")
        return 2
    worst = 0.0
    for x, y in zip(a, b):
        if not (math.isfinite(x) and math.isfinite(y)):
            print("ERROR non-finite value (NaN/inf)")
            return 2
        den = max(abs(x), abs(y), 1e-300)
        worst = max(worst, abs(x - y) / den)
    print(f"max_rel_err={worst:.3e} tol={tol:.0e} n={len(a)}")
    return 0 if worst <= tol else 1


if __name__ == "__main__":
    sys.exit(main())
