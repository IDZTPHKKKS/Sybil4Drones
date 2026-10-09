#!/usr/bin/env python3
"""95% bootstrap intervals over maps for every condition; writes results/intervals.md."""
import glob
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = 5000


def ci(x, stat):
    rng = np.random.default_rng(0)
    x = np.asarray(x, float)
    b = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(B)]
    return stat(x), *np.percentile(b, [2.5, 97.5])


def main():
    lines = ["# 95% bootstrap intervals over maps", "",
             "Explored space: mean over maps (%). False cells: median over maps of the per-run value. "
             f"{B} resamples of the maps.", ""]
    for exp in ("sybil", "storey", "bind", "teamsize"):
        runs = defaultdict(list)
        for f in glob.glob(os.path.join(ROOT, "results", exp, "*.json")):
            r = json.load(open(f))
            if r.get("rep", 0) == 0:
                key = r["condition"] if exp != "storey" else f"{r['family']} {r['condition']}"
                runs[key].append(r)
        lines += [f"## {exp}", "", "| condition | maps | explored [95% CI] | false cells [95% CI] |", "|---|---|---|---|"]
        for k in sorted(runs):
            rs = runs[k]
            e = ci([100 * r["coverage_honest"][-1] for r in rs], np.mean)
            w = ci([r["wrong"][-1] for r in rs], np.median)
            lines.append(f"| {k} | {len(rs)} | {e[0]:.1f} [{e[1]:.1f}, {e[2]:.1f}] | {w[0]:.0f} [{w[1]:.0f}, {w[2]:.0f}] |")
        lines.append("")
    open(os.path.join(ROOT, "results", "intervals.md"), "w").write("\n".join(lines))


if __name__ == "__main__":
    main()
