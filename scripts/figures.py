#!/usr/bin/env python3
"""Paper figures for the Sybil study.

    python3 scripts/figures.py --seeds 0 1 2 3 4
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "media")
C = "coverage_honest"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.direction": "in",
    "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "lines.linewidth": 1.1,
    "lines.markersize": 3.5, "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

STYLE = {
    "none": dict(color="0.0", ls="-", marker="", label="no attacker"),
    "sybil_weak": dict(color="#b2182b", ls="--", marker="o", label="weak"),
    "sybil_strong": dict(color="#2166ac", ls="-", marker="s", label="strong"),
    "sybil_stealth": dict(color="0.45", ls="-.", marker="^", label="stealth"),
}


def load(exp, seeds):
    runs = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(RES, exp, "*.json"))):
        r = json.load(open(f))
        if r["seed"] in seeds and r.get("rep", 0) == 0:
            runs[r["condition"]].append(r)
    return runs


def series(r, key, grid):
    t, y = np.array(r["t"]), np.array(r[key], float)
    return np.interp(grid, t, y, right=y[-1])


def final(rs, key=C, scale=100):
    return np.array([r[key][-1] for r in rs]) * scale


def ci(v):
    return 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300)
    plt.close(fig)


def coverage_time(runs):
    grid = np.arange(0, 601, 2.0)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2), sharey=True)
    for ax, d, tag in ((axes[0], "D0", "(a) no defence"), (axes[1], "D3", "(b) defence D3")):
        for a in ("none", "sybil_weak", "sybil_strong", "sybil_stealth"):
            rs = runs.get(f"{a}-{d}" if a == "none" else f"{a}4-{d}")
            if not rs:
                continue
            M = np.array([series(r, C, grid) for r in rs]) * 100
            st = STYLE[a]
            ax.plot(grid, np.median(M, 0), color=st["color"], ls=st["ls"], label=st["label"])
            ax.fill_between(grid, *np.percentile(M, [25, 75], 0), color=st["color"], alpha=0.12, lw=0)
        ax.set_xlim(0, 600), ax.set_ylim(0, 101)
        ax.set_xlabel("time [s]")
        ax.set_title(tag, loc="left", fontsize=8)
    axes[0].set_ylabel("explored by honest drones [%]")
    axes[0].legend(loc="center right")
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_coverage_time")


def damage(runs):
    ks = np.array([1, 2, 4, 8])
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for j, a in enumerate(("sybil_weak", "sybil_strong", "sybil_stealth")):
        m, h = [], []
        for k in ks:
            v = 100 - final(runs.get(f"{a}{k}-D0", []))
            m.append(v.mean() if len(v) else np.nan)
            h.append(ci(v) if len(v) else np.nan)
        st = STYLE[a]
        x = ks * (1 + 0.04 * (j - 1))
        ax.errorbar(x, m, yerr=h, color=st["color"], ls=st["ls"], marker=st["marker"], capsize=1.5,
                    elinewidth=0.6, label=st["label"])
    base = 100 - final(runs.get("none-D0", []))
    if len(base):
        ax.axhline(base.mean(), color="0.0", lw=0.6, ls=":", label="no attacker")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks), ax.set_xticklabels([str(k) for k in ks]), ax.minorticks_off()
    ax.set_xlabel("fake identities per compromised drone")
    ax.set_ylabel("unexplored [%]")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "fig_damage")


def defences(runs):
    levels = ["D0", "D1", "D2", "D3", "D4"]
    have = [d for d in levels if any(runs.get(f"{a}4-{d}") for a in ("sybil_weak", "sybil_strong", "sybil_stealth"))]
    x = np.arange(len(have))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2))
    for j, a in enumerate(("sybil_weak", "sybil_strong", "sybil_stealth")):
        st = STYLE[a]
        cov, cov_h, wr, wr_lo, wr_hi = [], [], [], [], []
        for d in have:
            rs = runs.get(f"{a}4-{d}", [])
            v = final(rs)
            w = final(rs, "wrong", 1)
            cov.append(v.mean() if len(v) else np.nan)
            cov_h.append(ci(v) if len(v) else np.nan)
            wr.append(np.median(w) if len(w) else np.nan)
            wr_lo.append(np.percentile(w, 25) if len(w) else np.nan)
            wr_hi.append(np.percentile(w, 75) if len(w) else np.nan)
        xs = x + 0.06 * (j - 1)
        axes[0].errorbar(xs, cov, yerr=cov_h, color=st["color"], ls=st["ls"], marker=st["marker"], capsize=1.5,
                         elinewidth=0.6, label=st["label"])
        wr, wr_lo, wr_hi = np.array(wr), np.array(wr_lo), np.array(wr_hi)
        axes[1].errorbar(xs, wr, yerr=[wr - wr_lo, wr_hi - wr], color=st["color"], ls=st["ls"], marker=st["marker"],
                         capsize=1.5, elinewidth=0.6)
    base = runs.get("none-D0", [])
    if base:
        axes[0].axhline(final(base).mean(), color="0.0", lw=0.6, ls=":", label="no attacker")
        axes[1].axhline(np.median(final(base, "wrong", 1)), color="0.0", lw=0.6, ls=":")
    for ax in axes:
        ax.set_xticks(x), ax.set_xticklabels(["none" if d == "D0" else d for d in have])
        ax.set_xlabel("defence level")
    axes[0].set_ylabel("explored by honest drones [%]")
    axes[0].set_ylim(60, 101)
    axes[0].legend(loc="lower right")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("false cells per honest map")
    axes[0].set_title("(a) exploration", loc="left", fontsize=8)
    axes[1].set_title("(b) false map data", loc="left", fontsize=8)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_defences")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=list(range(10)))
    a = ap.parse_args()
    cam = load("sybil", a.seeds)
    coverage_time(cam)
    damage(cam)
    defences(cam)
    print("maps per condition:", sorted({len(v) for v in cam.values()}))


if __name__ == "__main__":
    main()
