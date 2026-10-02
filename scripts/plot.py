#!/usr/bin/env python3
"""Figures and summary tables from results/*/ (plots whatever has finished).

    python3 scripts/plot.py        -> media/fig_*.png, results/summary.md
"""
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
MEDIA = os.path.join(ROOT, "media")

OI = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
ATTACKS = ["none", "passive", "blackhole", "greyhole", "claimjam", "fakefree", "fakewall"]
LABEL = {"none": "no attacker", "passive": "passive (control)", "blackhole": "blackhole (drop relays)",
         "greyhole": "greyhole (drop 50%)", "claimjam": "Sybil frontier claims", "fakefree": "fake free space",
         "fakewall": "fake walls", "fakewall_frame": "fake walls, framing others"}
COLOR = {a: OI[i] for i, a in enumerate(ATTACKS)}
COLOR["fakewall_frame"] = OI[7]
GRID = np.arange(0, 901, 2.0)
FAMS = ("office", "warehouse", "forest", "tunnels")

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False})


def load(exp):
    runs = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(RES, exp, "*.json"))):
        r = json.load(open(f))
        if r.get("rep", 0) == 0:
            runs[r["condition"]].append(r)
    return runs


COV = "coverage_trusted"


def is_attacker_id(r, o):
    return o in r["attackers"] or o >= r["params"]["n_drones"]


def series(r, key, grid=GRID):
    t, y = np.array(r["t"]), np.array(r[key], float)
    return np.interp(grid, t, y, right=y[-1])


def t_reach(r, level=0.9):
    c = np.array(r[COV])
    k = np.flatnonzero(c >= level)
    return r["t"][k[0]] if len(k) else np.nan


def flown_at(r, level=0.95):
    c = np.array(r[COV])
    k = np.flatnonzero(c >= level)
    return r["flown"][k[0]] if len(k) else np.nan


def mean_ci(M):
    M = np.asarray(M, float)
    m = np.nanmean(M, 0)
    h = 1.96 * np.nanstd(M, 0, ddof=1) / np.sqrt(np.sum(~np.isnan(M), 0)) if len(M) > 1 else 0 * m
    return m, h


def band(ax, runs, key, color, label, scale=1.0, grid=GRID):
    m, h = mean_ci([series(r, key, grid) * scale for r in runs])
    ax.plot(grid, m, color=color, lw=1.6, label=label)
    ax.fill_between(grid, m - h, m + h, color=color, alpha=0.15, lw=0)


def fig_attacks(runs, table):
    if not runs:
        return
    grid = np.arange(0, 601, 2.0)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    for ax, mode, title in ((axes[0], "vanilla", "Usual cooperative mapping"),
                            (axes[1], "secure", "Signed messages + first-hand audit")):
        for a in ATTACKS:
            rs = runs.get(f"{a}-{mode}")
            if rs:
                band(ax, rs, COV, COLOR[a], LABEL[a], 100, grid)
        ax.set_title(f"{title} (n={len(runs.get(f'none-{mode}', []))} maps)", fontsize=9)
        ax.set_xlabel("time [s]")
        ax.set_ylim(0, 102)
    axes[0].set_ylabel("space seen first-hand [%]")
    axes[1].legend(loc="lower right", fontsize=7)
    ax = axes[2]
    xs = np.arange(len(ATTACKS))
    for j, (mode, hatch) in enumerate((("vanilla", None), ("secure", "//"))):
        vals, errs = [], []
        for a in ATTACKS:
            rs = runs.get(f"{a}-{mode}", [])
            v = [r["wrong"][-1] for r in rs]
            vals.append(np.mean(v) if v else np.nan)
            errs.append(1.96 * np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
        ax.bar(xs + (j - 0.5) * 0.38, vals, 0.36, yerr=errs, color=[COLOR[a] for a in ATTACKS],
               hatch=hatch, edgecolor="white", linewidth=0.5, label=mode, error_kw=dict(lw=0.8))
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[a].split(" (")[0] for a in ATTACKS], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("wrong cells per honest map at the end")
    ax.set_title("Wrong cells left (plain: usual, hatched: secured)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(MEDIA, "fig_attacks.png"), dpi=150)
    plt.close(fig)

    table.append("## Attacks (1 compromised drone of 8)\n")
    table.append("| condition | maps | space seen at end | time to 90% | wrong cells / map | crashes | attacker revoked by | honest drones revoked |")
    table.append("|---|---|---|---|---|---|---|---|")
    order = [f"{a}-{m}" for m in ("vanilla", "secure") for a in ATTACKS] + \
            ["fakewall_frame-audit_noauth", "fakewall_frame-secure"]
    for c in order:
        rs = runs.get(c)
        if not rs:
            continue
        cov = np.array([r[COV][-1] for r in rs]) * 100
        t90 = np.array([t_reach(r) for r in rs])
        wr = np.array([r["wrong"][-1] for r in rs])
        cr = sum(r["crashed"][-1] for r in rs)
        n_h = [r["params"]["n_drones"] - r["n_attackers"] for r in rs]
        rev_att = [len({v["by"] for v in r["revocations"] if is_attacker_id(r, v["origin"])}) for r in rs]
        rev_hon = sum(sum(not is_attacker_id(r, v["origin"]) for v in r["revocations"]) for r in rs)
        reached = np.isfinite(t90)
        t90s = f"{np.nanmean(t90):.0f} s" + ("" if reached.all() else f" ({reached.sum()}/{len(rs)} reach)") if reached.any() else "never"
        revs = f"{np.mean(rev_att):.1f} of {np.mean(n_h):.0f}" if rs[0]["n_attackers"] else "-"
        table.append(f"| {c} | {len(rs)} | {cov.mean():.1f}% (min {cov.min():.1f}%) | {t90s} | {wr.mean():.0f} | {cr} | {revs} | {rev_hon} |")
    table.append("")
    table.append("### By map type: space seen at the end (mean, min)\n")
    table.append("| condition | " + " | ".join(FAMS) + " |")
    table.append("|---|" + "---|" * len(FAMS))
    for c in [f"{a}-{m}" for m in ("vanilla", "secure") for a in ("none", "fakewall", "fakefree", "claimjam")]:
        rs = runs.get(c, [])
        cells = []
        for fam in FAMS:
            v = np.array([r[COV][-1] for r in rs if r["family"] == fam]) * 100
            cells.append(f"{v.mean():.1f}% ({v.min():.1f}%)" if len(v) else "-")
        table.append(f"| {c} | " + " | ".join(cells) + " |")
    table.append("")


def fig_network(runs, table):
    if not runs:
        return
    conds = [f"{a}-{m}" for m in ("vanilla", "secure") for a in ("none", "blackhole", "greyhole")]
    grid = np.arange(0, 601, 2.0)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.1), sharey=True)
    for ax, mode, title in ((axes[0], "vanilla", "Usual team"), (axes[1], "secure", "Secured team")):
        for a in ("none", "blackhole", "greyhole"):
            rs = runs.get(f"{a}-{mode}")
            if rs:
                band(ax, rs, COV, COLOR[a], LABEL[a], 100, grid)
        ax.set_title(f"{title} (n={len(runs.get(f'none-{mode}', []))} maps)", fontsize=9)
        ax.set_xlabel("time [s]")
        ax.set_ylim(0, 102)
    axes[0].set_ylabel("space seen first-hand [%]")
    axes[1].legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(MEDIA, "fig_network.png"), dpi=150)
    plt.close(fig)
    table.append("## Packet-dropping insider (1 of 8 drones)\n")
    table.append("| condition | maps | space seen at end | time to 90% | time to 95% | flown until 95% | honest drones revoked |")
    table.append("|---|---|---|---|---|---|---|")
    for c in conds:
        rs = runs.get(c)
        if not rs:
            continue
        cov = np.array([r[COV][-1] for r in rs]) * 100
        t90 = np.array([t_reach(r) for r in rs])
        t95 = np.array([t_reach(r, 0.95) for r in rs])
        rev_hon = sum(sum(not is_attacker_id(r, v["origin"]) for v in r["revocations"]) for r in rs)
        fmt = lambda t: (f"{np.nanmean(t):.0f} s" + ("" if np.isfinite(t).all() else f" ({np.isfinite(t).sum()}/{len(t)} reach)")) if np.isfinite(t).any() else "never"
        table.append(f"| {c} | {len(rs)} | {cov.mean():.1f}% (min {cov.min():.1f}%) | {fmt(t90)} | {fmt(t95)} | "
                     f"{np.nanmean([flown_at(r) for r in rs]):.0f} m | {rev_hon} |")
    table.append("")
    table.append("### By map type: time to 90% [s]\n")
    table.append("| condition | " + " | ".join(FAMS) + " |")
    table.append("|---|" + "---|" * len(FAMS))
    for c in conds:
        rs = runs.get(c, [])
        cells = []
        for fam in FAMS:
            t = np.array([t_reach(r) for r in rs if r["family"] == fam])
            cells.append(f"{np.nanmean(t):.0f}" if len(t) and np.isfinite(t).any() else "-")
        table.append(f"| {c} | " + " | ".join(cells) + " |")
    table.append("")


SYB = (("sybil_weak", "#E69F00", "weak"), ("sybil_strong", "#D55E00", "strong"), ("sybil_stealth", "#0072B2", "stealth"))


def fig_sybil(runs, table, lidar=None):
    if not runs:
        return
    C = "coverage_honest"
    grid = np.arange(0, 601, 2.0)
    ks = (1, 2, 4, 8)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5))
    ax = axes[0]
    if runs.get("none-D0"):
        band(ax, runs["none-D0"], C, OI[0], "no attacker", 100, grid)
    for a, col, lab in SYB:
        rs = runs.get(f"{a}4-D0")
        if rs:
            band(ax, rs, C, col, f"{lab}, 4 fake identities", 100, grid)
    ax.set_title("No defence: coverage over time", fontsize=9)
    ax.set_xlabel("time [s]"), ax.set_ylabel("space seen by honest drones [%]"), ax.set_ylim(0, 102)
    ax.legend(loc="lower right", fontsize=7)
    ax = axes[1]
    for a, col, lab in SYB:
        m, h = [], []
        for k in ks:
            v = np.array([100 - 100 * r[C][-1] for r in runs.get(f"{a}{k}-D0", [])])
            m.append(v.mean() if len(v) else np.nan)
            h.append(1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
        ax.errorbar(ks, m, yerr=h, color=col, marker="o", ms=4, lw=1.4, capsize=2, label=lab)
        v = [100 - 100 * r[C][-1] for r in runs.get(f"{a}8-D3", [])]
        if v:
            ax.plot([8.6], [np.mean(v)], marker="s", ms=6, color=col, mfc="white", ls="none")
    base = [100 - 100 * r[C][-1] for r in runs.get("none-D0", [])]
    if base:
        ax.axhline(np.mean(base), color=OI[0], lw=0.8, ls=":", label="no attacker")
    ax.set_xscale("log", base=2), ax.set_xticks(ks), ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("fake identities (squares: 8 with full defence)"), ax.set_ylabel("space never seen by honest drones [%]")
    ax.set_title("Damage vs number of fake identities", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[2]
    levels = ("D0", "D1", "D2", "D3", "D4")
    xs = np.arange(len(levels))
    for j, (a, col, lab) in enumerate(SYB):
        vals, errs = [], []
        for d in levels:
            v = np.array([100 * r[C][-1] for r in runs.get(f"{a}4-{d}", [])])
            vals.append(v.mean() if len(v) else np.nan)
            errs.append(1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
        ax.bar(xs + (j - 1) * 0.27, vals, 0.25, yerr=errs, color=col, label=lab, error_kw=dict(lw=0.8))
    if base:
        ax.axhline(100 - np.mean(base), color=OI[0], lw=0.8, ls=":")
    ax.set_xticks(xs), ax.set_xticklabels(["none", "audit", "+vouch\n+presence", "+body+walls\n+check areas", "+quarantine"], fontsize=7)
    ax.set_ylabel("space seen by honest drones [%]"), ax.set_ylim(0, 102)
    ax.set_title("4 fake identities: which defence stops them", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(MEDIA, "fig_sybil.png"), dpi=150)
    plt.close(fig)

    def row(c, rs):
        cov = np.array([r[C][-1] for r in rs]) * 100
        t90 = []
        for r in rs:
            cc = np.array(r[C]); k = np.flatnonzero(cc >= 0.9)
            t90.append(r["t"][k[0]] if len(k) else np.nan)
        t90 = np.array(t90)
        n = rs[0]["params"]["n_drones"]
        caught = [len({v["origin"] for v in r["revocations"] if v["origin"] >= n}) for r in rs]
        first = [min([v["t"] for v in r["revocations"] if v["origin"] >= n], default=np.nan) for r in rs]
        hon = sum(sum(not is_attacker_id(r, v["origin"]) for v in r["revocations"]) for r in rs)
        t90s = (f"{np.nanmean(t90):.0f} s" + ("" if np.isfinite(t90).all() else f" ({np.isfinite(t90).sum()}/{len(rs)})")) if np.isfinite(t90).any() else "never"
        fs = f"{np.nanmedian(first):.0f} s" if np.isfinite(first).any() else "-"
        return (f"| {c} | {len(rs)} | {cov.mean():.1f}% | {cov.min():.1f}% | {t90s} | {np.mean([r['wrong'][-1] for r in rs]):.0f} | "
                f"{np.mean(caught):.1f} | {fs} | {hon} | {sum(r['crashed'][-1] for r in rs)} |")

    head = ["| condition | maps | seen at end | min | time to 90% | wrong cells / map | fake identities caught | first caught after | honest drones revoked | crashes |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    table.append("## Sybil attack (camera, 8 real drones, 1 compromised)\n")
    table.append("'Seen' = space the honest drones observed with their own sensors. 'Fake identities caught' counts every identity used, including replacements.\n")
    table += head
    order = ["none-D0", "none-D3", "none-D4"] + [f"{a}{k}-D0" for a, _, _ in SYB for k in ks] + \
            [f"{a}{k}-{d}" for a, _, _ in SYB for k, d in ((4, "D1"), (4, "D2"), (4, "D3"), (4, "D4"), (8, "D3"), (8, "D4"))] + \
            ["2x_sybil_strong4-D0", "2x_sybil_strong4-D3", "2x_sybil_strong4-D4"]
    for c in order:
        if runs.get(c):
            table.append(row(c, runs[c]))
    table.append("")
    if lidar:
        table.append("## Camera vs LiDAR (4 fake identities)\n")
        table.append("| condition | camera: seen | LiDAR: seen | camera: wrong cells | LiDAR: wrong cells |")
        table.append("|---|---|---|---|---|")
        for c in ["none-D0", "none-D3", "none-D4"] + [f"{a}4-{d}" for a, _, _ in SYB for d in ("D0", "D3", "D4")]:
            cam, lid = runs.get(c, []), lidar.get(c, [])
            if cam and lid:
                f = lambda rs, k: np.mean([r[k][-1] for r in rs])
                table.append(f"| {c} | {100 * f(cam, C):.1f}% | {100 * f(lid, C):.1f}% | {f(cam, 'wrong'):.0f} | {f(lid, 'wrong'):.0f} |")
        table.append("")


def fig_stealth(runs, table):
    if not runs:
        return
    rates = sorted({int(c.split("-")[0][4:]) for c in runs})
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))
    for mode, col, lab in (("vanilla", OI[6], "usual"), ("secure", OI[5], "secure")):
        m, h, det = [], [], []
        for r_ in rates:
            rs = runs.get(f"rate{r_}-{mode}", [])
            v = np.array([r[COV][-1] for r in rs]) * 100
            m.append(v.mean() if len(v) else np.nan)
            h.append(1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
            if mode == "secure":
                ts = []
                for r in rs:
                    tt = [v_["t"] for v_ in r["revocations"] if is_attacker_id(r, v_["origin"])]
                    ts.append(min(tt) if tt else np.nan)
                det.append(ts)
        axes[0].errorbar(rates, m, yerr=h, color=col, marker="o", ms=4, lw=1.4, capsize=2, label=lab)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("fake-wall injection rate [cells/s]")
    axes[0].set_ylabel("space seen first-hand at the end [%]")
    axes[0].set_ylim(0, 102)
    axes[0].legend()
    frac = [np.mean(np.isfinite(d)) * 100 for d in det]
    med = [np.nanmedian(d) if np.isfinite(d).any() else np.nan for d in det]
    axes[1].plot(rates, med, color=OI[5], marker="o", ms=4, lw=1.4)
    for x, y, f in zip(rates, med, frac):
        if np.isfinite(y):
            axes[1].annotate(f"{f:.0f}% caught", (x, y), textcoords="offset points", xytext=(4, 5), fontsize=7)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("fake-wall injection rate [cells/s]")
    axes[1].set_ylabel("time to first revocation [s] (median)")
    fig.tight_layout()
    fig.savefig(os.path.join(MEDIA, "fig_stealth.png"), dpi=150)
    plt.close(fig)
    table.append("## Stealth: fake-wall injection rate\n")
    table.append("| rate [cells/s] | usual: space seen | secure: space seen | secure: caught | secure: median time to catch |")
    table.append("|---|---|---|---|---|")
    for i, r_ in enumerate(rates):
        cv = [np.mean([r[COV][-1] for r in runs.get(f"rate{r_}-{m}", [])] or [np.nan]) * 100 for m in ("vanilla", "secure")]
        table.append(f"| {r_} | {cv[0]:.1f}% | {cv[1]:.1f}% | {frac[i]:.0f}% | {med[i]:.0f} s |")
    table.append("")


def fig_multi(runs, table):
    if not runs:
        return
    ks = sorted(int(c[1:].split("-")[0]) for c in runs)
    table.append("## Several fake-wall attackers among 8 drones (secure team)\n")
    table.append("| attackers | maps | space seen at end | time to 90% | wrong cells / map | honest drones revoked |")
    table.append("|---|---|---|---|---|---|")
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for i, k in enumerate(ks):
        rs = runs[f"k{k}-secure"]
        band(ax, rs, COV, OI[i + 1], f"{k} attacker{'s' if k > 1 else ''}", 100, np.arange(0, 601, 2.0))
        cov = np.array([r[COV][-1] for r in rs]) * 100
        t90 = np.array([t_reach(r) for r in rs])
        rev_hon = sum(sum(not is_attacker_id(r, v["origin"]) for v in r["revocations"]) for r in rs)
        table.append(f"| {k} | {len(rs)} | {cov.mean():.1f}% (min {cov.min():.1f}%) | {np.nanmean(t90):.0f} s | "
                     f"{np.mean([r['wrong'][-1] for r in rs]):.0f} | {rev_hon} |")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("space seen first-hand [%]")
    ax.set_ylim(0, 102)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(MEDIA, "fig_multi.png"), dpi=150)
    plt.close(fig)
    table.append("")


def fig_team(scaling, loss, table):
    if not scaling and not loss:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    if scaling:
        ns = sorted(int(c[1:]) for c in scaling)
        for i, n in enumerate(ns):
            rs = scaling[f"n{n}"]
            band(axes[0], rs, COV, OI[i + 1], f"{n} drone{'s' if n > 1 else ''}", 100)
            band(axes[1], rs, "flown", OI[i + 1], f"{n} drone{'s' if n > 1 else ''}")
        axes[0].set_ylabel("space seen first-hand [%]")
        axes[0].set_title("Team size", fontsize=9)
        axes[1].set_ylabel("distance flown by the team [m]")
        axes[1].set_title("Distance flown over time", fontsize=9)
        axes[0].legend(loc="lower right")
        for ax in axes[:2]:
            ax.set_xlabel("time [s]")
        axes[0].set_ylim(0, 102)
        table.append("## Team size (honest, no packet loss)\n")
        table.append("| drones | maps | space seen at end | time to 90% | time to 95% | flown until 95% |")
        table.append("|---|---|---|---|---|---|")
        for n in ns:
            rs = scaling[f"n{n}"]
            cov = np.array([r[COV][-1] for r in rs]) * 100
            t90 = np.array([t_reach(r) for r in rs])
            t95 = np.array([t_reach(r, 0.95) for r in rs])
            fmt = lambda t: f"{np.nanmean(t):.0f} s" + ("" if np.isfinite(t).all() else f" ({np.isfinite(t).sum()}/{len(t)} reach)") if np.isfinite(t).any() else "never"
            table.append(f"| {n} | {len(rs)} | {cov.mean():.1f}% | {fmt(t90)} | {fmt(t95)} | {np.nanmean([flown_at(r) for r in rs]):.0f} m |")
        table.append("")
    if loss:
        qs = sorted(int(c[4:]) for c in loss)
        for i, q in enumerate(qs):
            band(axes[2], loss[f"loss{q}"], COV, OI[i + 1], f"{q}% packets lost", 100, np.arange(0, 601, 2.0))
        axes[2].set_xlabel("time [s]")
        axes[2].set_ylabel("space seen first-hand [%]")
        axes[2].set_title("Packet loss (8 drones)", fontsize=9)
        axes[2].set_ylim(0, 102)
        axes[2].legend(loc="lower right")
        table.append("## Packet loss (8 honest drones)\n")
        table.append("| loss | maps | space seen at end | time to 90% | flown until 95% | map cells sent |")
        table.append("|---|---|---|---|---|")
        for q in qs:
            rs = loss[f"loss{q}"]
            cov = np.array([r[COV][-1] for r in rs]) * 100
            table.append(f"| {q}% | {len(rs)} | {cov.mean():.1f}% | {np.nanmean([t_reach(r) for r in rs]):.0f} s | "
                         f"{np.nanmean([flown_at(r) for r in rs]):.0f} m | {np.mean([r['sent'][-1] for r in rs]) / 1e6:.1f} M |")
        table.append("")
    fig.tight_layout()
    fig.savefig(os.path.join(MEDIA, "fig_team.png"), dpi=150)
    plt.close(fig)


def main():
    table = ["# Results summary\n", "Means over maps (10 seeds x 4 families unless fewer have finished). "
             "'Space seen' counts only cells observed with its own camera by a drone that reports truthfully and that most honest drones still trust at that time.\n"]
    fig_sybil(load("sybil"), table, load("sybil_lidar"))
    fig_network(load("network"), table)
    fig_attacks(load("attacks"), table)
    fig_stealth(load("stealth"), table)
    fig_multi(load("multi"), table)
    fig_team(load("scaling"), load("loss"), table)
    open(os.path.join(RES, "summary.md"), "w").write("\n".join(table) + "\n")
    print("\n".join(table))


if __name__ == "__main__":
    main()
