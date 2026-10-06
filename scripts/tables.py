#!/usr/bin/env python3
"""Result tables (explored % / false cells) from the run files, as in the README.

    python3 scripts/tables.py                                         # sybil, the four main families
    python3 scripts/tables.py --families multistorey atrium cave
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from swarm import worlds  # noqa: E402


def load(exp, families, seeds):
    runs = defaultdict(list)
    for f in glob.glob(os.path.join(ROOT, "results", exp, "*.json")):
        r = json.load(open(f))
        if r["family"] in families and r["seed"] in seeds and r.get("rep", 0) == 0:
            runs[r["condition"]].append(r)
    return runs


def explored(rs):
    return np.mean([r["coverage_honest"][-1] for r in rs]) * 100


def cell(rs):
    return f"{explored(rs):.1f}% / {np.median([r['wrong'][-1] for r in rs]):.0f}" if rs else "-"


def t90(rs):
    out = []
    for r in rs:
        t, c = np.array(r["t"]), np.array(r["coverage_honest"])
        out.append(t[np.argmax(c >= 0.9)] if c.max() >= 0.9 else np.nan)
    return np.nanmean(out) if rs else np.nan


def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(r) + " |")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="sybil")
    ap.add_argument("--families", nargs="*", default=list(worlds.FAMILIES), choices=worlds.ALL_FAMILIES)
    ap.add_argument("--seeds", nargs="*", type=int, default=list(range(10)))
    a = ap.parse_args()
    runs = load(a.exp, a.families, a.seeds)
    if not runs:
        sys.exit("no runs found")
    n = sorted({len(v) for v in runs.values()})
    print(f"{', '.join(a.families)}: {sum(map(len, runs.values()))} runs, {n} maps per condition\n")
    print("Explored space without any defence, by number of fake identities:\n")
    table(["attack", "1", "2", "4", "8"],
          [[k] + [f"{explored(runs[f'sybil_{k}{i}-D0']):.1f}%" if runs.get(f"sybil_{k}{i}-D0") else "-"
                  for i in (1, 2, 4, 8)] for k in ("weak", "strong", "stealth")])
    print("Explored / false cells by defence level:\n")
    rows = [("no attacker", "none"), ("weak, 8 identities", "sybil_weak8"), ("strong, 8 identities", "sybil_strong8"),
            ("stealth, 8 identities", "sybil_stealth8"), ("2 compromised drones, 4 identities each", "2x_sybil_strong4")]
    table(["attack", "none", "D3", "D4"], [[name] + [cell(runs.get(f"{c}-{d}", [])) for d in ("D0", "D3", "D4")]
                                          for name, c in rows])
    print("4 fake identities, explored / false cells by defence level:\n")
    table(["attack", "none", "D1", "D2", "D3", "D4"],
          [[k] + [cell(runs.get(f"sybil_{k}4-{d}", [])) for d in ("D0", "D1", "D2", "D3", "D4")]
           for k in ("weak", "strong", "stealth")])
    print("Time to 90% explored without an attacker: " + ", ".join(
        f"{d} {t90(runs.get(f'none-{d}', [])):.0f} s" for d in ("D0", "D3", "D4")))
    crashes = sum(r["crashed"][-1] > 0 for v in runs.values() for r in v)
    revoked = sum(1 for v in runs.values() for r in v for x in r["revocations"]
                  if x["origin"] < len(r["per_drone"]) and r["per_drone"][x["origin"]]["honest"])
    print(f"Runs with a crash: {crashes}. Honest drones revoked: {revoked}.")


if __name__ == "__main__":
    main()
