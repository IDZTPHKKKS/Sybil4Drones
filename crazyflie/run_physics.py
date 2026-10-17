#!/usr/bin/env python3
"""Key static experiments with Crazyflie physics flight, on the same maps and seeds as scripts/run.py.

    python3 crazyflie/run_physics.py                          # 3 maps per family, 5 conditions
    python3 crazyflie/run_physics.py --seeds 0 1 2 3 4        # more maps
    python3 crazyflie/run_physics.py --check                  # kinematic rerun of one stored run
    python3 crazyflie/run_physics.py --summary                # physics vs kinematic, paired by map
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from crazyflie.drone import physics  # noqa: E402
from run import DEFENCES  # noqa: E402
from swarm import worlds  # noqa: E402
from swarm.attacks import ATTACKS  # noqa: E402
from swarm.sim import Drone, Params, Sim  # noqa: E402

OUT = os.path.join(ROOT, "results/physics")
FAMILIES = ("office", "warehouse", "forest", "tunnels")
SPEC = ("sybil_strong", 1, dict(n_sybil=4))
CONDITIONS = {"none-D0": ("D0", None, "sybil"), "none-D4": ("D4", None, "sybil"), "sybil_strong4-D0": ("D0", SPEC, "sybil"),
              "sybil_strong4-D4": ("D4", SPEC, "sybil"), "sybil_strong4-D5": ("D5", SPEC, "bind")}


def run_one(family, seed, cond, flight="physics"):
    defence, spec, _ = CONDITIONS[cond]
    p = Params(seed=seed, t_max=600.0)
    for k, v in DEFENCES[defence].items():
        setattr(p, k, v)
    wrap = physics if flight == "physics" else (lambda c: c)
    types = [wrap(Drone)] * p.n_drones
    if spec:
        attack, n_att, attrs = spec
        cls = wrap(type(attack, (ATTACKS[attack],), dict(attrs)))
        for k in range(n_att):
            types[p.n_drones - 1 - k] = cls
    s = Sim(worlds.make(family, seed), p, drone_types=types)
    m = s.run()
    out = {k: getattr(m, k) for k in ("t", "coverage", "coverage_trusted", "coverage_honest", "coverage_mean",
                                      "wrong", "flown", "crashed", "sent", "overlap", "infected", "idle")}
    out["per_drone"] = [dict(id=d.id, honest=d.honest, flown=d.flown, crashed=d.crashed, guard_stops=d.guard_stops,
                             track_err_median=float(np.median(d.track_err)) if getattr(d, "track_err", None) else None,
                             track_err_max=float(np.max(d.track_err)) if getattr(d, "track_err", None) else None,
                             max_tilt=float(getattr(d, "max_tilt", 0.0))) for d in s.drones]
    out.update(family=family, seed=seed, flight=flight, condition=cond, params=p.__dict__,
               attackers=[d.id for d in s.drones if not d.honest], revocations=s.revocations)
    return out


def summary():
    rows = []
    for cond, (_, spec, kin_dir) in CONDITIONS.items():
        P, Kn = [], []
        for f in sorted(glob.glob(os.path.join(OUT, f"*_{cond}.json"))):
            d = json.load(open(f))
            k = os.path.join(ROOT, "results", kin_dir, f"{d['family']}_{d['seed']}_{cond}.json")
            if os.path.exists(k):
                P.append(d); Kn.append(json.load(open(k)))
        if not P:
            continue

        def stats(runs):
            A = [set(r["attackers"]) for r in runs]
            return dict(cov=100 * np.mean([r["coverage_honest"][-1] for r in runs]),
                        wrong=np.median([r["wrong"][-1] for r in runs]),
                        crashed=sum(r["crashed"][-1] for r in runs),
                        false_rev=sum(any(v["origin"] < 100 and v["origin"] not in a for v in r["revocations"])
                                      for r, a in zip(runs, A)),
                        t90=np.median([next((t for t, c in zip(r["t"], r["coverage_honest"]) if c >= 0.9), np.nan)
                                       for r in runs]))
        sp, sk = stats(P), stats(Kn)
        trk = [x["track_err_median"] for r in P for x in r["per_drone"] if x["track_err_median"] is not None]
        tilt = [np.degrees(x["max_tilt"]) for r in P for x in r["per_drone"]]
        rows.append((cond, len(P), sk, sp, np.median(trk), np.median(tilt)))
    print(f"{'condition':18s} {'maps':>4s} | {'explored %':>15s} | {'false cells':>15s} | {'crashes':>9s} | "
          f"{'honest expelled':>15s} | {'t to 90% [s]':>13s} | track err, max tilt")
    print(f"{'':18s} {'':>4s} | {'kin -> phys':>15s} | {'kin -> phys':>15s} | {'kin->phys':>9s} | {'kin -> phys':>15s} | {'kin -> phys':>13s} |")
    for cond, n, k, p, trk, tilt in rows:
        print(f"{cond:18s} {n:4d} | {k['cov']:6.1f} -> {p['cov']:5.1f} | {k['wrong']:6.0f} -> {p['wrong']:6.0f} | "
              f"{k['crashed']:3d} -> {p['crashed']:2d} | {k['false_rev']:6d} -> {p['false_rev']:5d} | "
              f"{k['t90']:5.0f} -> {p['t90']:4.0f} | {100 * trk:.1f} cm, {tilt:.0f} deg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        return summary()
    if a.check:
        r = run_one("office", 0, "none-D4", flight="kinematic")
        k = json.load(open(os.path.join(ROOT, "results/sybil/office_0_none-D4.json")))
        same = r["coverage_honest"] == k["coverage_honest"] and r["wrong"] == k["wrong"]
        print("kinematic rerun matches the stored baseline:", same,
              f"(explored {r['coverage_honest'][-1]:.4f} vs {k['coverage_honest'][-1]:.4f})")
        return
    os.makedirs(OUT, exist_ok=True)
    for seed in a.seeds:
        for fam in a.families:
            for cond in a.conditions:
                path = os.path.join(OUT, f"{fam}_{seed}_{cond}.json")
                if os.path.exists(path):
                    continue
                t0 = time.time()
                r = run_one(fam, seed, cond)
                json.dump(r, open(path, "w"))
                trk = np.median([x["track_err_median"] for x in r["per_drone"]])
                print(f"{fam:9s} {seed} {cond:18s} explored {100 * r['coverage_honest'][-1]:5.1f}%  false cells "
                      f"{r['wrong'][-1]:6.0f}  crashed {r['crashed'][-1]}  track err {100 * trk:.1f} cm  "
                      f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
