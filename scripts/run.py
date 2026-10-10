#!/usr/bin/env python3
"""Run an experiment over many maps; one JSON file per run, existing runs are skipped (resumable).

    python3 scripts/run.py sybil        # weak, strong and stealth Sybil attacks vs defence levels D0-D4
    python3 scripts/run.py bind         # shadow attack (fakes next to real drones) and D5 (one body, one identity)
    python3 scripts/run.py teamsize     # strong attacker with 4 and 16 drones, D0, D4 and D5
    python3 scripts/run.py storey       # targeted attack in multi-floor worlds (use --families multistorey atrium cave)
    python3 scripts/run.py network      # packet dropping attackers (blackhole, greyhole), usual vs secure
    python3 scripts/run.py attacks      # one compromised drone of 8, every attack, vanilla vs secure
    python3 scripts/run.py stealth      # fake-wall injection rate vs damage and detection
    python3 scripts/run.py multi        # 1-4 fake-wall attackers of 8, secure team
    python3 scripts/run.py loss         # packet loss 0..0.8, 8 honest drones
    python3 scripts/run.py scaling      # team size 1..16, honest
    python3 scripts/run.py attacks --families office --seeds 0 1     # a subset
    python3 scripts/run.py attacks --seeds 5 6 7 8 9                 # more maps, added to the rest
    python3 scripts/run.py attacks --seeds 0 1 --reps 3              # repeats with new noise seeds
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from swarm import worlds  # noqa: E402
from swarm.attacks import ATTACKS  # noqa: E402
from swarm.sim import Drone, Params, Sim  # noqa: E402

SEEDS = range(10)
T_MAX = dict(multistorey=1200.0, atrium=1200.0, metro=1200.0, carpark=1200.0, castle=1200.0)


VANILLA = dict(auth=False, defence="none")
SECURE = dict(auth=True, defence="audit")
ATTACK_LIST = ("passive", "blackhole", "greyhole", "claimjam", "fakefree", "fakewall")
D0 = dict(auth=False, defence="none")
D1 = dict(auth=False, defence="audit")
D2 = dict(D1, audit_vouched=True, presence=True)
D3 = dict(D2, trust_needs_body=True, trust_by_type=True, verify_free=True)
D4 = dict(D3, quarantine=True)
D5 = dict(D4, bind=True)
S1 = dict(D1, auth=True)
S3 = dict(D3, auth=True)
DEFENCES = dict(D0=D0, D1=D1, D2=D2, D3=D3, D4=D4, D5=D5, S1=S1, S3=S3)


def conditions(exp):
    if exp == "network":
        out = []
        for mode, over in (("vanilla", VANILLA), ("secure", SECURE)):
            out.append((f"none-{mode}", over, None))
            out += [(f"{a}-{mode}", over, (a, 1, {})) for a in ("blackhole", "greyhole")]
        return out
    if exp == "sybil_lidar":
        L = dict(sensor="lidar")
        out = [(f"none-{d}", dict(DEFENCES[d], **L), None) for d in ("D0", "D3", "D4")]
        for a in ("sybil_weak", "sybil_strong", "sybil_stealth"):
            for d in ("D0", "D3", "D4"):
                out.append((f"{a}4-{d}", dict(DEFENCES[d], **L), (a, 1, dict(n_sybil=4))))
        return out
    if exp == "storey":
        spec = lambda a: (a, 1, dict(n_sybil=4))
        return [("none-D0", D0, None), ("none-D4", D4, None),
                ("sybil_strong4-D0", D0, spec("sybil_strong")),
                ("sybil_targeted4-D0", D0, spec("sybil_targeted")),
                ("sybil_targeted4-D3", D3, spec("sybil_targeted")),
                ("sybil_targeted4-D4", D4, spec("sybil_targeted")),
                ("sybil_targeted4-D5", D5, spec("sybil_targeted"))]
    if exp == "teamsize":
        out = []
        for n in (4, 16):
            T = dict(n_drones=n)
            out.append((f"n{n}-none-D0", dict(D0, **T), None))
            out += [(f"n{n}-sybil_strong4-{d}", dict(DEFENCES[d], **T), ("sybil_strong", 1, dict(n_sybil=4)))
                    for d in ("D0", "D4", "D5")]
        return out
    if exp == "bind":
        spec = lambda a: (a, 1, dict(n_sybil=4))
        out = [("none-D5", D5, None)]
        out += [(f"sybil_shadow4-{d}", DEFENCES[d], spec("sybil_shadow")) for d in ("D0", "D3", "D4", "D5")]
        out += [(f"{a}4-D5", D5, spec(a)) for a in ("sybil_strong", "sybil_stealth", "sybil_targeted")]
        return out
    if exp == "sybil":
        out = [(f"none-{d}", DEFENCES[d], None) for d in ("D0", "D3", "D4")]
        for a in ("sybil_weak", "sybil_strong", "sybil_stealth"):
            for k in (1, 2, 4, 8):
                out.append((f"{a}{k}-D0", D0, (a, 1, dict(n_sybil=k))))
            for d in ("D1", "D2", "D3", "D4"):
                out.append((f"{a}4-{d}", DEFENCES[d], (a, 1, dict(n_sybil=4))))
            for d in ("D3", "D4"):
                out.append((f"{a}8-{d}", DEFENCES[d], (a, 1, dict(n_sybil=8))))
        for d in ("D0", "D3", "D4"):
            out.append((f"2x_sybil_strong4-{d}", DEFENCES[d], ("sybil_strong", 2, dict(n_sybil=4))))
        return out
    if exp == "scaling":
        return [(f"n{n}", dict(n_drones=n, t_max=900.0), None) for n in (1, 2, 4, 8, 16)]
    if exp == "loss":
        return [(f"loss{int(100 * q)}", dict(loss=q), None) for q in (0.0, 0.2, 0.4, 0.6, 0.8)]
    if exp == "attacks":
        out = []
        for mode, over in (("vanilla", VANILLA), ("secure", SECURE)):
            out.append((f"none-{mode}", over, None))
            out += [(f"{a}-{mode}", over, (a, 1, {})) for a in ATTACK_LIST]
        out.append(("fakewall_frame-audit_noauth", dict(auth=False, defence="audit"), ("fakewall", 1, dict(forge="honest"))))
        out.append(("fakewall_frame-secure", SECURE, ("fakewall", 1, dict(forge="honest"))))
        return out
    if exp == "stealth":
        return [(f"rate{r}-{mode}", over, ("fakewall", 1, dict(rate=r)))
                for r in (100, 300, 1000, 3000, 10000) for mode, over in (("vanilla", VANILLA), ("secure", SECURE))]
    if exp == "multi":
        return [(f"k{k}-secure", SECURE, ("fakewall", k, {})) for k in (1, 2, 3, 4)]
    raise SystemExit(f"unknown experiment {exp}")


def run_one(family, seed, overrides, spec, rep=0):
    w = worlds.make(family, seed)
    p = Params(seed=seed + 1000 * rep, t_max=600.0)
    for k, v in overrides.items():
        setattr(p, k, v)
    types = [Drone] * p.n_drones
    if spec:
        attack, n_att, attrs = spec
        cls = type(attack, (ATTACKS[attack],), dict(attrs))
        for k in range(n_att):
            types[p.n_drones - 1 - k] = cls
    s = Sim(w, p, drone_types=types)
    m = s.run()
    out = {k: getattr(m, k) for k in ("t", "coverage", "coverage_trusted", "coverage_honest", "coverage_mean",
                                      "wrong", "flown", "crashed", "sent", "overlap", "infected", "idle")}
    out["per_drone"] = [dict(id=d.id, honest=d.honest, seen_share=d.seen_share, flown=d.flown,
                             idle_time=d.idle_time, guard_stops=d.guard_stops, escapes=d.escapes, crashed=d.crashed)
                        for d in s.drones]
    out.update(family=family, seed=seed, rep=rep, attack=spec[0] if spec else None, n_attackers=spec[1] if spec else 0,
               attack_attrs=spec[2] if spec else {}, params=p.__dict__,
               attackers=[d.id for d in s.drones if not d.honest], revocations=s.revocations,
               guard_stops=int(sum(d.guard_stops for d in s.drones if d.honest)),
               end_modes=[d.mode for d in s.drones])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--families", nargs="*", default=list(worlds.FAMILIES), choices=worlds.ALL_FAMILIES)
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    ap.add_argument("--reps", type=int, default=1, help="runs per map with different noise seeds")
    ap.add_argument("--t-max", type=float, default=None, help="mission length in s (default 600, 1200 for multistorey and atrium)")
    a = ap.parse_args()
    outdir = os.path.join(ROOT, "results", a.exp)
    os.makedirs(outdir, exist_ok=True)
    for seed in a.seeds:
        for family in a.families:
            for name, over, spec in conditions(a.exp):
                for rep in range(a.reps):
                    suffix = f"_r{rep}" if rep else ""
                    path = os.path.join(outdir, f"{family}_{seed}_{name}{suffix}.json")
                    if os.path.exists(path):
                        continue
                    t0 = time.time()
                    t_max = a.t_max or T_MAX.get(family, 600.0)
                    r = run_one(family, seed, dict(over, t_max=t_max), spec, rep)
                    r["condition"] = name
                    r["wall"] = time.time() - t0
                    json.dump(r, open(path, "w"))
                    c = np.array(r["coverage"])
                    print(f"{a.exp} {family} {seed} {name:28s} coverage {100 * c[-1]:5.1f}% at t={r['t'][-1]:4.0f}s  "
                          f"crashed {r['crashed'][-1]}  wrong/drone {r['wrong'][-1]:7.0f}  ({r['wall']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
