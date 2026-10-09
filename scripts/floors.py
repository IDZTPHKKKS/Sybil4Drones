#!/usr/bin/env python3
"""Explored space per floor in the three-storey buildings; writes results/floors/*.json."""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
from run import DEFENCES  # noqa: E402
from swarm import worlds  # noqa: E402
from swarm.attacks import ATTACKS  # noqa: E402
from swarm.sim import Drone, Params, Sim  # noqa: E402

FH = 3.2
CONDS = [("none-D0", "D0", None), ("sybil_strong4-D0", "D0", "sybil_strong"),
         ("sybil_targeted4-D0", "D0", "sybil_targeted"), ("sybil_targeted4-D5", "D5", "sybil_targeted")]


def run(seed, name, d, attack):
    p = Params(seed=seed, t_max=1200.0)
    for k, v in DEFENCES[d].items():
        setattr(p, k, v)
    types = [Drone] * p.n_drones
    if attack:
        types[-1] = type(attack, (ATTACKS[attack],), dict(n_sybil=4))
    s = Sim(worlds.make("multistorey", seed), p, drone_types=types)
    s.run()
    T = s.target_cells
    z = (T % s.world.occ.shape[2]) * s.world.res
    floor = np.minimum((z // FH).astype(int), 2)
    seen = np.zeros(len(T), bool)
    for dr in s.honest():
        seen |= dr.src[T] == dr.id
    per = [float(100 * seen[floor == f].mean()) for f in range(3)]
    return dict(seed=seed, condition=name, floors=per, total=float(100 * s.m.coverage_honest[-1]))


def main():
    out = os.path.join(ROOT, "results", "floors")
    for seed in range(10):
        for name, d, attack in CONDS:
            path = os.path.join(out, f"multistorey_{seed}_{name}.json")
            if os.path.exists(path):
                continue
            r = run(seed, name, d, attack)
            json.dump(r, open(path, "w"))
            print(seed, name, [round(x, 1) for x in r["floors"]], round(r["total"], 1), flush=True)


if __name__ == "__main__":
    main()
