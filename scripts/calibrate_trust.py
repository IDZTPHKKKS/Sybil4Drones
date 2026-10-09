#!/usr/bin/env python3
"""Calibrate the trust tolerance on held-out maps (seeds 100-104, never used in experiments).

    python3 scripts/calibrate_trust.py                                   -> results/calibration_camera.json
    python3 scripts/calibrate_trust.py --families multistorey atrium cave -> results/calibration_camera_multistorey-atrium-cave.json
"""
import json
import os
import sys

import numpy as np
from scipy.stats import beta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from swarm import worlds  # noqa: E402
from swarm.sim import Drone, Params, Sim  # noqa: E402

SEEDS = range(100, 105)
SENSOR = "lidar" if "--lidar" in sys.argv else "camera"
CONF = Params().trust_conf
T_MAX = dict(multistorey=1200.0, atrium=1200.0, metro=1200.0, carpark=1200.0)


def families():
    if "--families" not in sys.argv:
        return list(worlds.FAMILIES)
    out = []
    for x in sys.argv[sys.argv.index("--families") + 1:]:
        if x.startswith("--"):
            break
        out.append(x)
    return out


def main():
    worst = []
    orig_audit = Drone.audit

    def audit(self):
        orig_audit(self)
        checked = self.agree + self.disagree
        rec = self.sim.calib
        for o in np.flatnonzero(self.far > 0):
            if o != self.id:
                q = beta.ppf(1 - CONF, 1 + self.far[o], 1 + checked[o] - self.far[o])
                if q > rec["q"]:
                    rec.update(q=float(q), far=int(self.far[o]), checked=int(checked[o]), t=self.sim.t)
        for o in np.flatnonzero(self.far_occ > 0):
            if o != self.id:
                q = beta.ppf(1 - CONF, 1 + self.far_occ[o], 1 + self.checked_occ[o] - self.far_occ[o])
                rec["q_walls"] = max(rec["q_walls"], float(q))
        rec["absent"] = max(rec["absent"], int(self.absent[:self.sim.p.n_drones].max()))

    Drone.audit = audit
    fams = families()
    for f in fams:
        for seed in SEEDS:
            p = Params(seed=seed, t_max=T_MAX.get(f, 600.0), auth=False, defence="audit", audit_vouched=True, presence=True,
                       trust_needs_body=True, trust_by_type=True, verify_free=True,
                       trust_conf=1.1, presence_conf=1.1, sensor=SENSOR)
            s = Sim(worlds.make(f, seed), p)
            s.calib = dict(q=0.0, q_walls=0.0, absent=0)
            s.run()
            worst.append(dict(family=f, seed=seed, **s.calib))
            print(f, seed, {k: round(v, 4) if isinstance(v, float) else v for k, v in s.calib.items()}, flush=True)
    q = max(w["q"] for w in worst)
    print(f"\nlargest 1st-percentile far rate of an honest drone: {q:.4f}  (walls only: "
          f"{max(w['q_walls'] for w in worst):.4f}); most presence checks an honest drone failed: "
          f"{max(w['absent'] for w in worst)}")
    json.dump(dict(runs=worst, max_q=q, sensor=SENSOR),
              open(os.path.join(ROOT, "results", f"calibration_{SENSOR}" + ("" if fams == list(worlds.FAMILIES)
                                                       else "_" + "-".join(fams)) + ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
