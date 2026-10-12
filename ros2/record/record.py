import argparse
import gzip
import json
import os
import sys

import numpy as np

SIM = os.environ.get("SYBIL_SIM", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, SIM)
sys.path.insert(0, os.path.join(SIM, "scripts"))
from swarm import kernels as K  # noqa: E402
from swarm import worlds  # noqa: E402
from swarm.attacks import ATTACKS  # noqa: E402
from swarm.sim import Drone, Params, Sim  # noqa: E402
from run import DEFENCES, T_MAX  # noqa: E402

state = dict(sim=None, frame=None, frames=[], maps={}, voxels={}, next_map={}, band=None)
MAP_EVERY = 2.0


def voxel_code(d):
    occ = d.state == K.OCC
    return np.where(occ & ~d.sim.occ_near, 2, occ.astype(np.int8)).astype(np.int8)


def map_code(d):
    s = d.sim
    if state["band"] is None:
        shape = s.world.occ.shape
        lo, hi = int(round(0.4 / s.world.res)), int(round(2.4 / s.world.res))
        state["band"] = (shape, lo, hi, ~s.occ_near.reshape(shape)[:, :, lo:hi])
    shape, lo, hi, not_near = state["band"]
    st = d.state.reshape(shape)[:, :, lo:hi]
    occ = (st == K.OCC).any(2)
    free = (st == K.FREE).any(2)
    false = ((st == K.OCC) & not_near).any(2)
    return np.where(false, 3, np.where(occ, 2, np.where(free, 1, 0))).astype(np.int8).ravel()


def pt(q):
    return [float(v) for v in q]


_audit_kernel = K.audit


def audit_kernel(*args):
    out = _audit_kernel(*args)
    f = state["frame"]
    if f is not None and args[3] == f["observer"]:
        d = state["sim"].drones[f["observer"]]
        checked = d.agree + d.disagree
        ids = np.flatnonzero((checked > 0) | (d.far > 0) | (d.far_occ > 0))
        f["counts"] = [[int(o), int(checked[o]), int(d.far[o]), int(d.checked_occ[o]), int(d.far_occ[o])] for o in ids]
        f["witness"] = [[int(a), sorted(int(i) for i in ids_)] for a, ids_ in d.attest_by.items()]
    return out


_presence = Drone._presence_check


def presence(self):
    s = self.sim
    f = state["frame"]
    if f is not None and f["observer"] == self.id:
        f["beacons"] = [[int(o), pt(q), float(t0)] for o, (q, t0) in self.beacons.items()
                        if o != self.id and s.t - t0 <= 2 * s.p.dt]
        f["bodies"] = [[int(d.id), pt(d.pos)] for d in s.drones if not d.crashed and d is not self]
    return _presence(self)


_audit = Drone.audit


def audit(self):
    s = self.sim
    if not self.honest or s.p.defence != "audit":
        return _audit(self)
    n0 = len(s.revocations)
    f = dict(observer=int(self.id), t=float(s.t), position=pt(self.pos), yaw=float(self.yaw), counts=[],
             beacons=[], bodies=[], witness=[])
    state["frame"] = f
    try:
        _audit(self)
    finally:
        state["frame"] = None
    f["revoked"] = [[int(r["origin"]), r["why"]] for r in s.revocations[n0:]]
    if s.t >= state["next_map"].get(self.id, 0.0):
        state["next_map"][self.id] = s.t + MAP_EVERY
        code = map_code(self)
        prev = state["maps"].get(self.id)
        idx = np.flatnonzero(code != prev) if prev is not None else np.flatnonzero(code)
        f["map"] = [idx.tolist(), code[idx].tolist()]
        state["maps"][self.id] = code
        vox = voxel_code(self)
        prev = state["voxels"].get(self.id)
        idx = np.flatnonzero(vox != prev) if prev is not None else np.flatnonzero(vox)
        f["voxels"] = [idx.tolist(), vox[idx].tolist()]
        state["voxels"][self.id] = vox
    f["trusted"] = np.packbits(self.trusted_source()).tobytes().hex()
    state["frames"].append(f)


K.audit = audit_kernel
Drone._presence_check = presence
Drone.audit = audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="office")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attack", default="sybil_strong")
    ap.add_argument("--n-sybil", type=int, default=4)
    ap.add_argument("--attackers", type=int, default=1)
    ap.add_argument("--level", default="D4", choices=["D1", "D2", "D3", "D4", "D5"])
    ap.add_argument("--drones", type=int, default=8)
    ap.add_argument("--detect", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    w = worlds.make(a.family, a.seed)
    p = Params(seed=a.seed, t_max=T_MAX.get(a.family, 600.0), n_drones=a.drones)
    for k, v in DEFENCES[a.level].items():
        setattr(p, k, v)
    p.presence_detect = a.detect
    types = [Drone] * p.n_drones
    if a.attack != "none":
        cls = type(a.attack, (ATTACKS[a.attack],), dict(n_sybil=a.n_sybil))
        for k in range(a.attackers):
            types[p.n_drones - 1 - k] = cls
    s = Sim(w, p, drone_types=types)
    state["sim"] = s
    s.run()
    honest = [d.id for d in s.drones if d.honest]
    world = dict(res=w.res, shape=list(w.occ.shape), bits=np.packbits(w.occ.ravel()).tobytes().hex())
    head = dict(family=a.family, seed=a.seed, attack=a.attack, n_sybil=a.n_sybil, level=a.level, n_ids=s.n_ids,
                honest=honest, attackers=[d.id for d in s.drones if not d.honest], frames=len(state["frames"]),
                revocations=len(s.revocations), dt=p.dt, v_max=p.v_max, detect=p.presence_detect,
                max_range=p.max_range, hfov=p.hfov, vfov=p.vfov, coverage=s.m.coverage_honest[-1], world=world)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with gzip.open(a.out, "wt") as fh:
        fh.write(json.dumps(head) + "\n")
        for f in state["frames"]:
            fh.write(json.dumps(f) + "\n")
    print(f"{os.path.basename(a.out)}: {len(state['frames'])} frames, {len(s.revocations)} revocations, "
          f"explored {100 * s.m.coverage_honest[-1]:.1f}%")


if __name__ == "__main__":
    main()
