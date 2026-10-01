#!/usr/bin/env python3
"""Watch a mission live in a window: the team's map from above, the drones' tracks, and

    python3 scripts/watch.py                          # office map 0, 8 honest drones
    python3 scripts/watch.py warehouse 2              # another map
    python3 scripts/watch.py office 0 fakewall        # one compromised drone (red), usual team
    python3 scripts/watch.py office 0 fakewall secure # same attacker, secured team
    python3 scripts/watch.py forest 1 none --drones 16
"""
import argparse
import os
import sys

import matplotlib
import numpy as np

try:
    matplotlib.use("macosx" if sys.platform == "darwin" else "TkAgg")
except Exception:
    pass
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from swarm import viz, worlds  # noqa: E402
from swarm.attacks import ATTACKS  # noqa: E402
from swarm.sim import Drone, Params, Sim  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", nargs="?", default="office", choices=worlds.FAMILIES)
    ap.add_argument("seed", nargs="?", type=int, default=0)
    ap.add_argument("attack", nargs="?", default="none", choices=["none"] + list(ATTACKS))
    ap.add_argument("mode", nargs="?", default="usual", choices=["usual", "secure"])
    ap.add_argument("--drones", type=int, default=8)
    ap.add_argument("--speed", type=float, default=2.0, help="simulated seconds per screen update")
    ap.add_argument("--frames", type=int, default=0, help="stop after this many updates (0 = run to the end)")
    a = ap.parse_args()

    secure = a.mode == "secure"
    p = Params(n_drones=a.drones, seed=a.seed, auth=secure, defence="audit" if secure else "none")
    types = [Drone] * p.n_drones
    if a.attack != "none":
        types[-1] = ATTACKS[a.attack]
    s = Sim(worlds.make(a.family, a.seed), p, drone_types=types)

    plt.ion()
    fig = plt.figure(figsize=(12, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1])
    ax_map = fig.add_subplot(gs[:, 0])
    ax_cov = fig.add_subplot(gs[0, 1])
    ax_fly = fig.add_subplot(gs[1, 1])
    who = "no attacker" if a.attack == "none" else f"attacker (red): {a.attack}, {a.mode} team"
    n = 0
    while s.t < p.t_max and plt.fignum_exists(fig.number):
        t_next = s.t + a.speed
        while s.t < t_next:
            s.step()
        s.record()
        ax_map.clear()
        rev = {r["by"] for r in s.revocations if r["origin"] < len(s.drones) and not s.drones[r["origin"]].honest}
        extra = f"   attacker revoked by {len(rev)} drones" if rev else ""
        viz.topdown(s, ax_map, f"{a.family} map {a.seed}, {len(s.drones)} drones, {who}   "
                               f"t = {s.t:.0f} s   seen {100 * s.m.coverage[-1]:.1f}%{extra}")
        ax_cov.clear()
        ax_cov.plot(s.m.t, 100 * np.array(s.m.coverage), color="#0072B2", lw=1.5, label="seen first-hand (team)")
        ax_cov.plot(s.m.t, 100 * np.array(s.m.coverage_mean), color="#E69F00", lw=1.2, label="known by average drone")
        ax_cov.set_ylim(0, 102), ax_cov.set_ylabel("%"), ax_cov.legend(fontsize=7, frameon=False, loc="lower right")
        ax_cov.grid(alpha=0.3)
        ax_fly.clear()
        ax_fly.plot(s.m.t, s.m.flown, color="#009E73", lw=1.5)
        ax_fly.set_xlabel("time [s]"), ax_fly.set_ylabel("distance flown by team [m]"), ax_fly.grid(alpha=0.3)
        fig.tight_layout()
        plt.pause(0.001)
        n += 1
        if s.done() or (a.frames and n >= a.frames):
            break
    print(f"end: t = {s.t:.0f} s, seen {100 * s.m.coverage[-1]:.1f}%, flown {s.m.flown[-1]:.0f} m, "
          f"crashed {s.m.crashed[-1]}, revocations {len(s.revocations)}")
    if plt.fignum_exists(fig.number) and not a.frames:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
