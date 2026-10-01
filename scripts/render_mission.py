#!/usr/bin/env python3
"""Side-by-side animation: the same map and attacker, a usual team vs a secured team.

    python3 scripts/render_mission.py office 0 fakewall     -> media/mission_office0_fakewall.{gif,mp4}
    python3 scripts/render_mission.py office 0 none
"""
import os
import sys

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from swarm import viz, worlds  # noqa: E402
from swarm.attacks import ATTACKS  # noqa: E402
from swarm.sim import Drone, Params, Sim  # noqa: E402

FRAME_EVERY = 2.0


def make(family, seed, attack, secure, t_max):
    p = Params(seed=seed, t_max=t_max, auth=secure, defence="audit" if secure else "none")
    types = [Drone] * p.n_drones
    if attack != "none":
        types[-1] = ATTACKS[attack]
    return Sim(worlds.make(family, seed), p, drone_types=types)


def main(family="office", seed=0, attack="fakewall", t_max=360.0):
    sims = [make(family, seed, attack, False, t_max), make(family, seed, attack, True, t_max)]
    names = ["usual team", "secured team"]
    frames = []
    t = 0.0
    while t <= t_max:
        for s in sims:
            while s.t < t - 1e-9 and not s.done():
                s.step()
            if not s.done() or s.m.t[-1] < s.t:
                s.record()
        fig = plt.figure(figsize=(12, 5.4))
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.55], height_ratios=[1, 1])
        for i, (s, name) in enumerate(zip(sims, names)):
            ax = fig.add_subplot(gs[:, i])
            rev = [r for r in s.revocations if r["origin"] < len(s.drones) and not s.drones[r["origin"]].honest] if r_ok(s) else []
            extra = f", liar caught by {len({r['by'] for r in rev})}" if rev else ""
            end = (", finished" if s.m.coverage_trusted[-1] > 0.9 else ", team gave up") if s.done() else ""
            viz.topdown(s, ax, f"{name}: seen {100 * s.m.coverage_trusted[-1]:.0f}%{extra}{end}")
        ax = fig.add_subplot(gs[0, 2])
        for s, name, c in zip(sims, names, ("#D55E00", "#0072B2")):
            ax.plot(s.m.t, 100 * np.array(s.m.coverage_trusted), color=c, lw=1.5, label=name.split(":")[0])
        ax.set_xlim(0, t_max), ax.set_ylim(0, 102)
        ax.set_ylabel("space seen [%]", fontsize=8)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        ax.tick_params(labelsize=7)
        ax = fig.add_subplot(gs[1, 2])
        for s, c in zip(sims, ("#D55E00", "#0072B2")):
            ax.plot(s.m.t, s.m.flown, color=c, lw=1.5)
        ax.set_xlim(0, t_max)
        ax.set_xlabel("time [s]", fontsize=8), ax.set_ylabel("distance flown [m]", fontsize=8)
        ax.tick_params(labelsize=7)
        who = "no attacker" if attack == "none" else f"attacker (red): {attack}"
        fig.suptitle(f"{family} map {seed}, 8 drones, {who}, t = {t:.0f} s", fontsize=10)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(fig)
        t += FRAME_EVERY
    base = os.path.join(ROOT, "media", f"mission_{family}{seed}_{attack}")
    imageio.mimsave(base + ".mp4", frames, fps=12, quality=8, macro_block_size=1)
    imageio.mimsave(base + ".gif", frames[::2], duration=1 / 6, loop=0)
    imageio.imwrite(base + "_final.png", frames[-1])
    print(base, len(frames), "frames")


def r_ok(s):
    return bool(s.revocations)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "office", int(a[1]) if len(a) > 1 else 0, a[2] if len(a) > 2 else "fakewall",
         float(a[3]) if len(a) > 3 else 360.0)
