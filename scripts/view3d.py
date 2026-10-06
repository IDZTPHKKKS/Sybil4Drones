#!/usr/bin/env python3
"""Live 3D view of a mission in MuJoCo's viewer: rotate, pan and zoom with the mouse while the

    mjpython scripts/view3d.py                                  # office map 0, 8 honest drones
    mjpython scripts/view3d.py warehouse 2
    mjpython scripts/view3d.py office 0 sybil_weak --fakes 4            # 4 fake drones, no defence
    mjpython scripts/view3d.py office 0 sybil_weak --fakes 4 --defence D3   # same, full defence
    mjpython scripts/view3d.py tunnels 3 sybil_strong --fakes 8 --defence D3
    mjpython scripts/view3d.py office 0 sybil_stealth --fakes 4 --sensor lidar
    mjpython scripts/view3d.py forest 1 none --speed 3          # three times real time
"""
import argparse
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from swarm import worlds  # noqa: E402
from swarm.attacks import ATTACKS  # noqa: E402
from swarm.scene3d import Scene3D  # noqa: E402
from swarm.sim import Drone, Params, Sim  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from run import DEFENCES  # noqa: E402

MAP_EVERY = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", nargs="?", default="office", choices=worlds.ALL_FAMILIES)
    ap.add_argument("seed", nargs="?", type=int, default=0)
    ap.add_argument("attack", nargs="?", default="none", choices=["none"] + list(ATTACKS))
    ap.add_argument("--defence", default="D0", choices=list(DEFENCES))
    ap.add_argument("--fakes", type=int, default=4, help="fake identities active at a time (Sybil attacks)")
    ap.add_argument("--sensor", default="camera", choices=["camera", "lidar"])
    ap.add_argument("--drones", type=int, default=8)
    ap.add_argument("--speed", type=float, default=1.0, help="1 = real time")
    a = ap.parse_args()

    p = Params(n_drones=a.drones, seed=a.seed, t_max=600.0)
    for k, v in DEFENCES[a.defence].items():
        setattr(p, k, v)
    p.sensor = a.sensor
    types = [Drone] * p.n_drones
    if a.attack != "none":
        cls = ATTACKS[a.attack]
        types[-1] = type(a.attack, (cls,), dict(n_sybil=a.fakes)) if a.attack.startswith("sybil") else cls
    s = Sim(worlds.make(a.family, a.seed), p, drone_types=types)
    s.record()
    sc = Scene3D(s)
    who = "no attacker" if a.attack == "none" else (f"{a.attack} with {a.fakes} fake drones" if a.attack.startswith("sybil") else a.attack)
    title = f"{a.family} map {a.seed} | {who} | defence {a.defence} | {a.sensor}"

    fig = mujoco.MjvFigure()
    mujoco.mjv_defaultFigure(fig)
    fig.title = "space seen [%]"
    fig.flg_extend = 1
    fig.range[1][0], fig.range[1][1] = 0, 100
    fig.figurergba[:] = [0, 0, 0, 0.5]
    fig.linergb[0][:] = [0.2, 0.6, 1.0]
    fig.linergb[1][:] = [1.0, 0.7, 0.2]
    fig.linename[0] = "team, first-hand"
    fig.linename[1] = "average drone"

    with mujoco.viewer.launch_passive(sc.model, sc.data, show_left_ui=False, show_right_ui=False) as v:
        v.cam.lookat[:] = [sc.L / 2, sc.W / 2, 0]
        v.cam.distance = 0.9 * sc.L
        v.cam.elevation = -55
        v.cam.azimuth = 100
        start = time.time()
        next_map = 0.0
        while v.is_running():
            now = time.time() - start
            target = now * a.speed
            while s.t + p.dt <= target and not s.done():
                sc.remember()
                s.step()
                if len(s.m.t) == 0 or s.t - s.m.t[-1] >= 1.0:
                    s.record()
            alpha = 1.0 if s.done() else float(np.clip((target - s.t) / p.dt + 1.0, 0.0, 1.0))
            with v.lock():
                sc.move_drones(alpha)
                if now >= next_map:
                    next_map = now + MAP_EVERY
                    sc.update_map(v.user_scn)
                    n = min(len(s.m.t), 1000)
                    for k, series in enumerate((s.m.coverage, s.m.coverage_mean)):
                        fig.linepnt[k] = n
                        fig.linedata[k][0:2 * n:2] = s.m.t[-n:]
                        fig.linedata[k][1:2 * n:2] = 100 * np.array(series[-n:])
                    left, right = sc.hud()
                    v.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                  title + "\n" + left, "\n" + right)])
                    w, h = v.viewport.width, v.viewport.height
                    v.set_figures((mujoco.MjrRect(w - w // 3 - 10, 10, w // 3, h // 3), fig))
                    update_tex = True
                else:
                    update_tex = False
            if update_tex:
                v.update_texture(sc.tex_id)
            v.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()
