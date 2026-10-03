#!/usr/bin/env python3
"""Render a side-by-side 3D video of one mission with two defence levels.

    python3 scripts/render3d.py office 0 sybil_strong 4 D0 D3
"""
import os
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from run import DEFENCES  # noqa: E402
from swarm import worlds  # noqa: E402
from swarm.attacks import ATTACKS  # noqa: E402
from swarm.scene3d import Scene3D  # noqa: E402
from swarm.sim import Drone, Params, Sim  # noqa: E402

W, H = 800, 520
STEP = 1.0


def make(family, seed, attack, fakes, defence):
    p = Params(seed=seed, t_max=600.0)
    for k, v in DEFENCES[defence].items():
        setattr(p, k, v)
    types = [Drone] * p.n_drones
    if attack != "none":
        types[-1] = type(attack, (ATTACKS[attack],), dict(n_sybil=fakes))
    s = Sim(worlds.make(family, seed), p, drone_types=types)
    s.record()
    return s


def frame(sc, r, cam, label):
    r.update_scene(sc.data, cam)
    sc.update_map(r.scene, r.scene.ngeom)
    mujoco.mjr_uploadTexture(sc.model, r._mjr_context, sc.tex_id)
    img = Image.fromarray(r.render())
    d = ImageDraw.Draw(img)
    s = sc.sim
    text = f"{label}\nt = {s.t:4.0f} s    seen by honest drones {100 * s.m.coverage_honest[-1]:5.1f}%"
    fakes = [f for a in s.drones if not a.honest for f in getattr(a, "fakes", [])]
    if fakes:
        caught = sum(all(not h.accept[f.id] for h in s.drones if h.honest) for f in fakes)
        text += f"\nfake identities used {len(fakes)}, caught {caught}"
    d.rectangle([0, 0, W, 64], fill=(0, 0, 0))
    d.multiline_text((10, 6), text, fill=(235, 235, 235), font=ImageFont.load_default(), spacing=4)
    return np.asarray(img)


def main(family="office", seed=0, attack="sybil_strong", fakes=4, left="D0", right="D3", t_end=600.0):
    names = {"D0": "no defence", "D1": "audit", "D2": "audit + presence", "D3": "full defence", "D4": "full defence + quarantine"}
    sims = [make(family, seed, attack, fakes, d) for d in (left, right)]
    scenes = [Scene3D(s) for s in sims]
    rens = [mujoco.Renderer(sc.model, H, W, max_geom=100000) for sc in scenes]
    frames = []
    t = 0.0
    while t <= t_end and not all(s.done() for s in sims):
        row = []
        for s, sc, r, d in zip(sims, scenes, rens, (left, right)):
            while s.t < t - 1e-9 and not s.done():
                sc.remember()
                s.step()
            if not s.m.t or s.t - s.m.t[-1] >= 1.0:
                s.record()
            sc.move_drones(1.0)
            cam = mujoco.MjvCamera()
            cam.lookat[:] = [sc.L / 2, sc.W / 2, 0]
            cam.distance, cam.elevation, cam.azimuth = 0.95 * sc.L, -50, 100 + 0.08 * t
            row.append(frame(sc, r, cam, names[d]))
        frames.append(np.concatenate(row, 1))
        t += STEP
    base = os.path.join(ROOT, "media", f"{family}{seed}_{attack}{fakes}_{left}_{right}")
    imageio.mimsave(base + ".mp4", frames, fps=20, quality=8, macro_block_size=1)
    small = [np.asarray(Image.fromarray(f).resize((f.shape[1] * 2 // 5, f.shape[0] * 2 // 5))) for f in frames[::5]]
    imageio.mimsave(base + ".gif", small, duration=0.1, loop=0)
    imageio.imwrite(base + "_end.png", frames[-1])
    print(base)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]), a[2], int(a[3]), a[4], a[5]) if len(a) >= 6 else main()
