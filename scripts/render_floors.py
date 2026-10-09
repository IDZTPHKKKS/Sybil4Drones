#!/usr/bin/env python3
"""Video of a mission in a multi-floor world: the team's map, floor by floor or as the whole building.

    python3 scripts/render_floors.py multistorey 0 D0          # floors side by side -> media/
    python3 scripts/render_floors.py atrium 0 D4 2 stacked     # whole building, frame every 2 s
"""
import os
import sys
import time

import imageio.v2 as imageio
import mujoco
import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
from run import DEFENCES  # noqa: E402
from swarm import kernels as K  # noqa: E402
from swarm import worlds  # noqa: E402
from swarm.attacks import ATTACKS  # noqa: E402
from swarm.sim import Drone, Params, Sim  # noqa: E402
from swarm.viz import team_state  # noqa: E402

F = 2
FLOOR_H = dict(multistorey=3.2, atrium=3.6, metro=5.0, carpark=3.0)
TEAM = [colormaps["tab10"](i)[:3] for i in (0, 2, 4, 6, 8, 9, 1, 5, 7, 3)]
W_PX, H_PX = 1600, 600
STACK_COLORS = [(0.20, 0.45, 0.85), (0.20, 0.70, 0.35), (0.95, 0.60, 0.15)]


def build(family, seed, attack, fakes, defence, t_max):
    p = Params(seed=seed, t_max=t_max)
    for k, v in DEFENCES[defence].items():
        setattr(p, k, v)
    types = [Drone] * p.n_drones
    if attack != "none":
        types[-1] = type(attack, (ATTACKS[attack],), dict(n_sybil=fakes))
    s = Sim(worlds.make(family, seed), p, drone_types=types)
    s.record()
    return s


class Floors:
    def __init__(self, sim, family, stacked=False):
        self.s = sim
        self.stacked = stacked
        w = sim.world
        self.L, self.Wd, self.H = w.size
        self.fh = FLOOR_H.get(family, self.H)
        self.n = int(round(self.H / self.fh))
        self.gap = 6.0
        self.cave = family not in FLOOR_H
        spec = mujoco.MjSpec()
        spec.visual.headlight.ambient = [0.5, 0.5, 0.5]
        spec.visual.headlight.diffuse = [0.45, 0.45, 0.45]
        spec.visual.global_.offwidth, spec.visual.global_.offheight = W_PX, H_PX
        spec.visual.map.zfar = 400
        spec.worldbody.add_light(pos=[0, 0, 80], dir=[0, 0, -1], diffuse=[0.4, 0.4, 0.4],
                                 type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0], rgba=[0, 0, 0, 0])
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        cx, cz = self.wrap()
        k = np.arange(cz)
        zc = (k + 0.5) * F * w.res
        self.floor_of_z = np.minimum((zc // self.fh).astype(int), self.n - 1)
        near_slab = np.abs(zc - self.fh * np.round(zc / self.fh)) < 0.3
        self.hide = near_slab if not self.cave else (zc > self.H - 1.2)
        T = sim.target_cells
        tz = (T % w.occ.shape[2]) * w.res
        self.t_floor = np.minimum((tz // self.fh).astype(int), self.n - 1)
        self.cmap = colormaps["viridis"]

    def wrap(self):
        nx, ny, nz = self.s.world.occ.shape
        return nx // F, nz // F

    def offset(self, z):
        f = min(int(z // self.fh), self.n - 1)
        if self.stacked:
            return np.zeros(3), f
        return np.array([f * (self.L + self.gap), 0.0, -f * self.fh]), f

    def floor_coverage(self):
        honest = np.zeros(len(self.s.target_cells), bool)
        for d in self.s.honest():
            honest |= d.src[self.s.target_cells] == d.id
        return [100 * honest[self.t_floor == f].mean() for f in range(self.n)]

    def draw(self, scn):
        s, w = self.s, self.s.world
        st = team_state(s)
        cx, cy, cz = (n // F for n in st.shape)
        occ = (st[:cx * F, :cy * F, :cz * F] == K.OCC).reshape(cx, F, cy, F, cz, F).any(axis=(1, 3, 5))
        faked = np.zeros(st.size, bool)
        for d in s.drones:
            if hasattr(d, "faked"):
                faked |= d.faked
        bad = ((st.ravel() == K.OCC) & faked & ~s.occ_near).reshape(st.shape)[:cx * F, :cy * F, :cz * F]
        wrong = bad.reshape(cx, F, cy, F, cz, F).any(axis=(1, 3, 5))
        occ[:, :, self.hide] = False
        if self.stacked:
            occ[:, :2, :] = False
        eye = np.eye(3).ravel()
        half = np.full(3, 0.5 * F * w.res * 0.95)
        alpha = 0.35 if self.cave else 1.0
        n = scn.ngeom
        for f in range(1 if self.stacked else self.n):
            off = np.array([f * (self.L + self.gap), 0, 0])
            mujoco.mjv_initGeom(scn.geoms[n], mujoco.mjtGeom.mjGEOM_BOX, np.array([self.L / 2, self.Wd / 2, 0.02]),
                                off + [self.L / 2, self.Wd / 2, -0.05], eye, np.array([0.85, 0.86, 0.88, 1], np.float32))
            n += 1
        cells = np.argwhere(occ)
        for c in cells[: scn.maxgeom - n - 2000]:
            p = (c + 0.5) * F * w.res
            off, f = self.offset(p[2])
            if wrong[tuple(c)]:
                rgba = np.array([0.9, 0.08, 0.08, 1.0], np.float32)
            elif self.stacked and not self.cave:
                rgba = np.array([*STACK_COLORS[f % 3], 0.45 if f < self.n - 1 else 0.3], np.float32)
            else:
                rgba = np.array([*self.cmap(0.1 + 0.8 * ((p[2] - f * self.fh) / self.fh))[:3], alpha], np.float32)
            mujoco.mjv_initGeom(scn.geoms[n], mujoco.mjtGeom.mjGEOM_BOX, half, p + off, eye, rgba)
            n += 1
        for d in s.drones:
            col = np.array([0.9, 0.08, 0.08, 1] if not d.honest else [*TEAM[d.id % 10], 1], np.float32)
            tr = (d.trail + [d.pos])[-150:]
            for a, b in zip(tr[:-1], tr[1:]):
                oa, fa = self.offset(a[2])
                ob, fb = self.offset(b[2])
                if fa != fb or n >= scn.maxgeom - 50:
                    continue
                mujoco.mjv_initGeom(scn.geoms[n], mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), eye, col)
                mujoco.mjv_connector(scn.geoms[n], mujoco.mjtGeom.mjGEOM_LINE, 3.0, a + oa, b + ob)
                n += 1
            off, _ = self.offset(d.pos[2])
            mujoco.mjv_initGeom(scn.geoms[n], mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, 0.55), d.pos + off, eye, col)
            n += 1
            for fk in getattr(d, "fakes", []):
                if fk.alive and n < scn.maxgeom - 1:
                    off, _ = self.offset(fk.pos[2])
                    mujoco.mjv_initGeom(scn.geoms[n], mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, 0.5), fk.pos + off, eye,
                                        np.array([1.0, 0.3, 0.3, 0.55], np.float32))
                    n += 1
        scn.ngeom = n


def main(family, seed=0, attack="sybil_strong", fakes=4, defence="D0", step=2.0, stacked=False):
    t_max = 1200.0 if family in FLOOR_H else 600.0
    s = build(family, seed, attack, fakes, defence, t_max)
    fl = Floors(s, family, stacked=stacked)
    r = mujoco.Renderer(fl.model, H_PX, W_PX, max_geom=60000)
    cam = mujoco.MjvCamera()
    width = fl.n * fl.L + (fl.n - 1) * fl.gap
    if stacked:
        cam.lookat[:] = [fl.L / 2, fl.Wd / 2, fl.H / 3]
        cam.distance, cam.elevation, cam.azimuth = 1.1 * fl.L, -30, 90
    elif fl.n > 1:
        cam.lookat[:] = [width / 2, fl.Wd / 2 - 3, 0]
        cam.distance, cam.elevation, cam.azimuth = 0.56 * width, -64, 90
    else:
        cam.lookat[:] = [width / 2, fl.Wd / 2 - 3, 0]
        cam.distance, cam.elevation, cam.azimuth = 1.05 * fl.L, -55, 100
    out = os.path.join(ROOT, "media", f"{family}{seed}_{attack}{fakes}_{defence}{'_stacked' if stacked else ''}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    video = imageio.get_writer(out, fps=20, quality=8, macro_block_size=1)
    font = ImageFont.load_default()
    t, t0 = 0.0, time.time()
    while t <= t_max and not s.done():
        while s.t < t - 1e-9 and not s.done():
            s.step()
            if s.t - s.m.t[-1] >= 1.0:
                s.record()
        r.update_scene(fl.data, cam)
        fl.draw(r.scene)
        img = Image.fromarray(r.render())
        dr = ImageDraw.Draw(img)
        names = ("ground floor", "first floor", "second floor") if fl.n > 1 else ("cave",)
        cov = fl.floor_coverage()
        label = f"{family}, {attack} with {fakes} fake identities, {'no defence' if defence == 'D0' else defence}    t = {s.t:4.0f} s"
        label += "    explored: " + ",  ".join(f"{nm} {c:.0f}%" for nm, c in zip(names, cov))
        dr.rectangle([0, 0, W_PX, 22], fill=(0, 0, 0))
        dr.text((10, 5), label, fill=(235, 235, 235), font=font)
        video.append_data(np.asarray(img))
        t += step
    video.close()
    print(f"{out}  ({time.time() - t0:.0f} s, ended at t = {s.t:.0f} s, explored {100 * s.m.coverage_honest[-1]:.1f}%)", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]) if len(a) > 1 else 0, defence=a[2] if len(a) > 2 else "D0",
         step=float(a[3]) if len(a) > 3 else 2.0, stacked=len(a) > 4 and a[4] == "stacked")
