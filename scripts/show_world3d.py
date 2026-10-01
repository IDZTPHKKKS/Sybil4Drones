#!/usr/bin/env python3
"""Browse the maps in 3D (the true world, as 0.4 m blocks coloured by height).

    mjpython scripts/show_world3d.py                 # office, seed 0
    mjpython scripts/show_world3d.py warehouse 3
"""
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
from matplotlib import colormaps
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from swarm import worlds  # noqa: E402

F = 2
L, W = 48.0, 32.0


def blocks(world):
    occ = world.occ
    nx, ny, nz = (s // F for s in occ.shape)
    b = occ[:nx * F, :ny * F, :nz * F].reshape(nx, F, ny, F, nz, F).any(axis=(1, 3, 5))
    b[:, :, -1] = False  # ceiling
    b[:, :, 0] = False
    inner = ndimage.binary_erosion(b, border_value=1)
    return np.argwhere(b & ~inner), nz


def fill(scn, world):
    cells, nz = blocks(world)
    cmap = colormaps["viridis"]
    half = np.full(3, 0.5 * F * world.res * 0.96)
    eye = np.eye(3).ravel()
    n = 0
    for c in cells[: scn.maxgeom - 64]:
        rgba = np.array(cmap(0.1 + 0.85 * (c[2] - 1) / max(nz - 2, 1)), np.float32)
        mujoco.mjv_initGeom(scn.geoms[n], mujoco.mjtGeom.mjGEOM_BOX, half, (c + 0.5) * F * world.res, eye, rgba)
        n += 1
    for p in world.starts[:16]:
        mujoco.mjv_initGeom(scn.geoms[n], mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, 0.15), p, eye,
                            np.array([0.9, 0.1, 0.1, 1], np.float32))
        n += 1
    scn.ngeom = n
    return len(cells)


def main():
    fam = sys.argv[1] if len(sys.argv) > 1 else "office"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    state = dict(fam=worlds.FAMILIES.index(fam), seed=seed, dirty=True)

    def key(code):
        c = chr(code) if code < 256 else ""
        if c in "Nn":
            state["seed"] += 1
        elif c in "Pp":
            state["seed"] = max(0, state["seed"] - 1)
        elif c in "Ff":
            state["fam"] = (state["fam"] + 1) % len(worlds.FAMILIES)
        else:
            return
        state["dirty"] = True

    spec = mujoco.MjSpec()
    spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
    spec.stat.center = [L / 2, W / 2, 0]
    spec.stat.extent = L / 2
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     width=256, height=256, rgb1=[0.82, 0.83, 0.86], rgb2=[0.74, 0.75, 0.79])
    spec.add_material(name="grid", texrepeat=[24, 16], texuniform=False).textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[L / 2, W / 2, 0.1], pos=[L / 2, W / 2, 0],
                            material="grid")
    spec.worldbody.add_light(pos=[L / 2, W / 2, 30], dir=[0, 0, -1], diffuse=[0.5, 0.5, 0.5],
                             type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    m = spec.compile()
    d = mujoco.MjData(m)
    with mujoco.viewer.launch_passive(m, d, key_callback=key, show_left_ui=False, show_right_ui=False) as v:
        v.cam.lookat[:] = [L / 2, W / 2, 0]
        v.cam.distance = 45
        v.cam.elevation = -50
        v.cam.azimuth = 100
        while v.is_running():
            if state["dirty"]:
                state["dirty"] = False
                f = worlds.FAMILIES[state["fam"]]
                w = worlds.make(f, state["seed"])
                with v.lock():
                    nb = fill(v.user_scn, w)
                    v.set_texts((None, None, f"{f}, seed {state['seed']}   ({w.size[0]:.0f} x {w.size[1]:.0f} x {w.size[2]:.1f} m)\n"
                                             "N / P: next / previous seed    F: next family", ""))
                print(f"{f} seed {state['seed']}: {nb} blocks", flush=True)
            v.sync()
            time.sleep(1 / 30)


if __name__ == "__main__":
    main()
