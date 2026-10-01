"""Top-down pictures of the team map and the drones' tracks."""
import matplotlib.pyplot as plt
import numpy as np

from . import kernels as K

COLORS = plt.get_cmap("tab20").colors


def team_state(sim):
    occ = np.zeros(sim.world.occ.size, np.int16)
    known = np.zeros(sim.world.occ.size, np.int16)
    for d in sim.honest():
        if d.crashed:
            continue
        occ += d.state == K.OCC
        known += d.state != K.UNKNOWN
    st = np.full(sim.world.occ.size, K.UNKNOWN, np.int8)
    st[known > 0] = K.FREE
    st[2 * occ > known] = K.OCC
    return st.reshape(sim.world.occ.shape)


def topdown(sim, ax, title=None):
    w = sim.world
    st = team_state(sim)[:, :, 2:-2]
    img = np.full(st.shape[:2] + (3,), 0.16)
    known = (st != K.UNKNOWN).any(2)
    img[known] = 0.93
    obst = (st == K.OCC).any(2)
    img[obst] = 0.55
    at_h = st[:, :, int(1.2 / w.res) - 2] == K.OCC
    img[at_h] = 0.08
    ax.imshow(img.transpose(1, 0, 2), origin="lower", extent=[0, w.size[0], 0, w.size[1]], interpolation="nearest")
    for d in sim.drones:
        tr = np.array(d.trail + [d.pos])
        c = COLORS[d.id % 20] if d.honest else (0.85, 0.05, 0.05)
        ax.plot(tr[:, 0], tr[:, 1], "-", lw=0.7, color=c, alpha=0.9)
        ax.plot(*d.pos[:2], "x" if d.crashed else "o", ms=4, color=c, mec="k", mew=0.4)
    ax.set_xlim(0, w.size[0]), ax.set_ylim(0, w.size[1])
    ax.set_xticks([]), ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)
