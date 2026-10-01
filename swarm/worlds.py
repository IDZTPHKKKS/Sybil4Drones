"""Procedural 3D worlds on a voxel grid."""
from dataclasses import dataclass

import numpy as np

RES = 0.2
FAMILIES = ("office", "warehouse", "forest", "tunnels")


@dataclass
class World:
    family: str
    seed: int
    occ: np.ndarray
    res: float
    starts: np.ndarray

    @property
    def size(self):
        return np.array(self.occ.shape) * self.res


class _Grid:
    def __init__(self, lx, ly, lz):
        self.occ = np.zeros((int(round(lx / RES)), int(round(ly / RES)), int(round(lz / RES))), dtype=bool)

    def box(self, x0, y0, z0, x1, y1, z1, value=True):
        i0 = np.floor(np.array([x0, y0, z0]) / RES + 1e-6).astype(int)
        i1 = np.ceil(np.array([x1, y1, z1]) / RES - 1e-6).astype(int)
        i0 = np.clip(i0, 0, self.occ.shape)
        i1 = np.clip(i1, 0, self.occ.shape)
        self.occ[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]] = value

    def cylinder(self, cx, cy, r, z0, z1):
        nx, ny, _ = self.occ.shape
        x = (np.arange(nx) + 0.5) * RES
        y = (np.arange(ny) + 0.5) * RES
        m = (x[:, None] - cx) ** 2 + (y[None, :] - cy) ** 2 <= r ** 2
        k0, k1 = int(np.floor(z0 / RES)), int(np.ceil(z1 / RES))
        self.occ[:, :, k0:k1] |= m[:, :, None]

    def shell(self):
        self.occ[[0, -1], :, :] = True
        self.occ[:, [0, -1], :] = True
        self.occ[:, :, [0, -1]] = True


def _starts(x, y0, y1, z, n=32):
    ys = np.arange(y0, y1, 0.8)
    pts = [(x + 0.8 * c, yy, z) for c in range(8) for yy in ys]
    return np.array(pts[:n])


def office(seed):
    rng = np.random.default_rng(seed)
    L, W, H = 48.0, 32.0, 3.2
    g = _Grid(L, W, H)
    g.shell()
    t = 0.2
    door_w, door_h = 1.6, 2.4
    cy = sorted(rng.uniform([9.0, 20.0], [11.0, 23.0]))
    cw = 2.4
    spine_x = rng.uniform(20, 28)
    bands = [(0.0, cy[0]), (cy[0] + cw, cy[1]), (cy[1] + cw, W)]
    launch_x = 5.0
    for bi, (y0, y1) in enumerate(bands):
        for yw in (y0, y1):
            if 0.5 < yw < W - 0.5:
                g.box(launch_x, yw - t / 2, 0, L, yw + t / 2, H)
        xs = [launch_x]
        x = launch_x
        while x < L - 5:
            x += rng.uniform(5.0, 9.0)
            xs.append(min(x, L))
        xs[-1] = L
        for x0, x1 in zip(xs[:-1], xs[1:]):
            if x1 - x0 < 2.5:
                continue
            g.box(x1 - t / 2, y0, 0, x1 + t / 2, y1, H)
            for yw in (y0, y1):
                if 0.5 < yw < W - 0.5:
                    dx = rng.uniform(x0 + 0.6, x1 - door_w - 0.6)
                    g.box(dx, yw - t, 0, dx + door_w, yw + t, door_h, value=False)
            if rng.random() < 0.4 and x1 < L - 1:
                dy = rng.uniform(y0 + 0.6, y1 - door_w - 0.6)
                g.box(x1 - t, dy, 0, x1 + t, dy + door_w, door_h, value=False)
            for _ in range(rng.integers(1, 4)):
                w, d = rng.uniform(1.2, 2.0), rng.uniform(0.7, 1.0)
                px, py = rng.uniform(x0 + 0.4, max(x0 + 0.5, x1 - w - 0.4)), rng.uniform(y0 + 0.4, max(y0 + 0.5, y1 - d - 0.4))
                g.box(px, py, 0.7, px + w, py + d, 0.8)
                for lx_, ly_ in ((px, py), (px + w - 0.1, py), (px, py + d - 0.1), (px + w - 0.1, py + d - 0.1)):
                    g.box(lx_, ly_, 0, lx_ + 0.1, ly_ + 0.1, 0.7)
            for _ in range(rng.integers(0, 3)):
                side = rng.integers(2)
                px = rng.uniform(x0 + 0.3, x1 - 1.0)
                py = y0 + t / 2 if side == 0 else y1 - t / 2 - 0.5
                g.box(px, py, 0, px + 0.8, py + 0.5, rng.uniform(1.8, 2.2))
    g.box(spine_x, t, 0, spine_x + cw, W - t, H, value=False)
    g.box(spine_x - t / 2, t, 0, spine_x + t / 2, W - t, H)
    g.box(spine_x + cw - t / 2, t, 0, spine_x + cw + t / 2, W - t, H)
    for yc in cy:
        g.box(spine_x - t, yc, 0, spine_x + cw + t, yc + cw, H, value=False)
    for (y0, y1) in bands:
        yy = rng.uniform(y0 + 0.6, y1 - door_w - 0.6)
        for xw in (spine_x, spine_x + cw):
            g.box(xw - t, yy, 0, xw + t, yy + door_w, door_h, value=False)
    g.shell()
    return g.occ, _starts(1.0, cy[0] - 5.0, cy[0] + cw + 5.0, 1.2)


def warehouse(seed):
    rng = np.random.default_rng(seed)
    L, W, H = 48.0, 32.0, 6.0
    g = _Grid(L, W, H)
    g.shell()
    x_start = 7.0
    depth = 1.2
    y = 2.0
    while y < W - 3.0:
        aisle = rng.uniform(2.4, 3.2)
        x = x_start
        while x < L - 3.0:
            seg = rng.uniform(7.0, 14.0)
            x1 = min(x + seg, L - 1.5)
            top = rng.uniform(3.8, 5.0)
            for xu in np.arange(x, x1 + 1e-6, 2.4):
                g.box(xu, y, 0, xu + 0.1, y + depth, top)
            for zl in np.arange(0.0, top, rng.uniform(1.2, 1.6)):
                g.box(x, y, zl, x1, y + depth, zl + 0.12)
                for _ in range(int((x1 - x) // 1.2)):
                    if rng.random() < 0.6:
                        gx = rng.uniform(x, x1 - 1.0)
                        g.box(gx, y + 0.1, zl + 0.12, gx + rng.uniform(0.6, 1.0), y + depth - 0.1, zl + rng.uniform(0.5, 1.0))
            if rng.random() < 0.3:
                g.box(x, y + depth / 2 - 0.05, 0, x1, y + depth / 2 + 0.05, top)
            x = x1 + rng.uniform(2.4, 3.6)
        y += depth + aisle
    for _ in range(rng.integers(3, 7)):
        px, py = rng.uniform(3.0, x_start - 1.5), rng.uniform(1.0, W - 2.0)
        g.box(px, py, 0, px + 1.2, py + 1.0, rng.uniform(0.6, 1.8))
    for cx in np.arange(8.0, L, 8.0):
        for cyy in np.arange(8.0, W, 8.0):
            g.box(cx - 0.2, cyy - 0.2, 0, cx + 0.2, cyy + 0.2, H)
    g.box(0.8, 11.0, 0, 5.5, 21.0, H, value=False)
    g.shell()
    return g.occ, _starts(1.0, 12.0, 20.0, 1.5)


def forest(seed):
    rng = np.random.default_rng(seed)
    L, W, H = 48.0, 32.0, 4.8
    g = _Grid(L, W, H)
    g.shell()
    trees = []
    for _ in range(6000):
        p = rng.uniform([6.0, 0.8], [L - 0.8, W - 0.8])
        dmin = 2.4 - 0.9 * (p[0] / L)
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > dmin ** 2 for q in trees):
            trees.append(p)
    for p in trees:
        r = rng.uniform(0.12, 0.35)
        g.cylinder(p[0], p[1], r, 0, H)
        for _ in range(rng.integers(0, 3)):
            a = rng.uniform(0, 2 * np.pi)
            ln = rng.uniform(0.6, 1.4)
            z = rng.uniform(1.0, 3.5)
            for s in np.linspace(0, ln, 8):
                bx, by = p[0] + s * np.cos(a), p[1] + s * np.sin(a)
                g.box(bx - 0.08, by - 0.08, z, bx + 0.08, by + 0.08, z + 0.12)
    for _ in range(rng.integers(15, 30)):  # bushes
        p = rng.uniform([6.0, 1.0], [L - 1.5, W - 1.5])
        g.box(p[0], p[1], 0, p[0] + rng.uniform(0.6, 1.4), p[1] + rng.uniform(0.6, 1.4), rng.uniform(0.5, 1.2))
    for _ in range(rng.integers(3, 7)):
        p = rng.uniform([8.0, 2.0], [L - 6, W - 2])
        a = rng.uniform(0, np.pi)
        for s in np.linspace(0, rng.uniform(3, 6), 30):
            bx, by = p[0] + s * np.cos(a), p[1] + s * np.sin(a)
            g.box(bx - 0.2, by - 0.2, 0, bx + 0.2, by + 0.2, 0.4)
    g.box(0.8, 11.0, 0, 5.5, 21.0, H, value=False)
    g.shell()
    return g.occ, _starts(1.0, 12.0, 20.0, 1.5)


def tunnels(seed):
    rng = np.random.default_rng(seed)
    L, W, H = 48.0, 32.0, 4.0
    g = _Grid(L, W, H)
    g.occ[:] = True
    cs = 4.0
    nx, ny = int(L // cs), int(W // cs)
    ceil = rng.uniform(2.4, 3.8, (nx, ny))
    seen = np.zeros((nx, ny), bool)
    edges = set()
    stack = [(0, ny // 2)]
    seen[stack[0]] = True
    while stack:
        x, y = stack[-1]
        nb = [(x + dx, y + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
              if 0 <= x + dx < nx and 0 <= y + dy < ny and not seen[x + dx, y + dy]]
        if not nb:
            stack.pop()
            continue
        n = nb[rng.integers(len(nb))]
        edges.add(tuple(sorted(((x, y), n))))
        seen[n] = True
        stack.append(n)
    for x in range(nx):
        for y in range(ny):
            for n in ((x + 1, y), (x, y + 1)):
                if n[0] < nx and n[1] < ny and rng.random() < 0.12:
                    edges.add(tuple(sorted(((x, y), n))))

    def centre(c):
        return (c[0] + 0.5) * cs, (c[1] + 0.5) * cs

    for x in range(nx):
        for y in range(ny):
            cx, cy = centre((x, y))
            r = rng.uniform(1.6, 3.4) / 2 if rng.random() < 0.2 else rng.uniform(0.9, 1.3)
            g.box(cx - r, cy - r, 0.2, cx + r, cy + r, ceil[x, y], value=False)
    for a, b in edges:
        (ax, ay), (bx, by) = centre(a), centre(b)
        w = rng.uniform(1.4, 2.6) / 2
        h = min(ceil[a], ceil[b])
        g.box(min(ax, bx) - w, min(ay, by) - w, 0.2, max(ax, bx) + w, max(ay, by) + w, h, value=False)
    free = np.argwhere(~g.occ[:, :, 3])
    for _ in range(rng.integers(25, 45)):
        i, j = free[rng.integers(len(free))] * RES
        if i < 6.0:
            continue
        if rng.random() < 0.5:
            g.box(i, j, 0, i + rng.uniform(0.3, 0.9), j + rng.uniform(0.3, 0.9), rng.uniform(0.4, 1.0))
        else:
            top = H
            g.box(i, j, rng.uniform(1.9, 2.6), i + rng.uniform(0.3, 0.7), j + rng.uniform(0.3, 0.7), top)
    g.box(0.8, 11.0, 0.2, 5.5, 21.0, 3.0, value=False)
    g.shell()
    return g.occ, _starts(1.0, 12.0, 20.0, 1.5)


def make(family, seed):
    occ, starts = {"office": office, "warehouse": warehouse, "forest": forest, "tunnels": tunnels}[family](seed)
    return World(family, seed, occ, RES, starts)
