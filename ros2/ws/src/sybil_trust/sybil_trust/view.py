import math

import numpy as np


class VoxelMap:
    def __init__(self, occ, res):
        self.occ = occ
        self.res = res

    @classmethod
    def from_points(cls, xyz, res):
        idx = np.floor(np.asarray(xyz, float) / res).astype(int)
        occ = np.zeros(idx.max(0) + 1, bool) if len(idx) else np.zeros((1, 1, 1), bool)
        occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        return cls(occ, res)

    def centres(self):
        return (np.argwhere(self.occ) + 0.5) * self.res

    def wall_crossings(self, p, q):
        nx, ny, nz = self.occ.shape
        res = self.res
        L = math.sqrt((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 + (q[2] - p[2]) ** 2)
        steps = int(L / (0.5 * res)) + 1
        prev, n = False, 0
        for s in range(steps + 1):
            t = s / steps
            i = int((p[0] + t * (q[0] - p[0])) / res)
            j = int((p[1] + t * (q[1] - p[1])) / res)
            k = int((p[2] + t * (q[2] - p[2])) / res)
            if i < 0 or j < 0 or k < 0 or i >= nx or j >= ny or k >= nz:
                continue
            o = bool(self.occ[i, j, k])
            if o and not prev:
                n += 1
            prev = o
        return n


class Camera:
    def __init__(self, max_range=6.0, hfov=math.radians(90), vfov=math.radians(60)):
        self.max_range, self.hfov, self.vfov = max_range, hfov, vfov

    def sees(self, pos, yaw, q, voxels):
        v = np.asarray(q, float) - np.asarray(pos, float)
        r = float(np.linalg.norm(v))
        if r < 0.5 or r > self.max_range:
            return False
        el = math.asin(max(-1.0, min(1.0, v[2] / r)))
        c, s = math.cos(yaw), math.sin(yaw)
        fwd = v[0] * c + v[1] * s
        side = -v[0] * s + v[1] * c
        if fwd <= 0 or abs(math.atan2(side, fwd)) > self.hfov / 2 or abs(el) > self.vfov / 2:
            return False
        return voxels is not None and voxels.wall_crossings(pos, q) == 0
