"""Multi-drone cooperative exploration with a shared voxel map over an imperfect radio network."""
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.stats import beta

from . import kernels as K

F = 2


@dataclass
class Params:
    n_drones: int = 8
    dt: float = 0.2
    t_max: float = 900.0
    v_max: float = 1.5  # m/s
    a_max: float = 3.0  # m/s^2
    yaw_rate: float = np.radians(120)
    radius: float = 0.12
    hfov: float = np.radians(90)
    vfov: float = np.radians(60)
    rays_h: int = 48
    rays_v: int = 32
    max_range: float = 6.0
    depth_noise: float = 0.015
    dropout: float = 0.02
    clear_range: float = 5.5
    sensor: str = "camera"
    lidar_h: int = 120
    lidar_v: int = 16
    lidar_vmin: float = np.radians(-7)
    lidar_vmax: float = np.radians(52)
    lidar_range: float = 30.0
    lidar_clear: float = 28.0
    lidar_noise: float = 0.002
    presence_range_lidar: float = 15.0
    ranger_range: float = 4.0
    ranger_half_angle: float = np.radians(13.5)
    comm_range: float = 25.0
    bandwidth: float = 50_000.0
    packet: int = 500
    loss: float = 0.0
    claim_radius: float = 6.0
    claim_ttl: float = 15.0
    replan_every: float = 4.0
    scan_time: float = 3.0
    launch_gap: float = 1.0
    segment: int = 6
    auth: bool = False
    defence: str = "none"
    trust_tol: float = 0.02
    trust_conf: float = 0.99
    audit_every: float = 1.0
    audit_tol: int = 2
    audit_complete: float = 0.7
    vouch_checks: int = 2000
    audit_vouched: bool = False
    presence: bool = False
    presence_detect: float = 0.9
    presence_conf: float = 0.99
    trust_needs_body: bool = False
    trust_by_type: bool = False
    verify_free: bool = False
    verify_weight: float = 0.3
    seed: int = 0


@dataclass
class Metrics:
    t: list = field(default_factory=list)
    coverage: list = field(default_factory=list)
    coverage_trusted: list = field(default_factory=list)
    coverage_honest: list = field(default_factory=list)
    coverage_mean: list = field(default_factory=list)
    wrong: list = field(default_factory=list)
    flown: list = field(default_factory=list)
    crashed: list = field(default_factory=list)
    sent: list = field(default_factory=list)
    overlap: list = field(default_factory=list)
    infected: list = field(default_factory=list)
    idle: list = field(default_factory=list)


def _ranger_dirs(p):
    out = []
    for sgn in (1.0, -1.0):
        out.append([0.0, 0.0, sgn])
        for ang in (p.ranger_half_angle / 2, p.ranger_half_angle):
            for az in (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi) if ang == p.ranger_half_angle else (0.25 * np.pi, 0.75 * np.pi, 1.25 * np.pi, 1.75 * np.pi):
                out.append([np.sin(ang) * np.cos(az), np.sin(ang) * np.sin(az), sgn * np.cos(ang)])
    return np.array(out)


def _lidar_dirs(p):
    az = np.linspace(-np.pi, np.pi, p.lidar_h, endpoint=False)
    el = np.linspace(p.lidar_vmin, p.lidar_vmax, p.lidar_v)
    A, E = np.meshgrid(az, el)
    return np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)


def _camera_dirs(p):
    az = np.linspace(-p.hfov / 2, p.hfov / 2, p.rays_h)
    el = np.linspace(-p.vfov / 2, p.vfov / 2, p.rays_v)
    A, E = np.meshgrid(az, el)
    d = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)
    return d


class Drone:
    honest = True
    truthful_own = True

    def __init__(self, ident, pos, sim):
        self.id = ident
        self.sim = sim
        shape = sim.world.occ.shape
        n = int(np.prod(shape))
        self.state = np.full(n, K.UNKNOWN, np.int8)
        self.src = np.full(n, -1, np.int16)
        self.lo = np.zeros(n, np.int8)
        cap = 1 << 16
        self.log_idx = np.empty(cap, np.int32)
        self.log_val = np.empty(cap, np.int8)
        self.log_org = np.empty(cap, np.int16)
        self.n_log = 0
        self.agree = np.zeros(sim.n_ids, np.int64)
        self.disagree = np.zeros(sim.n_ids, np.int64)
        self.accept = np.ones(sim.n_ids, np.bool_)
        self.far = np.zeros(sim.n_ids, np.int64)
        self.far_occ = np.zeros(sim.n_ids, np.int64)
        self.checked_occ = np.zeros(sim.n_ids, np.int64)
        cap_ev = 1 << 18
        self.ev = [(np.empty(cap_ev, np.int32), np.empty(cap_ev, np.int16), np.empty(cap_ev, np.int8), np.zeros(1, np.int64))
                   for _ in range(2)]
        self.next_audit = sim.p.audit_every
        self.claims = {}
        self.beacons = {}
        self.seen = np.zeros(sim.n_ids, np.int64)
        self.absent = np.zeros(sim.n_ids, np.int64)
        self.pos = np.array(pos, float)
        self.vel = np.zeros(3)
        self.yaw = 0.0
        self.mode = "scan"
        self.scan_left = sim.p.scan_time * 2 + sim.p.launch_gap * ident
        self.path = None
        self.target = None
        self.next_plan = 0.0
        self.blocked = np.zeros(tuple(s // F for s in shape), bool)
        self.flown = 0.0
        self.crashed = False
        self.idle_since = None
        self.trail = [self.pos.copy()]
        self.guard_stops = 0
        self.escapes = 0
        self.escape_until = 0.0
        self.idle_time = 0.0
        self.seen_share = 0.0

    def _log(self, idx, val, org):
        m = len(idx)
        if self.n_log + m > len(self.log_idx):
            cap = max(2 * len(self.log_idx), self.n_log + m)
            for name in ("log_idx", "log_val", "log_org"):
                old = getattr(self, name)
                new = np.empty(cap, old.dtype)
                new[:self.n_log] = old[:self.n_log]
                setattr(self, name, new)
        self.log_idx[self.n_log:self.n_log + m] = idx
        self.log_val[self.n_log:self.n_log + m] = val
        self.log_org[self.n_log:self.n_log + m] = org
        self.n_log += m

    def sense(self):
        s = self.sim
        c, sn = np.cos(self.yaw), np.sin(self.yaw)
        R = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])
        dirs = s.cam @ R.T
        for d, rng, clear, noise in ((dirs, s.range_, s.clear_, s.noise_),
                                     (s.rangers, s.p.ranger_range, s.p.ranger_range - 4 * s.p.depth_noise * s.p.ranger_range,
                                      s.p.depth_noise)):
            n = K.raycast(s.world.occ, s.world.res, self.pos, d, rng, noise, s.p.dropout,
                          s.buf_idx, s.buf_val, clear)
            m = K.apply_own(self.state, self.src, self.lo, s.buf_idx, s.buf_val, n, self.id, s.buf_chg,
                            self.agree, self.disagree, s.stamp, s.frame, *self.ev[0], self.checked_occ)
            s.frame += 1
            f = s.buf_idx[s.buf_chg[:m]]
            self._log(f, self.state[f], self.id)

    def outgoing(self, a, b, peer):
        idx, val, org = self.log_idx[a:b], self.log_val[a:b], self.log_org[a:b]
        keep = self.accept[org]
        if keep.all():
            return idx, val, org
        return idx[keep], val[keep], org[keep]

    def suspects(self):
        p = self.sim.p
        checked = self.agree + self.disagree
        sus = (checked < p.vouch_checks) | (self.far > 0.5 * p.trust_tol * np.maximum(checked, 1))
        sus &= self.accept
        sus[self.id] = False
        return sus

    def vouched(self):
        v = ~self.suspects() & self.accept
        if self.sim.p.trust_needs_body:
            v &= self.seen > 0
        v[self.id] = True
        return v

    def audit(self):
        s = self.sim
        cell, org, val, n = self.ev[1]
        vouched = self.vouched() if s.p.audit_vouched else s.all_true
        K.audit(self.state, self.src, s.world.occ.shape, self.id, cell, org, val, int(n[0]), self.far, s.p.audit_tol,
                s.p.audit_complete, vouched, self.far_occ)
        n[0] = 0
        self.ev.reverse()
        if s.p.defence != "audit" or not self.honest:
            return
        checked = self.agree + self.disagree
        for o in np.flatnonzero((self.far > 0) & self.accept):
            if o == self.id:
                continue
            if beta.sf(s.p.trust_tol, 1 + self.far[o], 1 + checked[o] - self.far[o]) > s.p.trust_conf:
                self._revoke(o, "audit", far=int(self.far[o]), checked=int(checked[o]))
            elif s.p.trust_by_type and self.far_occ[o] > 0 and beta.sf(
                    s.p.trust_tol, 1 + self.far_occ[o], 1 + self.checked_occ[o] - self.far_occ[o]) > s.p.trust_conf:
                self._revoke(o, "audit-walls", far=int(self.far_occ[o]), checked=int(self.checked_occ[o]))
        if s.p.presence:
            self._presence_check()
            for o in np.flatnonzero((self.absent > 0) & self.accept):
                if o != self.id and beta.sf(0.5, 1 + self.absent[o], 1 + self.seen[o]) > s.p.presence_conf:
                    self._revoke(o, "absent", seen=int(self.seen[o]), absent=int(self.absent[o]))

    def _revoke(self, o, why, **info):
        s = self.sim
        self.accept[o] = False
        reset = K.revoke(self.state, self.src, self.lo, o)
        s.cursor[:, self.id] = 0
        s.revocations.append(dict(t=s.t, by=self.id, origin=int(o), reset=int(reset), why=why, **info))

    def _presence_check(self):
        s = self.sim
        p = s.p
        bodies = [d.pos for d in s.drones if not d.crashed and d is not self]
        used = set()
        fresh = [(o, q, s.t - t0) for o, (q, t0) in self.beacons.items()
                 if o != self.id and self.accept[o] and s.t - t0 <= 2 * p.dt]
        fresh.sort(key=lambda x: np.linalg.norm(x[1] - self.pos))
        for o, q, age in fresh:
            if not s.in_view(self, q):
                continue
            if K.wall_crossings(s.world.occ, s.world.res, self.pos, q) > 0:
                continue
            tol = 0.4 + p.v_max * age
            match = None
            for b, bp in enumerate(bodies):
                if b not in used and np.linalg.norm(bp - q) <= tol:
                    match = b
                    break
            if match is not None and s.rng.random() < p.presence_detect:
                used.add(match)
                self.seen[o] += 1
            elif match is None:
                self.absent[o] += 1

    def receive(self, idx, val, org, sender):
        chg = np.empty(len(idx), np.int64)
        m = K.apply_received(self.state, self.src, idx, val, org, self.accept, chg)
        q = chg[:m]
        self._log(idx[q], val[q], org[q])

    def claim(self):
        return [] if self.target is None else [(self.id, self.target)]

    def beacon(self):
        return [(self.id, self.pos.copy())]

    def plan(self):
        s = self.sim
        cstate, trav = K.planning_grid(self.state, s.world.occ.shape, F)
        start = self._coarse(self.pos)
        if not trav[start]:
            nb = [tuple(np.add(start, d)) for d in s.nbhd]
            nb = [c for c in nb if all(0 <= c[i] < trav.shape[i] for i in range(3)) and trav[c]]
            if not nb:
                return self._escape(cstate, trav, start)
            start = min(nb, key=lambda c: np.linalg.norm(self._center(c) - self.pos))
        fr = K.frontiers(cstate, trav, self.blocked)
        dist, parent = K.bfs(trav, np.array(start))
        dist = dist.reshape(trav.shape)
        fr &= dist >= 0
        weight = fr.astype(float)
        if s.p.defence == "audit":
            sus = self.suspects()
            if s.p.trust_needs_body:
                sus |= self.accept & (self.seen == 0)
                sus[self.id] = False
            if sus.any():
                sc = K.suspect_cells(self.state, self.src, s.world.occ.shape, F, self.id, sus)
                vf = K.verify_frontiers(trav, sc, self.blocked, fr) & (dist >= 0)
                if s.p.verify_free:
                    sf = K.suspect_free_cells(self.state, self.src, s.world.occ.shape, F, self.id, sus)
                    vf |= sf & trav & ~fr & ~self.blocked & (dist >= 0)
                weight[vf] = s.p.verify_weight
                fr |= vf
        if not fr.any():
            return False
        lab, nl = ndimage.label(fr, structure=np.ones((3, 3, 3)))
        cells = np.argwhere(fr)
        blk = cells[:, :2] // s.p.segment
        key = np.stack([lab[fr], blk[:, 0], blk[:, 1]], 1)
        _, L = np.unique(key, axis=0, return_inverse=True)
        L = L.ravel() + 1
        nl = int(L.max())
        D = dist[fr]
        size = np.bincount(L, weights=weight[fr], minlength=nl + 1)
        order = np.lexsort((D, L))
        first = np.ones(len(order), bool)
        first[1:] = L[order][1:] != L[order][:-1]
        cand = order[first]
        tc = cells[cand]
        tpos = s.world.res * F * (tc + 0.5)
        util = size[L[cand]].astype(float)
        now = s.t
        for org, (q, t0) in list(self.claims.items()):
            if now - t0 > s.p.claim_ttl:
                del self.claims[org]
                continue
            d = np.linalg.norm(tpos - q, axis=1)
            util *= np.clip(d / s.p.claim_radius, 0.0, 1.0) * 0.9 + 0.1
        score = util / (D[cand] * s.world.res * F + 2.0)
        best = int(np.argmax(score))
        goal = tuple(tc[best])
        cy, cz = trav.shape[1], trav.shape[2]
        u = (goal[0] * cy + goal[1]) * cz + goal[2]
        chain = []
        while u >= 0:
            chain.append((u // (cy * cz), (u // cz) % cy, u % cz))
            u = parent[u]
        chain = chain[::-1]
        pts = [chain[0]]
        i = 0
        while i < len(chain) - 1:
            j = len(chain) - 1
            while j > i + 1 and not K.segment_clear(trav, np.array(chain[i], float), np.array(chain[j], float)):
                j -= 1
            pts.append(chain[j])
            i = j
        self.path = [self._center(c) for c in pts]
        self.anchor = self.pos.copy()
        self.target = self._center(goal)
        self.goal_cell = goal
        self.escape_until = 0.0
        self.mode = "fly"
        self.next_plan = s.t + s.p.replan_every
        return True

    def escaping(self):
        return self.mode == "fly" and self.sim.t < self.escape_until

    def _escape(self, cstate, trav, start):
        mask = (cstate != 1) | trav
        mask[start] = True
        dist, parent = K.bfs6(mask, np.array(start), 15)
        dist = dist.reshape(trav.shape)
        cand = np.argwhere(trav & (dist > 0))
        if len(cand) == 0:
            return False
        goal = tuple(cand[np.argmin(dist[tuple(cand.T)])])
        cy, cz = trav.shape[1], trav.shape[2]
        u = (goal[0] * cy + goal[1]) * cz + goal[2]
        chain = []
        while u >= 0:
            chain.append((u // (cy * cz), (u // cz) % cy, u % cz))
            u = parent[u]
        chain = chain[::-1][1:]
        self.path = [self._center(c) for c in chain]
        self.anchor = self.pos.copy()
        self.target = self._center(goal)
        self.goal_cell = goal
        self.mode = "fly"
        self.escapes += 1
        hop = len(chain) * self.sim.world.res * F / 0.3 + 2.0
        self.escape_until = self.next_plan = self.sim.t + hop
        return True

    def _coarse(self, p):
        c = np.floor(p / (self.sim.world.res * F)).astype(int)
        return tuple(np.clip(c, 0, np.array(self.blocked.shape) - 1))

    def _center(self, c):
        return self.sim.world.res * F * (np.asarray(c) + 0.5)

    def step(self):
        s = self.sim
        p = s.p
        if self.crashed:
            return
        self.sense()
        if s.t >= self.next_audit:
            self.next_audit += p.audit_every
            self.audit()
        if self.mode == "scan":
            self.yaw += p.yaw_rate * p.dt
            self.scan_left -= p.dt
            self._brake()
            if self.scan_left <= 0:
                self.mode = "plan"
        if self.mode == "fly" and s.t >= self.next_plan:
            self.mode = "plan"
        if self.mode == "fly" and not self.escaping() and K.path_blocked(self.state, s.world.occ.shape, s.world.res,
                                                                         np.array([self.pos] + self.path[:4]), 0.2):
            self.mode = "plan"
        if self.mode in ("plan", "idle"):
            if self.mode == "plan" or s.t >= self.next_plan:
                if self.plan():
                    self.idle_since = None
                else:
                    self.mode = "idle"
                    self.target = None
                    self.next_plan = s.t + 2.0
                    if self.idle_since is None:
                        self.idle_since = s.t
            if self.mode == "idle":
                self.idle_time += p.dt
                self._brake()
                self.yaw += 0.25 * p.yaw_rate * p.dt
        if self.mode == "fly":
            self._follow()
        if K.sphere_hits(s.world.occ, s.world.res, self.pos, p.radius):
            self.crashed = True
            self.mode = "crashed"
            self.vel[:] = 0

    def _brake(self):
        p = self.sim.p
        dv = -self.vel
        n = np.linalg.norm(dv)
        if n > p.a_max * p.dt:
            dv *= p.a_max * p.dt / n
        self._move(dv)

    def _follow(self):
        p = self.sim.p
        while self.path:
            a, b = self.anchor, self.path[0]
            ab = b - a
            L2 = ab @ ab
            t = 1.0 if L2 < 1e-9 else (self.pos - a) @ ab / L2
            if t >= 1.0 or np.linalg.norm(self.pos - b) < (0.2 if len(self.path) == 1 else 0.1):
                self.anchor = self.path.pop(0)
            else:
                break
        if not self.path:
            self.mode = "scan"
            self.scan_left = p.scan_time
            self.path = None
            g = self.goal_cell
            self.blocked[max(g[0] - 1, 0):g[0] + 2, max(g[1] - 1, 0):g[1] + 2, max(g[2] - 1, 0):g[2] + 2] = True
            self._brake()
            return
        a, b = self.anchor, self.path[0]
        ab = b - a
        L = np.linalg.norm(ab)
        t = np.clip(0.0 if L < 1e-9 else (self.pos - a) @ ab / L ** 2, 0.0, 1.0)
        proj = a + t * ab
        look = 0.1 if self.escaping() else 0.3
        carrot = proj
        pts = [b] + self.path[1:]
        cur = proj
        for q in pts:
            seg = np.linalg.norm(q - cur)
            if seg >= look:
                carrot = cur + (q - cur) * (look / seg)
                break
            look -= seg
            cur = carrot = q
        to = carrot - self.pos
        d_to = np.linalg.norm(to)
        d_next = np.linalg.norm(b - self.pos)
        if len(self.path) == 1:
            v_corner = 0.0
        else:
            u1 = ab / max(L, 1e-9)
            u2 = self.path[1] - b
            u2 = u2 / max(np.linalg.norm(u2), 1e-9)
            v_corner = p.v_max * max(0.1, u1 @ u2) ** 3
        v_lim = np.sqrt(v_corner ** 2 + 2 * 0.5 * p.a_max * d_next)
        want = np.arctan2(to[1], to[0]) if np.linalg.norm(to[:2]) > 0.05 else self.yaw
        err = (want - self.yaw + np.pi) % (2 * np.pi) - np.pi
        self.yaw += np.clip(err, -p.yaw_rate * p.dt, p.yaw_rate * p.dt)
        speed = min(p.v_max, v_lim) * max(np.cos(err), 0.0) ** 2
        v_des = to / max(d_to, 1e-9) * speed
        dv = v_des - self.vel
        n = np.linalg.norm(dv)
        if n > p.a_max * p.dt:
            dv *= p.a_max * p.dt / n
        v = self.vel + dv
        sp = np.linalg.norm(v)
        shape, res = self.sim.world.occ.shape, self.sim.world.res
        if self.escaping():
            if sp > 0.3:
                dv = v * (0.3 / sp) - self.vel
        elif K.near_known_occ(self.state, shape, res, self.pos, p.radius + 0.05):
            if sp > 0.3:
                dv = v * (0.3 / sp) - self.vel
        elif sp > 1e-6:
            d_stop = sp * p.dt + sp ** 2 / (2 * p.a_max)
            for f in np.linspace(0.25, 1.0, 4):
                if K.near_known_occ(self.state, shape, res, self.pos + v / sp * (d_stop * f), p.radius + 0.05):
                    self.guard_stops += 1
                    self.mode = "plan"
                    self._brake()
                    return
        self._move(dv)

    def _move(self, dv):
        dt = self.sim.p.dt
        v0 = self.vel.copy()
        self.vel += dv
        step = 0.5 * (v0 + self.vel) * dt
        self.pos += step
        self.flown += np.linalg.norm(step)
        if np.linalg.norm(self.pos - self.trail[-1]) > 0.5:
            self.trail.append(self.pos.copy())


class Sim:
    def __init__(self, world, params=Params(), drone_types=None):
        self.world = world
        self.p = params
        self.t = 0.0
        self.rng = np.random.default_rng(params.seed)
        K.seed(params.seed)
        self.n_ids = 256
        self.all_true = np.ones(self.n_ids, np.bool_)
        if params.sensor == "lidar":
            self.cam = _lidar_dirs(params)
            self.range_, self.clear_, self.noise_ = params.lidar_range, params.lidar_clear, params.lidar_noise
        else:
            self.cam = _camera_dirs(params)
            self.range_, self.clear_, self.noise_ = params.max_range, params.clear_range, params.depth_noise
        self.rangers = _ranger_dirs(params)
        cap = len(self.cam) * int(3 * self.range_ / world.res + 4)
        self.buf_idx = np.empty(cap, np.int32)
        self.buf_val = np.empty(cap, np.int8)
        self.buf_chg = np.empty(cap, np.int64)
        self.stamp = np.zeros(world.occ.size, np.int64)
        self.frame = 1
        self.nbhd = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)]
        types = drone_types or [Drone] * params.n_drones
        self.drones = [T(i, world.starts[i], self) for i, T in enumerate(types)]
        n = len(self.drones)
        self.cursor = np.zeros((n, n), np.int64)
        self.pending = [[[] for _ in range(n)] for _ in range(n)]
        self.sent_total = 0
        self.links = np.zeros((n, n), bool)
        self.revocations = []
        self._truth_setup()
        self.m = Metrics()

    def _truth_setup(self):
        occ = self.world.occ
        truth = np.where(occ, K.OCC, K.FREE).astype(np.int8).ravel()
        self.truth = truth
        _, trav = K.planning_grid(truth, occ.shape, F)
        start = tuple(np.floor(self.world.starts[0] / (self.world.res * F)).astype(int))
        dist, _ = K.bfs(trav, np.array(start))
        reach = (dist >= 0).reshape(trav.shape)
        fine = np.repeat(np.repeat(np.repeat(reach, F, 0), F, 1), F, 2)
        full = np.zeros(occ.shape, bool)
        full[:fine.shape[0], :fine.shape[1], :fine.shape[2]] = fine
        surf = ndimage.binary_dilation(full) & occ
        self.target_cells = np.flatnonzero(full | surf)
        self.reach = reach
        cube = np.ones((3, 3, 3), bool)
        self.occ_near = ndimage.binary_dilation(occ, cube).ravel()
        self.free_near = ndimage.binary_dilation(~occ, cube).ravel()

    def in_view(self, d, q):
        p = self.p
        v = q - d.pos
        r = np.linalg.norm(v)
        if r < 0.5:
            return False
        el = np.arcsin(np.clip(v[2] / r, -1, 1))
        if p.sensor == "lidar":
            if r > p.presence_range_lidar or el < p.lidar_vmin or el > p.lidar_vmax:
                return False
        else:
            if r > p.max_range:
                return False
            c, sn = np.cos(d.yaw), np.sin(d.yaw)
            fwd = v[0] * c + v[1] * sn
            side = -v[0] * sn + v[1] * c
            if fwd <= 0 or abs(np.arctan2(side, fwd)) > p.hfov / 2 or abs(el) > p.vfov / 2:
                return False
        return K.wall_crossings(self.world.occ, self.world.res, d.pos, q) == 0

    def _update_links(self):
        n = len(self.drones)
        self.links[:] = False
        for i in range(n):
            if self.drones[i].crashed:
                continue
            for j in range(i + 1, n):
                if self.drones[j].crashed:
                    continue
                a, b = self.drones[i].pos, self.drones[j].pos
                d = np.linalg.norm(a - b)
                if d > self.p.comm_range:
                    continue
                w = K.wall_crossings(self.world.occ, self.world.res, a, b)
                if d <= self.p.comm_range * 0.5 ** w:
                    self.links[i, j] = self.links[j, i] = True

    def _exchange(self):
        p = self.p
        n = len(self.drones)
        for i in range(n):
            nb = np.flatnonzero(self.links[i])
            if len(nb) == 0:
                continue
            di = self.drones[i]
            budget = int(p.bandwidth * p.dt / len(nb))
            for j in self.rng.permutation(nb):
                dj = self.drones[j]
                for org, q in di.claim():
                    if self.rng.random() >= p.loss:
                        dj.claims[org] = (q, self.t)
                for org, q in di.beacon():
                    if self.rng.random() >= p.loss:
                        dj.beacons[org] = (q, self.t)
                ranges = self.pending[i][j]
                self.pending[i][j] = []
                left = budget
                todo = []
                while ranges and left > 0:
                    a, b = ranges.pop(0)
                    take = min(b - a, left)
                    todo.append((a, a + take))
                    if a + take < b:
                        ranges.insert(0, (a + take, b))
                    left -= take
                self.pending[i][j] = ranges
                if left > 0 and self.cursor[i, j] < di.n_log:
                    a = int(self.cursor[i, j])
                    b = min(di.n_log, a + left)
                    todo.append((a, b))
                    self.cursor[i, j] = b
                for a, b in todo:
                    for pa in range(a, b, p.packet):
                        pb = min(pa + p.packet, b)
                        self.sent_total += pb - pa
                        if self.rng.random() < p.loss:
                            self.pending[i][j].append((pa, pb))
                            continue
                        idx, val, org = di.outgoing(pa, pb, j)
                        keep = org != j
                        if keep.any():
                            dj.receive(idx[keep], val[keep], org[keep], i)

    def honest(self):
        return [d for d in self.drones if d.honest]

    def record(self):
        H = self.honest()
        T = self.target_cells
        live = [d for d in H if not d.crashed]
        seen = np.zeros(len(T), bool)
        trusted = np.zeros(len(T), bool)
        honest = np.zeros(len(T), bool)
        times = np.zeros(len(T), np.int16)
        for d in self.drones:
            if not d.truthful_own:
                continue
            own = d.src[T] == d.id
            d.seen_share = float(own.mean())
            times += own
            seen |= own
            if sum(not h.accept[d.id] for h in live) <= len(live) / 2:
                trusted |= own
            if d.honest:
                honest |= own
        per = []
        wrong = 0
        for d in H:
            st = d.state[T]
            good = ((st == K.OCC) & self.occ_near[T]) | ((st == K.FREE) & self.free_near[T])
            per.append(np.mean(good))
            wrong += int(((d.state == K.OCC) & ~self.occ_near).sum() + ((d.state == K.FREE) & ~self.free_near).sum())
        m = self.m
        m.t.append(self.t)
        m.coverage.append(float(seen.mean()))
        m.coverage_trusted.append(float(trusted.mean()))
        m.coverage_honest.append(float(honest.mean()))
        m.coverage_mean.append(float(np.mean(per)))
        m.overlap.append(float((times >= 2).sum() / max(seen.sum(), 1)))
        faked = np.zeros(self.world.occ.size, bool)
        for d in self.drones:
            if hasattr(d, "faked"):
                faked |= d.faked
        m.infected.append(sum(int((faked & (d.state != K.UNKNOWN) & (d.src != d.id)).sum() >= 100) for d in H)
                          if faked.any() else 0)
        m.idle.append(sum(d.mode == "idle" for d in H))
        m.wrong.append(wrong / max(len(H), 1))
        m.flown.append(float(sum(d.flown for d in H)))
        m.crashed.append(sum(d.crashed for d in H))
        m.sent.append(self.sent_total)

    def done(self):
        H = [d for d in self.honest() if not d.crashed]
        return not H or all(d.mode == "idle" and d.idle_since is not None and self.t - d.idle_since > 10 for d in H)

    def step(self):
        for d in self.drones:
            d.step()
        self._update_links()
        self._exchange()
        self.t += self.p.dt

    def run(self, record_every=2.0, callback=None):
        next_rec = 0.0
        while self.t < self.p.t_max and not self.done():
            if self.t >= next_rec:
                self.record()
                next_rec += record_every
                if callback:
                    callback(self)
            self.step()
        self.record()
        return self.m
