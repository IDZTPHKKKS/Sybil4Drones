"""Compromised team members."""
import numpy as np
from scipy import ndimage

from . import kernels as K
from .sim import Drone

SYBIL_BASE = 100
_CROSS = ndimage.generate_binary_structure(3, 1)


class Attacker(Drone):
    honest = False
    forge = "self"

    def _origins(self, n):
        rng = self.sim.rng
        if self.forge == "self" or self.sim.p.auth:
            return np.full(n, self.id, np.int16)
        if self.forge == "honest":
            ids = [d.id for d in self.sim.drones if d.honest]
            return rng.choice(ids, n).astype(np.int16)
        return (SYBIL_BASE + rng.integers(0, 50, n)).astype(np.int16)


class Passive(Attacker):
    pass


class Blackhole(Attacker):
    drop = 1.0

    def outgoing(self, a, b, peer):
        idx, val, org = super().outgoing(a, b, peer)
        relayed = org != self.id
        keep = ~relayed | (self.sim.rng.random(len(org)) >= self.drop)
        return idx[keep], val[keep], org[keep]


class Greyhole(Blackhole):
    drop = 0.5


class ClaimJam(Attacker):
    forge = "sybil"
    n_extra = 8

    def claim(self):
        if self.sim.p.auth:
            vic = [q for org, (q, t0) in self.claims.items() if self.sim.t - t0 < 2.0]
            if vic:
                return [(self.id, min(vic, key=lambda q: np.linalg.norm(q - self.pos)))]
            return super().claim()
        out = []
        for org, (q, t0) in self.claims.items():
            if org < SYBIL_BASE and self.sim.t - t0 < 2.0:
                out.append((SYBIL_BASE + org, q))
        for i, q in enumerate(getattr(self, "fake_targets", [])[:self.n_extra]):
            out.append((SYBIL_BASE + 50 + i, q))
        if self.target is not None:
            out.append((self.id, self.target))
        return out

    def plan(self):
        ok = super().plan()
        if ok:
            cstate, trav = K.planning_grid(self.state, self.sim.world.occ.shape, 2)
            fr = K.frontiers(cstate, trav, self.blocked)
            lab, nl = ndimage.label(fr, structure=np.ones((3, 3, 3)))
            if nl:
                c = ndimage.center_of_mass(fr, lab, range(1, nl + 1))
                self.fake_targets = [self.sim.world.res * 2 * (np.array(x) + 0.5) for x in c]
        return ok


class _Injector(Attacker):
    rate = 10_000
    every = 1.0

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.faked = np.zeros(self.state.size, bool)
        self.next_inject = 0.0
        shape = self.sim.world.occ.shape
        interior = np.zeros(shape, bool)
        interior[1:-1, 1:-1, 1:-1] = True
        self.interior = interior.ravel()

    def receive(self, idx, val, org, sender):
        keep = ~self.faked[idx]
        super().receive(idx[keep], val[keep], org[keep], sender)

    def step(self):
        super().step()
        if not self.crashed and self.sim.t >= self.next_inject:
            self.next_inject = self.sim.t + self.every
            cells = self.choose()
            if len(cells):
                k = int(self.rate * self.every)
                if len(cells) > k:
                    cells = self.sim.rng.choice(cells, k, replace=False)
                self.faked[cells] = True
                self._log(cells.astype(np.int32), np.full(len(cells), self.value, np.int8), self._origins(len(cells)))

    def _shell(self, known):
        shape = self.sim.world.occ.shape
        grown = ndimage.binary_dilation(known.reshape(shape), _CROSS).ravel()
        return np.flatnonzero(grown & ~known & self.interior)


class FakeFree(_Injector):
    value = K.FREE

    def choose(self):
        known = (self.state != K.UNKNOWN) | self.faked
        return self._shell(known)


class FakeWall(_Injector):
    value = K.OCC

    def choose(self):
        free = self.state == K.FREE
        cand = self._shell(free)
        return cand[(self.state[cand] == K.UNKNOWN) & ~self.faked[cand]]


ATTACKS = {
    "passive": Passive,
    "blackhole": Blackhole,
    "greyhole": Greyhole,
    "claimjam": ClaimJam,
    "fakefree": FakeFree,
    "fakewall": FakeWall,
}


class FakeDrone:
    def __init__(self, ident, pos, t):
        self.id = ident
        self.pos = np.array(pos, float)
        self.yaw = 0.0
        self.path = []
        self.target = None
        self.alive = True
        self.born = t
        self.wait = 0.0
        self.room = None


class _Sybil(Attacker):
    n_sybil = 4
    speed = 1.2
    strong = False

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        s = self.sim
        self.faked = np.zeros(self.state.size, bool)
        self.fab_val = np.zeros(self.state.size, np.int8)
        self.fakes = []
        self.next_id = SYBIL_BASE
        self.next_spawn = 0.0
        self.next_assign = 0.0
        self.retired = []
        if not self.strong:
            for n in range(self.n_sybil):
                f = self._spawn(self.pos + np.array([0.0, 0.0, 0.0]))
                f.wait = s.p.launch_gap * (n + 1) + s.p.scan_time * 2

    def _spawn(self, pos):
        f = FakeDrone(self.next_id, pos, self.sim.t)
        self.next_id = min(self.next_id + 1, self.sim.n_ids - 1)
        self.fakes.append(f)
        return f

    def alive(self):
        return [f for f in self.fakes if f.alive]

    def _revoked(self, f):
        live = [h for h in self.sim.drones if h.honest and not h.crashed]
        return sum(not h.accept[f.id] for h in live) > len(live) / 2

    def claim(self):
        out = [] if self.target is None else [(self.id, self.target)]
        if not self.sim.p.auth:
            out += [(f.id, f.target) for f in self.alive() if f.target is not None]
        return out

    def beacon(self):
        out = [(self.id, self.pos.copy())]
        if not self.sim.p.auth:
            out += [(f.id, f.pos.copy()) for f in self.alive()]
        return out

    def receive(self, idx, val, org, sender):
        keep = ~self.faked[idx]
        super().receive(idx[keep], val[keep], org[keep], sender)

    def witnessed(self):
        fakes = [f.id for a in self.sim.drones if isinstance(a, _Sybil) for f in a.alive()]
        return super().witnessed() + fakes

    def step(self):
        Drone.step(self)
        s = self.sim
        if self.crashed or s.p.auth:
            return
        self._manage()
        if s.t >= self.next_assign:
            self.next_assign = s.t + 2.0
            self._assign()
        for f in self.alive():
            if f.wait > 0:
                f.wait -= s.p.dt
                continue
            self._move_fake(f)
            self._scan(f)

    def _manage(self):
        pass

    def _believed(self):
        b = self.state.copy()
        m = self.faked & (b == K.UNKNOWN)
        b[m] = self.fab_val[m]
        return b

    def _assign(self):
        s = self.sim
        shape = s.world.occ.shape
        b = self._believed()
        cstate, trav = K.planning_grid(b, shape, 2)
        fr = K.frontiers(cstate, trav, np.zeros_like(trav))
        cells = np.argwhere(fr)
        if len(cells) == 0:
            return
        tpos = s.world.res * 2 * (cells + 0.5)
        taken = [q for o, (q, t0) in self.claims.items() if o < SYBIL_BASE and s.t - t0 < s.p.claim_ttl]
        for f in self.alive():
            if f.wait > 0 or (f.path and f.target is not None):
                if f.target is not None:
                    taken.append(f.target)
                continue
            start = tuple(np.clip((f.pos / (s.world.res * 2)).astype(int), 0, np.array(trav.shape) - 1))
            dist, parent = K.bfs(trav | (cstate == 0), np.array(start))
            dist = dist.reshape(trav.shape)
            ok = dist[tuple(cells.T)] >= 0
            if not ok.any():
                continue
            far = np.full(len(cells), 50.0)
            for q in taken:
                far = np.minimum(far, np.linalg.norm(tpos - q, axis=1))
            score = np.where(ok, np.minimum(far, 12.0) - 0.15 * dist[tuple(cells.T)] * 0.4, -np.inf)
            g = tuple(cells[int(np.argmax(score))])
            cy, cz = trav.shape[1], trav.shape[2]
            u = (g[0] * cy + g[1]) * cz + g[2]
            chain = []
            while u >= 0:
                chain.append((u // (cy * cz), (u // cz) % cy, u % cz))
                u = parent[u]
            f.path = [s.world.res * 2 * (np.array(c) + 0.5) for c in chain[::-1]]
            f.target = s.world.res * 2 * (np.array(g) + 0.5)
            taken.append(f.target)

    def _visible(self, q):
        s = self.sim
        return any(s.in_view(h, q) for h in s.drones if h.honest and not h.crashed)

    def _move_fake(self, f):
        s = self.sim
        if not f.path:
            f.target = None
            return
        step = self.speed * s.p.dt
        w = f.path[0]
        v = w - f.pos
        d = np.linalg.norm(v)
        nxt = w if d <= step else f.pos + v / d * step
        if self.strong and getattr(self, "hide", False) and self._visible(nxt):
            f.wait = 0.0
            f.blocked = getattr(f, "blocked", 0.0) + s.p.dt
            if f.blocked > 3.0:
                f.path, f.target, f.blocked = [], None, 0.0
            return
        f.blocked = 0.0
        if d > 1e-6:
            f.yaw = np.arctan2(v[1], v[0])
        f.pos = nxt
        if d <= step:
            f.path.pop(0)

    def _lengths(self, f, dirs):
        raise NotImplementedError

    def _scan(self, f):
        s = self.sim
        if s.p.sensor == "lidar":
            dirs = s.cam
        else:
            c, sn = np.cos(f.yaw), np.sin(f.yaw)
            dirs = s.cam @ np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]]).T
        L = self._lengths(f, dirs)
        n = K.fabricate_scan(s.world.occ.shape, s.world.res, f.pos, dirs, L, s.buf_idx, s.buf_val)
        idx, val = s.buf_idx[:n], s.buf_val[:n]
        new = (self.state[idx] == K.UNKNOWN) & ~self.faked[idx]
        idx, first = np.unique(idx[new], return_index=True)
        val = val[new][first]
        if len(idx):
            self.faked[idx] = True
            self.fab_val[idx] = val
            self._log(idx.astype(np.int32), val, np.full(len(idx), f.id, np.int16))


class SybilWeak(_Sybil):

    def _lengths(self, f, dirs):
        return self.sim.rng.uniform(1.5, self.sim.range_, len(dirs))


class SybilStrong(_Sybil):
    hide = False
    strong = True
    relabel = 0.8
    spawn_every = 4.0
    first_spawn = 8.0

    def _manage(self):
        s = self.sim
        for f in self.alive():
            if self._revoked(f):
                f.alive = False
                self.retired.append((s.t, f.id))
                self.next_spawn = max(self.next_spawn, s.t + 10.0)
        if s.t >= max(self.next_spawn, self.first_spawn) and len(self.alive()) < self.n_sybil:
            q = self._hidden_spot()
            if q is not None:
                self._spawn(q)
                self.next_spawn = s.t + self.spawn_every

    def _hidden_spot(self):
        s = self.sim
        shape = s.world.occ.shape
        free = np.flatnonzero(self.state == K.FREE)
        if len(free) == 0:
            return None
        for _ in range(30):
            g = free[s.rng.integers(len(free))]
            q = (np.array(np.unravel_index(g, shape)) + 0.5) * s.world.res
            if np.linalg.norm(q - self.pos) < 10.0 and q[2] > 0.8 and not (self.hide and self._visible(q)):
                return q
        return None

    def _lengths(self, f, dirs):
        s = self.sim
        if f.room is None or np.any(np.abs(f.pos - f.room[0]) > f.room[1] - 0.5):
            half = np.array([s.rng.uniform(4, 7), s.rng.uniform(4, 7), 1.6])
            f.room = (f.pos.copy(), half)
        c, h = f.room
        lo, hi = c - h, c + h
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo - f.pos) / dirs
            t2 = (hi - f.pos) / dirs
        t = np.where(dirs > 0, t2, t1)
        t[~np.isfinite(t)] = np.inf
        L = np.clip(t.min(1), 0.3, s.range_)
        return L

    def outgoing(self, a, b, peer):
        idx, val, org = super().outgoing(a, b, peer)
        live = self.alive()
        if self.sim.p.auth or not live:
            return idx, val, org
        ids = np.array([f.id for f in live], np.int16)
        org = org.copy()
        relab = ~self.faked[idx] & (org != peer) & (self.sim.rng.random(len(org)) < self.relabel)
        org[relab] = ids[idx[relab] % len(ids)]
        return idx, val, org


class SybilStealth(SybilStrong):
    hide = True


class SybilTargeted(SybilStrong):
    reach = 12
    seal_at = 1.0

    def _assign(self):
        s = self.sim
        shape = s.world.occ.shape
        cstate, trav = K.planning_grid(self._believed(), shape, 2)
        fr = K.frontiers(cstate, trav, np.zeros_like(trav))
        cells = np.argwhere(fr)
        if len(cells) == 0:
            return
        unk = np.pad((cstate == -1).astype(np.int32), ((1, 0), (1, 0), (1, 0))).cumsum(0).cumsum(1).cumsum(2)
        r = self.reach
        lo = np.maximum(cells - r, 0)
        hi = np.minimum(cells + r + 1, np.array(trav.shape))
        value = np.zeros(len(cells))
        for sx in (0, 1):
            for sy in (0, 1):
                for sz in (0, 1):
                    c = np.stack([hi[:, 0] if sx else lo[:, 0], hi[:, 1] if sy else lo[:, 1], hi[:, 2] if sz else lo[:, 2]], 1)
                    value += (-1) ** (3 - sx - sy - sz) * unk[c[:, 0], c[:, 1], c[:, 2]]
        cell_size = s.world.res * 2
        tpos = cell_size * (cells + 0.5)
        taken = [q for o, (q, t0) in self.claims.items() if o < SYBIL_BASE and s.t - t0 < s.p.claim_ttl]
        for f in self.alive():
            if f.wait > 0 or (f.path and f.target is not None):
                if f.target is not None:
                    taken.append(f.target)
                continue
            start = tuple(np.clip((f.pos / cell_size).astype(int), 0, np.array(trav.shape) - 1))
            dist, parent = K.bfs(trav | (cstate == 0), np.array(start))
            dist = dist.reshape(trav.shape)
            d = dist[tuple(cells.T)]
            ok = d >= 0
            for q in taken:
                ok &= np.linalg.norm(tpos - q, axis=1) > s.p.claim_radius
            if not ok.any():
                continue
            k = int(np.argmax(np.where(ok, value - 0.5 * d, -np.inf)))
            g = tuple(cells[k])
            cy, cz = trav.shape[1], trav.shape[2]
            u = (g[0] * cy + g[1]) * cz + g[2]
            chain = []
            while u >= 0:
                chain.append((u // (cy * cz), (u // cz) % cy, u % cz))
                u = parent[u]
            f.path = [cell_size * (np.array(c) + 0.5) for c in chain[::-1]]
            f.target = tpos[k]
            box = cstate[max(g[0] - 3, 0):g[0] + 4, max(g[1] - 3, 0):g[1] + 4, max(g[2] - 3, 0):g[2] + 4]
            unknown = np.argwhere(box == -1) + np.array([max(g[0] - 3, 0), max(g[1] - 3, 0), max(g[2] - 3, 0)])
            v = (unknown.mean(0) - np.array(g)) if len(unknown) else np.array([1.0, 0.0, 0.0])
            f.gate = (tpos[k], v / (np.linalg.norm(v) + 1e-9))
            taken.append(f.target)

    def _lengths(self, f, dirs):
        L = super()._lengths(f, dirs)
        gate = getattr(f, "gate", None)
        if gate is None or np.linalg.norm(f.pos - gate[0]) > 1.5:
            return L
        q, n = gate[0] + self.seal_at * gate[1], gate[1]
        cos = dirs @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((q - f.pos) @ n) / cos
        hit = (cos > 0.3) & (t > 0.3)
        L[hit] = np.minimum(L[hit], t[hit])
        return L


class SybilShadow(SybilTargeted):
    offset = 0.3

    def beacon(self):
        out = [(self.id, self.pos.copy())]
        if self.sim.p.auth:
            return out
        s = self.sim
        hosts = sorted(o for o, (q, t0) in self.beacons.items()
                       if o < SYBIL_BASE and o != self.id and s.t - t0 <= 2 * s.p.dt)
        for k, f in enumerate(self.alive()):
            if not hosts:
                out.append((f.id, f.pos.copy()))
                continue
            q = self.beacons[hosts[k % len(hosts)]][0]
            a = 2 * np.pi * k / max(len(self.alive()), 1)
            out.append((f.id, q + self.offset * np.array([np.cos(a), np.sin(a), 0.0])))
        return out


ATTACKS.update(sybil_weak=SybilWeak, sybil_strong=SybilStrong, sybil_stealth=SybilStealth,
               sybil_targeted=SybilTargeted, sybil_shadow=SybilShadow)
