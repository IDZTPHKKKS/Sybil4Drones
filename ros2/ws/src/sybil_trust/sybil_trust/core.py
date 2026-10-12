import numpy as np
from scipy.stats import beta

LEVELS = dict(
    D1=dict(presence=False, by_type=False, quarantine=False, bind=False),
    D2=dict(presence=True, by_type=False, quarantine=False, bind=False),
    D3=dict(presence=True, by_type=True, quarantine=False, bind=False),
    D4=dict(presence=True, by_type=True, quarantine=True, bind=False),
    D5=dict(presence=True, by_type=True, quarantine=True, bind=True),
)


class Trust:
    def __init__(self, me, level="D4", n_ids=256, tol=0.02, conf=0.99, detect=0.9, presence_conf=0.99,
                 quorum=2, v_max=1.5, bind_stale=float("inf"), seed=0):
        cfg = LEVELS[level]
        self.me = me
        self.presence = cfg["presence"]
        self.by_type = cfg["by_type"]
        self.quarantine = cfg["quarantine"]
        self.bind = cfg["bind"]
        self.tol, self.conf, self.detect, self.presence_conf = tol, conf, detect, presence_conf
        self.quorum, self.v_max, self.bind_stale = quorum, v_max, bind_stale
        self.accept = np.ones(n_ids, bool)
        self.seen = np.zeros(n_ids, np.int64)
        self.absent = np.zeros(n_ids, np.int64)
        self.phys = np.zeros(n_ids, bool)
        self.phys[me] = True
        self.bound = {}
        self.rng = np.random.default_rng(seed)

    def step(self, t, pos, counts, beacons, bodies, witness):
        out = []

        def revoke(o, why, bad, total):
            self.accept[o] = False
            out.append((int(o), why, int(bad), int(total)))

        for o in sorted(counts):
            checked, wrong, checked_walls, wrong_walls = counts[o]
            if wrong <= 0 or not self.accept[o] or o == self.me:
                continue
            if beta.sf(self.tol, 1 + wrong, 1 + checked - wrong) > self.conf:
                revoke(o, "audit", wrong, checked)
            elif self.by_type and wrong_walls > 0 and beta.sf(
                    self.tol, 1 + wrong_walls, 1 + checked_walls - wrong_walls) > self.conf:
                revoke(o, "audit-walls", wrong_walls, checked_walls)

        if self.presence:
            pos = np.asarray(pos, float)
            bodies = [(b, np.asarray(q, float)) for b, q in bodies]
            fresh = [(o, np.asarray(q, float), age, vis) for o, q, age, vis in beacons
                     if o != self.me and self.accept[o]]
            if self.bind:
                fresh.sort(key=lambda x: min([np.linalg.norm(bp - x[1]) for _, bp in bodies] or [np.inf]))
            else:
                fresh.sort(key=lambda x: np.linalg.norm(x[1] - pos))
            used = set()
            for o, q, age, vis in fresh:
                if not vis:
                    continue
                match, best = None, 0.4 + self.v_max * age
                for b, bp in bodies:
                    d = np.linalg.norm(bp - q)
                    if b not in used and d <= best:
                        match, best = b, d
                        if not self.bind:
                            break
                if match is not None and self.bind:
                    owner = self.bound.get(match)
                    if owner is not None and owner[0] != o and self.accept[owner[0]] and t - owner[1] <= self.bind_stale:
                        self.absent[o] += 1
                        continue
                if match is not None and self.rng.random() < self.detect:
                    used.add(match)
                    self.seen[o] += 1
                    if self.bind:
                        self.bound[match] = (o, t)
                elif match is None:
                    self.absent[o] += 1
            for o in np.flatnonzero((self.absent > 0) & self.accept):
                if o != self.me and beta.sf(0.5, 1 + self.absent[o], 1 + self.seen[o]) > self.presence_conf:
                    revoke(o, "absent", self.absent[o], self.absent[o] + self.seen[o])

        if self.quarantine:
            votes = np.zeros(len(self.accept), np.int64)
            for a, ids in witness.items():
                if a != self.me and self.seen[a] > 0 and self.accept[a]:
                    votes[list(ids)] += 1
            self.phys = ((self.seen > 0) | (votes >= self.quorum)) & self.accept
            self.phys[self.me] = True
        return out

    def trusted(self):
        return self.accept & self.phys if self.quarantine else self.accept.copy()

    def witnessed(self):
        return [int(o) for o in np.flatnonzero((self.seen > 0) & self.accept)]
