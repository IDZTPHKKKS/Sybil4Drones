import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from swarm import kernels as K  # noqa: E402
from swarm import worlds  # noqa: E402
from swarm.attacks import FakeWall  # noqa: E402
from swarm.sim import Drone, Params, Sim  # noqa: E402


def test_worlds_are_deterministic_and_starts_are_free():
    for f in worlds.FAMILIES:
        a, b = worlds.make(f, 3), worlds.make(f, 3)
        assert np.array_equal(a.occ, b.occ)
        assert not a.occ[tuple((a.starts / a.res).astype(int).T)].any()
        assert not np.array_equal(a.occ, worlds.make(f, 4).occ)


def test_noise_free_ray_stops_at_the_first_wall():
    occ = np.zeros((50, 10, 10), bool)
    occ[30, :, :] = True
    idx = np.empty(1000, np.int32)
    val = np.empty(1000, np.int8)
    n = K.raycast(occ, 0.2, np.array([1.01, 1.01, 1.01]), np.array([[1.0, 0.0, 0.0]]), 20.0, 0.0, 0.0, idx, val, 19.0)
    i = idx[:n] // (10 * 10)
    assert val[n - 1] == K.OCC and i[n - 1] == 30
    assert (val[:n - 1] == K.FREE).all() and list(i[:n - 1]) == list(range(5, 30))


def test_own_observations_override_and_credit_the_claimed_origin():
    n = 10
    state = np.full(n, K.UNKNOWN, np.int8)
    src = np.full(n, -1, np.int16)
    lo = np.zeros(n, np.int8)
    state[3], src[3] = K.OCC, 5
    agree, disagree = np.zeros(8, np.int64), np.zeros(8, np.int64)
    ev = (np.empty(10, np.int32), np.empty(10, np.int16), np.empty(10, np.int8), np.zeros(1, np.int64))
    stamp = np.zeros(n, np.int64)
    m = K.apply_own(state, src, lo, np.array([3], np.int32), np.array([K.FREE], np.int8), 1, 0,
                    np.empty(1, np.int64), agree, disagree, stamp, 1, *ev, np.zeros(8, np.int64))
    assert m == 1 and state[3] == K.FREE and src[3] == 0
    assert disagree[5] == 1 and ev[3][0] == 1 and ev[1][0] == 5


def test_honest_team_explores_without_crashing_or_false_revocations():
    w = worlds.make("forest", 0)
    s = Sim(w, Params(n_drones=4, t_max=90.0, defence="audit", auth=True))
    m = s.run()
    assert m.crashed[-1] == 0
    assert m.coverage[-1] > 0.3
    assert s.revocations == []


def test_fake_wall_attacker_is_revoked_and_rolled_back():
    w = worlds.make("office", 1)
    p = Params(n_drones=4, t_max=40.0, defence="audit", auth=True)
    s = Sim(w, p, drone_types=[Drone, Drone, Drone, FakeWall])
    s.run()
    assert any(r["origin"] == 3 for r in s.revocations)
    assert all(r["origin"] == 3 for r in s.revocations)
    for d in s.drones[:3]:
        if not d.accept[3]:
            assert not (d.src == 3).any()
