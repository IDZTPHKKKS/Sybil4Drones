import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crazyflie.controller import GeometricController  # noqa: E402
from crazyflie.drone import physics  # noqa: E402
from crazyflie.dynamics import G, CFBody, CFParams  # noqa: E402
from swarm import worlds  # noqa: E402
from swarm.sim import Drone, Params, Sim  # noqa: E402


def fly(body, ctrl, ref, seconds):
    out = []
    for _ in range(int(seconds / body.prm.h)):
        p, v, a, yaw = ref(body.n * body.prm.h)
        body.step(*ctrl(body, p, v, a, yaw, body.prm.h))
        out.append(body.p.copy())
    return np.array(out)


def test_free_fall_without_thrust_or_drag():
    b = CFBody([0, 0, 10], prm=CFParams(drag=0.0))
    b.T = 0.0
    for _ in range(500):
        b.step(0.0, np.zeros(3))
    assert abs(b.p[2] - (10 - 0.5 * G)) < 0.05


def test_hover_holds_position():
    b, prm = CFBody([1, 2, 1]), CFParams()
    c = GeometricController(prm.mass, prm.inertia)
    P = fly(b, c, lambda t: (np.array([1.0, 2, 1]), np.zeros(3), np.zeros(3), 0.0), 5.0)
    assert np.abs(P[-1] - [1, 2, 1]).max() < 0.01
    assert b.tilt() < 0.01


def test_step_response_settles_with_small_overshoot():
    b, prm = CFBody([0, 0, 1]), CFParams()
    c = GeometricController(prm.mass, prm.inertia)
    P = fly(b, c, lambda t: (np.array([1.0, 0, 1]), np.zeros(3), np.zeros(3), 0.0), 4.0)
    assert abs(P[-1, 0] - 1.0) < 0.02
    assert P[:, 0].max() < 1.15
    assert abs(P[:, 2] - 1.0).max() < 0.1


def test_tracks_a_1p5_m_per_s_line():
    b, prm = CFBody([0, 0, 1]), CFParams()
    c = GeometricController(prm.mass, prm.inertia)
    ref = lambda t: (np.array([1.5 * t, 0, 1]), np.array([1.5, 0, 0]), np.zeros(3), 0.0)
    P = fly(b, c, ref, 4.0)
    t = np.arange(1, len(P) + 1) * prm.h
    e = np.abs(P[:, 0] - 1.5 * t)
    assert e.max() < 0.4
    assert e[t > 3.0].max() < 0.05


def test_yaw_command_turns_the_body():
    b, prm = CFBody([0, 0, 1]), CFParams()
    c = GeometricController(prm.mass, prm.inertia)
    fly(b, c, lambda t: (np.array([0.0, 0, 1]), np.zeros(3), np.zeros(3), 1.0), 3.0)
    assert abs(b.yaw() - 1.0) < 0.02


def test_short_mission_with_physics_explores_without_crashing():
    p = Params(seed=0, t_max=60.0)
    s = Sim(worlds.make("office", 0), p, drone_types=[physics(Drone)] * p.n_drones)
    s.run()
    assert s.m.coverage[-1] > 0.15
    assert sum(d.crashed for d in s.drones) == 0
    assert max(d.max_tilt for d in s.drones) > 0.01
    assert np.median([np.median(d.track_err) for d in s.drones]) < 0.1
