"""Crazyflie 2 rigid-body dynamics (MuJoCo Menagerie cf2 parameters) with motor lag and rotor drag."""
from dataclasses import dataclass

import numpy as np

G = 9.81


@dataclass
class CFParams:
    mass: float = 0.027
    inertia: tuple = (2.3951e-5, 2.3951e-5, 3.2347e-5)
    thrust_max: float = 0.35
    moment_max: float = 4e-3
    motor_tau: float = 0.02
    drag: float = 0.01
    h: float = 0.002


def skew(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def expm_so3(w, h):
    th = np.linalg.norm(w) * h
    if th < 1e-9:
        return np.eye(3) + skew(w) * h
    k = skew(w / np.linalg.norm(w))
    return np.eye(3) + np.sin(th) * k + (1.0 - np.cos(th)) * (k @ k)


def yaw_rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class CFBody:
    def __init__(self, pos, yaw=0.0, prm=None):
        self.prm = prm or CFParams()
        self.J = np.asarray(self.prm.inertia, float)
        self.p = np.array(pos, float)
        self.v = np.zeros(3)
        self.R = yaw_rot(yaw)
        self.w = np.zeros(3)
        self.T = self.prm.mass * G
        self.tau = np.zeros(3)
        self.n = 0

    def yaw(self):
        return float(np.arctan2(self.R[1, 0], self.R[0, 0]))

    def tilt(self):
        return float(np.arccos(np.clip(self.R[2, 2], -1.0, 1.0)))

    def step(self, T_cmd, tau_cmd):
        q = self.prm
        T_cmd = float(np.clip(T_cmd, 0.0, q.thrust_max))
        tau_cmd = np.clip(tau_cmd, -q.moment_max, q.moment_max)
        a = q.h / (q.motor_tau + q.h)
        self.T += a * (T_cmd - self.T)
        self.tau += a * (tau_cmd - self.tau)
        acc = (self.T * self.R[:, 2] - q.drag * self.v) / q.mass - np.array([0.0, 0.0, G])
        self.v += acc * q.h
        self.p += self.v * q.h
        wdot = (self.tau - np.cross(self.w, self.J * self.w)) / self.J
        self.w += wdot * q.h
        self.R = self.R @ expm_so3(self.w, q.h)
        self.n += 1
        if self.n % 200 == 0:
            u, _, vt = np.linalg.svd(self.R)
            self.R = u @ vt
