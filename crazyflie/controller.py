"""Geometric tracking controller on SO(3) (from wind_drone_mujoco) with velocity and acceleration feed-forward."""
import numpy as np

from .dynamics import G


def vee(m):
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


class GeometricController:
    def __init__(self, mass, inertia, kp=9.0, kd=5.0, ki=2.0, k_r=0.015, k_w=1.0e-3, i_limit=0.5,
                 max_tilt=0.6, i_zone=0.15):
        self.m, self.J = mass, np.asarray(inertia, float)
        self.kp, self.kd, self.ki, self.k_r, self.k_w = kp, kd, ki, k_r, k_w
        self.i_limit, self.max_tilt, self.i_zone = i_limit, max_tilt, i_zone
        self.integral = np.zeros(3)

    def __call__(self, body, p_ref, v_ref, a_ref, yaw_ref, h):
        e = p_ref - body.p
        near = np.abs(e) < self.i_zone
        self.integral = np.clip(self.integral + np.where(near, e, 0.0) * h, -self.i_limit, self.i_limit)
        acc = a_ref + self.kp * e + self.kd * (v_ref - body.v) + self.ki * self.integral
        horiz = np.hypot(acc[0], acc[1])
        max_h = np.tan(self.max_tilt) * (G + acc[2])
        if horiz > max_h > 0:
            acc[:2] *= max_h / horiz
        force = self.m * (acc + np.array([0.0, 0.0, G]))
        thrust = float(force @ body.R[:, 2])
        z_d = force / np.linalg.norm(force)
        x_c = np.array([np.cos(yaw_ref), np.sin(yaw_ref), 0.0])
        y_d = np.cross(z_d, x_c)
        y_d /= np.linalg.norm(y_d)
        x_d = np.cross(y_d, z_d)
        r_d = np.column_stack([x_d, y_d, z_d])
        e_r = 0.5 * vee(r_d.T @ body.R - body.R.T @ r_d)
        tau = -self.k_r * e_r - self.k_w * body.w + np.cross(body.w, self.J * body.w)
        return thrust, tau
