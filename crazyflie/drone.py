"""Crazyflie physics flight for any drone class: tracked guidance commands and body-fixed sensors."""
import numpy as np

from swarm import kernels as K

from .controller import GeometricController
from .dynamics import CFBody, CFParams


class PhysicsFlight:
    cf_params = CFParams()

    def __init__(self, ident, pos, sim):
        super().__init__(ident, pos, sim)
        self.body = CFBody(self.pos, self.yaw, self.cf_params)
        self.ctrl = GeometricController(self.cf_params.mass, self.cf_params.inertia)
        self._moved = False
        self.track_err = []
        self.max_tilt = 0.0

    def _fly(self, dv, hold=None):
        dt = self.sim.p.dt
        b = self.body
        p0, v0 = b.p.copy(), b.v.copy()
        a = dv / dt
        n = max(1, int(round(dt / b.prm.h)))
        err = 0.0
        prev = b.p.copy()
        flown = 0.0
        for k in range(1, n + 1):
            t = k * b.prm.h
            if hold is not None:
                p_ref, v_ref, a_ref = hold, np.zeros(3), np.zeros(3)
            else:
                p_ref, v_ref, a_ref = p0 + v0 * t + 0.5 * a * t * t, v0 + a * t, a
            T, tau = self.ctrl(b, p_ref, v_ref, a_ref, self.yaw, b.prm.h)
            b.step(T, tau)
            err = max(err, float(np.linalg.norm(b.p - p_ref)))
            flown += float(np.linalg.norm(b.p - prev))
            prev = b.p.copy()
        self.track_err.append(err)
        self.max_tilt = max(self.max_tilt, b.tilt())
        self.pos[:] = b.p
        self.vel[:] = b.v
        self.flown += flown
        if np.linalg.norm(self.pos - self.trail[-1]) > 0.5:
            self.trail.append(self.pos.copy())

    def _move(self, dv):
        self._moved = True
        self._fly(dv)

    def step(self):
        self._moved = False
        if not self.crashed:
            self._hold_at = self.body.p.copy()
        super().step()
        if not self.crashed and not self._moved:
            self._fly(np.zeros(3), hold=self._hold_at)
            if K.sphere_hits(self.sim.world.occ, self.sim.world.res, self.pos, self.sim.p.radius):
                self.crashed = True
                self.mode = "crashed"
                self.vel[:] = 0

    def sense(self):
        s = self.sim
        R = self.body.R
        dirs = s.cam @ R.T
        rangers = s.rangers @ R.T
        for d, rng, clear, noise in ((dirs, s.range_, s.clear_, s.noise_),
                                     (rangers, s.p.ranger_range, s.p.ranger_range - 4 * s.p.depth_noise * s.p.ranger_range,
                                      s.p.depth_noise)):
            n = K.raycast(s.world.occ, s.world.res, self.pos, np.ascontiguousarray(d), rng, noise, s.p.dropout,
                          s.buf_idx, s.buf_val, clear)
            m = K.apply_own(self.state, self.src, self.lo, s.buf_idx, s.buf_val, n, self.id, s.buf_chg,
                            self.agree, self.disagree, s.stamp, s.frame, *self.ev[0], self.checked_occ)
            s.frame += 1
            f = s.buf_idx[s.buf_chg[:m]]
            self._log(f, self.state[f], self.id)


def physics(cls):
    return type(f"CF_{cls.__name__}", (PhysicsFlight, cls), {})
