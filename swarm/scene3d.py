"""3D view of a running mission: drones, the team's voxel map (majority view), tracks, explored floor."""
import mujoco
import numpy as np
from matplotlib import colormaps

from . import kernels as K
from .viz import team_state

F = 2
TRAIL_POINTS = 120
DRONE_SCALE = 2.5
TEAM = [colormaps["tab10"](i)[:3] for i in (0, 2, 4, 6, 8, 9, 1, 5, 7, 3)]
RED = (0.9, 0.08, 0.08)


class Scene3D:
    def __init__(self, sim):
        self.sim = sim
        w = sim.world
        self.L, self.W, self.H = w.size
        self.nx, self.ny, self.nz = w.occ.shape
        self.tex_w, self.tex_h = self.nx, self.ny
        self.model, self.data = self._build()
        self.tex_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TEXTURE, "explored")
        self.cmap = colormaps["viridis"]
        self.shown = {}
        self.n_trail = len(sim.drones) * TRAIL_POINTS
        self.prev = np.array([d.pos for d in sim.drones])
        self.prev_yaw = np.array([d.yaw for d in sim.drones])

    def _build(self):
        s = self.sim
        spec = mujoco.MjSpec()
        spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
        spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
        spec.visual.map.znear = 0.01
        spec.visual.quality.shadowsize = 0
        spec.visual.global_.offwidth = 1920
        spec.visual.global_.offheight = 1080
        spec.stat.center = [self.L / 2, self.W / 2, 0]
        spec.stat.extent = max(self.L, self.W) / 2
        tex = spec.add_texture(name="explored", type=mujoco.mjtTexture.mjTEXTURE_2D,
                               builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, width=self.tex_w, height=self.tex_h,
                               rgb1=[0.12, 0.12, 0.14], rgb2=[0.12, 0.12, 0.14])
        mat = spec.add_material(name="floor", texrepeat=[1, 1], texuniform=False)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "explored"
        wb = spec.worldbody
        wb.add_light(pos=[self.L / 2, self.W / 2, 30], dir=[0, 0, -1], diffuse=[0.5, 0.5, 0.5],
                     type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
        wb.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[self.L / 2, self.W / 2, 0.1],
                    pos=[self.L / 2, self.W / 2, 0], material="floor", contype=0, conaffinity=0)
        k = DRONE_SCALE
        for d in s.drones:
            c = list(RED if not d.honest else TEAM[d.id % len(TEAM)]) + [1]
            b = wb.add_body(name=f"drone{d.id}", mocap=True, pos=list(d.pos))
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03 * k, 0.03 * k, 0.008 * k], rgba=[0.15, 0.15, 0.15, 1],
                       contype=0, conaffinity=0)
            for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                p = np.array([sx, sy, 0]) * 0.032 * k
                b.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, 0, *p], size=[0.004 * k, 0, 0],
                           rgba=[0.2, 0.2, 0.2, 1], contype=0, conaffinity=0)
                b.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=list(p + [0, 0, 0.006 * k]),
                           size=[0.022 * k, 0.001 * k, 0], rgba=c, contype=0, conaffinity=0)
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=[0.035 * k, 0, 0], size=[0.008 * k, 0, 0],
                       rgba=[1, 1, 1, 1], contype=0, conaffinity=0)
        m = spec.compile()
        return m, mujoco.MjData(m)

    def move_drones(self, alpha):
        for i, d in enumerate(self.sim.drones):
            p = (1 - alpha) * self.prev[i] + alpha * d.pos
            dy = (d.yaw - self.prev_yaw[i] + np.pi) % (2 * np.pi) - np.pi
            yaw = self.prev_yaw[i] + alpha * dy
            self.data.mocap_pos[i] = p
            self.data.mocap_quat[i] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_kinematics(self.model, self.data)

    def remember(self):
        self.prev = np.array([d.pos for d in self.sim.drones])
        self.prev_yaw = np.array([d.yaw for d in self.sim.drones])

    def update_map(self, scn, base=0):
        s = self.sim
        if base != getattr(self, "_base", 0):
            self.shown = {}
        self._base = base
        st = team_state(s)
        cx, cy, cz = self.nx // F, self.ny // F, self.nz // F
        g = st[:cx * F, :cy * F, :cz * F].reshape(cx, F, cy, F, cz, F)
        occ = (g == K.OCC).any(axis=(1, 3, 5))
        faked = np.zeros(st.size, bool)
        for d in s.drones:
            if hasattr(d, "faked"):
                faked |= d.faked
        bad = ((st.ravel() == K.OCC) & faked & ~s.occ_near).reshape(st.shape)[:cx * F, :cy * F, :cz * F]
        wrong = bad.reshape(cx, F, cy, F, cz, F).any(axis=(1, 3, 5))
        occ[:, :, 0] = occ[:, :, -1] = False
        cells = set(map(tuple, np.argwhere(occ)))
        stale = [c for c in self.shown if c not in cells or self.shown[c][1] != bool(wrong[c])]
        if stale:
            self.shown = {}
        new = [c for c in cells if c not in self.shown]
        room = scn.maxgeom - base - self.n_trail - len(self.shown)
        new = new[: max(room, 0)]
        eye = np.eye(3).ravel()
        half = np.full(3, 0.5 * F * s.world.res * 0.96)
        top = max(cz - 2, 1)
        for c in new:
            slot = base + self.n_trail + len(self.shown)
            w = bool(wrong[c])
            rgba = np.array([*RED, 1.0]) if w else np.array(self.cmap(0.1 + 0.85 * (c[2] - 1) / top), float)
            pos = (np.array(c) + 0.5) * F * s.world.res
            mujoco.mjv_initGeom(scn.geoms[slot], mujoco.mjtGeom.mjGEOM_BOX, half, pos, eye, rgba.astype(np.float32))
            self.shown[c] = (slot, w)
        scn.ngeom = base + self.n_trail + len(self.shown)
        for i, d in enumerate(s.drones):
            tr = np.array((d.trail + [d.pos])[-TRAIL_POINTS - 1:])
            col = np.array([*(RED if not d.honest else TEAM[d.id % len(TEAM)]), 0.9], np.float32)
            for j in range(TRAIL_POINTS):
                geom = scn.geoms[base + i * TRAIL_POINTS + j]
                if j < len(tr) - 1:
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), eye, col)
                    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 2.0, tr[j], tr[j + 1])
                else:
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, 1e-4), np.zeros(3), eye,
                                        np.zeros(4, np.float32))
        n = scn.ngeom
        for d in s.drones:
            if not hasattr(d, "fakes") or s.p.auth:
                continue
            for f in d.fakes:
                i, q = f.id, f.pos
                if n >= scn.maxgeom:
                    break
                revoked = not f.alive or all(not h.accept[i] for h in s.drones if h.honest and not h.crashed)
                col = np.array([0.6, 0.6, 0.6, 0.35] if revoked else [0.95, 0.1, 0.1, 0.45], np.float32)
                mujoco.mjv_initGeom(scn.geoms[n], mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, 0.35), q, eye, col)
                n += 1
            for z in getattr(d, "zones", []):
                corners = [(z[0], z[1]), (z[2], z[1]), (z[2], z[3]), (z[0], z[3]), (z[0], z[1])]
                for (x0, y0), (x1, y1) in zip(corners[:-1], corners[1:]):
                    if n >= scn.maxgeom:
                        break
                    mujoco.mjv_initGeom(scn.geoms[n], mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), eye,
                                        np.array([0.95, 0.15, 0.15, 0.9], np.float32))
                    mujoco.mjv_connector(scn.geoms[n], mujoco.mjtGeom.mjGEOM_LINE, 4.0,
                                         np.array([x0, y0, 0.25]), np.array([x1, y1, 0.25]))
                    n += 1
        scn.ngeom = n
        known = (st[:, :, 1:-1] != K.UNKNOWN).any(2)
        img = np.where(known[..., None], np.array([200, 205, 212], np.uint8), np.array([30, 30, 36], np.uint8))
        img = np.ascontiguousarray(img.transpose(1, 0, 2))
        a = self.model.tex_adr[self.tex_id]
        self.model.tex_data[a:a + img.size] = img.ravel()

    def hud(self):
        s = self.sim
        m = s.m
        H = s.honest()
        att = [d for d in s.drones if not d.honest]
        n_real = len(s.drones)
        rev = {r["by"] for r in s.revocations if r["origin"] < n_real and not s.drones[r["origin"]].honest}
        left = "time\nspace seen (first-hand)\nknown by average drone\ndistance flown\ndrones\ncrashed\nwrong cells per map"
        right = (f"{s.t:.0f} s\n{100 * m.coverage[-1]:.1f} %\n{100 * m.coverage_mean[-1]:.1f} %\n{m.flown[-1]:.0f} m\n"
                 f"{len(H)} honest" + (f" + {len(att)} attacker" if att else "") +
                 f"\n{m.crashed[-1]}\n{m.wrong[-1]:.0f}")
        if att:
            left += "\nattacker revoked by"
            right += f"\n{len(rev)} of {len(H)}"
            fakes = [f for d in att for f in getattr(d, "fakes", [])]
            if fakes and not s.p.auth:
                caught = sum(all(not h.accept[f.id] for h in H if not h.crashed) for f in fakes)
                left += "\nfake identities: active / caught by all"
                right += f"\n{sum(f.alive for f in fakes)} / {caught}"
        left += "\nseen by honest drones"
        right += f"\n{100 * m.coverage_honest[-1]:.1f} %"
        if s.done():
            left += "\n\nmission complete"
        return left, right
