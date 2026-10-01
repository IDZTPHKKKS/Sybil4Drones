"""Compiled inner loops: ray casting, map updates, planning grid, breadth-first search, line checks."""
import numpy as np
from numba import njit

UNKNOWN, FREE, OCC = -1, 0, 1


@njit(cache=True)
def seed(n):
    np.random.seed(n)


@njit(cache=True)
def _first_hit(occ, p0, p1, p2, d0, d1, d2, tmax):
    nx, ny, nz = occ.shape
    i, j, k = int(np.floor(p0)), int(np.floor(p1)), int(np.floor(p2))
    si = 1 if d0 > 0 else -1
    sj = 1 if d1 > 0 else -1
    sk = 1 if d2 > 0 else -1
    big = 1e30
    tdi = abs(1.0 / d0) if d0 != 0 else big
    tdj = abs(1.0 / d1) if d1 != 0 else big
    tdk = abs(1.0 / d2) if d2 != 0 else big
    tmi = ((i + (si > 0)) - p0) / d0 if d0 != 0 else big
    tmj = ((j + (sj > 0)) - p1) / d1 if d1 != 0 else big
    tmk = ((k + (sk > 0)) - p2) / d2 if d2 != 0 else big
    t = 0.0
    while t <= tmax:
        if i < 0 or j < 0 or k < 0 or i >= nx or j >= ny or k >= nz:
            return -1.0
        if occ[i, j, k]:
            return t
        if tmi < tmj and tmi < tmk:
            t = tmi
            tmi += tdi
            i += si
        elif tmj < tmk:
            t = tmj
            tmj += tdj
            j += sj
        else:
            t = tmk
            tmk += tdk
            k += sk
    return -1.0


@njit(cache=True)
def raycast(occ, res, origin, dirs, max_range, noise, dropout, out_idx, out_val, clear_range):
    nx, ny, nz = occ.shape
    n = 0
    cap = out_idx.shape[0]
    tmax = max_range / res
    p0, p1, p2 = origin[0] / res, origin[1] / res, origin[2] / res
    for r in range(dirs.shape[0]):
        if dropout > 0 and np.random.random() < dropout:
            continue
        d0, d1, d2 = dirs[r, 0], dirs[r, 1], dirs[r, 2]
        th = _first_hit(occ, p0, p1, p2, d0, d1, d2, tmax)
        hit = th >= 0
        tm = clear_range / res
        if hit:
            tm = th + 1e-6 + noise * th * np.random.standard_normal()
            if tm < 0.5:
                tm = 0.5
            if tm > tmax:
                hit = False
                tm = clear_range / res
        i, j, k = int(np.floor(p0)), int(np.floor(p1)), int(np.floor(p2))
        si = 1 if d0 > 0 else -1
        sj = 1 if d1 > 0 else -1
        sk = 1 if d2 > 0 else -1
        big = 1e30
        tdi = abs(1.0 / d0) if d0 != 0 else big
        tdj = abs(1.0 / d1) if d1 != 0 else big
        tdk = abs(1.0 / d2) if d2 != 0 else big
        tmi = ((i + (si > 0)) - p0) / d0 if d0 != 0 else big
        tmj = ((j + (sj > 0)) - p1) / d1 if d1 != 0 else big
        tmk = ((k + (sk > 0)) - p2) / d2 if d2 != 0 else big
        while n < cap:
            if i < 0 or j < 0 or k < 0 or i >= nx or j >= ny or k >= nz:
                if hit and n > 0:
                    out_val[n - 1] = OCC
                break
            flat = (i * ny + j) * nz + k
            tnext = min(tmi, tmj, tmk)
            if tnext >= tm:
                if hit:
                    out_idx[n] = flat
                    out_val[n] = OCC
                    n += 1
                break
            out_idx[n] = flat
            out_val[n] = FREE
            n += 1
            if tmi < tmj and tmi < tmk:
                tmi += tdi
                i += si
            elif tmj < tmk:
                tmj += tdj
                j += sj
            else:
                tmk += tdk
                k += sk
    return n


LO_HIT, LO_MISS, LO_MIN, LO_MAX = 3, -2, -8, 14


@njit(cache=True)
def apply_own(state, src, lo, idx, val, n, own, changed, agree, disagree, stamp, frame, ev_cell, ev_org, ev_val, ev_n,
              checked_occ):
    hit_mark = 2 * frame
    miss_mark = 2 * frame + 1
    m = 0
    for pas in range(2):
        for q in range(n):
            f = idx[q]
            is_hit = val[q] == OCC
            if pas == 0:
                if not is_hit or stamp[f] == hit_mark:
                    continue
                stamp[f] = hit_mark
            else:
                if is_hit or stamp[f] == hit_mark or stamp[f] == miss_mark:
                    continue
                stamp[f] = miss_mark
            o = src[f]
            if o != own:
                if o >= 0:
                    if state[f] == OCC:
                        checked_occ[o] += 1
                    if state[f] == val[q]:
                        agree[o] += 1
                    else:
                        disagree[o] += 1
                        e = ev_n[0]
                        if e < ev_cell.shape[0]:
                            ev_cell[e] = f
                            ev_org[e] = o
                            ev_val[e] = state[f]
                            ev_n[0] = e + 1
                src[f] = own
                lo[f] = 0
            v = lo[f] + (LO_HIT if is_hit else LO_MISS)
            lo[f] = min(max(v, LO_MIN), LO_MAX)
            s_new = OCC if lo[f] > 0 else FREE
            if state[f] != s_new:
                changed[m] = q
                m += 1
                state[f] = s_new
    return m


@njit(cache=True)
def apply_received(state, src, idx, val, origin, accept, changed):
    m = 0
    for q in range(idx.shape[0]):
        f = idx[q]
        if state[f] == UNKNOWN and accept[origin[q]]:
            state[f] = val[q]
            src[f] = origin[q]
            changed[m] = q
            m += 1
    return m


@njit(cache=True)
def planning_grid(state, shape, f):
    nx, ny, nz = shape
    cx, cy, cz = nx // f, ny // f, nz // f
    cstate = np.full((cx, cy, cz), -1, np.int8)
    trav = np.zeros((cx, cy, cz), np.bool_)
    for a in range(cx):
        for b in range(cy):
            for c in range(cz):
                anyocc = False
                allfree = True
                for i in range(a * f, a * f + f):
                    for j in range(b * f, b * f + f):
                        for k in range(c * f, c * f + f):
                            s = state[(i * ny + j) * nz + k]
                            if s == OCC:
                                anyocc = True
                            if s != FREE:
                                allfree = False
                if anyocc:
                    cstate[a, b, c] = 1
                elif allfree:
                    cstate[a, b, c] = 0
                if not allfree:
                    continue
                ok = True
                for i in range(max(a * f - 1, 0), min(a * f + f + 1, nx)):
                    for j in range(max(b * f - 1, 0), min(b * f + f + 1, ny)):
                        for k in range(max(c * f - 1, 0), min(c * f + f + 1, nz)):
                            if state[(i * ny + j) * nz + k] == OCC:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                trav[a, b, c] = ok
    return cstate, trav


@njit(cache=True)
def frontiers(cstate, trav, blocked):
    cx, cy, cz = trav.shape
    fr = np.zeros(trav.shape, np.bool_)
    for a in range(cx):
        for b in range(cy):
            for c in range(cz):
                if not trav[a, b, c] or blocked[a, b, c]:
                    continue
                if (a > 0 and cstate[a - 1, b, c] == -1) or (a < cx - 1 and cstate[a + 1, b, c] == -1) or \
                   (b > 0 and cstate[a, b - 1, c] == -1) or (b < cy - 1 and cstate[a, b + 1, c] == -1):
                    fr[a, b, c] = True
    return fr


@njit(cache=True)
def bfs(trav, start):
    cx, cy, cz = trav.shape
    n = cx * cy * cz
    dist = np.full(n, -1, np.int32)
    parent = np.full(n, -1, np.int32)
    queue = np.empty(n, np.int32)
    s = (start[0] * cy + start[1]) * cz + start[2]
    dist[s] = 0
    queue[0] = s
    head, tail = 0, 1
    while head < tail:
        u = queue[head]
        head += 1
        a = u // (cy * cz)
        b = (u // cz) % cy
        c = u % cz
        for da in range(-1, 2):
            for db in range(-1, 2):
                for dc in range(-1, 2):
                    if da == 0 and db == 0 and dc == 0:
                        continue
                    x, y, z = a + da, b + db, c + dc
                    if x < 0 or y < 0 or z < 0 or x >= cx or y >= cy or z >= cz:
                        continue
                    if not trav[x, y, z]:
                        continue
                    v = (x * cy + y) * cz + z
                    if dist[v] >= 0:
                        continue
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue[tail] = v
                    tail += 1
    return dist, parent


@njit(cache=True)
def segment_clear(trav, p, q):
    d0, d1, d2 = q[0] - p[0], q[1] - p[1], q[2] - p[2]
    steps = int(2 * max(abs(d0), abs(d1), abs(d2))) + 1
    for s in range(steps + 1):
        t = s / steps
        a = int(np.floor(p[0] + 0.5 + t * d0))
        b = int(np.floor(p[1] + 0.5 + t * d1))
        c = int(np.floor(p[2] + 0.5 + t * d2))
        if not trav[a, b, c]:
            return False
    return True


@njit(cache=True)
def wall_crossings(occ, res, p, q):
    nx, ny, nz = occ.shape
    L = np.sqrt((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 + (q[2] - p[2]) ** 2)
    steps = int(L / (0.5 * res)) + 1
    prev = False
    n = 0
    for s in range(steps + 1):
        t = s / steps
        i = int((p[0] + t * (q[0] - p[0])) / res)
        j = int((p[1] + t * (q[1] - p[1])) / res)
        k = int((p[2] + t * (q[2] - p[2])) / res)
        if i < 0 or j < 0 or k < 0 or i >= nx or j >= ny or k >= nz:
            continue
        o = occ[i, j, k]
        if o and not prev:
            n += 1
        prev = o
    return n


@njit(cache=True)
def sphere_hits(occ, res, p, radius):
    nx, ny, nz = occ.shape
    r = radius / res
    ci, cj, ck = p[0] / res, p[1] / res, p[2] / res
    for i in range(max(int(ci - r) - 1, 0), min(int(ci + r) + 2, nx)):
        for j in range(max(int(cj - r) - 1, 0), min(int(cj + r) + 2, ny)):
            for k in range(max(int(ck - r) - 1, 0), min(int(ck + r) + 2, nz)):
                if not occ[i, j, k]:
                    continue
                dx = max(i - ci, 0.0, ci - (i + 1))
                dy = max(j - cj, 0.0, cj - (j + 1))
                dz = max(k - ck, 0.0, ck - (k + 1))
                if dx * dx + dy * dy + dz * dz < r * r:
                    return True
    return False


@njit(cache=True)
def path_blocked(state, shape, res, pts, margin):
    nx, ny, nz = shape
    r = int(np.ceil(margin / res))
    for s in range(pts.shape[0] - 1):
        L = np.sqrt(((pts[s + 1] - pts[s]) ** 2).sum())
        steps = int(L / (0.5 * res)) + 1
        for u in range(steps + 1):
            t = u / steps
            ci = int((pts[s, 0] + t * (pts[s + 1, 0] - pts[s, 0])) / res)
            cj = int((pts[s, 1] + t * (pts[s + 1, 1] - pts[s, 1])) / res)
            ck = int((pts[s, 2] + t * (pts[s + 1, 2] - pts[s, 2])) / res)
            for i in range(max(ci - r, 0), min(ci + r + 1, nx)):
                for j in range(max(cj - r, 0), min(cj + r + 1, ny)):
                    for k in range(max(ck - r, 0), min(ck + r + 1, nz)):
                        if state[(i * ny + j) * nz + k] == OCC:
                            return True
    return False


@njit(cache=True)
def near_known_occ(state, shape, res, p, radius):
    nx, ny, nz = shape
    r = radius / res
    ci, cj, ck = p[0] / res, p[1] / res, p[2] / res
    for i in range(max(int(ci - r) - 1, 0), min(int(ci + r) + 2, nx)):
        for j in range(max(int(cj - r) - 1, 0), min(int(cj + r) + 2, ny)):
            for k in range(max(int(ck - r) - 1, 0), min(int(ck + r) + 2, nz)):
                if state[(i * ny + j) * nz + k] != OCC:
                    continue
                dx = max(i - ci, 0.0, ci - (i + 1))
                dy = max(j - cj, 0.0, cj - (j + 1))
                dz = max(k - ck, 0.0, ck - (k + 1))
                if dx * dx + dy * dy + dz * dz < r * r:
                    return True
    return False


@njit(cache=True)
def audit(state, src, shape, own, ev_cell, ev_org, ev_val, n, far, tol, complete, vouched, far_occ):
    nx, ny, nz = shape
    for e in range(n):
        f = ev_cell[e]
        claimed = ev_val[e]
        o = ev_org[e]
        if state[f] == claimed:
            continue
        i = f // (ny * nz)
        j = (f // nz) % ny
        k = f % nz
        near = False
        for a in range(max(i - tol, 0), min(i + tol + 1, nx)):
            for b in range(max(j - tol, 0), min(j + tol + 1, ny)):
                for c in range(max(k - tol, 0), min(k + tol + 1, nz)):
                    g = (a * ny + b) * nz + c
                    sg = src[g]
                    if state[g] == claimed and sg >= 0 and sg != o and (sg == own or vouched[sg]):
                        near = True
                        break
                if near:
                    break
            if near:
                break
        if near:
            continue
        seen = 0
        tot = 0
        for a in range(max(i - 1, 0), min(i + 2, nx)):
            for b in range(max(j - 1, 0), min(j + 2, ny)):
                for c in range(max(k - 1, 0), min(k + 2, nz)):
                    g = (a * ny + b) * nz + c
                    if g == f:
                        continue
                    tot += 1
                    if src[g] == own:
                        seen += 1
        if seen >= complete * tot:
            far[o] += 1
            if claimed == OCC:
                far_occ[o] += 1


@njit(cache=True)
def revoke(state, src, lo, org):
    m = 0
    for f in range(state.shape[0]):
        if src[f] == org:
            state[f] = UNKNOWN
            src[f] = -1
            lo[f] = 0
            m += 1
    return m


@njit(cache=True)
def suspect_cells(state, src, shape, f, own, suspect):
    nx, ny, nz = shape
    cx, cy, cz = nx // f, ny // f, nz // f
    out = np.zeros((cx, cy, cz), np.bool_)
    for i in range(cx * f):
        for j in range(cy * f):
            for k in range(cz * f):
                g = (i * ny + j) * nz + k
                o = src[g]
                if o >= 0 and o != own and state[g] == OCC and suspect[o]:
                    out[i // f, j // f, k // f] = True
    return out


@njit(cache=True)
def verify_frontiers(trav, sus, blocked, explore):
    cx, cy, cz = trav.shape
    fr = np.zeros(trav.shape, np.bool_)
    for a in range(cx):
        for b in range(cy):
            for c in range(cz):
                if not trav[a, b, c] or blocked[a, b, c] or explore[a, b, c]:
                    continue
                if (a > 0 and sus[a - 1, b, c]) or (a < cx - 1 and sus[a + 1, b, c]) or \
                   (b > 0 and sus[a, b - 1, c]) or (b < cy - 1 and sus[a, b + 1, c]):
                    fr[a, b, c] = True
    return fr


@njit(cache=True)
def suspect_free_cells(state, src, shape, f, own, suspect):
    nx, ny, nz = shape
    cx, cy, cz = nx // f, ny // f, nz // f
    out = np.zeros((cx, cy, cz), np.bool_)
    for i in range(cx * f):
        for j in range(cy * f):
            for k in range(cz * f):
                g = (i * ny + j) * nz + k
                o = src[g]
                if o >= 0 and o != own and state[g] == FREE and suspect[o]:
                    out[i // f, j // f, k // f] = True
    return out


@njit(cache=True)
def bfs6(trav, start, max_steps):
    cx, cy, cz = trav.shape
    n = cx * cy * cz
    dist = np.full(n, -1, np.int32)
    parent = np.full(n, -1, np.int32)
    queue = np.empty(n, np.int32)
    s = (start[0] * cy + start[1]) * cz + start[2]
    dist[s] = 0
    queue[0] = s
    head, tail = 0, 1
    while head < tail:
        u = queue[head]
        head += 1
        if dist[u] >= max_steps:
            continue
        a = u // (cy * cz)
        b = (u // cz) % cy
        c = u % cz
        for d in range(6):
            x, y, z = a, b, c
            if d == 0:
                x += 1
            elif d == 1:
                x -= 1
            elif d == 2:
                y += 1
            elif d == 3:
                y -= 1
            elif d == 4:
                z += 1
            else:
                z -= 1
            if x < 0 or y < 0 or z < 0 or x >= cx or y >= cy or z >= cz or not trav[x, y, z]:
                continue
            v = (x * cy + y) * cz + z
            if dist[v] >= 0:
                continue
            dist[v] = dist[u] + 1
            parent[v] = u
            queue[tail] = v
            tail += 1
    return dist, parent


@njit(cache=True)
def fabricate_scan(shape, res, origin, dirs, lengths, out_idx, out_val):
    nx, ny, nz = shape
    n = 0
    cap = out_idx.shape[0]
    p0, p1, p2 = origin[0] / res, origin[1] / res, origin[2] / res
    for r in range(dirs.shape[0]):
        d0, d1, d2 = dirs[r, 0], dirs[r, 1], dirs[r, 2]
        tm = lengths[r] / res
        i, j, k = int(np.floor(p0)), int(np.floor(p1)), int(np.floor(p2))
        si = 1 if d0 > 0 else -1
        sj = 1 if d1 > 0 else -1
        sk = 1 if d2 > 0 else -1
        big = 1e30
        tdi = abs(1.0 / d0) if d0 != 0 else big
        tdj = abs(1.0 / d1) if d1 != 0 else big
        tdk = abs(1.0 / d2) if d2 != 0 else big
        tmi = ((i + (si > 0)) - p0) / d0 if d0 != 0 else big
        tmj = ((j + (sj > 0)) - p1) / d1 if d1 != 0 else big
        tmk = ((k + (sk > 0)) - p2) / d2 if d2 != 0 else big
        while n < cap:
            if i < 1 or j < 1 or k < 1 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
                break
            flat = (i * ny + j) * nz + k
            if min(tmi, tmj, tmk) >= tm:
                out_idx[n] = flat
                out_val[n] = OCC
                n += 1
                break
            out_idx[n] = flat
            out_val[n] = FREE
            n += 1
            if tmi < tmj and tmi < tmk:
                tmi += tdi
                i += si
            elif tmj < tmk:
                tmj += tdj
                j += sj
            else:
                tmk += tdk
                k += sk
    return n
