import argparse
import gzip
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 10})

STATUS = {"trusted": "#1a9e3a", "quarantined": "#e0b000", "revoked": "#d01c1c"}


def load(path):
    with gzip.open(path, "rt") as fh:
        head = json.loads(fh.readline())
        frames = [json.loads(line) for line in fh]
    return head, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--viewer", type=int, default=None)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    head, frames = load(a.recording)
    viewer = head["honest"][0] if a.viewer is None else a.viewer
    name = os.path.basename(a.recording).replace(".jsonl.gz", "")
    out = a.out or os.path.join(os.path.dirname(a.recording), "videos", name + ".mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    w = head["world"]
    n = int(np.prod(w["shape"]))
    occ = np.unpackbits(np.frombuffer(bytes.fromhex(w["bits"]), np.uint8))[:n].astype(bool).reshape(w["shape"])
    res = w["res"]
    band = occ[:, :, int(round(0.4 / res)):int(round(2.4 / res))].any(2)
    W, H = band.shape[0] * res, band.shape[1] * res
    grid = np.zeros(band.size, np.int8)
    have_map = any("map" in f for f in frames)
    cmap = ListedColormap(["#e4e4e4", "#ffffff", "#3a3a3a", "#d01c1c"])

    by_sec = defaultdict(list)
    for f in frames:
        by_sec[int(round(f["t"]))].append(f)
    secs = sorted(by_sec)

    fig, ax = plt.subplots(figsize=(8, 6.2), dpi=120)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.08)
    trails = defaultdict(list)
    revoked, log = set(), []
    trusted = np.ones(head["n_ids"], bool)
    attackers = set(head.get("attackers", []))
    honest = set(head["honest"])
    title = (f"{head['family']} {head['seed']}, {head['attack'].replace('sybil_', '')} attacker, "
             f"defence {head['level']}, seen by drone {viewer}")

    writer = FFMpegWriter(fps=a.fps, bitrate=2400)
    with writer.saving(fig, out, dpi=120):
        for s in secs:
            pos, claims = {}, {}
            for f in by_sec[s]:
                pos[f["observer"]] = f["position"]
                for b, q in f["bodies"]:
                    if b in attackers:
                        pos[b] = q
                for o, q, _ in f["beacons"]:
                    claims[o] = q
                for o, why in f["revoked"]:
                    tag = "honest drone" if o in honest else "identity"
                    log = (log + [f"{f['t']:5.0f} s  drone {f['observer']} revoked {tag} {o} ({why})"])[-4:]
                if f["observer"] == viewer and "map" in f:
                    idx, vals = f["map"]
                    grid[np.asarray(idx, int)] = vals
                if f["observer"] == viewer:
                    trusted = np.unpackbits(np.frombuffer(bytes.fromhex(f["trusted"]), np.uint8))[:head["n_ids"]]
                    revoked |= {o for o, _ in f["revoked"]}
            for i, q in pos.items():
                trails[i].append(q[:2])

            ax.clear()
            if have_map:
                ax.imshow(grid.reshape(band.shape).T, origin="lower", extent=(0, W, 0, H), cmap=cmap, vmin=-0.5,
                          vmax=3.5, interpolation="nearest")
                ax.contour(band.T, levels=[0.5], extent=(0, W, 0, H), colors="#9a9a9a", linewidths=0.3)
            else:
                ax.imshow(band.T, origin="lower", extent=(0, W, 0, H), cmap="Greys", vmin=0, vmax=1.6,
                          interpolation="nearest")
            for i, tr in trails.items():
                tr = np.array(tr)
                ax.plot(tr[:, 0], tr[:, 1], ":", lw=0.9, color="#d9822b" if i in attackers else "#2c5bb8")
            counts = dict(trusted=0, quarantined=0, revoked=0)
            for o, q in claims.items():
                if o in pos or o in honest or o in attackers:
                    continue
                st = "revoked" if o in revoked else ("trusted" if trusted[o] else "quarantined")
                counts[st] += 1
                ax.plot(q[0], q[1], "s", ms=7, mfc=STATUS[st], mec="k", mew=0.6)
                ax.annotate(str(o), (q[0], q[1]), xytext=(4, 4), textcoords="offset points", fontsize=7,
                            color=STATUS[st])
            for i, q in pos.items():
                atk = i in attackers
                ax.plot(q[0], q[1], "o", ms=8 if i == viewer else 6.5, mfc="#d9822b" if atk else "#2c5bb8",
                        mec="k", mew=1.2 if i == viewer else 0.6)
                ax.annotate(f"{i}" + (" (attacker)" if atk else ""), (q[0], q[1]), xytext=(5, -9),
                            textcoords="offset points", fontsize=7)
            ax.set_xlim(0, W)
            ax.set_ylim(0, H)
            ax.set_aspect("equal")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            known = (grid > 0).mean() * 100
            false = int((grid == 3).sum())
            ax.set_title(f"{title}\nt = {s:3d} s    map of drone {viewer}: {known:.0f}% known, {false} false wall cells"
                         f"    fake identities: {counts['trusted']} trusted, "
                         f"{counts['quarantined']} quarantined, {counts['revoked']} revoked", fontsize=10)
            for k, line in enumerate(log):
                ax.text(0.01, 0.01 + 0.04 * (len(log) - 1 - k), line, transform=ax.transAxes, fontsize=7,
                        family="monospace", va="bottom", bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
            h = [plt.Line2D([], [], ls="", marker="o", mfc="#2c5bb8", mec="k", label="honest drone"),
                 plt.Line2D([], [], ls="", marker="o", mfc="#d9822b", mec="k", label="compromised drone")]
            h += [plt.Line2D([], [], ls="", marker="s", mfc=c, mec="k", label=f"fake, {k}") for k, c in STATUS.items()]
            if have_map:
                h += [plt.Line2D([], [], ls="", marker="s", mfc=c, mec="#999", label=k) for k, c in
                      (("unknown", "#e4e4e4"), ("free", "#ffffff"), ("wall", "#3a3a3a"), ("false wall", "#d01c1c"))]
            ax.legend(handles=h, loc="upper right", fontsize=6.5, framealpha=0.9, ncol=2)
            writer.grab_frame()
    plt.close(fig)
    print(f"{out}: {len(secs)} frames, {len(secs) / a.fps:.0f} s")


if __name__ == "__main__":
    main()
