#!/usr/bin/env python3
"""Top-down views of the world families (obstacles at flight height, and at any height).

    python3 scripts/show_worlds.py            # seeds 0-2 of every family -> media/worlds.png
    python3 scripts/show_worlds.py 0 9        # seeds 0..9 (all 40 experiment maps) -> media/worlds_0-9.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from swarm import worlds  # noqa: E402


def main(seeds=(0, 1, 2)):
    fam = worlds.FAMILIES
    fig, axes = plt.subplots(len(fam), len(seeds), figsize=(4.2 * len(seeds), 3.1 * len(fam)), squeeze=False)
    for i, f in enumerate(fam):
        for j, s in enumerate(seeds):
            w = worlds.make(f, s)
            occ = w.occ[:, :, 1:-1]
            img = 0.35 * occ.any(2) + 0.65 * w.occ[:, :, int(1.2 / w.res)]
            ax = axes[i, j]
            ax.imshow(img.T, origin="lower", cmap="Greys", extent=[0, w.size[0], 0, w.size[1]], vmin=0, vmax=1)
            ax.plot(w.starts[:16, 0], w.starts[:16, 1], ".", color="tab:red", ms=3)
            ax.set_title(f"{f}, seed {s}", fontsize=9)
            ax.set_xticks([]), ax.set_yticks([])
            assert not w.occ[tuple((w.starts / w.res).astype(int).T)].any(), "start inside obstacle"
    fig.tight_layout()
    name = "worlds.png" if tuple(seeds) == (0, 1, 2) else f"worlds_{seeds[0]}-{seeds[-1]}.png"
    out = os.path.join(ROOT, "media", name)
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:]]
    main(tuple(range(a[0], a[1] + 1)) if len(a) == 2 else (0, 1, 2))
