import glob
import os

import pytest

from sybil_replay.recording import compare, load, stamp_ns, world
from sybil_trust.core import Trust
from sybil_trust.view import Camera

DATA = sorted(glob.glob(os.environ.get("SYBIL_DATA", "/data") + "/*.jsonl.gz"))


@pytest.mark.parametrize("path", DATA, ids=[os.path.basename(p) for p in DATA])
def test_trust_core_matches_simulator(path):
    head, frames = load(path)
    voxels = world(head)
    cam = Camera(head["max_range"], head["hfov"], head["vfov"])
    nodes = {i: Trust(i, head["level"], head["n_ids"], detect=head["detect"], v_max=head["v_max"])
             for i in head["honest"]}
    results = {}
    for f in frames:
        core = nodes[f["observer"]]
        counts = {o: tuple(v) for o, *v in f["counts"]}
        beacons = [(o, q, f["t"] - t0, cam.sees(f["position"], f["yaw"], q, voxels)) for o, q, t0 in f["beacons"]]
        revs = core.step(f["t"], f["position"], counts, beacons, f["bodies"], dict(f["witness"]))
        results[(f["observer"], stamp_ns(f["t"]))] = ([(o, w) for o, w, _, _ in revs], core.trusted())
    r = compare(head, frames, results)
    assert r["ok"], r
