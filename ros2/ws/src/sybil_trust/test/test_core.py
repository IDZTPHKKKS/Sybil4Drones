import math

import numpy as np

from sybil_trust.core import Trust
from sybil_trust.view import Camera, VoxelMap


def test_audit_revokes_liar():
    t = Trust(0, "D5")
    assert (1, "audit", 200, 3000) in t.step(1.0, [0, 0, 1], {1: (3000, 200, 100, 10)}, [], [], {})


def test_audit_keeps_noisy_honest_drone():
    t = Trust(0, "D5")
    assert t.step(1.0, [0, 0, 1], {1: (3000, 2, 100, 0)}, [], [], {}) == []


def test_missing_body_is_revoked():
    t = Trust(0, "D2")
    out = []
    for k in range(10):
        out += t.step(float(k), [0, 0, 1], {}, [(5, [2.0, 0, 1], 0.0, True)], [], {})
    assert (5, "absent", 6, 6) in out


def test_binding_rejects_shadow():
    t = Trust(0, "D5", detect=1.0)
    body = [(1, [2.0, 0.0, 1.0])]
    beacons = [(1, [2.0, 0.0, 1.0], 0.0, True), (9, [2.3, 0.0, 1.0], 0.0, True)]
    out = []
    for k in range(12):
        out += t.step(float(k), [0, 0, 1], {}, beacons, body, {})
    assert any(o == 9 and why == "absent" for o, why, _, _ in out)
    assert t.accept[1]


def test_quarantine_needs_quorum():
    t = Trust(0, "D4", detect=1.0)
    seen = [(1, [2.0, 0, 1], 0.0, True), (2, [2.0, 1.0, 1], 0.0, True)]
    t.step(0.0, [0, 0, 1], {}, seen, [(1, [2.0, 0, 1]), (2, [2.0, 1.0, 1])], {})
    t.step(1.0, [0, 0, 1], {}, [], [], {1: [7]})
    assert not t.trusted()[7]
    t.step(2.0, [0, 0, 1], {}, [], [], {1: [7], 2: [7]})
    assert t.trusted()[7]


def test_wall_blocks_view():
    occ = np.zeros((40, 40, 10), bool)
    occ[20, :, :] = True
    vox, cam = VoxelMap(occ, 0.2), Camera()
    assert cam.sees([3.0, 4.0, 1.0], 0.0, [3.8, 4.0, 1.0], vox)
    assert not cam.sees([3.0, 4.0, 1.0], 0.0, [5.0, 4.0, 1.0], vox)
    assert not cam.sees([3.0, 4.0, 1.0], math.pi, [3.8, 4.0, 1.0], vox)


def test_voxel_points_roundtrip():
    occ = np.zeros((10, 8, 5), bool)
    occ[3, 2, 1] = occ[9, 7, 4] = True
    vox = VoxelMap(occ, 0.2)
    back = VoxelMap.from_points(vox.centres().astype(np.float32), 0.2)
    assert np.array_equal(back.occ, occ)
