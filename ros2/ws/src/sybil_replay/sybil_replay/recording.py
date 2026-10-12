import gzip
import json

import numpy as np

from sybil_trust.view import VoxelMap


def load(path):
    with gzip.open(path, "rt") as fh:
        head = json.loads(fh.readline())
        frames = [json.loads(line) for line in fh]
    return head, frames


def world(head):
    w = head["world"]
    n = int(np.prod(w["shape"]))
    occ = np.unpackbits(np.frombuffer(bytes.fromhex(w["bits"]), np.uint8))[:n].astype(bool).reshape(w["shape"])
    return VoxelMap(occ, w["res"])


def stamp_ns(t):
    return int(round(t * 1e9))


def unpack(bits, n):
    if isinstance(bits, str):
        bits = bytes.fromhex(bits)
    return np.unpackbits(np.frombuffer(bytes(bits), np.uint8))[:n].astype(bool)


def compare(head, frames, results):
    bad_trust, bad_rev, missing, n_rev = 0, 0, 0, 0
    gaps, diffs = [], []
    for f in frames:
        got = results.get((f["observer"], stamp_ns(f["t"])))
        if got is None:
            missing += 1
            gaps.append(f["t"])
            continue
        revs, trusted = got
        if sorted(map(tuple, f["revoked"])) != sorted(map(tuple, revs)):
            bad_rev += 1
        want = unpack(f["trusted"], head["n_ids"])
        if not np.array_equal(want, trusted):
            bad_trust += 1
            if len(diffs) < 8:
                diffs.append([f["observer"], round(f["t"], 1), np.flatnonzero(want != trusted).tolist()])
        n_rev += len(revs)
    return dict(frames=len(frames), missing=missing, revocations=n_rev, expected_revocations=head["revocations"],
                trust_mismatch=bad_trust, revocation_mismatch=bad_rev,
                trust_diffs=diffs,
                missing_t=[round(min(gaps), 1), round(max(gaps), 1)] if gaps else [],
                ok=missing == 0 and bad_trust == 0 and bad_rev == 0 and n_rev == head["revocations"])
