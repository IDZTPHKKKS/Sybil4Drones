import json
import os

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from sybil_msgs.msg import Revocation, TrustState

from .recording import compare, load, unpack


def ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


class Checker(Node):
    def __init__(self):
        super().__init__("checker")
        self.path = self.declare_parameter("recording", "").value
        self.out = self.declare_parameter("out", "").value
        self.head, self.frames = load(self.path)
        self.revs, self.states = {}, {}
        for i in self.head["honest"]:
            self.create_subscription(Revocation, f"/drone_{i}/trust/revocations", self.on_rev, 1000)
            self.create_subscription(TrustState, f"/drone_{i}/trust/state", self.on_state, 1000)
        self.last, self.idle = 0, 0
        self.done = False
        self.create_timer(1.0, self.check)

    def on_rev(self, m):
        self.revs.setdefault((m.observer, ns(m.header.stamp)), []).append((m.origin, m.reason))

    def on_state(self, m):
        self.states[(m.observer, ns(m.header.stamp))] = unpack(m.trusted_bits, self.head["n_ids"])

    def check(self):
        n = len(self.states)
        self.idle = self.idle + 1 if n == self.last else 0
        self.last = n
        if n < len(self.frames) and self.idle < (10 if n else 60):
            return
        results = {k: (self.revs.get(k, []), v) for k, v in self.states.items()}
        r = compare(self.head, self.frames, results)
        name = os.path.basename(self.path).replace(".jsonl.gz", "")
        self.get_logger().info(f"{name}: {json.dumps(r)}")
        if self.out:
            with open(self.out, "a") as fh:
                fh.write(json.dumps(dict(mission=name, **r)) + "\n")
        self.done = True


def main():
    rclpy.init()
    node = Checker()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.2)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
