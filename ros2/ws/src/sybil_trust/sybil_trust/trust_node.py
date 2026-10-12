import math

import numpy as np
import rclpy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from vision_msgs.msg import Detection3DArray

from sybil_msgs.msg import AuditCounts, BeaconArray, Revocation, TrustState, WitnessArray, WitnessReport

from .core import Trust
from .view import Camera, VoxelMap


def ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class TrustNode(Node):
    def __init__(self):
        super().__init__("trust")
        p = lambda name, default: self.declare_parameter(name, default).value
        self.me = p("drone_id", 0)
        level = p("level", "D4")
        self.core = Trust(self.me, level, n_ids=p("n_ids", 256), tol=p("tol", 0.02), conf=p("conf", 0.99),
                          detect=p("detect", 0.9), quorum=p("quorum", 2), v_max=p("v_max", 1.5),
                          seed=p("seed", self.me))
        self.camera = Camera(p("max_range", 6.0), p("hfov", math.radians(90)), p("vfov", math.radians(60)))
        self.res = p("map_resolution", 0.2)
        self.voxels = None
        self.sync = None
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(PointCloud2, "/map_3d", self.on_map, latched)
        self.rev_pub = self.create_publisher(Revocation, "trust/revocations", 100)
        self.state_pub = self.create_publisher(TrustState, "trust/state", 100)
        self.wit_pub = self.create_publisher(WitnessReport, "radio/witness_out", 10)
        self.get_logger().info(f"defence {level}")

    def on_map(self, m):
        xyz = point_cloud2.read_points_numpy(m, field_names=("x", "y", "z"))
        self.voxels = VoxelMap.from_points(xyz, self.res)
        if self.sync is None:
            inputs = [Subscriber(self, Odometry, "odom", qos_profile=100),
                      Subscriber(self, AuditCounts, "audit/counts", qos_profile=100),
                      Subscriber(self, BeaconArray, "radio/beacons", qos_profile=100),
                      Subscriber(self, WitnessArray, "radio/witness", qos_profile=100),
                      Subscriber(self, Detection3DArray, "perception/drones", qos_profile=100)]
            self.sync = ApproximateTimeSynchronizer(inputs, queue_size=100, slop=0.01)
            self.sync.registerCallback(self.on_inputs)
            self.get_logger().info(f"map received ({len(xyz)} voxels), listening")

    def on_inputs(self, odom, counts, beacons, witness, drones):
        stamp = counts.header.stamp
        t = ns(stamp) * 1e-9
        p = odom.pose.pose.position
        pos = np.array([p.x, p.y, p.z])
        yaw = yaw_of(odom.pose.pose.orientation)
        audit = {o: (c, w, cw, ww) for o, c, w, cw, ww in
                 zip(counts.origin, counts.checked, counts.wrong, counts.checked_walls, counts.wrong_walls)}
        claims = []
        for b in beacons.beacons:
            q = np.array([b.position.x, b.position.y, b.position.z])
            claims.append((b.id, q, (ns(stamp) - ns(b.received)) * 1e-9, self.camera.sees(pos, yaw, q, self.voxels)))
        bodies = [(int(d.id), np.array([d.bbox.center.position.x, d.bbox.center.position.y,
                                         d.bbox.center.position.z])) for d in drones.detections]
        reports = {r.sender: list(r.seen) for r in witness.reports}
        for o, why, bad, total in self.core.step(t, pos, audit, claims, bodies, reports):
            self.rev_pub.publish(Revocation(header=counts.header, observer=self.me, origin=o, reason=why,
                                            bad=bad, total=total))
            self.get_logger().info(f"t={t:.1f}s revoked {o} ({why}, {bad}/{total})")
        self.state_pub.publish(TrustState(header=counts.header, observer=self.me,
                                          trusted_bits=np.packbits(self.core.trusted()).tolist(),
                                          revoked=[int(o) for o in np.flatnonzero(~self.core.accept)]))
        self.wit_pub.publish(WitnessReport(sender=self.me, seen=self.core.witnessed()))


def main():
    rclpy.init()
    try:
        rclpy.spin(TrustNode())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
