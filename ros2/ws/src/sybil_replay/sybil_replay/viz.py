import numpy as np
import rclpy
from geometry_msgs.msg import Point, TransformStamped
from tf2_ros import TransformBroadcaster
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

from sybil_msgs.msg import BeaconArray, Revocation, TrustState

BLUE = ColorRGBA(r=0.15, g=0.35, b=0.85, a=1.0)
ORANGE = ColorRGBA(r=0.95, g=0.55, b=0.1, a=1.0)
GREEN = ColorRGBA(r=0.1, g=0.75, b=0.2, a=0.9)
YELLOW = ColorRGBA(r=0.95, g=0.85, b=0.1, a=0.9)
RED = ColorRGBA(r=0.85, g=0.1, b=0.1, a=0.9)
WHITE = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
TRAIL = ColorRGBA(r=0.15, g=0.35, b=0.85, a=0.4)


def secs(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def xyz(p):
    return (p.x, p.y, p.z)


class Viz(Node):
    def __init__(self):
        super().__init__("viz")
        honest = self.declare_parameter("honest", [0]).value
        self.viewer = self.declare_parameter("viewer", honest[0]).value
        self.n_ids = self.declare_parameter("n_ids", 256).value
        self.honest = set(honest)
        self.t = 0.0
        self.bodies, self.claims, self.trails = {}, {}, {}
        self.trusted = np.ones(self.n_ids, bool)
        self.revoked = set()
        self.log = []
        for i in honest:
            self.create_subscription(Odometry, f"/drone_{i}/odom", lambda m, i=i: self.on_odom(i, m), 50)
            self.create_subscription(Detection3DArray, f"/drone_{i}/perception/drones", self.on_drones, 50)
            self.create_subscription(BeaconArray, f"/drone_{i}/radio/beacons", self.on_beacons, 50)
            self.create_subscription(Revocation, f"/drone_{i}/trust/revocations", self.on_rev, 50)
        self.create_subscription(TrustState, f"/drone_{self.viewer}/trust/state", self.on_trust, 50)
        self.create_subscription(PointCloud2, f"/eval/drone_{self.viewer}/false_cells", self.on_false, 10)
        self.false_cells = np.zeros((0, 3))
        self.pub = self.create_publisher(MarkerArray, "/viz/markers", 10)
        self.tf = TransformBroadcaster(self)
        self.create_timer(0.1, self.draw)

    def on_odom(self, i, m):
        self.t = max(self.t, secs(m.header.stamp))
        self.bodies[i] = xyz(m.pose.pose.position)
        self.trails.setdefault(i, []).append(self.bodies[i])

    def on_drones(self, m):
        for d in m.detections:
            if int(d.id) not in self.honest:
                self.bodies[int(d.id)] = xyz(d.bbox.center.position)

    def on_beacons(self, m):
        for b in m.beacons:
            self.claims[b.id] = (xyz(b.position), secs(b.received))

    def on_false(self, m):
        self.false_cells = point_cloud2.read_points_numpy(m, field_names=("x", "y", "z"))

    def on_trust(self, m):
        self.trusted = np.unpackbits(np.frombuffer(bytes(m.trusted_bits), np.uint8))[:self.n_ids].astype(bool)
        self.revoked = set(m.revoked)

    def on_rev(self, m):
        what = "honest drone" if m.origin in self.honest else "identity"
        self.log = (self.log + [f"t={secs(m.header.stamp):5.1f}s  drone {m.observer} revoked {what} {m.origin} ({m.reason})"])[-6:]

    def status(self, o):
        if o in self.revoked:
            return "revoked", RED
        return ("trusted", GREEN) if self.trusted[o] else ("quarantined", YELLOW)

    def marker(self, ns, mid, kind, pos, scale, color, text=""):
        k = Marker(ns=ns, id=mid, type=kind, action=Marker.ADD, text=text, color=color)
        k.header.frame_id = "map"
        k.header.stamp = self.get_clock().now().to_msg()
        k.pose.position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
        k.pose.orientation.w = 1.0
        k.scale.x = k.scale.y = k.scale.z = scale
        return k

    def frames(self):
        now = self.get_clock().now().to_msg()
        out = []
        root = TransformStamped(child_frame_id="map")
        root.header.frame_id, root.header.stamp = "world", now
        root.transform.rotation.w = 1.0
        out.append(root)
        for b, q in self.bodies.items():
            t = TransformStamped(child_frame_id=f"drone_{b}/base_link")
            t.header.frame_id, t.header.stamp = "map", now
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(float, q)
            t.transform.rotation.w = 1.0
            out.append(t)
        self.tf.sendTransform(out)

    def draw(self):
        self.frames()
        clear = Marker(action=Marker.DELETEALL)
        clear.header.frame_id = "map"
        out = MarkerArray(markers=[clear])
        for b, q in self.bodies.items():
            attacker = b not in self.honest
            out.markers.append(self.marker("bodies", b, Marker.SPHERE, q, 0.6, ORANGE if attacker else BLUE))
            label = f"{b} attacker" if attacker else (f"{b} viewer" if b == self.viewer else str(b))
            out.markers.append(self.marker("labels", b, Marker.TEXT_VIEW_FACING, (q[0], q[1], q[2] + 0.8), 0.6,
                                           WHITE, label))
        counts = dict(trusted=0, quarantined=0, revoked=0)
        for o, (q, t0) in self.claims.items():
            if o in self.bodies or self.t - t0 > 3.0:
                continue
            st, color = self.status(o)
            counts[st] += 1
            out.markers.append(self.marker("fakes", o, Marker.CUBE, q, 0.5, color))
            out.markers.append(self.marker("fake_labels", o, Marker.TEXT_VIEW_FACING, (q[0], q[1], q[2] + 0.7),
                                           0.45, color, f"fake {o}: {st}"))
        for i, tr in self.trails.items():
            k = self.marker("trails", i, Marker.LINE_STRIP, (0, 0, 0), 0.06, TRAIL)
            k.points = [Point(x=q[0], y=q[1], z=q[2]) for q in tr[::2]]
            out.markers.append(k)
        k = self.marker("false_walls", 0, Marker.CUBE_LIST, (0, 0, 0), 0.2, RED)
        k.points = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in self.false_cells]
        out.markers.append(k)
        hud = (f"t = {self.t:.0f} s   drone {self.viewer} sees fakes: {counts['trusted']} trusted, "
               f"{counts['quarantined']} quarantined, {counts['revoked']} revoked\n" + "\n".join(self.log))
        out.markers.append(self.marker("hud", 0, Marker.TEXT_VIEW_FACING, (24.0, 36.0, 2.0), 0.9, WHITE, hud))
        self.pub.publish(out)


def main():
    rclpy.init()
    try:
        rclpy.spin(Viz())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
