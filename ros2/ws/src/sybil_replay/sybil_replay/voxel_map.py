import colorsys

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

from .recording import load


def height_colour(h):
    r, g, b = colorsys.hsv_to_rgb(0.67 * (1.0 - min(max(h, 0.0), 1.0)), 1.0, 1.0)
    return ColorRGBA(r=r, g=g, b=b, a=1.0)


class VoxelMap(Node):
    def __init__(self):
        super().__init__("voxel_map")
        head, frames = load(self.declare_parameter("recording", "").value)
        self.drone = self.declare_parameter("drone", head["honest"][0]).value
        self.block = self.declare_parameter("resolution", 0.4).value
        w = head["world"]
        self.shape, self.res = tuple(w["shape"]), w["res"]
        self.updates = [(f["t"], np.asarray(f["voxels"][0], int), np.asarray(f["voxels"][1], np.int8))
                        for f in frames if f["observer"] == self.drone and "voxels" in f]
        self.reset()
        self.occ_pub = self.create_publisher(PointCloud2, f"/drone_{self.drone}/octomap_points", 1)
        self.false_pub = self.create_publisher(PointCloud2, f"/eval/drone_{self.drone}/false_voxels", 1)
        self.cubes_pub = self.create_publisher(MarkerArray, f"/drone_{self.drone}/occupied_cells_vis_array", 1)

        self.create_timer(0.5, self.tick)
        self.get_logger().info(f"drone {self.drone}: {len(self.updates)} map updates, {self.block} m cubes")

    def reset(self):
        self.code = np.zeros(int(np.prod(self.shape)), np.int8)
        self.next, self.last_t = 0, -1.0

    def centres(self, mask):
        ijk = np.column_stack(np.unravel_index(np.flatnonzero(mask), self.shape))
        k = max(1, int(round(self.block / self.res)))
        blocks = np.unique(ijk // k, axis=0) if len(ijk) else np.zeros((0, 3), int)
        return (blocks + 0.5) * k * self.res

    def cloud(self, pts, stamp):
        return point_cloud2.create_cloud_xyz32(Header(stamp=stamp, frame_id="map"), pts.astype(np.float32).tolist())

    def cubes(self, occ, false, stamp):
        out = MarkerArray()
        top = self.shape[2] * self.res
        for mid, (pts, red) in enumerate(((occ, False), (false, True))):
            m = Marker(ns="occupied" if not red else "false", id=mid, type=Marker.CUBE_LIST, action=Marker.ADD)
            m.header = Header(stamp=stamp, frame_id="map")
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = self.block
            m.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            m.points = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in pts]
            if red:
                m.colors = [ColorRGBA(r=0.9, g=0.05, b=0.05, a=1.0)] * len(pts)
            else:
                m.colors = [height_colour(z / top) for z in pts[:, 2]]
            out.markers.append(m)
        return out

    def tick(self):
        now = self.get_clock().now()
        t = now.nanoseconds * 1e-9
        if t < self.last_t:
            self.reset()
        self.last_t = t
        changed = False
        while self.next < len(self.updates) and self.updates[self.next][0] <= t:
            _, idx, vals = self.updates[self.next]
            self.code[idx] = vals
            self.next += 1
            changed = True
        if changed:
            stamp = now.to_msg()
            occ, false = self.centres(self.code == 1), self.centres(self.code == 2)
            self.occ_pub.publish(self.cloud(occ, stamp))
            self.false_pub.publish(self.cloud(false, stamp))
            self.cubes_pub.publish(self.cubes(occ, false, stamp))


def main():
    rclpy.init()
    try:
        rclpy.spin(VoxelMap())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
