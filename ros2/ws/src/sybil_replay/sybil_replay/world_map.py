import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from .recording import load, world


class WorldMap(Node):
    def __init__(self):
        super().__init__("world_map")
        head, _ = load(self.declare_parameter("recording", "").value)
        pub = self.create_publisher(PointCloud2, "/map_3d",
                                    QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        pts = world(head).centres()
        pub.publish(point_cloud2.create_cloud_xyz32(Header(frame_id="map"), pts.tolist()))
        self.get_logger().info(f"published {head['family']} {head['seed']}: {len(pts)} voxels")


def main():
    rclpy.init()
    try:
        rclpy.spin(WorldMap())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
