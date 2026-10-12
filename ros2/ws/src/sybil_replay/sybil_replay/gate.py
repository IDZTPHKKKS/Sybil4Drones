import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rosbag2_interfaces.srv import Resume


class Gate(Node):
    def __init__(self):
        super().__init__("gate")
        self.topics = self.declare_parameter("topics", [""]).value
        self.need = self.declare_parameter("subscribers", 1).value
        self.client = self.create_client(Resume, "/rosbag2_player/resume")
        self.ready_since = None
        self.done = False
        self.timer = self.create_timer(0.5, self.poll)

    def poll(self):
        ready = self.client.service_is_ready() and all(
            self.count_publishers(t) >= 1 and self.count_subscribers(t) >= self.need for t in self.topics)
        if not ready:
            self.ready_since = None
            return
        now = self.get_clock().now()
        if self.ready_since is None:
            self.ready_since = now
            return
        if (now - self.ready_since).nanoseconds < 1_000_000_000:
            return
        self.timer.cancel()
        self.client.call_async(Resume.Request()).add_done_callback(self.resumed)

    def resumed(self, _):
        self.get_logger().info(f"all {len(self.topics)} topics connected, playback resumed")
        self.done = True


def main():
    rclpy.init()
    node = Gate()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.2)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
