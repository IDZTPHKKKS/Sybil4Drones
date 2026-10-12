import argparse
import math
import os
import shutil

import rosbag2_py
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point
import numpy as np
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.serialization import serialize_message
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from vision_msgs.msg import Detection3D, Detection3DArray

from sybil_msgs.msg import AuditCounts, Beacon, BeaconArray, WitnessArray, WitnessReport

from .recording import load, stamp_ns

DRONE_TOPICS = [("odom", "nav_msgs/msg/Odometry"),
                ("audit/counts", "sybil_msgs/msg/AuditCounts"),
                ("radio/beacons", "sybil_msgs/msg/BeaconArray"),
                ("radio/witness", "sybil_msgs/msg/WitnessArray"),
                ("perception/drones", "vision_msgs/msg/Detection3DArray")]


def time_of(t):
    n = stamp_ns(t)
    return Time(sec=n // 1_000_000_000, nanosec=n % 1_000_000_000)


def point(q):
    return Point(x=float(q[0]), y=float(q[1]), z=float(q[2]))


def messages(f, head):
    i = f["observer"]
    h = Header(stamp=time_of(f["t"]), frame_id="map")
    odom = Odometry(header=h, child_frame_id=f"drone_{i}/base_link")
    odom.pose.pose.position = point(f["position"])
    odom.pose.pose.orientation.z = math.sin(f["yaw"] / 2)
    odom.pose.pose.orientation.w = math.cos(f["yaw"] / 2)
    counts = AuditCounts(header=h)
    for o, c, w, cw, ww in f["counts"]:
        counts.origin.append(o)
        counts.checked.append(c)
        counts.wrong.append(w)
        counts.checked_walls.append(cw)
        counts.wrong_walls.append(ww)
    beacons = BeaconArray(header=h, beacons=[Beacon(id=o, position=point(q), received=time_of(t0))
                                             for o, q, t0 in f["beacons"]])
    witness = WitnessArray(header=h, reports=[WitnessReport(sender=a, seen=ids) for a, ids in f["witness"]])
    drones = Detection3DArray(header=h)
    for b, q in f["bodies"]:
        d = Detection3D(header=h, id=str(b))
        d.bbox.center.position = point(q)
        d.bbox.size.x = d.bbox.size.y = d.bbox.size.z = 0.24
        drones.detections.append(d)
    return dict(zip([t for t, _ in DRONE_TOPICS], [odom, counts, beacons, witness, drones]))


OCC_VALUE = np.array([-1, 0, 100, 100], np.int8)


def map_messages(f, grid, head):
    i = f["observer"]
    idx, vals = f["map"]
    grid[np.asarray(idx, int)] = np.asarray(vals, np.int8)
    w = head["world"]
    nx, ny = w["shape"][0], w["shape"][1]
    h = Header(stamp=time_of(f["t"]), frame_id="map")
    m = OccupancyGrid(header=h)
    m.info.resolution = float(w["res"])
    m.info.width, m.info.height = nx, ny
    m.info.origin.orientation.w = 1.0
    m.data = OCC_VALUE[grid.reshape(nx, ny).T.ravel()].tolist()
    cells = np.argwhere(grid.reshape(nx, ny) == 3)
    pts = np.column_stack([(cells + 0.5) * w["res"], np.full(len(cells), 1.4)]) if len(cells) else np.zeros((0, 3))
    return {f"/drone_{i}/map": m, f"/eval/drone_{i}/false_cells": point_cloud2.create_cloud_xyz32(h, pts.tolist())}


def write_qos(path, topics):
    with open(path, "w") as fh:
        for name in topics:
            fh.write(f"{name}:\n  history: keep_last\n  depth: 200\n  reliability: reliable\n  durability: volatile\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("out")
    a = ap.parse_args()
    head, frames = load(a.recording)
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=a.out, storage_id="mcap"),
                rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"))
    topics = [(f"/drone_{i}/{t}", typ) for i in head["honest"] for t, typ in DRONE_TOPICS]
    for k, (name, typ) in enumerate(topics):
        writer.create_topic(rosbag2_py.TopicMetadata(id=k, name=name, type=typ, serialization_format="cdr"))
    maps = [(f"/drone_{i}/map", "nav_msgs/msg/OccupancyGrid") for i in head["honest"]]
    maps += [(f"/eval/drone_{i}/false_cells", "sensor_msgs/msg/PointCloud2") for i in head["honest"]]
    for k, (name, typ) in enumerate(maps, start=len(topics)):
        writer.create_topic(rosbag2_py.TopicMetadata(id=k, name=name, type=typ, serialization_format="cdr"))
    names = [name for name, _ in topics + maps]
    w = head["world"]
    grids = {i: np.zeros(w["shape"][0] * w["shape"][1], np.int8) for i in head["honest"]}
    for f in frames:
        when = stamp_ns(f["t"])
        for t, msg in messages(f, head).items():
            writer.write(f"/drone_{f['observer']}/{t}", serialize_message(msg), when)
        if "map" in f:
            for name, msg in map_messages(f, grids[f["observer"]], head).items():
                writer.write(name, serialize_message(msg), when)
    del writer
    write_qos(os.path.join(a.out, "qos.yaml"), names)
    print(f"{a.out}: {len(frames)} frames, {len(names)} topics")


if __name__ == "__main__":
    main()
