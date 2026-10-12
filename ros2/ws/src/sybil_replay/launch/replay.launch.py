import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from sybil_replay.recording import load
from sybil_replay.sim_to_bag import DRONE_TOPICS


def nodes(context):
    arg = lambda name: LaunchConfiguration(name).perform(context)
    recording, bag, rate, show = arg("recording"), arg("bag"), arg("rate"), arg("show") == "true"
    head, frames = load(recording)
    sim_time = dict(use_sim_time=True)
    out = [Node(package="sybil_trust", executable="trust_node", name="trust", namespace=f"drone_{i}", output="log",
                parameters=[sim_time, dict(drone_id=i, level=head["level"], n_ids=head["n_ids"], detect=head["detect"],
                                           v_max=head["v_max"], max_range=head["max_range"], hfov=head["hfov"],
                                           vfov=head["vfov"])])
           for i in head["honest"]]
    out.append(Node(package="sybil_replay", executable="world_map", output="log",
                    parameters=[dict(recording=recording)]))
    loop = ["--loop"] if show else []
    play = ExecuteProcess(cmd=["ros2", "bag", "play", bag, "--clock", "--rate", rate, "--start-paused", *loop,
                               "--qos-profile-overrides-path", os.path.join(bag, "qos.yaml")], output="log")
    topics = [f"/drone_{i}/{t}" for i in head["honest"] for t in [t for t, _ in DRONE_TOPICS]]
    gate = Node(package="sybil_replay", executable="gate", output="screen", parameters=[dict(topics=topics)])
    out += [play, gate]
    if show:
        viewer = arg("viewer")
        viewer = int(viewer) if viewer else head["honest"][0]
        if any("voxels" in f for f in frames[:200]):
            out.append(Node(package="sybil_replay", executable="voxel_map", output="log",
                            parameters=[sim_time, dict(recording=recording, drone=viewer)]))
        out += [Node(package="sybil_replay", executable="viz", output="log",
                     parameters=[sim_time, dict(honest=head["honest"], n_ids=head["n_ids"],
                                                viewer=viewer)]),
                Node(package="rosbridge_server", executable="rosbridge_websocket", output="log",
                     parameters=[dict(port=9090, address="0.0.0.0")]),
                Node(package="rosapi", executable="rosapi_node", output="log")]
    else:
        checker = Node(package="sybil_replay", executable="checker", output="screen",
                       parameters=[dict(recording=recording, out=arg("out"))])
        out += [checker, RegisterEventHandler(OnProcessExit(target_action=checker,
                                                            on_exit=[EmitEvent(event=Shutdown())]))]
    return out


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("recording"),
        DeclareLaunchArgument("bag"),
        DeclareLaunchArgument("rate", default_value="10.0"),
        DeclareLaunchArgument("show", default_value="false"),
        DeclareLaunchArgument("viewer", default_value=""),
        DeclareLaunchArgument("out", default_value=""),
        OpaqueFunction(function=nodes),
    ])
