# sybil-ros2

The D1–D5 defences from Sybil4Drones as ROS 2 (Jazzy) nodes, running in Docker.

Missions from the simulator are converted to rosbag2 files and played through one trust node per honest
drone. The nodes have to make exactly the same decisions as the simulator: every revocation and the trust
state of every drone once per second.

## Layout

| path | contents |
|---|---|
| `ws/src/sybil_msgs` | radio and trust messages (`AuditCounts`, `BeaconArray`, `WitnessArray`, `Revocation`, `TrustState`) |
| `ws/src/sybil_trust` | `core.py` (D1–D5, no ROS), `view.py` (camera frustum and line of sight), `trust_node` |
| `ws/src/sybil_replay` | `sim_to_bag`, `world_map` (latched map), `gate` (resumes playback once every trust node is listening), `checker`, `viz`, launch file |
| `record/record.py` | runs a mission in the simulator and saves what each drone's trust logic receives |
| `record/video.py` | top-view MP4 of a recorded mission |
| `data/` | recorded missions, bags and videos (not in the repository, made by `record`, `bags` and `video`) |

## Topics per drone (`/drone_<i>/...`)

| topic | type | in / out |
|---|---|---|
| `odom` | `nav_msgs/Odometry` | in |
| `audit/counts` | `sybil_msgs/AuditCounts` | in, from the map audit |
| `radio/beacons` | `sybil_msgs/BeaconArray` | in |
| `radio/witness` | `sybil_msgs/WitnessArray` | in |
| `perception/drones` | `vision_msgs/Detection3DArray` | in |
| `trust/revocations` | `sybil_msgs/Revocation` | out |
| `trust/state` | `sybil_msgs/TrustState` | out |
| `radio/witness_out` | `sybil_msgs/WitnessReport` | out |

`/map_3d` (`sensor_msgs/PointCloud2`, latched) is used for line-of-sight checks. A trust node subscribes to its
drone's inputs only after it has a map.

## Results

| mission | level | frames | revocations | matches simulator |
|---|---|---|---|---|
| office 0, weak, 4 fakes | D1 | 2,772 | 28 | yes |
| warehouse 0, strong, 4 fakes | D2 | 4,193 | 45 | yes |
| forest 0, stealth, 4 fakes | D3 | 4,193 | 54 | yes |
| tunnels 0, strong, 4 fakes | D4 | 2,114 | 2 | yes |
| office 0, two colluding attackers | D4 | 3,246 | 68 | yes |
| office 0, shadow, 4 fakes | D5 | 3,248 | 15 | yes |
| tunnels 3, targeted, 4 fakes | D5 | 3,710 | 3 | yes |
| office 0, no attacker | D5 | 1,840 | 0 | yes |

Played at 20x through `ros2 bag play`; every trust state and every revocation is identical.

## Inputs that still come from the simulator

| input | in the simulator | needed on a real drone |
|---|---|---|
| drone detections | every body near a claimed position, with its true identity as the track id | detector and tracker on the depth images |
| `/map_3d` | the true world | the drone's own map (OctoMap) |
| audit counts | voxel map audit inside the simulator | audit node on the drone's OctoMap |
| pose | exact | state estimate with drift |

## Use

```bash
./ros.sh build                    # docker image and colcon build
./ros.sh test                     # unit tests and the recorded missions against the trust core
./ros.sh record --attack sybil_shadow --level D5 --out data/office0_shadow4_D5.jsonl.gz
./ros.sh bags                     # recordings to rosbag2 (mcap)
./ros.sh replay "" 20             # every mission through the ROS 2 nodes at 20x, compared with the simulator
./ros.sh show office0_shadow4_D5      # live view in Lichtblick: Open connection > Rosbridge > ws://localhost:9090
./ros.sh video office0_shadow4_D5     # MP4 in data/videos
```

Recordings use a detection probability of 1.0 so the comparison is exact; the nodes default to 0.9.

## Remove

```bash
./ros.sh uninstall                # removes the docker image and build files, then asks before deleting data/
```
