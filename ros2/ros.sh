#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
IMAGE=sybil-ros2
DIR="$PWD"
ARGS=(--rm --label sybil-ros2 -v "$DIR/ws:/ws" -v "$DIR/data:/data")
in_ros() { docker run "${ARGS[@]}" "$IMAGE" bash -c "source /opt/ros/jazzy/setup.bash && { [ ! -f install/setup.bash ] || source install/setup.bash; } && $1"; }
missions() { for f in data/${1:-*}.jsonl.gz; do basename "$f" .jsonl.gz; done; }

case "$1" in
  build)
    docker build -t "$IMAGE" .
    in_ros "colcon build --symlink-install"
    ;;
  test)
    in_ros "python3 -m pytest -q src/sybil_trust/test src/sybil_replay/test"
    ;;
  record)
    shift
    python3 record/record.py "$@"
    ;;
  bags)
    for m in $(missions "$2"); do in_ros "ros2 run sybil_replay sim_to_bag /data/$m.jsonl.gz /data/bags/$m"; done
    ;;
  replay)
    rm -f data/replay_results.jsonl
    for m in $(missions "$2"); do
      in_ros "ros2 launch sybil_replay replay.launch.py recording:=/data/$m.jsonl.gz bag:=/data/bags/$m rate:=${3:-10} out:=/data/replay_results.jsonl" \
        2>&1 | grep -E '"frames"|Traceback|Error' | sed 's/.*\]: //'
    done
    ;;
  show)
    m=${2:-office0_shadow4_D5}
    v=""; [ -n "$4" ] && v="viewer:=$4"
    echo "in Lichtblick: Open connection > Rosbridge > ws://localhost:9090 (ctrl-c to stop)"
    docker run "${ARGS[@]}" -it -p 9090:9090 "$IMAGE" bash -c "source /opt/ros/jazzy/setup.bash && source install/setup.bash && \
      ros2 launch sybil_replay replay.launch.py recording:=/data/$m.jsonl.gz bag:=/data/bags/$m rate:=${3:-2} show:=true $v"
    ;;
  video)
    shift
    for m in $(missions "$1"); do python3 record/video.py "data/$m.jsonl.gz" "${@:2}"; done
    ;;
  shell)
    docker run "${ARGS[@]}" -it "$IMAGE" bash
    ;;
  uninstall)
    docker ps -aq --filter label=sybil-ros2 | xargs docker rm -f 2>/dev/null || true
    docker image rm -f "$IMAGE" ros:jazzy-ros-base 2>/dev/null || true
    docker builder prune -f >/dev/null
    rm -rf ws/build ws/install ws/log
    echo "docker image and build files removed"
    read -r -p "Also delete recordings, bags and videos in $DIR/data? [y/N] " a
    if [ "$a" = "y" ]; then rm -rf data; echo "removed $DIR/data"; fi
    ;;
  *)
    echo "usage: ./ros.sh build | test | record ... | bags [mission] | replay [mission] [rate] | show [mission] [rate] [viewer] | video [mission] | shell | uninstall"
    ;;
esac
