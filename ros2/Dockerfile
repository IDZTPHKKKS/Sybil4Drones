FROM ros:jazzy-ros-base
RUN apt-get update && apt-get install -y --no-install-recommends python3-numpy python3-scipy python3-pytest ros-jazzy-rosbridge-server ros-jazzy-vision-msgs \
    && rm -rf /var/lib/apt/lists/*
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc \
    && echo "[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash" >> /root/.bashrc
WORKDIR /ws
