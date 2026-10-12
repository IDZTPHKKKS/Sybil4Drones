cmake_minimum_required(VERSION 3.8)
project(sybil_msgs)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/AuditCounts.msg"
  "msg/Beacon.msg"
  "msg/BeaconArray.msg"
  "msg/WitnessReport.msg"
  "msg/WitnessArray.msg"
  "msg/Revocation.msg"
  "msg/TrustState.msg"
  DEPENDENCIES builtin_interfaces geometry_msgs std_msgs)
ament_package()
