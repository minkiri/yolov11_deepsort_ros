cmake_minimum_required(VERSION 3.0.2)
project(yolov11_deepsort)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  src/yolov11_deepsort.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

