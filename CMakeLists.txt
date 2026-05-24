cmake_minimum_required(VERSION 3.0.2)
project(odometry_monitor)

find_package(catkin REQUIRED COMPONENTS
  rospy
  nav_msgs
)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
  src/odometry_monitor/odometry_monitor_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
