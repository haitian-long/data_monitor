cmake_minimum_required(VERSION 3.0.2)
project(data_monitor)

find_package(catkin REQUIRED COMPONENTS
  rospy
  nav_msgs
)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
  src/odom_monitor/odom_monitor_node.py
  src/pcd_monitor/pcd_monitor_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
