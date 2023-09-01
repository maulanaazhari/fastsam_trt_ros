cmake_minimum_required(VERSION 3.0.2)
project(fastsam_trt_ros)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  rospy
  sensor_msgs
  vision_msgs
  fastsam_ros_msgs
)


catkin_package(
 INCLUDE_DIRS include
 LIBRARIES fastsam_trt_ros
 CATKIN_DEPENDS cv_bridge image_transport roscpp rospy sensor_msgs vision_msgs fastsam_ros_msgs
 DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)


catkin_python_setup()
catkin_package(CATKIN_DEPENDS rospy)

catkin_install_python(PROGRAMS src/fastsam_trt_ros/fastsam_ros.py
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})