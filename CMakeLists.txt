cmake_minimum_required(VERSION 2.8.3)
set(CMAKE_CXX_STANDARD 17)
project(cam_lidar_calib)

find_package(catkin REQUIRED COMPONENTS
  pcl_conversions
  pcl_ros
  roscpp
  sensor_msgs
  message_filters
  cv_bridge
  tf
  tf_conversions
)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Open3D REQUIRED)

find_package(Ceres REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(FMT REQUIRED)
find_package(open3d_ros)

pkg_check_modules(JSONCPP jsoncpp)
add_definitions(${Open3D_DEFINITIONS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Open3D_CXX_FLAGS} -Wall -Wextra")

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS
  DEPENDS PCL
  DEPENDS Open3D

)
link_directories(${Open3D_LIBRARY_DIRS})

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${Open3D_INCLUDE_DIRS}
  ${open3d_ros_INCLUDE_DIRS}
)

add_executable(cam_lidar_calib src/cam_lidar_calib_node.cpp)
add_executable(custom_calib src/custom_calib.cpp)

add_executable(cam_lidar_proj_node src/cam_lidar_proj_node.cpp)

target_link_libraries(cam_lidar_calib
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
  ${JSONCPP_LIBRARIES}
  fmt::fmt
  ${Open3D_LIBRARIES}
  ${open3d_ros_LIBRARIES}
)

target_link_libraries(cam_lidar_proj_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${JSONCPP_LIBRARIES}
  fmt::fmt
  ${Open3D_LIBRARIES}
  ${open3d_ros_LIBRARIES}
)

target_link_libraries(custom_calib
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
  ${JSONCPP_LIBRARIES}
  fmt::fmt
  ${Open3D_LIBRARIES}
  ${open3d_ros_LIBRARIES}
)
