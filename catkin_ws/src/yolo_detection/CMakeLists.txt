cmake_minimum_required(VERSION 3.0.2)
project(yolo_detection)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
  roscpp
  sensor_msgs
  image_transport
  cv_bridge
)

find_package(OpenCV REQUIRED)

## TensorRT + CUDA
find_library(TENSORRT_LIB nvinfer)
find_library(TENSORRT_PARSERS nvonnxparser)
find_library(TENSORRT_PLUGIN nvinfer_plugin)
find_package(CUDA REQUIRED)


add_message_files(
  FILES
  detection.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
   DEPENDENCIES
   std_msgs  # Or other packages containing msgs
)

catkin_package(
  CATKIN_DEPENDS roscpp std_msgs message_runtime sensor_msgs image_transport cv_bridge
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)

add_executable(yolo_detection
  src/yolo_detection.cpp
)

add_dependencies(yolo_detection ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(yolo_detection
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${TENSORRT_LIB}
  ${TENSORRT_PARSERS}
  ${TENSORRT_PLUGIN}
)


