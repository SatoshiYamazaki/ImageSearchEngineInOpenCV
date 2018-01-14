cmake_minimum_required(VERSION 2.8)
project(CMakeSample)
find_package( OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})
add_executable(CMakeSample 
  cpp/kdtree_recognition.cpp
  cpp/index.cpp
)
set(CMAKE_BUILD_TYPE Release)
target_link_libraries(CMakeSample ${OpenCV_LIBS})

