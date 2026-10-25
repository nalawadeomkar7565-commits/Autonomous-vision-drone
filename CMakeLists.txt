cmake_minimum_required(VERSION 3.16)
project(drone_hud_vision CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Find Required Packages
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0 gstreamer-app-1.0)

# HailoRT C++ Header and Library Paths
find_path(HAILORT_INCLUDE_DIR hailort/hailort.hpp PATHS /usr/include /usr/local/include)
find_library(HAILORT_LIB hailort PATHS /usr/lib /usr/local/lib)

if(NOT HAILORT_INCLUDE_DIR OR NOT HAILORT_LIB)
    message(FATAL_ERROR "HailoRT C++ SDK not found! Check hailo-all installation.")
endif()

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${GSTREAMER_INCLUDE_DIRS}
    ${HAILORT_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# 2. Add Executable
add_executable(drone_hud_node 
    src/main.cpp 
    src/HailoYOLOv8.cpp
)

# 3. Link Shared Libraries
target_link_libraries(drone_hud_node 
    PRIVATE 
    ${OpenCV_LIBS} 
    ${GSTREAMER_LIBRARIES} 
    ${HAILORT_LIB}
)
