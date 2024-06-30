cmake_minimum_required(VERSION 3.12)
project(auto_aim VERSION 1.0 LANGUAGES C CXX)

# Set compilation flags
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Set configure_file
set (CONFIGURE_DIR_PATH ${PROJECT_SOURCE_DIR}/src)
# configure_file (
#         "${PROJECT_SOURCE_DIR}/src/config.h.in"
#         "${CONFIGURE_DIR_PATH}/config.h")

# Initialize custom options
# option (ENABLE_DEBUG_CANVAS "Enable debug canvas to draw debug image" ON)
# option (ENABLE_RECORDING "Enable recording of raw camera image" OFF)
# option (ENABLE_OPENVINO "Enable openvino to identify buff" ON)
# option (ENABLE_ROS "Enable ROS to visualize positions" ON)

set(EXECUTABLE_NAME ${PROJECT_NAME})

# Find non-ros packages
find_package(OpenCV 4.5 REQUIRED)
# find_package(HikCameraSDK REQUIRED)

find_package(OpenVINO REQUIRED)
set(OpenVINO_LIB openvino::runtime)
find_package(Ceres REQUIRED)
set(Ceres_LIB Ceres::ceres)

# Include project source directory
include_directories(${PROJECT_SOURCE_DIR}/src ${CONFIGURE_DIR_PATH})

# Recursively search for all source files under the 'src' folder and store them into AUTO_AIM_SOURCE variable
# Flag 'CONFIGURE_DEPENDS' asks cmake to detect GLOB result changes so no need to rerun cmake when adding a new source file.
file(GLOB_RECURSE AUTO_AIM_SOURCE CONFIGURE_DEPENDS
        ${PROJECT_SOURCE_DIR}/src/*.cpp
        ${PROJECT_SOURCE_DIR}/src/*.cc)

find_package (ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies ()
ament_auto_add_library(${EXECUTABLE_NAME} SHARED ${AUTO_AIM_SOURCE})

# Link libraries
target_link_libraries(${EXECUTABLE_NAME} ${OpenCV_LIBS} ${HikCameraSDK_LIB} ${OpenVINO_LIB} ${Ceres_LIB} -lpthread)

install(
    DIRECTORY models/
    DESTINATION share/${PROJECT_NAME}/models
)

pluginlib_export_plugin_description_file(rmcs_executor plugins.xml)

ament_auto_package()
