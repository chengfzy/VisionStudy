cmake_minimum_required(VERSION 3.2)
set(CMAKE_CXX_STANDARD 14)
project(VisionStudy)

# cmake modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)


###############################################################################
# Libraries
###############################################################################
# find packages for basic libraries
find_package(Gflags REQUIRED)                                   # gflags
find_package(Glog REQUIRED)                                     # glog
find_package(Boost COMPONENTS filesystem date_time REQUIRED)    # boost
find_package(Eigen3 REQUIRED)                                   # Eigen
find_package(OpenCV REQUIRED)                                   # OpenCV

# prive dependency include directories and libraries
list(APPEND PRIVATE_DEPEND_INCLUDES
        ${GFLAGS_INCLUDE_DIRS}
        ${GLOG_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        )
list(APPEND PRIVATE_DEPEND_LIBS
        ${GFLAGS_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${OpenCV_LIBRARIES}
        )

find_package(Gflags REQUIRED)                                   # gflags
find_package(Glog REQUIRED)                                     # glog
find_package(Boost COMPONENTS filesystem date_time REQUIRED)    # boost
find_package(Eigen3 REQUIRED)                                   # Eigen
find_package(OpenCV REQUIRED)                                   # OpenCV
find_package(ZBar REQUIRED)                                     # ZBar
include_directories(
        ${GFLAGS_INCLUDE_DIRS}
        ${GLOG_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${ZBAR_INCLUDE_DIRS})
set(EXTERNAL_LIBS ${EXTERNAL_LIBS}
        ${GFLAGS_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${Boost_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${ZBAR_LIBRARIES})

###############################################################################
# Project Setting
###############################################################################
# default build type
IF (NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE RELEASE)
ENDIF ()
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

# projects build setting
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


# project name
set(PRJ_LIB_COMMON common)   # common library

# project include
list(APPEND PRIVATE_DEPEND_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/common/include)

# projects
add_subdirectory(common)        # common library

add_subdirectory(Eg01_QrCodeDecode)
add_subdirectory(Eg02_AprilTagDetection)
add_subdirectory(Eg03_VisualDebugging)

# include folder
#include_directories(.)
#include_directories(./3rdparty/apriltag/include)
#
## internal 3rd party libraries
#set(INTERNAL_LIB_APRILTAG apriltag)
#set(INTERNAL_LIBS ${INTERAL_LIBS}
#        ${INTERNAL_LIB_APRILTAG})
#add_subdirectory(./3rdparty)
#
## add Folder
#add_subdirectory(Eg01_ChessboardDetection)
#add_subdirectory(Eg02_QrCodeDecode)
#add_subdirectory(Eg03_AprilTagDetection)
#add_subdirectory(Eg04_VisualDebugging)