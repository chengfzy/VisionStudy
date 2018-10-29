cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)
project(VisionStudy)

# cmake modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# library: boost
option(BUILD_USE_SYSTEM_BOOST "Use system boost in /usr/lib" ON)
if (BUILD_USE_SYSTEM_BOOST)
    message(STATUS "USE system boost in /usr/lib/")
    # boost: bypass default path for search another boost in /usr/lib
    # set(BOOST_NO_SYSTEM_PATHS FALSE)
    set(BOOST_ROOT "/usr")
    set(BOOST_INCLUDEDIR "${BOOST_ROOT}/include")
    set(BOOST_LIBRARYDIR "${BOOST_ROOT}/lib/x86_64-linux-gnu")
    find_package(Boost REQUIRED COMPONENTS filesystem date_time)
else ()
    message(STATUS "Use Boost in /usr/local")
    find_package(Boost REQUIRED COMPONENTS filesystem date_time REQUIRED)
endif ()
if (NOT Boost_FOUND)
    message(FATAL_ERROR "Cannot Found Boost")
endif ()
message(STATUS "Boost headers: ${Boost_INCLUDE_DIR}")
message(STATUS "Boost libs: ${Boost_LIBRARIES}")

# libraries
find_package(Eigen3 REQUIRED)                           # Eigen
find_package(OpenCV REQUIRED)                           # OpenCV
include_directories(${EIGEN3_INCLUDE_DIR}
        ${Boost_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS})
set(THIRD_LIBS ${THIRD_LIBS}
        gflags glog
        ${Boost_LIBRARIES}
        ${OpenCV_LIBS})


# projects build setting
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


# include folder
include_directories(.)

# Add Folder
add_subdirectory(Eg01_ChessboardDetection)
