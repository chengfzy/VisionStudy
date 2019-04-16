project(Eg02_AprilTagDetection)

# project include
list(APPEND PRIVATE_DEPEND_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/apriltag/include)

# april tag library
set(PRJ_APRIL_TAG apriltag)
add_subdirectory(apriltag)

# build project
add_executable(${PROJECT_NAME} main.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE ${PRIVATE_DEPEND_INCLUDES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${PRIVATE_DEPEND_LIBS} ${PRJ_APRIL_TAG})