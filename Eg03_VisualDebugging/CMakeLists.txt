project(Eg03_VisualDebugging)

option(CVV_DEBUG_MODE "OpenCV Visual Debug Mode" ON)
if (CVV_DEBUG_MODE MATCHES ON)
    add_definitions(-DCVVISUAL_DEBUGMODE)
endif ()

# build project
add_executable(${PROJECT_NAME} main.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE ${PRIVATE_DEPEND_INCLUDES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${PRIVATE_DEPEND_LIBS} ${PRJ_APRIL_TAG})
