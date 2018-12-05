project(Eg04_VisualDebugging)

option(CVV_DEBUG_MODE "OpenCV Visual Debug Mode" ON)
if (CVV_DEBUG_MODE MATCHES ON)
    add_definitions(-DCVVISUAL_DEBUGMODE)
endif ()

# build the target
aux_source_directory(./ DIR_SRCS)
add_executable(${PROJECT_NAME} ${DIR_SRCS})
target_link_libraries(${PROJECT_NAME} ${EXTERNAL_LIBS})
