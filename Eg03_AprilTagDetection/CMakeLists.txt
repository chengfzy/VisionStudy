# Application to Detect AprilTag
project(Eg03_AprilTagDetection)

# build the target
aux_source_directory(./ DIR_SRCS)
add_executable(${PROJECT_NAME} ${DIR_SRCS})
target_link_libraries(${PROJECT_NAME} ${EXTERNAL_LIBS} ${INTERNAL_LIBS})
add_dependencies(${PROJECT_NAME} ${INTERNAL_LIBS})