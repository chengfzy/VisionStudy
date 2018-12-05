project(${INTERNAL_LIB_APRILTAG})

# build library
file(GLOB_RECURSE FILE_SRCS "src/*.*")
add_library(${PROJECT_NAME} SHARED ${FILE_SRCS})
target_link_libraries(${PROJECT_NAME} ${EXTERNAL_LIBS})