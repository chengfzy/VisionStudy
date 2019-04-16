project(${PRJ_APRIL_TAG})

# build library
file(GLOB_RECURSE FILE_SRCS src/*.*)
add_library(${PROJECT_NAME} ${FILE_SRCS})
target_include_directories(${PROJECT_NAME} PRIVATE ${PRIVATE_DEPEND_INCLUDES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${PRIVATE_DEPEND_LIBS})