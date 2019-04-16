# find library: ZBar
find_package(ZBar)
if (${ZBAR_FOUND})
    project(Eg01_QrCodeDecode)

    # dependency
    list(APPEND PRIVATE_DEPEND_INCLUDES ${ZBAR_INCLUDE_DIRS})
    list(APPEND PRIVATE_DEPEND_LIBS ${ZBAR_LIBRARIES})

    # build project
    add_executable(${PROJECT_NAME} main.cpp)
    target_include_directories(${PROJECT_NAME} PRIVATE ${PRIVATE_DEPEND_INCLUDES})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${PRIVATE_DEPEND_LIBS})
endif ()
