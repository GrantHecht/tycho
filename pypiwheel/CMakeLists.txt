
set(CMAKE_VERBOSE_MAKEFILE TRUE)

add_custom_target(pypiwheel  ALL DEPENDS ${CMAKE_SOURCE_DIR}/tycho)
add_dependencies(pypiwheel _tycho)

set(WHEEL_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/tycho)

set(TYCHO_BINARY_NAME "_tycho${PYTHON_MODULE_EXTENSION}")


configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in"
    "${CMAKE_CURRENT_BINARY_DIR}/setup.py"
    @ONLY)


add_custom_command(TARGET pypiwheel
                    COMMAND ${CMAKE_COMMAND} -E rm -rf ${WHEEL_BUILD_DIR}/
                    COMMENT "Cleaning tycho install directory")


add_custom_command(TARGET pypiwheel
                    COMMAND ${CMAKE_COMMAND} -E copy_directory  ${CMAKE_SOURCE_DIR}/tycho
                                                ${WHEEL_BUILD_DIR}/tycho
                    DEPENDS ${CMAKE_SOURCE_DIR}/tycho
                    COMMENT "Copied tycho to ${WHEEL_BUILD_DIR}")

add_custom_command(TARGET pypiwheel
                    COMMAND ${CMAKE_COMMAND} -E copy  ${CMAKE_CURRENT_BINARY_DIR}/setup.py ${WHEEL_BUILD_DIR}

                    COMMENT "Copied setup.py to ${WHEEL_BUILD_DIR}")


add_custom_command(TARGET pypiwheel
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_tycho> ${WHEEL_BUILD_DIR}/
    COMMENT "Copied _tycho extension to ${WHEEL_BUILD_DIR}"
    )




add_custom_command(TARGET pypiwheel 
                    COMMAND
                   ${PYTHON_EXECUTABLE} ${WHEEL_BUILD_DIR}/setup.py bdist_wheel 
                   WORKING_DIRECTORY ${WHEEL_BUILD_DIR}
                   COMMENT "Generating Binary Wheel")


message(STATUS ${TYCHO_BINARY_NAME})


