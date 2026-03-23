cmake_minimum_required(VERSION 3.20)
project(tycho_fetch_test CXX)

if(NOT DEFINED TYCHO_SOURCE_DIR)
    message(FATAL_ERROR
        "TYCHO_SOURCE_DIR must be set. "
        "Example: cmake -DTYCHO_SOURCE_DIR=/path/to/tycho ..")
endif()

# Simulate FetchContent by adding tycho as a subdirectory
# (In real usage, this would be FetchContent_Declare/MakeAvailable)
add_subdirectory(${TYCHO_SOURCE_DIR} ${CMAKE_BINARY_DIR}/tycho-build)

# Verify subdirectory guards suppress bindings and examples
if(ENABLE_PYTHON_BINDINGS)
    message(FATAL_ERROR
        "ENABLE_PYTHON_BINDINGS should default OFF in subdirectory mode")
endif()
if(BUILD_CPP_EXAMPLES)
    message(FATAL_ERROR
        "BUILD_CPP_EXAMPLES should default OFF in subdirectory mode")
endif()

add_executable(fetch_test ../common/tycho_cmake_test.cpp)
target_link_libraries(fetch_test PRIVATE tycho::tycho)
target_compile_definitions(fetch_test PRIVATE TYCHO_TEST_NAME="FetchContent")
