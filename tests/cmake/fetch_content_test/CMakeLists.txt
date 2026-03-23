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

add_executable(fetch_test main.cpp)
target_link_libraries(fetch_test PRIVATE tycho::tycho)
