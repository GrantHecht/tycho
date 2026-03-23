cmake_minimum_required(VERSION 3.20)
project(tycho_fetch_test CXX)

# Simulate FetchContent by adding tycho as a subdirectory
# (In real usage, this would be FetchContent_Declare/MakeAvailable)
add_subdirectory(${TYCHO_SOURCE_DIR} ${CMAKE_BINARY_DIR}/tycho-build)

add_executable(fetch_test main.cpp)
target_link_libraries(fetch_test PRIVATE tycho::tycho)
