cmake_minimum_required(VERSION 3.20)
project(tycho_find_test CXX)

find_package(tycho REQUIRED)

add_executable(find_test main.cpp)
target_link_libraries(find_test PRIVATE tycho::tycho)
