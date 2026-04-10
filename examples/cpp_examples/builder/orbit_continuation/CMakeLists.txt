cmake_minimum_required(VERSION 3.20)
project(orbit_continuation_builder CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(orbit_continuation_builder main.cpp)
target_link_libraries(orbit_continuation_builder PRIVATE tycho::tycho)
