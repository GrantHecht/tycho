cmake_minimum_required(VERSION 3.20)
project(delta3_launch_builder CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(delta3_launch_builder main.cpp)
target_link_libraries(delta3_launch_builder PRIVATE tycho::tycho)
