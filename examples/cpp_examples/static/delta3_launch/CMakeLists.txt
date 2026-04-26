cmake_minimum_required(VERSION 3.20)
project(delta3_launch CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(delta3_launch_cpp main.cpp delta3_launch_ode.cpp)
target_link_libraries(delta3_launch_cpp PRIVATE tycho::tycho)
