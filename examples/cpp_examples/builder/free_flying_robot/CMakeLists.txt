cmake_minimum_required(VERSION 3.20)
project(free_flying_robot_builder CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(free_flying_robot_builder main.cpp)
target_link_libraries(free_flying_robot_builder PRIVATE tycho::tycho)
