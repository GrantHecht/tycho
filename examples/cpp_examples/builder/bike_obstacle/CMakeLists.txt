cmake_minimum_required(VERSION 3.20)
project(bike_obstacle_builder CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(bike_obstacle_builder main.cpp)
target_link_libraries(bike_obstacle_builder PRIVATE tycho::tycho)
