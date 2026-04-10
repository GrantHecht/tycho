cmake_minimum_required(VERSION 3.20)
project(minimum_time_to_climb_builder CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(minimum_time_to_climb_builder main.cpp)
target_link_libraries(minimum_time_to_climb_builder PRIVATE tycho::tycho)
