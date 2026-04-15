cmake_minimum_required(VERSION 3.20)
project(parallel_parking_builder CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(parallel_parking_builder main.cpp)
target_link_libraries(parallel_parking_builder PRIVATE tycho::tycho)
