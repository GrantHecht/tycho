cmake_minimum_required(VERSION 3.20)
project(goddard_rocket_builder CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(goddard_rocket_builder main.cpp)
target_link_libraries(goddard_rocket_builder PRIVATE tycho::tycho)
