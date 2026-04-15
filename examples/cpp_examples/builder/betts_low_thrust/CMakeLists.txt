cmake_minimum_required(VERSION 3.20)
project(betts_low_thrust_builder CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(betts_low_thrust_builder main.cpp)
target_link_libraries(betts_low_thrust_builder PRIVATE tycho::tycho)
