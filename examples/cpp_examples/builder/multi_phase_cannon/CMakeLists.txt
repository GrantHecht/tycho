cmake_minimum_required(VERSION 3.20)
project(multi_phase_cannon_builder CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(multi_phase_cannon_builder main.cpp)
target_link_libraries(multi_phase_cannon_builder PRIVATE tycho::tycho)
