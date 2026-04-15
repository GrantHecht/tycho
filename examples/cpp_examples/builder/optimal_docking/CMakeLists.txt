cmake_minimum_required(VERSION 3.20)
project(optimal_docking_builder CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(optimal_docking_builder main.cpp)
target_link_libraries(optimal_docking_builder PRIVATE tycho::tycho)
