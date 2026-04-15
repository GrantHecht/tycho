cmake_minimum_required(VERSION 3.20)
project(analytic_example_builder CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(analytic_example_builder main.cpp)
target_link_libraries(analytic_example_builder PRIVATE tycho::tycho)
