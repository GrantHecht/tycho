cmake_minimum_required(VERSION 3.20)
project(min_time_climb_tables_builder CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(min_time_climb_tables_builder main.cpp)
target_link_libraries(min_time_climb_tables_builder PRIVATE tycho::tycho)
