cmake_minimum_required(VERSION 3.20)
project(n2_repro CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(n2_repro main.cpp)
target_link_libraries(n2_repro PRIVATE tycho::tycho)
