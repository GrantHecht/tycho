cmake_minimum_required(VERSION 3.20)
project(zermelo_builder CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(zermelo_builder main.cpp)
target_link_libraries(zermelo_builder PRIVATE tycho::tycho)
