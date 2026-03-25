cmake_minimum_required(VERSION 3.20)
project(zermelo CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(zermelo_cpp main.cpp)
target_link_libraries(zermelo_cpp PRIVATE tycho::tycho)
