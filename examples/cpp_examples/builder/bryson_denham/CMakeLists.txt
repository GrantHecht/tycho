cmake_minimum_required(VERSION 3.20)
project(bryson_denham_builder CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(bryson_denham_builder main.cpp)
target_link_libraries(bryson_denham_builder PRIVATE tycho::tycho)
