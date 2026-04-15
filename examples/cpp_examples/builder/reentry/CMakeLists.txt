cmake_minimum_required(VERSION 3.20)
project(reentry_builder CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(reentry_builder main.cpp)
target_link_libraries(reentry_builder PRIVATE tycho::tycho)
