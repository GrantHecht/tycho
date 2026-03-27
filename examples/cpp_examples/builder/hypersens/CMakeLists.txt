cmake_minimum_required(VERSION 3.20)
project(hypersens_builder CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(hypersens_builder main.cpp)
target_link_libraries(hypersens_builder PRIVATE tycho::tycho)
