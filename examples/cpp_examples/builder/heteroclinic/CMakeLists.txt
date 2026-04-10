cmake_minimum_required(VERSION 3.20)
project(heteroclinic_builder CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(heteroclinic_builder main.cpp)
target_link_libraries(heteroclinic_builder PRIVATE tycho::tycho)
