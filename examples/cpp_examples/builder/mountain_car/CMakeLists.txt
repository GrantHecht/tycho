cmake_minimum_required(VERSION 3.20)
project(mountain_car_builder CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(mountain_car_builder main.cpp)
target_link_libraries(mountain_car_builder PRIVATE tycho::tycho)
