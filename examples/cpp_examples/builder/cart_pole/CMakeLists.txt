cmake_minimum_required(VERSION 3.20)
project(cart_pole_builder CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(cart_pole_builder main.cpp)
target_link_libraries(cart_pole_builder PRIVATE tycho::tycho)
