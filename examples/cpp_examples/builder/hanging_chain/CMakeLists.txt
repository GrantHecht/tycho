cmake_minimum_required(VERSION 3.20)
project(hanging_chain_builder CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(hanging_chain_builder main.cpp)
target_link_libraries(hanging_chain_builder PRIVATE tycho::tycho)
