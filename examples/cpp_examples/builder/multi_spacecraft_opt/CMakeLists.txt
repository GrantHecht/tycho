cmake_minimum_required(VERSION 3.20)
project(multi_spacecraft_opt_builder CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(multi_spacecraft_opt_builder main.cpp)
target_link_libraries(multi_spacecraft_opt_builder PRIVATE tycho::tycho)
