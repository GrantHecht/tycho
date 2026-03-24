cmake_minimum_required(VERSION 3.20)
project(brachistochrone CXX)

# If building standalone, find tycho. If in-tree, target already exists.
if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(brachistochrone_cpp main.cpp)
target_link_libraries(brachistochrone_cpp PRIVATE tycho::tycho)
