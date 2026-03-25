cmake_minimum_required(VERSION 3.20)
project(brachistochrone_builder CXX)

# If building standalone, find tycho. If in-tree, target already exists.
if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(brachistochrone_builder main.cpp)
target_link_libraries(brachistochrone_builder PRIVATE tycho::tycho)
