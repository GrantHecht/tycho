cmake_minimum_required(VERSION 3.20)
project(hypersens CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(hypersens_cpp main.cpp hypersens_ode.cpp)
target_link_libraries(hypersens_cpp PRIVATE tycho::tycho)
