cmake_minimum_required(VERSION 3.20)
project(van_der_pol_builder CXX)

if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(van_der_pol_builder main.cpp)
target_link_libraries(van_der_pol_builder PRIVATE tycho::tycho)
