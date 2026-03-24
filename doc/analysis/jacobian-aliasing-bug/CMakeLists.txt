cmake_minimum_required(VERSION 3.20)
project(bug_fix_diagnostics CXX)

# If building standalone, find tycho. If in-tree, target already exists.
if(NOT TARGET tycho::tycho)
    find_package(tycho REQUIRED)
endif()

add_executable(jac_alias_diag jac_alias_diag.cpp)
target_link_libraries(jac_alias_diag PRIVATE tycho::tycho)

