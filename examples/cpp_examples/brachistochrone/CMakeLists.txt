add_executable(brachistochrone_cpp main.cpp)

target_compile_options(brachistochrone_cpp PRIVATE ${COMPILE_FLAGS})
target_include_directories(brachistochrone_cpp PRIVATE ${TYCHO_INCLUDES} ${INCLUDE_DIRS})
target_link_libraries(brachistochrone_cpp PRIVATE tycho_core ${LINK_LIBS})
target_precompile_headers(brachistochrone_cpp REUSE_FROM pch)

if(UNIX)
    target_link_options(brachistochrone_cpp PRIVATE ${LINKER_FLAGS})
endif()

add_executable(brachistochrone_bench bench.cpp)

target_compile_options(brachistochrone_bench PRIVATE ${COMPILE_FLAGS})
target_include_directories(brachistochrone_bench PRIVATE ${TYCHO_INCLUDES} ${INCLUDE_DIRS})
target_link_libraries(brachistochrone_bench PRIVATE tycho_core ${LINK_LIBS})
target_precompile_headers(brachistochrone_bench REUSE_FROM pch)

if(UNIX)
    target_link_options(brachistochrone_bench PRIVATE ${LINKER_FLAGS})
endif()
