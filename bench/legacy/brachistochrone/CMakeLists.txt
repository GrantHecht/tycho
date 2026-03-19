add_executable(brachistochrone_bench bench.cpp)

target_compile_options(brachistochrone_bench PUBLIC ${COMPILE_FLAGS})
target_include_directories(brachistochrone_bench PRIVATE ${TYCHO_INCLUDES} ${INCLUDE_DIRS})
target_link_libraries(brachistochrone_bench PRIVATE tycho_core)
target_precompile_headers(brachistochrone_bench REUSE_FROM pch)

if(UNIX)
    target_link_options(brachistochrone_bench PRIVATE ${LINKER_FLAGS})
endif()
