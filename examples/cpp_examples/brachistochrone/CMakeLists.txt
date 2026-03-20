add_executable(brachistochrone_cpp main.cpp)

target_compile_options(brachistochrone_cpp PRIVATE ${COMPILE_FLAGS}
    # Force -fPIC to match the PCH (built as a static lib with -fPIC);
    # without this, Clang 22+ rejects the reused PCH due to PIC/PIE mismatch.
    $<$<CXX_COMPILER_ID:Clang>:-fPIC>)
target_include_directories(brachistochrone_cpp PRIVATE ${INCLUDE_DIRS})
target_link_libraries(brachistochrone_cpp PRIVATE tycho_core)
target_precompile_headers(brachistochrone_cpp REUSE_FROM pch)

if(UNIX)
    target_link_options(brachistochrone_cpp PRIVATE ${LINKER_FLAGS})
endif()

add_executable(brachistochrone_bench bench.cpp)

target_compile_options(brachistochrone_bench PRIVATE ${COMPILE_FLAGS}
    $<$<CXX_COMPILER_ID:Clang>:-fPIC>)
target_include_directories(brachistochrone_bench PRIVATE ${INCLUDE_DIRS})
target_link_libraries(brachistochrone_bench PRIVATE tycho_core)
target_precompile_headers(brachistochrone_bench REUSE_FROM pch)

if(UNIX)
    target_link_options(brachistochrone_bench PRIVATE ${LINKER_FLAGS})
endif()
