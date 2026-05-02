include(FetchContent)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.9.5
    GIT_SHALLOW    TRUE
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(benchmark)

# Separate PCH for benchmarks — the global CMAKE_POSITION_INDEPENDENT_CODE ON
# causes the library PCH to be built with -fPIC, which is incompatible with
# benchmark executables that should not use PIC/PIE (avoids indirection overhead
# that would skew measurements).
add_library(pch_bench STATIC ${CMAKE_SOURCE_DIR}/src/pch.h ${CMAKE_SOURCE_DIR}/src/pch.cpp)
set_target_properties(pch_bench PROPERTIES POSITION_INDEPENDENT_CODE OFF)
target_precompile_headers(pch_bench PRIVATE
    ${CMAKE_SOURCE_DIR}/src/pch.h
    ${CMAKE_SOURCE_DIR}/src/vf/tycho_vector_functions.h
)
target_include_directories(pch_bench PRIVATE ${TYCHO_INCLUDES} ${INCLUDE_DIRS})
target_compile_options(pch_bench PRIVATE ${COMPILE_FLAGS})

add_executable(bench_all
    bench_init.cpp
    kepler/bench_kepler.cpp
    vector_functions/bench_vector_functions.cpp
    type_erasure/bench_type_erasure.cpp
    integrators/bench_integrators.cpp
    integrators/bench_method_comparison.cpp
    integrators/bench_controller_comparison.cpp
    optimal_control/bench_optimal_control.cpp
    solvers/bench_solvers.cpp
    utils/bench_utils.cpp
)
set_target_properties(bench_all PROPERTIES POSITION_INDEPENDENT_CODE OFF)
target_compile_options(bench_all PRIVATE ${COMPILE_FLAGS})
target_include_directories(bench_all PRIVATE ${TYCHO_INCLUDES} ${INCLUDE_DIRS})
target_link_libraries(bench_all PRIVATE tycho benchmark::benchmark_main)
target_precompile_headers(bench_all REUSE_FROM pch_bench)
set_property(TARGET bench_all PROPERTY JOB_POOL_COMPILE heavy_compile)
if(UNIX)
    target_link_options(bench_all PRIVATE ${LINKER_FLAGS})
endif()

if(ENABLE_ENZYME_AD AND BUILD_CPP_BENCHMARKS)
    add_executable(bench_enzyme bench_enzyme.cpp)
    set_target_properties(bench_enzyme PROPERTIES POSITION_INDEPENDENT_CODE OFF)
    target_compile_options(bench_enzyme PRIVATE ${COMPILE_FLAGS} ${TYCHO_ENZYME_COMPILE_FLAGS})
    target_include_directories(bench_enzyme PRIVATE
        ${TYCHO_INCLUDES} ${INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/tests/cpp/enzyme   # reuse enzyme_test_dynamics.h
    )
    target_link_libraries(bench_enzyme PRIVATE tycho benchmark::benchmark_main)
    set_target_properties(bench_enzyme PROPERTIES
        JOB_POOL_COMPILE heavy_compile
        DISABLE_PRECOMPILE_HEADERS ON)
    if(UNIX)
        target_link_options(bench_enzyme PRIVATE ${LINKER_FLAGS})
    endif()
endif()
