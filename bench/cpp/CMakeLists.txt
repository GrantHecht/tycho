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

add_executable(bench_all
    kepler/bench_kepler.cpp
    vector_functions/bench_vector_functions.cpp
    type_erasure/bench_type_erasure.cpp
    integrators/bench_integrators.cpp
    optimal_control/bench_optimal_control.cpp
    solvers/bench_solvers.cpp
    utils/bench_utils.cpp
)
target_compile_options(bench_all PRIVATE ${COMPILE_FLAGS})
target_include_directories(bench_all PRIVATE ${TYCHO_INCLUDES} ${INCLUDE_DIRS})
target_link_libraries(bench_all PRIVATE tycho_core benchmark::benchmark_main)
target_precompile_headers(bench_all REUSE_FROM pch)
# bench_all is linked against benchmark (static), which adds BENCHMARK_STATIC_DEFINE
# via its interface; the reused pch was built without it, causing a harmless mismatch
# warning on clang-cl.  Suppress it.
if(WIN32)
    target_compile_options(bench_all PRIVATE -Wno-clang-cl-pch)
endif()
if(UNIX)
    target_link_options(bench_all PRIVATE ${LINKER_FLAGS})
endif()
