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

add_subdirectory(kepler)
add_subdirectory(vector_functions)
add_subdirectory(type_erasure)
add_subdirectory(integrators)
add_subdirectory(optimal_control)
add_subdirectory(solvers)
add_subdirectory(utils)
