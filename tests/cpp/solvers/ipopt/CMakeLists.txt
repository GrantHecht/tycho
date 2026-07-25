# =============================================================================
# Ipopt-backend test executable.  Only added when the build is configured with
# Ipopt support; the Ipopt headers and link are scoped to this target, so the
# rest of the test suite is unaffected.
# =============================================================================

set(TYCHO_IPOPT_TEST_SOURCES
    test_ipopt_adapter.cpp
)

# tycho_add_test_executable (tests/cpp/CMakeLists.txt) is a CMake function,
# so it is visible here even though it is defined in the parent listfile:
# CMake functions are global once defined, regardless of which subdirectory
# calls them. It covers everything common to every test executable (compile
# flags, ${_TYCHO_TEST_COMMON_INCLUDES}, linking against tycho + gtest_main,
# linker flags, and gtest_discover_tests registration); only the Ipopt-
# specific extras are layered on below.
tycho_add_test_executable(tycho_ipopt_tests TYCHO_IPOPT_TEST_SOURCES)

target_include_directories(tycho_ipopt_tests PRIVATE
    ${IPOPT_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/tests/cpp/solvers         # for solver_test_utils.h
    ${CMAKE_SOURCE_DIR}/tests/cpp/optimal_control # for oc_test_utils.h
)

target_link_libraries(tycho_ipopt_tests PRIVATE PkgConfig::IPOPT)

# Includes tycho/tycho.h -> heavy template instantiation; join the shared pool
# so concurrent builds cannot OOM the system.
set_property(TARGET tycho_ipopt_tests PROPERTY JOB_POOL_COMPILE heavy_compile)

# Tests don't need -O3.  Lower optimization for compile-time savings.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(tycho_ipopt_tests PRIVATE $<$<CONFIG:Release>:-O1>)
endif()
