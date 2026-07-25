# =============================================================================
# Ipopt-backend test executable.  Only added when the build is configured with
# Ipopt support; the Ipopt headers and link are scoped to this target, so the
# rest of the test suite is unaffected.
# =============================================================================

add_executable(tycho_ipopt_tests
    test_ipopt_adapter.cpp
)

target_compile_options(tycho_ipopt_tests PRIVATE ${COMPILE_FLAGS})

target_include_directories(tycho_ipopt_tests PRIVATE
    ${_TYCHO_TEST_COMMON_INCLUDES}
    ${IPOPT_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/tests/cpp/solvers         # for solver_test_utils.h
    ${CMAKE_SOURCE_DIR}/tests/cpp/optimal_control # for oc_test_utils.h
)

target_link_libraries(tycho_ipopt_tests PRIVATE tycho GTest::gtest_main PkgConfig::IPOPT)

if(UNIX)
    target_link_options(tycho_ipopt_tests PRIVATE ${LINKER_FLAGS})
endif()

# Includes tycho/tycho.h -> heavy template instantiation; join the shared pool
# so concurrent builds cannot OOM the system.
set_property(TARGET tycho_ipopt_tests PROPERTY JOB_POOL_COMPILE heavy_compile)

# Tests don't need -O3.  Lower optimization for compile-time savings.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(tycho_ipopt_tests PRIVATE $<$<CONFIG:Release>:-O1>)
endif()

include(GoogleTest)
gtest_discover_tests(tycho_ipopt_tests)
