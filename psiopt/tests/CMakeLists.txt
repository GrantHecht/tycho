################################################################################
# PSIOPT unit tests -- the solver-internal slice of the C++ test suite: pure
# globalization-mechanism / KKT / sparse-backend tests that touch nothing in
# Tycho's VectorFunction tree (see tests/cpp/solvers/ for the VF-coupled
# tests that stayed behind, and the psiopt CMakeLists.txt in the parent
# directory for why the split exists at all).
#
# Builds in both configurations psiopt itself supports:
#   Subproject (the Tycho build). tycho's tests/cpp/CMakeLists.txt calls
#   FetchContent_Declare(googletest) + FetchContent_MakeAvailable(googletest)
#   too; whichever of the two directories CMake processes first does the real
#   fetch, and the second FetchContent_MakeAvailable call is a documented
#   no-op (same NAME). COMPILE_FLAGS / INCLUDE_DIRS / LINK_LIBS are inherited
#   from the parent directory scope, same as psiopt's own CMakeLists.txt.
#
#   Standalone (`cmake -S psiopt`). This file fetches GoogleTest itself; the
#   parent psiopt/CMakeLists.txt has already resolved COMPILE_FLAGS,
#   INCLUDE_DIRS, LINK_LIBS and PSIOPT_DEP_INCLUDES (Eigen/fmt) before
#   add_subdirectory(tests) runs, so they are used as-is here too.
################################################################################

include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.17.0
    GIT_SHALLOW    TRUE
    EXCLUDE_FROM_ALL
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Clang 22.1.0 with MSVC headers: suppress char8_t->char32_t implicit conversion
# warning in gtest-printers.h.  gtest compiles with -WX so this becomes an error
# without the suppression. Mirrors tests/cpp/CMakeLists.txt.
if(WIN32)
    foreach(_t gtest gtest_main)
        if(TARGET ${_t})
            target_compile_options(${_t} PRIVATE -Wno-character-conversion)
        endif()
    endforeach()
endif()

include(GoogleTest)

#-------------------------------------------------------------------------------
# Test sources -- solver-internal suites with no VectorFunction dependency
# (globalization mechanisms, KKT/jet plumbing, the Accelerate sparse
# interface, progress-measure acceptance/restoration rules).
#-------------------------------------------------------------------------------
set(PSIOPT_TEST_SOURCES
    test_accelerate_interface.cpp
    test_acceptance_diagnostics.cpp
    test_classic_merit_restoration.cpp
    test_filter_acceptance.cpp
    test_funnel_acceptance.cpp
    test_jet_mkl_guard.cpp
    test_l1_restoration.cpp
    test_merit_rules.cpp
    test_monitored_governor.cpp
    test_nlp_adapter.cpp
    test_proximal_restoration.cpp
    test_recovery_dispatch_gate.cpp
    test_switching_acceptance.cpp
    test_watchdog.cpp
)

add_executable(psiopt_tests ${PSIOPT_TEST_SOURCES})
target_compile_options(psiopt_tests PRIVATE ${COMPILE_FLAGS})
target_include_directories(psiopt_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PSIOPT_DEP_INCLUDES}
    ${INCLUDE_DIRS}
)
# psiopt::psiopt is PRIVATE-linked to psiopt::utils and (Linux) never
# publishes MKL as a link requirement of its own -- that only happens for the
# tycho INTERFACE target on the Tycho side (src/CMakeLists.txt). An
# executable that actually resolves symbols (unlike the static libraries
# above, which never need to) must link LINK_LIBS itself.
target_link_libraries(psiopt_tests PRIVATE psiopt::psiopt psiopt::utils
    ${LINK_LIBS} GTest::gtest_main)
if(UNIX)
    target_link_options(psiopt_tests PRIVATE ${LINKER_FLAGS})
endif()

# Unity build -- merge test TUs to eliminate cross-TU template duplication,
# same pattern (and batch size) as tests/cpp/CMakeLists.txt's tycho_tests.
# These TUs are much lighter than the VF-heavy ones on the Tycho side (no
# tycho/tycho.h umbrella), so this target does not need the heavy_compile
# job pool: it is not registered in standalone builds, and even as a
# subproject these sources do not approach the RAM budget that pool exists
# to protect.
set_target_properties(psiopt_tests PROPERTIES
    UNITY_BUILD ON
    UNITY_BUILD_BATCH_SIZE 8
)

# Reduce optimization level for tests to improve compile time, matching
# tests/cpp/CMakeLists.txt.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(psiopt_tests PRIVATE $<$<CONFIG:Release>:-O1>)
endif()

gtest_discover_tests(psiopt_tests)
