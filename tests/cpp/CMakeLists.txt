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
# without the suppression.
if(WIN32)
    foreach(_t gtest gtest_main)
        if(TARGET ${_t})
            target_compile_options(${_t} PRIVATE -Wno-character-conversion)
        endif()
    endforeach()
endif()

include(GoogleTest)

#-------------------------------------------------------------------------------
# Test PCH: pre-parse Tycho.h (the full aggregate header) so that each test TU
# doesn't independently re-instantiate the entire template library.  Without
# this, every test that includes Tycho.h re-parses ODEPhase.h,
# DenseFunctionBase.h, all solvers, integrators, etc. — the dominant build cost.
#
# Uses the same COMPILE_FLAGS as production code so tests are compiled under
# identical optimizations.
#-------------------------------------------------------------------------------
add_library(pch_tests STATIC ${PROJECT_SOURCE_DIR}/src/pch.cpp)
target_include_directories(pch_tests PRIVATE ${TYCHO_INCLUDES} ${INCLUDE_DIRS})
target_compile_options(pch_tests PRIVATE ${COMPILE_FLAGS})
target_precompile_headers(pch_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/src/pch.h
    ${PROJECT_SOURCE_DIR}/src/Tycho.h
)

#-------------------------------------------------------------------------------
# Single test executable with all TUs — 42 small source files compile in
# parallel with low per-TU RAM, but only one link step (linking is ~5GB).
#-------------------------------------------------------------------------------
set(TYCHO_TEST_SOURCES
    # Utils
    utils/test_timer.cpp
    utils/test_math_functions.cpp
    utils/test_sizing_helpers.cpp
    utils/test_eigen_stl.cpp
    utils/test_lambda_jump_table.cpp
    utils/test_misc_utils.cpp
    utils/test_flat_map.cpp
    utils/test_type_storage.cpp
    utils/test_thread_pool.cpp
    utils/test_tuple_iterator.cpp
    utils/test_memory_management.cpp

    # VectorFunctions
    vector_functions/test_args_segment.cpp
    vector_functions/test_constant_scaled.cpp
    vector_functions/test_sum_dot_cross.cpp
    vector_functions/test_norms.cpp
    vector_functions/test_normalized_stacked_padded.cpp
    vector_functions/test_composition_trig.cpp
    vector_functions/test_elements_cwiseops.cpp
    vector_functions/test_generic_function_basic.cpp
    vector_functions/test_generic_function_copy_move.cpp
    vector_functions/test_generic_function_interfaces.cpp
    vector_functions/test_vf_ode_expressions.cpp
    vector_functions/test_vf_composition_chains.cpp
    vector_functions/test_vf_hessian.cpp
    vector_functions/test_vector_functions.cpp

    # Astro
    astro/test_kepler_conversions.cpp
    astro/test_kepler_edge_cases.cpp
    astro/test_kepler_propagation.cpp
    astro/test_lambert_solver.cpp
    astro/test_cr3bp.cpp

    # Integrators
    integrators/test_integration_basic.cpp
    integrators/test_integration_convergence.cpp
    integrators/test_integration_stm.cpp
    integrators/test_integration_adaptive.cpp
    integrators/test_integration_dense.cpp

    # OptimalControl
    optimal_control/test_phase_construction.cpp
    optimal_control/test_ocp_multiphase.cpp
    optimal_control/test_lgl_collocation.cpp
    optimal_control/test_mesh_refinement.cpp
    optimal_control/test_ode_sizes.cpp

    # Solvers
    solvers/test_psiopt_convergence.cpp
    solvers/test_solver_interfaces.cpp
    solvers/test_jet_batch.cpp
    solvers/test_nlp_structure.cpp
)

add_executable(tycho_tests ${TYCHO_TEST_SOURCES})
# Match PIC mode with pch_tests (static lib gets -fPIC, executable gets -fPIE
# by default — clang 20+ rejects the AST mismatch in reused PCH).
set_target_properties(tycho_tests PROPERTIES POSITION_INDEPENDENT_CODE OFF)
target_compile_options(tycho_tests PRIVATE ${COMPILE_FLAGS}
    $<$<NOT:$<PLATFORM_ID:Windows>>:-fPIC>)
target_include_directories(tycho_tests PRIVATE
    ${INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}                 # for test_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/astro           # for astro_test_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/integrators     # for integrator_test_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/optimal_control # for oc_test_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/solvers         # for solver_test_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/vector_functions # for vf_test_utils.h
)
target_link_libraries(tycho_tests PRIVATE tycho GTest::gtest_main)
target_precompile_headers(tycho_tests REUSE_FROM pch_tests)
if(UNIX)
    target_link_options(tycho_tests PRIVATE ${LINKER_FLAGS})
endif()
gtest_discover_tests(tycho_tests)
