set(_TYCHO_BINDINGS_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/vf
    ${CMAKE_CURRENT_SOURCE_DIR}/optimal_control
    ${CMAKE_CURRENT_SOURCE_DIR}/solvers
    ${CMAKE_CURRENT_SOURCE_DIR}/astro
    ${CMAKE_CURRENT_SOURCE_DIR}/integrators
)

# Ensure nanobind-static exists before pch_bindings uses it
nanobind_build_library(nanobind-static)

# Separate PCH compiled WITH TYCHO_PYTHON_BINDINGS so that nanobind headers
# are baked into the precompiled header used by _tychopy and its binding sources.
add_library(pch_bindings STATIC ${CMAKE_SOURCE_DIR}/src/pch.cpp)
target_precompile_headers(pch_bindings PRIVATE
    ${CMAKE_SOURCE_DIR}/src/pch.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pch_nb.h
    ${CMAKE_SOURCE_DIR}/src/vf/tycho_vector_functions.h
    ${CMAKE_CURRENT_SOURCE_DIR}/vector_functions_bind.h
)
target_compile_definitions(pch_bindings PRIVATE TYCHO_PYTHON_BINDINGS)
target_include_directories(pch_bindings PRIVATE ${_TYCHO_SRC_INCLUDES} ${_TYCHO_BINDINGS_INCLUDES} ${INCLUDE_DIRS})
target_compile_options(pch_bindings PUBLIC ${BINDING_COMPILE_FLAGS})
target_link_libraries(pch_bindings PRIVATE nanobind-static)

#-------------------------------------------------------------------------------
# Heavy binding TUs (>2.5 GB RSS) — compiled in the heavy_compile ninja pool
# so at most 2 run concurrently, preventing OOM on 32 GB systems.
#-------------------------------------------------------------------------------
add_library(_tychopy_heavy OBJECT
    tycho_module.cpp
    optimal_control/generic_odes_build_part3.cpp
    optimal_control/generic_odes_build_part4.cpp
    optimal_control/generic_odes_build_part5.cpp
    optimal_control/generic_odes_build_part6.cpp
    astro/tycho_astro.cpp
    astro/kepler_utils.cpp
    astro/kepler_model.cpp
    astro/kepler_integrator.cpp
)
target_compile_options(_tychopy_heavy PRIVATE ${BINDING_COMPILE_FLAGS})
target_compile_definitions(_tychopy_heavy PRIVATE TYCHO_PYTHON_BINDINGS)
set_target_properties(_tychopy_heavy PROPERTIES DEFINE_SYMBOL "")
target_include_directories(_tychopy_heavy PRIVATE ${_TYCHO_SRC_INCLUDES} ${_TYCHO_BINDINGS_INCLUDES} ${INCLUDE_DIRS})
target_precompile_headers(_tychopy_heavy REUSE_FROM pch_bindings)
set_property(TARGET _tychopy_heavy PROPERTY JOB_POOL_COMPILE heavy_compile)

#-------------------------------------------------------------------------------
# _tychopy nanobind module — light/medium binding TUs plus heavy objects.
#-------------------------------------------------------------------------------
nanobind_add_module(_tychopy
    # Utils bindings
    utils_bind.cpp
    bump_allocator_bind.cpp
    # VectorFunctions bindings
    vf/tycho_vector_functions.cpp
    vf/vector_function_build_part1.cpp
    vf/vector_function_build_part2.cpp
    vf/args_seg_build_part1.cpp
    vf/args_seg_build_part2.cpp
    vf/args_seg_build_part3.cpp
    vf/args_seg_build_part4.cpp
    vf/args_seg_build_part5.cpp
    vf/bulk_operations_build.cpp
    vf/free_functions_build.cpp
    vf/matrix_function_build.cpp
    vf/python_arg_parsing.cpp
    # OptimalControl bindings
    optimal_control/tycho_optimal_control.cpp
    optimal_control/generic_odes_build_part1.cpp
    optimal_control/generic_odes_build_part2.cpp
    optimal_control/generic_odes_integrator_part1.cpp
    optimal_control/generic_odes_integrator_part2.cpp
    optimal_control/generic_odes_integrator_part3.cpp
    optimal_control/generic_odes_integrator_part4.cpp
    optimal_control/generic_odes_integrator_part5.cpp
    optimal_control/generic_odes_integrator_part6.cpp
    optimal_control/ode_phase_base_bind.cpp
    optimal_control/optimal_control_problem_bind.cpp
    optimal_control/lgl_interp_table_bind.cpp
    optimal_control/mesh_iterate_info_bind.cpp
    # Solvers bindings
    solvers/tycho_solvers.cpp
    solvers/optimization_problem_base.cpp
    solvers/psiopt_bind.cpp
    solvers/optimization_problem_bind.cpp
    solvers/jet_bind.cpp
    # Astro bindings
    astro/lambert_solvers_bind.cpp
    # Integrators bindings
    integrators/rk_coeffs_bind.cpp
)

target_compile_options(_tychopy PRIVATE ${BINDING_COMPILE_FLAGS})
target_compile_definitions(_tychopy PUBLIC TYCHO_PYTHON_BINDINGS)
# Prevent CMake from auto-defining _tychopy_EXPORTS on the module target; the macro
# is absent from the reused pch_bindings PCH, which causes a -Wclang-cl-pch mismatch.
set_target_properties(_tychopy PROPERTIES DEFINE_SYMBOL "")
target_link_libraries(_tychopy PUBLIC tycho tycho_extensions_bindings)
target_sources(_tychopy PRIVATE $<TARGET_OBJECTS:_tychopy_heavy>)
target_include_directories(_tychopy PRIVATE ${_TYCHO_SRC_INCLUDES} ${_TYCHO_BINDINGS_INCLUDES} ${INCLUDE_DIRS})
target_precompile_headers(_tychopy REUSE_FROM pch_bindings)

if(APPLE)
    # macOS needs -bind_at_load and -undefined dynamic_lookup for Python modules
    target_link_options(_tychopy PRIVATE ${LINKER_FLAGS})
endif()
# Linux/Windows: tycho provides LINK_LIBS (MKL, threads, dl) transitively.
# Linux: no --no-undefined — Python symbols are resolved at runtime by the interpreter.

# Install rule for scikit-build-core: places _tychopy alongside the tychopy package in the wheel.
install(TARGETS _tychopy LIBRARY DESTINATION .)

# Dev-install shortcut: when PYTHON_LOCAL_INSTALL_DIR is set (e.g. by config_and_build.sh),
# copy _tychopy and the Python package directly to site-packages after every build so that
# changes are immediately usable without a separate pip install step.
if(PYTHON_LOCAL_INSTALL_DIR)
    add_custom_command(TARGET _tychopy POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_tychopy> ${PYTHON_LOCAL_INSTALL_DIR}/
        COMMAND ${CMAKE_COMMAND} -E rm -rf ${PYTHON_LOCAL_INSTALL_DIR}/tychopy
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/tychopy ${PYTHON_LOCAL_INSTALL_DIR}/tychopy
        COMMENT "Installed _tychopy and tychopy package to ${PYTHON_LOCAL_INSTALL_DIR}"
    )
endif()
