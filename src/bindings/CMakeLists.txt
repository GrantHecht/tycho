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
# Heavy binding TUs (>2.5 GB RSS) — compiled in the heavy_compile ninja pool,
# limiting concurrency to prevent OOM.
#-------------------------------------------------------------------------------
add_library(_tychopy_heavy OBJECT
    tycho_module.cpp
    optimal_control/generic_odes_build_part3.cpp
    optimal_control/generic_odes_build_part4.cpp
    optimal_control/generic_odes_build_part6.cpp
    astro/tycho_astro.cpp
    astro/kepler_utils.cpp
    astro/kepler_model.cpp
    astro/kepler_integrator.cpp
)
target_compile_options(_tychopy_heavy PRIVATE ${BINDING_COMPILE_FLAGS})
target_link_libraries(_tychopy_heavy PRIVATE nanobind-static)
target_compile_definitions(_tychopy_heavy PRIVATE TYCHO_PYTHON_BINDINGS)
set_target_properties(_tychopy_heavy PROPERTIES DEFINE_SYMBOL "")
target_include_directories(_tychopy_heavy PRIVATE ${_TYCHO_SRC_INCLUDES} ${_TYCHO_BINDINGS_INCLUDES} ${INCLUDE_DIRS})
target_precompile_headers(_tychopy_heavy REUSE_FROM pch_bindings)
set_property(TARGET _tychopy_heavy PROPERTY JOB_POOL_COMPILE heavy_compile)

#-------------------------------------------------------------------------------
# _tychopy nanobind module — light/medium binding TUs only.
# Heavy TUs are compiled separately in _tychopy_heavy and linked as object files below.
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
set_property(TARGET _tychopy PROPERTY JOB_POOL_COMPILE heavy_compile)
target_compile_definitions(_tychopy PUBLIC TYCHO_PYTHON_BINDINGS)
# Prevent CMake from auto-defining _tychopy_EXPORTS on the module target; the macro
# is absent from the reused pch_bindings PCH, which causes a -Wclang-cl-pch mismatch.
set_target_properties(_tychopy PROPERTIES DEFINE_SYMBOL "")
target_link_libraries(_tychopy PUBLIC tycho)
if(TARGET tycho_extensions_bindings)
    target_link_libraries(_tychopy PUBLIC tycho_extensions_bindings)
    target_compile_definitions(_tychopy PRIVATE TYCHO_HAS_EXTENSIONS)
    target_compile_definitions(_tychopy_heavy PRIVATE TYCHO_HAS_EXTENSIONS)
    message(STATUS "tycho: extensions bindings enabled (TYCHO_HAS_EXTENSIONS defined)")
else()
    message(STATUS "tycho: extensions bindings disabled (TYCHO_HAS_EXTENSIONS undefined)")
endif()
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
#
# Stubs for the _tychopy extension are also copied to site-packages ROOT so
# PEP 561 type-checkers (mypy/pyright) discover them next to _tychopy.cpython-*.so.
# We source from the committed source-tree snapshot at tychopy/_stubs/ — same
# files the wheel install rules use — so the dev and wheel installs stay
# consistent. (The fresh stubs in ${_TYCHOPY_STUB_DIR} are still useful for
# the snapshot-refresh target; dev installs rely on the committed snapshot.)
if(PYTHON_LOCAL_INSTALL_DIR)
    add_custom_command(TARGET _tychopy POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_tychopy> ${PYTHON_LOCAL_INSTALL_DIR}/
        COMMAND ${CMAKE_COMMAND} -E rm -rf ${PYTHON_LOCAL_INSTALL_DIR}/tychopy
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/tychopy ${PYTHON_LOCAL_INSTALL_DIR}/tychopy
        # PEP 561 stubs at site-packages root (next to the .so).
        COMMAND ${CMAKE_COMMAND} -E rm -rf ${PYTHON_LOCAL_INSTALL_DIR}/_tychopy
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/tychopy/_stubs/_tychopy.pyi
                ${PYTHON_LOCAL_INSTALL_DIR}/
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_SOURCE_DIR}/tychopy/_stubs/_tychopy
                ${PYTHON_LOCAL_INSTALL_DIR}/_tychopy
        COMMENT "Installed _tychopy and tychopy package to ${PYTHON_LOCAL_INSTALL_DIR}"
    )
endif()

#-------------------------------------------------------------------------------
# Stub generation (nanobind_add_stub)
#-------------------------------------------------------------------------------
# Generates .pyi stub files for _tychopy and each submodule so IDEs and
# type-checkers see the API surface that the .so exports. Committed snapshot
# at <repo>/tychopy/_stubs/ is regenerated by the tychopy_stubs_snapshot
# target; CI compares the freshly-generated stubs against the snapshot and
# fails on drift (Task 6 docs-build workflow).

set(_TYCHOPY_STUB_DIR "${CMAKE_CURRENT_BINARY_DIR}/stubs")
file(MAKE_DIRECTORY "${_TYCHOPY_STUB_DIR}/_tychopy")

# Top-level _tychopy.pyi
nanobind_add_stub(_tychopy_stub
    MODULE _tychopy
    OUTPUT "${_TYCHOPY_STUB_DIR}/_tychopy.pyi"
    PYTHON_PATH "$<TARGET_FILE_DIR:_tychopy>"
    DEPENDS _tychopy
    INCLUDE_PRIVATE FALSE
    EXCLUDE_DOCSTRINGS FALSE
)

# Submodule stubs. List matches the submodule names exposed in
# src/bindings/tycho_module.cpp / function_registry.h (vector_functions,
# optimal_control, solvers, astro, utils, extensions). Note: rk_coeffs_bind
# populates the optimal_control submodule (no separate _tychopy.integrators),
# so integrators is not a stub target. Add it here only if a future binding
# defines `m.def_submodule("integrators", ...)`. The `extensions` submodule
# is always created by function_registry.h (may be empty if no user extension
# is built — that's fine; its stub will simply be near-empty).
set(_TYCHOPY_SUBMODULES vector_functions optimal_control solvers astro utils extensions)

foreach(_submod IN LISTS _TYCHOPY_SUBMODULES)
    nanobind_add_stub(_tychopy_${_submod}_stub
        MODULE _tychopy.${_submod}
        OUTPUT "${_TYCHOPY_STUB_DIR}/_tychopy/${_submod}.pyi"
        PYTHON_PATH "$<TARGET_FILE_DIR:_tychopy>"
        DEPENDS _tychopy
        INCLUDE_PRIVATE FALSE
        EXCLUDE_DOCSTRINGS FALSE
    )
endforeach()

# Aggregate target — builds all stubs in one shot.
set(_TYCHOPY_ALL_STUB_TARGETS _tychopy_stub)
foreach(_submod IN LISTS _TYCHOPY_SUBMODULES)
    list(APPEND _TYCHOPY_ALL_STUB_TARGETS _tychopy_${_submod}_stub)
endforeach()
add_custom_target(tychopy_all_stubs DEPENDS ${_TYCHOPY_ALL_STUB_TARGETS})

# Snapshot target: regenerate all stubs and copy them into the source tree at
# tychopy/_stubs/. Run after changing any binding signature; commit the result.
# CI checks for drift between this snapshot and the freshly-built stubs.
#
# The post-process step (scripts/fix_stubs.py) sanitises nanobind stubgen
# output to work around two known upstream quirks:
#   - bare `import <submodule>` lines for the file's own submodule
#     (e.g. `import optimal_control` in _tychopy/optimal_control.pyi)
#   - same-submodule type references using a truncated qualifier
#     (e.g. `optimal_control.ScaleModes` instead of the unqualified `ScaleModes`)
# Both stem from nanobind's std::variant type-name composition stripping the
# package-root prefix; correcting them requires per-method nb::sig() overrides
# on every binding (~35 sites for auto_scale alone), which doesn't scale.
set(_TYCHOPY_SNAPSHOT_DIR "${PROJECT_SOURCE_DIR}/tychopy/_stubs")
add_custom_target(tychopy_stubs_snapshot
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${_TYCHOPY_SNAPSHOT_DIR}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_TYCHOPY_SNAPSHOT_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${_TYCHOPY_STUB_DIR}" "${_TYCHOPY_SNAPSHOT_DIR}"
    COMMAND ${Python_EXECUTABLE} "${PROJECT_SOURCE_DIR}/scripts/fix_stubs.py"
            "${_TYCHOPY_SNAPSHOT_DIR}"
    DEPENDS tychopy_all_stubs
    COMMENT "Refreshing tychopy/_stubs/ snapshot (with fix_stubs.py post-process) — commit the result"
    VERBATIM)

# Install stubs at site-packages ROOT (PEP 561 location for the _tychopy
# extension module). Type-checkers look for `_tychopy.pyi` next to
# `_tychopy.cpython-*.so` and a `_tychopy/` directory of submodule stubs at
# the same level — NOT nested under tychopy/_stubs/. We source from the
# committed snapshot in the source tree (kept in sync by the
# tychopy_stubs_snapshot target above) so wheel builds never depend on a live
# _tychopy import.
#
# For wheel builds (SKBUILD), `wheel.install-dir` defaults to "/" which maps
# CMake DESTINATION "." onto site-packages root. For dev installs, the same
# install rules + PYTHON_LOCAL_INSTALL_DIR land them next to the .so.
install(FILES "${_TYCHOPY_SNAPSHOT_DIR}/_tychopy.pyi"
        DESTINATION ".")
install(DIRECTORY "${_TYCHOPY_SNAPSHOT_DIR}/_tychopy"
        DESTINATION "."
        FILES_MATCHING PATTERN "*.pyi")

# py.typed marker (signals to type-checkers that the *tychopy* package ships
# type info). PEP 561 requires this file to live INSIDE the package directory.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/py.typed" "")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/py.typed"
        DESTINATION "tychopy")
