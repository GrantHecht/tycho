# The solver library owns the solver headers plus the Eigen typedefs and
# utility headers they need, so its header root sits alongside include/ for
# every target here. Its src/interior and src/interior/utils carry the two
# private headers (fmtlib.h, color_text.h) that src/pch.h and
# src/utils/tycho_utils.h still include unqualified -- the same arrangement
# the in-tree solver project had, at its new paths.
set(_TYCHO_SRC_INCLUDES
  ${PROJECT_SOURCE_DIR}/include
  ${TYCHO_HVEN_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/optimal_control
  ${CMAKE_CURRENT_SOURCE_DIR}/solvers
  ${CMAKE_CURRENT_SOURCE_DIR}/typedefs
  ${CMAKE_CURRENT_SOURCE_DIR}/utils
  ${CMAKE_CURRENT_SOURCE_DIR}/vf
  ${CMAKE_CURRENT_SOURCE_DIR}/integrators
  ${TYCHO_HVEN_DIR}/src/interior
  ${TYCHO_HVEN_DIR}/src/interior/utils
)
set(TYCHO_INCLUDES ${_TYCHO_SRC_INCLUDES} PARENT_SCOPE)

#-------------------------------------------------------------------------------
## PCH
add_library(pch STATIC pch.h pch.cpp)
target_precompile_headers(pch PRIVATE pch.h
    vf/tycho_vector_functions.h)
target_include_directories(pch PRIVATE ${_TYCHO_SRC_INCLUDES} ${INCLUDE_DIRS})
# hven publishes HVEN_DEFAULT_QP_THREADS as a PUBLIC compile definition because
# two of its public headers read it, so a target that compiles hven headers has
# to LINK hven, not merely receive its include directory. Build-tree only, like
# every other link to hven here.
target_link_libraries(pch PRIVATE $<BUILD_INTERFACE:hven::hven>)
target_compile_options(pch PRIVATE ${COMPILE_FLAGS})
# Omit the embedded timestamp from the generated PCH so ccache can reuse cached
# object files across PCH regenerations (with sloppiness=pch_defines,time_macros
# and pch_external_checksum in the user's ccache config, consumers hash the PCH
# CONTENT — a byte-identical PCH after an innocuous regen then costs nothing).
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(pch PRIVATE -Xclang -fno-pch-timestamp)
endif()


## Put all ".h"/".cpp" pairs here
# Example: add_library(myODE STATIC odes/myode.h odes/myode.cpp)

# vectorfunctions is header-only (all VF code is template-based);
# expose it as an INTERFACE library so dependents get the include path.
add_library(vectorfunctions INTERFACE)
target_include_directories(vectorfunctions INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/vf>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Explicit template instantiations for common VF types — reduces cross-TU
# template duplication. See include/tycho/detail/vf/extern_templates.h.
add_library(vf_instantiations STATIC vf/extern_template_instantiations.cpp)
target_include_directories(vf_instantiations PRIVATE ${_TYCHO_SRC_INCLUDES} ${INCLUDE_DIRS})
target_link_libraries(vf_instantiations PRIVATE $<BUILD_INTERFACE:hven::hven>)
target_compile_options(vf_instantiations PRIVATE ${COMPILE_FLAGS})
target_precompile_headers(vf_instantiations REUSE_FROM pch)
set_property(TARGET vf_instantiations PROPERTY JOB_POOL_COMPILE heavy_compile)

# Explicit template instantiations for common Integrator<DODE> types — reduces
# binding TU RAM pressure. See
# include/tycho/detail/integrators/extern_templates.h.
add_library(integrators_instantiations STATIC
    integrators/extern_template_instantiations.cpp)
target_include_directories(integrators_instantiations PRIVATE
    ${_TYCHO_SRC_INCLUDES} ${INCLUDE_DIRS})
target_link_libraries(integrators_instantiations PRIVATE $<BUILD_INTERFACE:hven::hven>)
target_compile_options(integrators_instantiations PRIVATE ${COMPILE_FLAGS})
target_precompile_headers(integrators_instantiations REUSE_FROM pch)
set_property(TARGET integrators_instantiations PROPERTY JOB_POOL_COMPILE heavy_compile)




# Tycho's own solver surface: everything about solving that the solver library
# does not carry. OptimizationProblem is the convenience layer that turns user
# VF expressions into the library's abstract constraint and objective
# functions, so it cannot live on the library's side of the boundary; the Ipopt
# backend cannot either, because the library's surface carries no interface to
# a foreign solver. Everything that used to get these symbols from the in-tree
# solver project links this target, which links hven in turn.
add_library(solvers_vf STATIC
    solvers/solve_types.cpp
    solvers/optimization_problem.cpp
    solvers/transcription_declaration.cpp
    solvers/transcribed_aggregate.cpp
)

# Alternative NLP solver backend. Exactly one of the two implementations of the
# backend entry points is compiled in: the real adapter when Ipopt is linked,
# the fallback that reports the missing build support otherwise.
#
# The Ipopt imported target is wrapped in BUILD_INTERFACE because solvers_vf is
# part of the installed export set: a PRIVATE dependency on a static library is
# still recorded (as $<LINK_ONLY:...>) in its interface, and install(EXPORT)
# rejects an imported target that is not itself exported. Build-tree consumers
# still link Ipopt transitively; the include directories are attached
# explicitly so they do not depend on that wrapping.
if(ENABLE_IPOPT)
    target_sources(solvers_vf PRIVATE solvers/ipopt_tnlp_adapter.cpp)
    target_compile_definitions(solvers_vf PRIVATE TYCHO_ENABLE_IPOPT)
    target_include_directories(solvers_vf PRIVATE ${IPOPT_INCLUDE_DIRS})
    target_link_libraries(solvers_vf PRIVATE $<BUILD_INTERFACE:PkgConfig::IPOPT>)
else()
    target_sources(solvers_vf PRIVATE solvers/ipopt_backend_stub.cpp)
endif()

target_include_directories(solvers_vf PRIVATE ${_TYCHO_SRC_INCLUDES} ${INCLUDE_DIRS})
target_compile_options(solvers_vf PRIVATE ${COMPILE_FLAGS})
# hven has its own install/export (hvenTargets.cmake, a separate export set
# from tychoTargets), but that target is not part of tychoTargets itself, so
# the link here stays build-tree only -- the same BUILD_INTERFACE guard the
# rest of the non-exportable link dependencies use. cmake/tychoConfig.cmake.in
# re-attaches hven::hven on tycho::tycho for install-tree consumers via its
# own find_dependency(hven) against hven's installed package config.
target_link_libraries(solvers_vf PUBLIC $<BUILD_INTERFACE:hven::hven>)
target_precompile_headers(solvers_vf REUSE_FROM pch)
set_property(TARGET solvers_vf PROPERTY JOB_POOL_COMPILE heavy_compile)


add_library(optimalcontrol STATIC
    optimal_control/tycho_optimal_control.h
    optimal_control/ode_phase_base.cpp
    optimal_control/phase_indexer.cpp
    optimal_control/optimal_control_problem.cpp
    optimal_control/lgl_interp_table.cpp
    optimal_control/runtime_ode.cpp
)

target_include_directories(optimalcontrol PRIVATE ${_TYCHO_SRC_INCLUDES} ${INCLUDE_DIRS})
target_precompile_headers(optimalcontrol REUSE_FROM pch)
target_link_libraries(optimalcontrol PRIVATE solvers_vf)
target_compile_options(optimalcontrol PRIVATE ${COMPILE_FLAGS})
set_property(TARGET optimalcontrol PROPERTY JOB_POOL_COMPILE heavy_compile)



add_library(astro STATIC
    astro/tycho_astro.h
    astro/lambert_solvers.cpp
    astro/accelleration_models.cpp
)

target_precompile_headers(astro REUSE_FROM pch)
target_include_directories(astro PRIVATE ${_TYCHO_SRC_INCLUDES} ${INCLUDE_DIRS})
target_link_libraries(astro PRIVATE $<BUILD_INTERFACE:hven::hven>)
target_link_libraries(astro PRIVATE optimalcontrol)
target_compile_options(astro PRIVATE ${COMPILE_FLAGS})


#-------------------------------------------------------------------------------
# tycho: public INTERFACE target bundling all core static libs.
# Used by Python bindings, C++ consumers (find_package/FetchContent), and future frontends.
add_library(tycho INTERFACE)
target_link_libraries(tycho INTERFACE solvers_vf optimalcontrol vectorfunctions vf_instantiations integrators_instantiations astro)
# hven and the Eigen target it publishes are not part of tychoTargets (hven
# has its own, separate export set: hvenTargets); the installed package
# config re-establishes both -- Eigen3::Eigen via the bundled-headers block
# above, hven::hven via find_dependency(hven) -- for install-tree consumers.
target_link_libraries(tycho INTERFACE $<BUILD_INTERFACE:hven::hven>)
# CRITICAL: LINK_LIBS is wrapped in BUILD_INTERFACE to avoid encoding
# host-specific absolute paths in the export set. ALL libraries in LINK_LIBS
# (set in root CMakeLists.txt) must be re-found and re-attached in
# cmake/tychoConfig.cmake.in. If you modify LINK_LIBS, update the config
# template to match.
target_link_libraries(tycho INTERFACE $<BUILD_INTERFACE:${LINK_LIBS}>)
target_link_libraries(tycho INTERFACE $<BUILD_INTERFACE:Eigen3::Eigen>)
target_include_directories(tycho INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
add_library(tycho::tycho ALIAS tycho)
target_compile_features(tycho INTERFACE cxx_std_20)

# Compile definitions required by public headers
target_compile_definitions(tycho INTERFACE
    FMT_HEADER_ONLY
    FMT_USE_LOCALE=0
    EIGEN_INITIALIZE_MATRICES_BY_ZERO
    EIGEN_DONT_PARALLELIZE
    TYCHO_DEFAULT_QP_THREADS=8
)
if(APPLE)
    target_compile_definitions(tycho INTERFACE USE_ACCELERATE_SPARSE)
endif()
# Note: TYCHO_DEFAULT_ARENA_SIZE is intentionally omitted — it is only
# used in src/utils/memory_management.cpp (not in any public header).

# fmt and sparse-solver include paths for build-tree consumers (FetchContent).
# Install-tree consumers get these via tychoConfig.cmake.in.
target_include_directories(tycho INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/dep/fmt/include>
)
# MKL include directories are needed by public headers. On Apple,
# AccelerateSparse headers are part of the SDK and propagated via the
# AccelerateSparse::AccelerateSparse target.
if(MKL_INCLUDE_DIRS)
    target_include_directories(tycho INTERFACE
        $<BUILD_INTERFACE:${MKL_INCLUDE_DIRS}>
    )
endif()

# Compile options required for correct ABI with header-template code.
# Tycho's public headers contain Eigen types and template definitions that are
# instantiated in the consumer's translation units. Consumers MUST compile with
# matching SIMD, FP, and OpenMP flags to ensure:
#   - Eigen vectorization/alignment matches the library (prevents segfaults)
#   - Floating-point semantics match for template ABI consistency
#   - OpenMP runtime is linked (static libs contain OpenMP symbols)
# Generator expressions ensure correct flags per compiler/platform, and work
# in both build-tree and install-tree usage (CMake stores genexes as-is in
# tychoTargets.cmake, evaluating them against the consumer's compiler).
# Note: ${CLANG_MAX_INLINE_DEPTH} is a CMake variable expanded at configure
# time, so the inline threshold value is baked into the installed target.

# SIMD flags — Eigen alignment/vectorization must match between library and consumer
if(SIMD_FLAGS)
    target_compile_options(tycho INTERFACE ${SIMD_FLAGS})
endif()
# FP mode flags
if(FP_FLAGS)
    target_compile_options(tycho INTERFACE ${FP_FLAGS})
endif()
# Inline threshold — critical for VectorFunction expression template performance.
# Without aggressive inlining, the compiler cannot flatten the DSL expression tree
# into efficient machine code, causing 5-10x slowdowns in NLP evaluation.
target_compile_options(tycho INTERFACE
    $<$<CXX_COMPILER_ID:Clang>:-mllvm>
    $<$<CXX_COMPILER_ID:Clang>:-inline-threshold=${CLANG_MAX_INLINE_DEPTH}>
    $<$<CXX_COMPILER_ID:AppleClang>:-mllvm>
    $<$<CXX_COMPILER_ID:AppleClang>:-inline-threshold=${CLANG_MAX_INLINE_DEPTH}>
)
# OpenMP compile flags
# TODO: Add Windows (clang-cl) OpenMP flags for INTERFACE consumers
target_compile_options(tycho INTERFACE
    $<$<AND:$<PLATFORM_ID:Linux>,$<CXX_COMPILER_ID:Clang>>:-fopenmp=libiomp5>
    $<$<AND:$<PLATFORM_ID:Linux>,$<CXX_COMPILER_ID:GNU>>:-fopenmp>
    $<$<AND:$<PLATFORM_ID:Darwin>,$<CXX_COMPILER_ID:AppleClang>>:-Xclang>
    $<$<AND:$<PLATFORM_ID:Darwin>,$<CXX_COMPILER_ID:AppleClang>>:-fopenmp>
    $<$<AND:$<PLATFORM_ID:Darwin>,$<CXX_COMPILER_ID:Clang>,$<NOT:$<CXX_COMPILER_ID:AppleClang>>>:-fopenmp>
)

#-------------------------------------------------------------------------------
# Install rules

install(TARGETS tycho solvers_vf optimalcontrol astro vectorfunctions
        vf_instantiations integrators_instantiations
    EXPORT tychoTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

#-------------------------------------------------------------------------------

if(ENABLE_PYTHON_BINDINGS)
    add_subdirectory(bindings)
endif()
