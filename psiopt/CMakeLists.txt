################################################################################
# PSIOPT — the interior-point NLP solver, as its own CMake project.
#
# Builds two static libraries:
#   psiopt::psiopt   the solver core (PSIOPT, NonLinearProgram, the
#                    globalization machinery, the sparse-KKT backends)
#   psiopt::utils    the small runtime support layer the core links
#                    (thread pool, core count, arena allocator, color text)
#
# Nothing here depends on Tycho's VectorFunction layer: the headers under
# include/ are the whole of what the solver needs, so this directory
# configures and builds on its own as well as it does as a subdirectory of
# the Tycho tree. The one piece of the old solver surface that does need
# VectorFunction — OptimizationProblem, the convenience layer that turns VF
# expressions into solver constraint/objective functions — stays on the
# Tycho side as the `solvers_vf_glue` target.
#
# Two configurations:
#
#   Subproject (the Tycho build). The parent has already called
#   tycho_compile_options() and tycho_resolve_sparse_backend(), so
#   COMPILE_FLAGS / INCLUDE_DIRS / the sparse-backend targets are already in
#   scope and are used as-is. Eigen, fmt and the project-wide preprocessor
#   definitions come from the parent's directory-level include_directories()
#   and add_definitions(), inherited here.
#
#   Standalone (`cmake -S psiopt`). This file resolves the same toolchain
#   itself: it sets the prerequisites the two shared cmake modules read from
#   caller scope, includes them from ../cmake, and adds the Eigen/fmt include
#   paths and preprocessor definitions the parent would otherwise have set
#   globally. Phase 1 expects a full Tycho checkout (Eigen and fmt come from
#   ../dep); standalone packaging is a later phase.
################################################################################

cmake_minimum_required(VERSION 3.16)
project(psiopt VERSION 0.1.0 LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(PSIOPT_STANDALONE ON)
else()
    set(PSIOPT_STANDALONE OFF)

    # Subproject mode assumes the Tycho super-build: the state the targets
    # below are built from -- COMPILE_FLAGS and the resolved sparse backend --
    # is set by the parent, not here. Any other parent leaves that state
    # missing, and the targets would then be configured with no optimization,
    # SIMD or fast-math flags and no sparse solver, which surfaces much later
    # as a link failure or a silently slow library. Check for it and say so
    # instead.
    #
    # The check is on the variables rather than on the macros being visible
    # (`if(NOT COMMAND tycho_compile_options)`): including a module makes the
    # command visible without having called it, and it is the call's results
    # this file consumes.
    set(_psiopt_missing "")
    if(NOT DEFINED COMPILE_FLAGS)
        list(APPEND _psiopt_missing "COMPILE_FLAGS (call tycho_compile_options())")
    endif()
    if(APPLE)
        if(NOT TARGET AccelerateSparse::AccelerateSparse)
            list(APPEND _psiopt_missing
                 "AccelerateSparse::AccelerateSparse (call tycho_resolve_sparse_backend())")
        endif()
    elseif(NOT MKL_FOUND)
        list(APPEND _psiopt_missing "MKL (call tycho_resolve_sparse_backend())")
    endif()
    if(_psiopt_missing)
        string(REPLACE ";" "\n    " _psiopt_missing_text "${_psiopt_missing}")
        message(FATAL_ERROR
            "PSIOPT was added as a subdirectory, but the state it expects a parent "
            "project to provide is missing:\n"
            "    ${_psiopt_missing_text}\n"
            "Two configurations are supported today:\n"
            "  1. Standalone -- configure psiopt/ as the top-level project "
            "(cmake -S psiopt -B <dir>). It then resolves the toolchain, the sparse "
            "backend and the header-only dependencies itself.\n"
            "  2. As part of the Tycho super-build, which resolves all of the above "
            "before add_subdirectory(psiopt).\n"
            "Consuming psiopt directly from a third-party project "
            "(add_subdirectory/FetchContent) is a planned later phase; it needs "
            "usage requirements this build does not carry yet.")
    endif()
    unset(_psiopt_missing)
    unset(_psiopt_missing_text)
endif()

include(GNUInstallDirs)

# Extra include paths and compile definitions that the parent supplies
# globally when this is a subproject; empty in that case, populated below
# when standalone.
set(PSIOPT_DEP_INCLUDES "")

if(PSIOPT_STANDALONE)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
    include(CheckCXXCompilerFlag)

    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        message(FATAL_ERROR
            "PSIOPT cannot be built with the Microsoft compiler. Use Clang or GCC.")
    endif()

    #---------------------------------------------------------------------------
    # Prerequisites tycho_compile_options() reads from caller scope.
    set(LINK_TIME_OPT FALSE)
    set(CLANG_MAX_INLINE_DEPTH 225)

    set(TYCHO_FP_MODE "SAFER_FAST" CACHE STRING
        "Floating-point optimization mode. Valid values: STRICT, SAFER_FAST, FAST.")
    set_property(CACHE TYCHO_FP_MODE PROPERTY STRINGS STRICT SAFER_FAST FAST)
    string(TOUPPER "${TYCHO_FP_MODE}" TYCHO_FP_MODE)
    if(NOT TYCHO_FP_MODE MATCHES "^(STRICT|SAFER_FAST|FAST)$")
        message(FATAL_ERROR
            "Invalid TYCHO_FP_MODE='${TYCHO_FP_MODE}'. Valid values: STRICT, SAFER_FAST, FAST.")
    endif()

    set(_TYCHO_WINDOWS_CLANG_NO_FINITE_MATH_ONLY_FLAG "-fno-finite-math-only")
    if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
       AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
        # clang-cl ignores raw GNU-style frontend flags; forward them explicitly.
        set(_TYCHO_WINDOWS_CLANG_NO_FINITE_MATH_ONLY_FLAG "/clang:-fno-finite-math-only")
    endif()

    option(COMPILE_TIME_TRACE "Enable Clang -ftime-trace for build profiling (Clang only)" OFF)
    option(ENABLE_IPOPT
        "Link against an installed Ipopt (found via pkg-config) to enable the alternative NLP solver backend"
        OFF)
    if(ENABLE_IPOPT)
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(IPOPT REQUIRED IMPORTED_TARGET ipopt)
    endif()

    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    endif()

    include(tycho_compile_options)
    include(tycho_sparse_backend)

    tycho_compile_options()

    find_package(Threads REQUIRED)
    # ENABLE_PYTHON_BINDINGS is deliberately left unset: the standalone solver
    # has no Python surface, so the macro's find_package(Python) is skipped.
    tycho_resolve_sparse_backend()

    if(UNIX AND NOT APPLE)
        # MKL's threading layer is Intel's libiomp5; match the parent build.
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            list(APPEND COMPILE_FLAGS "-fopenmp=libiomp5")
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "GCC|GNU")
            list(APPEND COMPILE_FLAGS "-fopenmp")
        endif()
    endif()

    #---------------------------------------------------------------------------
    # Header-only dependencies and the preprocessor definitions the public
    # headers assume. In a Tycho build these come from the root listfile.
    foreach(_dep eigen fmt)
        if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../dep/${_dep}")
            message(FATAL_ERROR
                "PSIOPT standalone expects the bundled ${_dep} at dep/${_dep}. "
                "Run: git submodule update --init dep/${_dep}")
        endif()
    endforeach()
    list(APPEND PSIOPT_DEP_INCLUDES
        "${CMAKE_CURRENT_SOURCE_DIR}/../dep/eigen"
        "${CMAKE_CURRENT_SOURCE_DIR}/../dep/fmt/include")

    add_definitions(-DEIGEN_INITIALIZE_MATRICES_BY_ZERO)
    add_definitions(-DEIGEN_DONT_PARALLELIZE)
    add_definitions(-DTYCHO_DEFAULT_QP_THREADS=8)
    add_definitions(-DTYCHO_DEFAULT_ARENA_SIZE=256)
    add_definitions(-DFMT_HEADER_ONLY)
    add_definitions(-DFMT_USE_LOCALE=0)
    if(WIN32)
        add_definitions(-D_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
    endif()

    enable_testing()
endif()

################################################################################
# Targets

add_library(utils STATIC
    src/utils/color_text.h
    src/utils/color_text.cpp
    src/utils/get_core_count.cpp
    src/utils/memory_management.cpp
    src/utils/thread_pool.cpp
)
add_library(psiopt::utils ALIAS utils)
# fmt is header-only (FMT_HEADER_ONLY is defined project-wide); its include
# path arrives with the rest of the dependency includes below. fmt::fmt-header-only
# is deliberately not linked — that would drag the fmt target into tycho's export set.

add_library(psiopt STATIC
    src/psiopt.cpp
    src/psiopt_globalization.cpp
    src/psiopt_print.cpp
    src/psiopt_settings.cpp
    src/non_linear_program.cpp
    src/solver_init.cpp
)
add_library(psiopt::psiopt ALIAS psiopt)
target_link_libraries(psiopt PRIVATE utils)

# Alternative NLP solver backend. Exactly one of the two implementations of the
# backend entry points is compiled in: the real adapter when Ipopt is linked,
# the fallback that reports the missing build support otherwise.
#
# The Ipopt imported target is wrapped in BUILD_INTERFACE because psiopt is part
# of the installed export set: a PRIVATE dependency on a static library is still
# recorded (as $<LINK_ONLY:...>) in its interface, and install(EXPORT) rejects an
# imported target that is not itself exported. Build-tree consumers still link
# Ipopt transitively; the include directories are attached explicitly so they do
# not depend on that wrapping.
if(ENABLE_IPOPT)
    target_sources(psiopt PRIVATE src/ipopt_tnlp_adapter.cpp)
    target_compile_definitions(psiopt PRIVATE TYCHO_ENABLE_IPOPT)
    target_include_directories(psiopt PRIVATE ${IPOPT_INCLUDE_DIRS})
    target_link_libraries(psiopt PRIVATE $<BUILD_INTERFACE:PkgConfig::IPOPT>)
else()
    target_sources(psiopt PRIVATE src/ipopt_backend_stub.cpp)
endif()

if(APPLE)
    target_link_libraries(psiopt PUBLIC AccelerateSparse::AccelerateSparse)
endif()

foreach(_t utils psiopt)
    # The public header root travels with the target, so a consumer that links
    # psiopt::psiopt needs no include path of its own.
    target_include_directories(${_t} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    # src/ and src/utils/ carry the library's private headers (fmtlib.h,
    # color_text.h), which are included unqualified.
    target_include_directories(${_t} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
        ${PSIOPT_DEP_INCLUDES}
        ${INCLUDE_DIRS}
    )
    target_compile_options(${_t} PRIVATE ${COMPILE_FLAGS})
endforeach()

################################################################################
# Install
#
# When built inside Tycho, psiopt and utils are contributed to the tychoTargets
# export set by src/CMakeLists.txt, alongside the rest of the core libraries;
# only the headers are installed from here. Standalone packaging (a psioptConfig
# package of its own) is a later phase.

install(DIRECTORY include/tycho
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)

if(PSIOPT_STANDALONE)
    install(TARGETS psiopt utils
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
endif()

################################################################################
# Tests — the solver-core slice of the C++ suite (populated in a later step of
# the standalone split; absent for now).

if(PSIOPT_STANDALONE OR BUILD_CPP_TESTS)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
        add_subdirectory(tests)
    endif()
endif()
