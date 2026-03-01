# Add Bindings/ itself to the include path so FunctionRegistry.h resolves bare
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Ensure nanobind-static exists before pch_bindings uses it
nanobind_build_library(nanobind-static)

# Separate PCH compiled WITH TYCHO_PYTHON_BINDINGS so that nanobind headers
# are baked into the precompiled header used by _tycho and its binding sources.
add_library(pch_bindings STATIC ${CMAKE_SOURCE_DIR}/src/pch.cpp)
target_precompile_headers(pch_bindings PRIVATE
    ${CMAKE_SOURCE_DIR}/src/pch.h
    ${CMAKE_SOURCE_DIR}/src/VectorFunctions/Tycho_VectorFunctions.h
)
target_compile_definitions(pch_bindings PRIVATE TYCHO_PYTHON_BINDINGS)
target_include_directories(pch_bindings PRIVATE ${INCLUDE_DIRS})
target_compile_options(pch_bindings PUBLIC ${COMPILE_FLAGS})
target_link_libraries(pch_bindings PRIVATE nanobind-static)

nanobind_add_module(_tycho
    TychoModule.cpp
    # VectorFunctions bindings
    Tycho_VectorFunctions.cpp
    VectorFunctionBuildPart1.cpp
    VectorFunctionBuildPart2.cpp
    ArgsSegBuildPart1.cpp
    ArgsSegBuildPart2.cpp
    ArgsSegBuildPart3.cpp
    ArgsSegBuildPart4.cpp
    ArgsSegBuildPart5.cpp
    BulkOperationsBuild.cpp
    FreeFunctionsBuild.cpp
    MatrixFunctionBuild.cpp
    PythonArgParsing.cpp
    # OptimalControl bindings
    Tycho_OptimalControl.cpp
    GenericODESBuildPart1.cpp
    GenericODESBuildPart2.cpp
    GenericODESBuildPart3.cpp
    GenericODESBuildPart4.cpp
    GenericODESBuildPart5.cpp
    GenericODESBuildPart6.cpp
    ODEPhaseBaseBind.cpp
    OptimalControlProblemBind.cpp
    LGLInterpTableBind.cpp
    # Solvers bindings
    OptimizationProblemBase.cpp
    PSIOPTBind.cpp
    OptimizationProblemBind.cpp
    # Astro bindings
    Tycho_Astro.cpp
    KeplerUtils.cpp
    KeplerModel.cpp
    LambertSolversBind.cpp
    # Utils bindings
    MemoryManagerBind.cpp
)

target_compile_options(_tycho PUBLIC ${COMPILE_FLAGS})
target_compile_definitions(_tycho PUBLIC TYCHO_PYTHON_BINDINGS)
target_link_libraries(_tycho PUBLIC psiopt)
target_link_libraries(_tycho PUBLIC utils)
target_link_libraries(_tycho PUBLIC optimalcontrol)
target_link_libraries(_tycho PUBLIC vectorfunctions)
target_link_libraries(_tycho PUBLIC astro)
target_link_libraries(_tycho PUBLIC tycho_extensions)
target_include_directories(_tycho PRIVATE ${INCLUDE_DIRS})
target_precompile_headers(_tycho REUSE_FROM pch_bindings)

if(UNIX)
    target_link_options(_tycho PRIVATE ${LINKER_FLAGS})
endif()

add_custom_target(pytychosrc ALL DEPENDS ${CMAKE_SOURCE_DIR}/tycho)

if(PYTHON_LOCAL_INSTALL_DIR)
    add_custom_command(TARGET _tycho POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_tycho> ${PYTHON_LOCAL_INSTALL_DIR}/
        COMMENT "Copied _tycho extension to ${PYTHON_LOCAL_INSTALL_DIR}"
    )
    message(STATUS "Cleaning tycho install directory")

    add_custom_command(TARGET pytychosrc POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E rm -rf ${PYTHON_LOCAL_INSTALL_DIR}/tycho
                    COMMENT "Cleaning tycho install directory")

    add_custom_command(TARGET pytychosrc POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_directory  ${CMAKE_SOURCE_DIR}/tycho
                                                ${PYTHON_LOCAL_INSTALL_DIR}/tycho
                    DEPENDS ${CMAKE_SOURCE_DIR}/tycho
                    COMMENT "Copied tycho to ${PYTHON_LOCAL_INSTALL_DIR}")
endif()
