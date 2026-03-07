# Add Bindings/ and each subdirectory to the include path so headers resolve
# with bare names regardless of which subdirectory the including file lives in.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/VectorFunctions)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/OptimalControl)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Solvers)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Astro)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Integrators)

# Ensure nanobind-static exists before pch_bindings uses it
nanobind_build_library(nanobind-static)

# Separate PCH compiled WITH TYCHO_PYTHON_BINDINGS so that nanobind headers
# are baked into the precompiled header used by _tycho and its binding sources.
add_library(pch_bindings STATIC ${CMAKE_SOURCE_DIR}/src/pch.cpp)
target_precompile_headers(pch_bindings PRIVATE
    ${CMAKE_SOURCE_DIR}/src/pch.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pch_nb.h
    ${CMAKE_SOURCE_DIR}/src/VectorFunctions/Tycho_VectorFunctions.h
    ${CMAKE_CURRENT_SOURCE_DIR}/VectorFunctionsBind.h
)
target_compile_definitions(pch_bindings PRIVATE TYCHO_PYTHON_BINDINGS)
target_include_directories(pch_bindings PRIVATE ${INCLUDE_DIRS})
target_compile_options(pch_bindings PUBLIC ${BINDING_COMPILE_FLAGS})
target_link_libraries(pch_bindings PRIVATE nanobind-static)

nanobind_add_module(_tycho
    TychoModule.cpp
    # Utils bindings
    UtilsBind.cpp
    MemoryManagerBind.cpp
    # VectorFunctions bindings
    VectorFunctions/Tycho_VectorFunctions.cpp
    VectorFunctions/VectorFunctionBuildPart1.cpp
    VectorFunctions/VectorFunctionBuildPart2.cpp
    VectorFunctions/ArgsSegBuildPart1.cpp
    VectorFunctions/ArgsSegBuildPart2.cpp
    VectorFunctions/ArgsSegBuildPart3.cpp
    VectorFunctions/ArgsSegBuildPart4.cpp
    VectorFunctions/ArgsSegBuildPart5.cpp
    VectorFunctions/BulkOperationsBuild.cpp
    VectorFunctions/FreeFunctionsBuild.cpp
    VectorFunctions/MatrixFunctionBuild.cpp
    VectorFunctions/PythonArgParsing.cpp
    # OptimalControl bindings
    OptimalControl/Tycho_OptimalControl.cpp
    OptimalControl/GenericODESBuildPart1.cpp
    OptimalControl/GenericODESBuildPart2.cpp
    OptimalControl/GenericODESBuildPart3.cpp
    OptimalControl/GenericODESBuildPart4.cpp
    OptimalControl/GenericODESBuildPart5.cpp
    OptimalControl/GenericODESBuildPart6.cpp
    OptimalControl/ODEPhaseBaseBind.cpp
    OptimalControl/OptimalControlProblemBind.cpp
    OptimalControl/LGLInterpTableBind.cpp
    OptimalControl/MeshIterateInfoBind.cpp
    # Solvers bindings
    Solvers/Tycho_Solvers.cpp
    Solvers/OptimizationProblemBase.cpp
    Solvers/PSIOPTBind.cpp
    Solvers/OptimizationProblemBind.cpp
    Solvers/JetBind.cpp
    # Astro bindings
    Astro/Tycho_Astro.cpp
    Astro/KeplerUtils.cpp
    Astro/KeplerModel.cpp
    Astro/LambertSolversBind.cpp
    # Integrators bindings
    Integrators/RKCoeffsBind.cpp
)

target_compile_options(_tycho PRIVATE ${BINDING_COMPILE_FLAGS})
target_compile_definitions(_tycho PUBLIC TYCHO_PYTHON_BINDINGS)
target_link_libraries(_tycho PUBLIC psiopt)
target_link_libraries(_tycho PUBLIC utils)
target_link_libraries(_tycho PUBLIC optimalcontrol)
target_link_libraries(_tycho PUBLIC vectorfunctions)
target_link_libraries(_tycho PUBLIC astro)
target_link_libraries(_tycho PUBLIC tycho_extensions_bindings)
target_include_directories(_tycho PRIVATE ${INCLUDE_DIRS})
target_precompile_headers(_tycho REUSE_FROM pch_bindings)

if(UNIX)
    target_link_options(_tycho PRIVATE ${LINKER_FLAGS})
elseif(WIN32)
    target_link_libraries(_tycho PUBLIC ${LINK_LIBS})
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
