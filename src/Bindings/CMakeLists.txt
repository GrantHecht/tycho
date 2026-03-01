# Add Bindings/ itself to the include path so FunctionRegistry.h resolves bare
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

pybind11_add_module(_tycho
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
    # Solvers bindings
    OptimizationProblemBase.cpp
    # Astro bindings
    Tycho_Astro.cpp
    KeplerUtils.cpp
    KeplerModel.cpp
)
# Note: MemoryManager::Build() is defined in Utils/MemoryManagement.cpp (guarded by
# #ifdef TYCHO_PYTHON_BINDINGS) and pulled in via the 'utils' static library link below.

target_compile_options(_tycho PUBLIC ${COMPILE_FLAGS})
target_compile_definitions(_tycho PUBLIC TYCHO_PYTHON_BINDINGS)
target_link_libraries(_tycho PUBLIC psiopt)
target_link_libraries(_tycho PUBLIC utils)
target_link_libraries(_tycho PUBLIC optimalcontrol)
target_link_libraries(_tycho PUBLIC vectorfunctions)
target_link_libraries(_tycho PUBLIC astro)
target_link_libraries(_tycho PUBLIC tycho_extensions)
target_include_directories(_tycho PRIVATE ${INCLUDE_DIRS})
target_precompile_headers(_tycho REUSE_FROM pch)

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
