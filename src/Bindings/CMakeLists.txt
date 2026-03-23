set(_TYCHO_BINDINGS_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/VectorFunctions
    ${CMAKE_CURRENT_SOURCE_DIR}/OptimalControl
    ${CMAKE_CURRENT_SOURCE_DIR}/Solvers
    ${CMAKE_CURRENT_SOURCE_DIR}/Astro
    ${CMAKE_CURRENT_SOURCE_DIR}/Integrators
)

# Ensure nanobind-static exists before pch_bindings uses it
nanobind_build_library(nanobind-static)

# Separate PCH compiled WITH TYCHO_PYTHON_BINDINGS so that nanobind headers
# are baked into the precompiled header used by _tychopy and its binding sources.
add_library(pch_bindings STATIC ${CMAKE_SOURCE_DIR}/src/pch.cpp)
target_precompile_headers(pch_bindings PRIVATE
    ${CMAKE_SOURCE_DIR}/src/pch.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pch_nb.h
    ${CMAKE_SOURCE_DIR}/src/VectorFunctions/Tycho_VectorFunctions.h
    ${CMAKE_CURRENT_SOURCE_DIR}/VectorFunctionsBind.h
)
target_compile_definitions(pch_bindings PRIVATE TYCHO_PYTHON_BINDINGS)
target_include_directories(pch_bindings PRIVATE ${_TYCHO_SRC_INCLUDES} ${_TYCHO_BINDINGS_INCLUDES} ${INCLUDE_DIRS})
target_compile_options(pch_bindings PUBLIC ${BINDING_COMPILE_FLAGS})
target_link_libraries(pch_bindings PRIVATE nanobind-static)

nanobind_add_module(_tychopy
    TychoModule.cpp
    # Utils bindings
    UtilsBind.cpp
    BumpAllocatorBind.cpp
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

target_compile_options(_tychopy PRIVATE ${BINDING_COMPILE_FLAGS})
target_compile_definitions(_tychopy PUBLIC TYCHO_PYTHON_BINDINGS)
# Prevent CMake from auto-defining _tychopy_EXPORTS on the module target; the macro
# is absent from the reused pch_bindings PCH, which causes a -Wclang-cl-pch mismatch.
set_target_properties(_tychopy PROPERTIES DEFINE_SYMBOL "")
target_link_libraries(_tychopy PUBLIC tycho tycho_extensions_bindings)
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
