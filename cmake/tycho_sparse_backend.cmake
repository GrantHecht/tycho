################################################################################
# tycho_resolve_sparse_backend()
#
# Resolves the platform sparse-linear-algebra backend: Apple Accelerate
# (AccelerateSparse) on macOS, Intel MKL everywhere else. Populates
# USE_ACCELERATE_SPARSE and the AccelerateSparse::AccelerateSparse imported
# target on Apple, or the MKL_* variables plus INCLUDE_DIRS/LINK_LIBS
# elsewhere, exactly as the root block this was extracted from.
#
# Implemented as a macro, not a function: find_package(AccelerateSparse) /
# find_package(MKL) create imported targets and set variables that must land
# in the caller's directory scope. A function() would confine any imported
# target it creates to a function-local scope that disappears on return,
# breaking target_link_libraries() calls elsewhere that expect
# AccelerateSparse::AccelerateSparse (or the MKL_LIBRARIES/LINK_LIBS
# variables) to still exist.
#
# Extracted verbatim from the root CMakeLists.txt MKL/Accelerate resolution
# block -- no behavior changes.
################################################################################

macro(tycho_resolve_sparse_backend)

if(APPLE)
  find_package(AccelerateSparse REQUIRED)
  add_compile_definitions(USE_ACCELERATE_SPARSE)
else()
  find_package(MKL)
  if(NOT MKL_FOUND)
      message(FATAL_ERROR
          "Tycho requires Intel MKL. Set MKLROOT to the MKL installation directory.\n"
          "  See CLAUDE.md for installation instructions.")
  endif()
  # Validate that no individual MKL library is NOTFOUND. FindMKL.cmake does not
  # include MKL_OMP_LIBRARY in FIND_PACKAGE_HANDLE_STANDARD_ARGS, so MKL_FOUND
  # can be TRUE while MKL_LIBRARIES contains a NOTFOUND entry (libiomp5).
  foreach(_mkl_lib IN LISTS MKL_LIBRARIES)
      if("${_mkl_lib}" MATCHES "NOTFOUND")
          message(FATAL_ERROR
              "MKL library component not found: ${_mkl_lib}\n"
              "Your MKL installation may be incomplete. "
              "Ensure MKLROOT points to a complete MKL installation.")
      endif()
  endforeach()
endif()

if(ENABLE_PYTHON_BINDINGS)
    find_package(Python ${PYVERSION_EXACT} REQUIRED COMPONENTS Interpreter Development)
endif()

# Set dependency variables
set(INCLUDE_DIRS ${MKL_INCLUDE_DIRS})
# On Linux, static MKL archives have circular dependencies and must be wrapped
# in --start-group/--end-group so the linker rescans them.
if(UNIX AND NOT APPLE)
  set(LINK_LIBS -Wl,--start-group ${MKL_LIBRARIES} -Wl,--end-group Threads::Threads ${CMAKE_DL_LIBS})
else()
  set(LINK_LIBS ${MKL_LIBRARIES} Threads::Threads ${CMAKE_DL_LIBS})
endif()

endmacro()
