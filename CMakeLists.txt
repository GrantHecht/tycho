cmake_minimum_required(VERSION 3.16)
project(Tycho VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set( CMAKE_MESSAGE_LOG_LEVEL "STATUS" )
################################################################################
################## Includes ####################################################
################################################################################

include(cmake/git-submodule-init.cmake)
include(cmake/git-submodule-update.cmake)


################################################################################
############### Warnings and Errors ############################################
################################################################################

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt.")
endif()

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
  message(FATAL_ERROR "You cannot build Tycho with the Microsoft Compiler. Please use Clang or GCC and try again.")
endif()

################################################################################
############### Build Settings #################################################
################################################################################


set(LINK_TIME_OPT FALSE)          ### DO LTO - Recommended for full release
set(CLANG_MAX_INLINE_DEPTH 225)
#set(BUILD_TYCHO_WHEEL TRUE)      ### CI Build for generic wheels
#set(BUILD_SPHINX_DOCS TRUE)

# Option to enable strict IEEE 754 compliance for proper NaN handling
# When OFF (default): Uses fast-math optimizations that may ignore NaNs
# When ON: Enforces strict IEEE 754 compliance, treating NaNs properly
option(STRICT_IEEE_COMPLIANCE "Enable strict IEEE 754 compliance for proper NaN handling" OFF)
option(ENABLE_PYTHON_BINDINGS "Build Python bindings (_tycho pybind11 module)" ON)
option(BUILD_CPP_EXAMPLES "Build standalone C++ example executables" ON)


################################################################################
##########################  Set Compiler Flags  ################################
################################################################################

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



# Release Flags
list(APPEND RELEASE_FLAGS "-O2")


## Generic Binary Flags
if(BUILD_TYCHO_WHEEL)
    # Detect architecture for wheel builds
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
        # ARM64 (Apple Silicon, ARM servers) - use NEON
        list(APPEND RELEASE_FLAGS "-march=armv8-a+simd")
        message(STATUS "Building ARM64 wheel with NEON support")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        # x86-64 (Intel/AMD) - use AVX2 (x86-64-v3 flags written out for older compilers)
        list(APPEND RELEASE_FLAGS "-mcx16")
        list(APPEND RELEASE_FLAGS "-mpopcnt")
        list(APPEND RELEASE_FLAGS "-msse3")
        list(APPEND RELEASE_FLAGS "-msse4.1")
        list(APPEND RELEASE_FLAGS "-msse4.2")
        list(APPEND RELEASE_FLAGS "-mssse3")
        list(APPEND RELEASE_FLAGS "-mavx")
        list(APPEND RELEASE_FLAGS "-mavx2")
        list(APPEND RELEASE_FLAGS "-mbmi")
        list(APPEND RELEASE_FLAGS "-mbmi2")
        list(APPEND RELEASE_FLAGS "-mf16c")
        list(APPEND RELEASE_FLAGS "-mfma")
        list(APPEND RELEASE_FLAGS "-mlzcnt")
        list(APPEND RELEASE_FLAGS "-mmovbe")
        list(APPEND RELEASE_FLAGS "-mxsave")
        message(STATUS "Building x86-64 wheel with AVX2 support")
    else()
        # Fallback for other architectures
        list(APPEND RELEASE_FLAGS "-march=native")
        message(WARNING "Unknown architecture ${CMAKE_SYSTEM_PROCESSOR}, using -march=native")
    endif()

    # Windows runs out of ram with LINK_TIME_OPT on gh-actions
    if(NOT WIN32)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
            list(APPEND RELEASE_FLAGS "-mtune=skylake")
        endif()
    endif()
    set(LINK_TIME_OPT TRUE)   ### DO LTO - Recommended for full release
    set(CLANG_MAX_INLINE_DEPTH 400)
    
else()
    list(APPEND RELEASE_FLAGS "-march=native")
    if(NOT WIN32)
        list(APPEND RELEASE_FLAGS "-mtune=native")
    endif()
endif()


if(WIN32)
     list(APPEND RELEASE_FLAGS "/MD")
     list(APPEND RELEASE_FLAGS "/DNDEBUG")
     list(APPEND RELEASE_FLAGS "/O2")
     list(APPEND RELEASE_FLAGS "/Ob2")
     list(APPEND RELEASE_FLAGS "/arch:AVX2")
     # Floating-point handling: strict IEEE 754 compliance or fast-math
     if(STRICT_IEEE_COMPLIANCE)
         list(APPEND RELEASE_FLAGS "/fp:strict")
     else()
         list(APPEND RELEASE_FLAGS "/fp:fast")
     endif()
     list(APPEND RELEASE_FLAGS "/GS-")
     list(APPEND RELEASE_FLAGS "/Gw")
endif()

if(NOT WIN32)
    # Debugging Flags
    set(DEBUG_FLAGS "-g" "-ggdb3")
    # Common Flags
    list(APPEND COMMON_FLAGS "-pthread")
endif()
 

if(LINK_TIME_OPT)
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    if(WIN32)
        ## clang-cl flag
        if(BUILD_TYCHO_WHEEL)
            ## Do full LTO for CI builds since it is single threaded anyway
            list(APPEND RELEASE_FLAGS "-flto=full")
            list(APPEND RELEASE_FLAGS "/opt:lldltojobs=1")
        else()
            list(APPEND RELEASE_FLAGS "-flto=thin")
            list(APPEND RELEASE_FLAGS "/opt:lldltojobs=8")
        endif()
    else()
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    endif()

  elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    set(CMAKE_CXX_COMPILE_OPTIONS_IPO ${CMAKE_CXX_COMPILE_OPTIONS_IPO} -flto=1)
  endif()

else()
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
endif()



list(APPEND COMMON_FLAGS "$<$<STREQUAL:${CMAKE_CXX_COMPILER_ID},Clang>:-mllvm>")
list(APPEND COMMON_FLAGS "$<$<STREQUAL:${CMAKE_CXX_COMPILER_ID},Clang>:-inline-threshold=${CLANG_MAX_INLINE_DEPTH}>")



if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" AND FALSE)
    list(APPEND RELEASE_FLAGS "-finline-limit=1000")
    list(APPEND RELEASE_FLAGS "--param=inline-unit-growth=500")    #40,40
    list(APPEND RELEASE_FLAGS "--param=early-inlining-insns=50")   #14,14
    list(APPEND RELEASE_FLAGS "--param=large-function-insns=3500") #2700,2700
    list(APPEND RELEASE_FLAGS "--param=large-function-growth=120") #100,100
    list(APPEND RELEASE_FLAGS "--param=inline-min-speedup=10")     #30,15
endif()


if(NOT WIN32)
  list(APPEND RELEASE_FLAGS "-fomit-frame-pointer")
  list(APPEND RELEASE_FLAGS "-fno-stack-protector")
  #list(APPEND RELEASE_FLAGS "-fcf-protection=none")
  list(APPEND RELEASE_FLAGS "-fno-asynchronous-unwind-tables")
  
  # Floating-point handling: strict IEEE 754 compliance or fast-math
  if(STRICT_IEEE_COMPLIANCE)
      # Enforce proper NaN handling with selective optimizations
      list(APPEND RELEASE_FLAGS "-fno-math-errno")
      list(APPEND RELEASE_FLAGS "-fno-trapping-math")
      list(APPEND RELEASE_FLAGS "-fno-signaling-nans")
      # Ensure NaN handling is correct (no finite-math-only)
      list(APPEND RELEASE_FLAGS "-fno-finite-math-only")
  else()
      # Use fast-math for performance (may not handle NaNs properly)
      list(APPEND RELEASE_FLAGS "-ffast-math")
  endif()
endif()
 

# Combine Flags
set(COMPILE_FLAGS PUBLIC ${COMMON_FLAGS})
list(APPEND COMPILE_FLAGS "$<$<OR:$<CONFIG:RELEASE>,$<CONFIG:RELWITHDEBINFO>>:${RELEASE_FLAGS}>")
list(APPEND COMPILE_FLAGS "$<$<OR:$<CONFIG:DEBUG>,$<CONFIG:RELWITHDEBINFO>>:${DEBUG_FLAGS}>")



################################################################################
################## TYCHO PREPROCCESSOR MACROS  #################################
################################################################################

add_definitions(-DEIGEN_INITIALIZE_MATRICES_BY_ZERO) ##NEVER EVER GET RID OF THIS EVER
add_definitions(-DEIGEN_DONT_PARALLELIZE) ##NEVER EVER GET RID OF THIS EVER


add_definitions(-DTYCHO_DEFAULT_QP_THREADS=8) ## no spaces between = !!!! otherwise it is set to one
add_definitions(-DTYCHO_DEFAULT_FUNC_THREADS=16) ## no spaces between = !!!! otherwise it is set to one

###  initial size of asset arena memory allocator in number of doubles/superscalars
add_definitions(-DTYCHO_DEFAULT_ARENA_SIZE=256) ## no spaces between = !!!! otherwise it is set to one

### Comment out to roll back to old memory allocator. We will get rid of this once confident in arena
add_definitions(-DTYCHO_MEMORYMAN)

add_definitions(-DTYCHO_COMPILERVERSION="${CMAKE_CXX_COMPILER_VERSION}")
add_definitions(-DTYCHO_VERSIONSTRING="${CMAKE_PROJECT_VERSION}")
add_definitions(-DTYCHO_COMPILERSTRING="${CMAKE_CXX_COMPILER_ID}")
add_definitions(-DTYCHO_COMPILERVERSION="${CMAKE_CXX_COMPILER_VERSION}")
add_definitions(-DTYCHO_OS="${CMAKE_SYSTEM_NAME}")
add_definitions(-DTYCHO_OSVERSION="${CMAKE_SYSTEM_VERSION}")
add_definitions(-DFMT_HEADER_ONLY)

if(WIN32)
## See : https://developercommunity.visualstudio.com/t/Access-violation-in-_Thrd_yield-after-up/10664660#T-N10669129-N10678728
add_definitions(-D_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif()



################################################################################
###################### Find and set our dependencies ###########################
################################################################################

#if(APPLE)
#  include(FindPythonEnv)
#endif()

# Set up submodule dependencies
include_directories(dep/eigen)
include_directories(dep/fmt/include)
include_directories(dep/autodiff)

if(ENABLE_PYTHON_BINDINGS)
    find_package(Python ${PYVERSION_EXACT} COMPONENTS Interpreter Development REQUIRED)
    add_subdirectory(dep)
endif()

# Set up external dependencies
find_package(Threads REQUIRED)

if(UNIX AND NOT APPLE)
  # find_package(OpenMP REQUIRED)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    list(APPEND COMPILE_FLAGS "-fopenmp=libiomp5")
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GCC")
    list(APPEND COMPILE_FLAGS "-fopenmp")
  endif()
elseif(APPLE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
      # fina_package(OpenMP REQUIRED) will fail with AppleClang so 
      # we need this to point towards the homebrew libomp
      include_directories("/opt/homebrew/Cellar/libomp/22.1.0/include/")
      link_directories("/opt/homebrew/Cellar/libomp/22.1.0/lib/")
      list(APPEND COMPILE_FLAGS "-Xclang")
      list(APPEND COMPILE_FLAGS "-fopenmp")
      list(APPEND OpenMP_CXX_FLAGS "-lomp")
    else()
      # With homebrew LLVM clang compiler on Mac, this seems to work well for 
      # finding homebrew OpenMP 
      find_package(OpenMP REQUIRED)

      list(APPEND COMPILE_FLAGS "-fopenmp")
      list(APPEND OpenMP_CXX_FLAGS "-lomp")
      
      # Add flags to make LLVM Clang more compatible with Apple SDK headers
      list(APPEND COMPILE_FLAGS "-Wno-elaborated-enum-base")
    endif()
  endif()
endif()

if(APPLE)
  find_package(AccelerateSparse REQUIRED)
  add_compile_definitions(USE_ACCELERATE_SPARSE)
else()
  find_package(MKL REQUIRED)
endif()

if(ENABLE_PYTHON_BINDINGS)
    find_package(Python ${PYVERSION_EXACT} REQUIRED COMPONENTS Interpreter Development)
endif()

# Set dependency variables
set(INCLUDE_DIRS ${MKL_INCLUDE_DIRS} ${AccelerateSparse_INCLUDE_DIRS})
set(LINK_LIBS ${MKL_LIBRARIES} Threads::Threads ${CMAKE_DL_LIBS} ${AccelerateSparse_LIBRARIES})

# Handle OpenMP for Apple
if (APPLE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
    list(APPEND LINK_LIBS ${OpenMP_CXX_FLAGS})
  else()
    list(APPEND LINK_LIBS OpenMP::OpenMP_CXX)
  endif()
endif()

if(ENABLE_PYTHON_BINDINGS)
    # Handle Python executable variable
    if (NOT DEFINED PYTHON_EXECUTABLE)
        set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})
    endif()
    if(DEFINED Python_INCLUDE_DIRS)
        list(APPEND INCLUDE_DIRS ${Python_INCLUDE_DIRS})
    elseif(DEFINED PYTHON_INCLUDE_DIRS)
        list(APPEND INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS})
    endif()
    if (NOT APPLE)
        # Don't directly link with python library on mac and instead rely on -undefined dynamic_lookup
        # to bind to the python symbols at runtime from the interpreter
        if(DEFINED Python_LIBRARIES)
            list(APPEND LINK_LIBS ${Python_LIBRARIES})
        elseif(DEFINED PYTHON_LIBRARIES)
            list(APPEND LINK_LIBS ${PYTHON_LIBRARIES})
        endif()
    endif()

endif()

if(UNIX)
  list(APPEND LINK_LIBS m)
endif()



################################################################################
################ Linker Flags ##################################################


if(BUILD_SHARED_LIBS AND NOT APPLE AND NOT WIN32)  # Windows and Linux Dynamic Linking
  list(APPEND LINKER_FLAGS "${OpenMP_CXX_FLAGS} -Wl,--no-undefined -Wl,--start-group ${MKL_LIBRARIES_LIST} -Wl, --end-group")
elseif(UNIX AND NOT APPLE)  # Linux Static Linking
  list(APPEND LINKER_FLAGS "${OpenMP_CXX_FLAGS} -Wl,--no-undefined -Wl,--start-group ${MKL_LIBRARIES_LIST} -Wl,--end-group")
elseif(APPLE)  # Apple Dynamic and Static Linking
  list(APPEND LINKER_FLAGS ${OpenMP_CXX_FLAGS})
  list(APPEND LINKER_FLAGS "-Wl,-bind_at_load")
  list(APPEND LINKER_FLAGS "-Wl,-undefined,dynamic_lookup")
else()  # Windows Static Linking
  list(APPEND LINKER_FLAGS "${OpenMP_CXX_FLAGS}")
endif()

###########################################################
######## Find Correct Python Install Location  ############

if(ENABLE_PYTHON_BINDINGS)
    if(NOT PIP_INSTALL)
      if(NOT PYTHON_LOCAL_INSTALL_DIR)
        if(WIN32)
          set(PYTHON_LOCAL_INSTALL_DIR ${Python_SITEARCH})
        else()
          execute_process ( COMMAND ${PYTHON_EXECUTABLE} -m site --user-site
          OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
          set(PYTHON_LOCAL_INSTALL_DIR ${PYTHON_SITE_PACKAGES})
        endif()
      endif()

      if(PYTHON_LOCAL_INSTALL_DIR)
        message(STATUS "Tycho will be installed to " ${PYTHON_LOCAL_INSTALL_DIR})
      else()
        message(STATUS "Could not find appropriate install directory")
      endif()
    endif()
endif()

message(STATUS "CMAKE DIR LOC" ${CMAKE_CURRENT_SOURCE_DIR})
message(STATUS ${OpenMP_CXX_FLAGS})
message(STATUS ${OpenMP_CXX_LIBRARIES})
message(STATUS ${PYTHON_EXECUTABLE})

# Display IEEE compliance setting
if(STRICT_IEEE_COMPLIANCE)
    message(STATUS "IEEE 754 Compliance: STRICT (NaNs will be properly handled)")
else()
    message(STATUS "IEEE 754 Compliance: FAST-MATH (NaNs may not be properly handled)")
endif()

################################################################################
# Code Directories
add_subdirectory(src)

if(ENABLE_PYTHON_BINDINGS)
    add_subdirectory(extensions)
endif()


# Documentation
# READ: pip install sphinx,breathe,furo before trying to build docs

if(BUILD_SPHINX_DOCS)
  add_subdirectory(doc)
endif()

if(ENABLE_PYTHON_BINDINGS)
    add_subdirectory(pypiwheel)
endif()

if(BUILD_CPP_EXAMPLES)
    add_subdirectory(examples/cpp_examples)
endif()

# Formatting
include(cmake/clang-format.cmake)
