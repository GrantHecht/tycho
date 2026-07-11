#pragma once

#ifndef EIGEN_INITIALIZE_MATRICES_BY_ZERO
#error "tycho requires -DEIGEN_INITIALIZE_MATRICES_BY_ZERO (set automatically by the CMake build; \
see CMakeLists.txt). Several VectorFunction/interp/defect code paths rely on zero-initialized \
Eigen temporaries and are undefined behavior without it."
#endif

// Tycho — High-performance trajectory design and optimal control
// Public API umbrella header

// fmt color/formatting support required by detail headers
#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/format.h>

#include "tycho/typedefs.h"
#include "tycho/utils.h"
#include "tycho/math.h"
#include "tycho/vector_functions.h"
#include "tycho/integrators.h"
#include "tycho/optimal_control.h"
#include "tycho/solvers.h"
#include "tycho/astro.h"
