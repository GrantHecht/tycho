// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
// =============================================================================

#pragma once

#include "tycho/detail/vf/core/expression_fwd_declarations.h"
#include <algorithm>
#include <array>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <variant>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Sparse>

#include "tycho/detail/typedefs/eigen_types.h"
#include "tycho/detail/utils/crtp_base.h"
#include "tycho/detail/utils/flat_map.h"
#include "tycho/detail/utils/function_return_type.h"
#include "tycho/detail/utils/get_core_count.h"
#include "tycho/detail/utils/math_functions.h"
#include "tycho/detail/utils/sizing_helpers.h"
#include "tycho/detail/utils/std_extensions.h"
#include "tycho/detail/utils/thread_pool.h"
#include "tycho/detail/utils/type_name.h"
#include "tycho/detail/utils/type_storage.h"

#ifdef TYCHO_PYTHON_BINDINGS

namespace tycho {

using namespace tycho::vf;

/*
 * Converts list of python objects into a vector of dynamically sized GenericFunctions.
 * Can accept any of the fundamental types exposed to python as well as Python and Numpy
 * vectors and scalars.
 */
std::vector<GenericFunction<-1, -1>> ParsePythonArgs(nb::args x, int irows = 0);

/*
 * Converts list of python objects into a vector of dynamically sized scalar GenericFunctions.
 * Can accept any of the fundamental scalar types exposed to python as well as Python and Numpy
 * scalars.
 */
std::vector<GenericFunction<-1, 1>> ParsePythonArgsScalar(nb::args x, int irows = 0);

} // namespace tycho

#endif // TYCHO_PYTHON_BINDINGS