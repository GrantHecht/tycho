// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Defines the type erasure spec SizableSpec defining the ability to query
// the Input/Output rows of a type-erased vectorfunction as well as its name
// and thread safety.
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> Tycho
//   - Python binding methods (Build(py::module)) moved to src/Bindings/ (PR 2)
//   - pybind11 / pybind11 header references removed
//   - PR 9: Removed dead Model<>/ExternalInterface<> boilerplate
// =============================================================================

#pragma once

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

#include "tycho/detail/eigen_types.h"
#include "tycho/detail/std_extensions.h"
#include "tycho/detail/math_functions.h"
#include "tycho/detail/type_name.h"
#include "tycho/detail/type_storage.h"
#include "tycho/detail/sizing_helpers.h"
#include "tycho/detail/thread_pool.h"
#include "tycho/detail/flat_map.h"
#include "tycho/detail/function_return_type.h"
#include "tycho/detail/get_core_count.h"
#include "tycho/detail/crtp_base.h"

namespace Tycho {

struct SizableSpec {
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct Concept { // abstract base class for model.
        virtual ~Concept() = default;
        // Your (internal) interface goes here.
        virtual std::string name() const = 0;

        virtual int IRows() const = 0;
        virtual int ORows() const = 0;
        virtual bool thread_safe() const = 0;
    };
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
};

} // namespace Tycho
