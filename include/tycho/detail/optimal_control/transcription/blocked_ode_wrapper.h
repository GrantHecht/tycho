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
#include "tycho/detail/utils/std_extensions.h"
#include "tycho/detail/utils/math_functions.h"
#include "tycho/detail/utils/type_name.h"
#include "tycho/detail/utils/type_storage.h"
#include "tycho/detail/utils/sizing_helpers.h"
#include "tycho/detail/utils/thread_pool.h"
#include "tycho/detail/utils/flat_map.h"
#include "tycho/detail/utils/function_return_type.h"
#include "tycho/detail/utils/get_core_count.h"
#include "tycho/detail/utils/crtp_base.h"

namespace tycho::oc {

// Import cross-namespace types from vf and utils.
using utils::SZ_SUM;
using utils::SZ_MAX;
using utils::SZ_PROD;
using vf::DenseDerivativeMode;
using vf::GenericFunction;
using vf::VectorExpression;
using vf::VectorFunction;
using vf::ThreadingFlags;

template <class DODE> struct Blocked_ODE_Wrapper : DODE {
    static const int UV = 0;
    static const int PV = SZ_SUM<DODE::PV, DODE::UV>::value;
    static const int XtUV = DODE::XtV;
    using Base = DODE;

    inline int XtUVars() const { return Base::XtVars(); }
    inline int UVars() const { return 0; }
    inline int PVars() const { return Base::UVars() + Base::PVars(); }

    Blocked_ODE_Wrapper() {};
    Blocked_ODE_Wrapper(const DODE &ode) : Base(ode) {}
};

} // namespace tycho::oc
