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
//   - PR 9: Replaced rubber_types with tycho::utils::TypeStorage<ConditionalBase<IR>>
// =============================================================================

#pragma once

#include "tycho/detail/vf/common/common_functions.h"
#include "tycho/detail/vf/type_erasure/conditional_base.h"
#include "tycho/detail/vf/type_erasure/generic_conditional.h"
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

namespace tycho::vf {

// GenericComparative<IR> has the same interface as GenericConditional<IR>
// but is a distinct type so that Python bindings register "Comparative"
// and "Conditional" as separate nanobind classes.
template <int IR> struct GenericComparative {
    using InType = Eigen::Ref<const Eigen::Matrix<double, IR, 1>>;

    static const bool is_conditional = true;
    static const int IRC = IR;

    tycho::utils::TypeStorage<ConditionalBase<IR>> storage;

    GenericComparative() = default;

    template <class T> GenericComparative(const T &t) {
        storage.template emplace<ConditionalModel<IR, std::decay_t<T>>>(t);
    }

    GenericComparative(const GenericComparative &) = default;
    GenericComparative(GenericComparative &&) noexcept = default;
    GenericComparative &operator=(const GenericComparative &) = default;
    GenericComparative &operator=(GenericComparative &&) noexcept = default;

    int input_rows() const { return storage.get().input_rows(); }

    template <class InTypeT> bool compute(const Eigen::MatrixBase<InTypeT> &x) const {
        InType xt(x.derived());
        return storage.get().compute(xt);
    }
};

} // namespace tycho::vf
