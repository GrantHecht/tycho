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
//   - Namespace renamed: asset -> Tycho
//   - Python binding methods (Build(py::module)) moved to src/Bindings/ (PR 2)
//   - pybind11 header references removed
// =============================================================================

#pragma once

#include "tycho/detail/vf/core/computable_base.h"

namespace tycho::vf {

template <class Derived, int IR, int OR> struct Computable : ComputableBase<Derived, IR, OR> {
    using Base = ComputableBase<Derived, IR, OR>;
    template <class Scalar> using Output = typename Base::template Output<Scalar>;
    template <class Scalar> using Input = typename Base::template Input<Scalar>;
    template <class Scalar> using Gradient = typename Base::template Gradient<Scalar>;

    template <class Scalar> using ConstVectorBaseRef = const Eigen::MatrixBase<Scalar> &;
    template <class Scalar> using VectorBaseRef = Eigen::MatrixBase<Scalar> &;
};

///// Scalar Specialization
template <class Derived, int IR>
struct Computable<Derived, IR, 1> : ComputableBase<Derived, IR, 1> {
    using Base = ComputableBase<Derived, IR, 1>;

    template <class Scalar> using Output = typename Base::template Output<Scalar>;
    template <class Scalar> using Input = typename Base::template Input<Scalar>;
    template <class Scalar> using Gradient = typename Base::template Gradient<Scalar>;

    template <class Scalar> using ConstVectorBaseRef = const Eigen::MatrixBase<Scalar> &;
    template <class Scalar> using VectorBaseRef = Eigen::MatrixBase<Scalar> &;
};

} // namespace tycho::vf
