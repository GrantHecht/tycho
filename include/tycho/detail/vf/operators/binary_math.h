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
#include "tycho/detail/vf/core/expression_fwd_declarations.h"

namespace Tycho {

template <class Func1, int IR1, int OR1, class Func2, int IR2, int OR2>
auto crossProduct(const DenseFunctionBase<Func1, IR1, OR1> &f1,
                  const DenseFunctionBase<Func2, IR2, OR2> &f2) {
    return FunctionCrossProduct<Func1, Func2>(f1.derived(), f2.derived());
}

template <class Func1, int IR1, int OR1, class Func2, int IR2, int OR2>
auto quatProduct(const DenseFunctionBase<Func1, IR1, OR1> &f1,
                 const DenseFunctionBase<Func2, IR2, OR2> &f2) {
    return FunctionQuatProduct<Func1, Func2>(f1.derived(), f2.derived());
}

template <class Func1, int IR1, int OR1, class Func2, int IR2, int OR2>
auto dotProduct(const DenseFunctionBase<Func1, IR1, OR1> &f1,
                const DenseFunctionBase<Func2, IR2, OR2> &f2) {
    return FunctionDotProduct<Func1, Func2>(f1.derived(), f2.derived());
}

template <class Func1, int IR1, int OR1, class Func2, int IR2, int OR2>
auto cwiseProduct(const DenseFunctionBase<Func1, IR1, OR1> &f1,
                  const DenseFunctionBase<Func2, IR2, OR2> &f2) {
    return CwiseFunctionProduct<Func1, Func2>(f1.derived(), f2.derived());
}
template <class Func1, int IR1, int OR1, class Func2, int IR2, int OR2>
auto cwiseQuotient(const DenseFunctionBase<Func1, IR1, OR1> &f1,
                   const DenseFunctionBase<Func2, IR2, OR2> &f2) {
    return CwiseFunctionProduct<Func1, CwiseInverse<Func2>>(f1.derived(),
                                                            CwiseInverse<Func2>(f2.derived()));
}

} // namespace Tycho