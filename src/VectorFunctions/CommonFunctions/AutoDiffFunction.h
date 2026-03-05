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
//   - pybind11 / pybind11 header references removed
// =============================================================================

#pragma once

#include "VectorFunction.h"

namespace Tycho {

template <class Func>
struct ADFun : VectorFunction<ADFun<Func>, Func::IRC, Func::ORC, FDiffCentArray, FDiffFwd> {
    using Base = VectorFunction<ADFun<Func>, Func::IRC, Func::ORC, FDiffCentArray, FDiffFwd>;
    DENSE_FUNCTION_BASE_TYPES(Base)

    Func func;
    ADFun(Func f) : func(std::move(f)) { this->setIORows(this->func.IRows(), this->func.ORows()); }
    template <class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
        this->func.compute(x, fx_);
    }

    void test() {
        Input<double> x;
        x.setRandom();
        return TestDerivs(x);
    };
    void test(Input<double> x) { return TestDerivs(x); };
    void TestDerivs(Input<double> x) {
        Output<double> l;
        l.setOnes();

        std::cout << "Jacobian Error" << std::endl;
        std::cout << this->jacobian(x) - this->func.jacobian(x) << std::endl << std::endl;
        ;
        std::cout << "Hessian Error" << std::endl;
        std::cout << (this->adjointhessian(x, l) - this->func.adjointhessian(x, l)) << std::endl
                  << std::endl;
        ;
    }
};

} // namespace Tycho
