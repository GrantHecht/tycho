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

#include "tycho/detail/vf/core/vector_function.h"

namespace tycho::vf {

template <class Func>
struct ADFun : VectorFunction<ADFun<Func>, Func::IRC, Func::ORC,
                              DenseDerivativeMode::FDiffCentArray, DenseDerivativeMode::FDiffFwd> {
    using Base = VectorFunction<ADFun<Func>, Func::IRC, Func::ORC,
                                DenseDerivativeMode::FDiffCentArray, DenseDerivativeMode::FDiffFwd>;
    DENSE_FUNCTION_BASE_TYPES(Base)

    Func func;
    ADFun(Func f) : func(std::move(f)) {
        this->set_io_rows(this->func.input_rows(), this->func.output_rows());
    }
    template <class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
        this->func.compute(x, fx_);
    }

    void test() {
        Input<double> x;
        x.setRandom();
        return test_derivs(x);
    };
    void test(Input<double> x) { return test_derivs(x); };
    void test_derivs(Input<double> x) {
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

} // namespace tycho::vf
