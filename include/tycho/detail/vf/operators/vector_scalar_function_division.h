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

template <class Derived, class VecFunc, class ScalFunc> struct VectorScalarFunctionDivision_Impl;

template <class VecFunc, class ScalFunc>
struct VectorScalarFunctionDivision
    : VectorScalarFunctionDivision_Impl<VectorScalarFunctionDivision<VecFunc, ScalFunc>, VecFunc,
                                        ScalFunc> {
    using Base = VectorScalarFunctionDivision_Impl<VectorScalarFunctionDivision<VecFunc, ScalFunc>,
                                                   VecFunc, ScalFunc>;
    VF_TYPE_ALIASES(Base);
    using Base::Base;
};

template <class Derived, class VecFunc, class ScalFunc>
struct VectorScalarFunctionDivision_Impl
    : VectorFunction<Derived, SZ_MAX<VecFunc::IRC, ScalFunc::IRC>::value, VecFunc::ORC> {
    using Base = VectorFunction<Derived, SZ_MAX<VecFunc::IRC, ScalFunc::IRC>::value, VecFunc::ORC>;
    VF_TYPE_ALIASES(Base);

    using Base::compute;

    VecFunc vectorfunc;
    ScalFunc scalarfunc;

    using INPUT_DOMAIN =
        CompositeDomain<Base::IRC, typename VecFunc::INPUT_DOMAIN, typename ScalFunc::INPUT_DOMAIN>;
    static constexpr bool is_vectorizable = VecFunc::is_vectorizable && ScalFunc::is_vectorizable;

    VectorScalarFunctionDivision_Impl() {}
    VectorScalarFunctionDivision_Impl(VecFunc f1, ScalFunc f2)
        : vectorfunc(std::move(f1)), scalarfunc(std::move(f2)) {
        int irtemp = std::max(this->vectorfunc.input_rows(), this->scalarfunc.input_rows());
        this->set_io_rows(irtemp, this->vectorfunc.output_rows());

        this->set_input_domain(this->input_rows(),
                               {scalarfunc.input_domain(), vectorfunc.input_domain()});

        if (this->scalarfunc.input_rows() != this->vectorfunc.input_rows()) {
            throw std::invalid_argument(
                fmt::format("VectorScalarFunctionDivision: input size mismatch "
                            "(VectorFunc IRows={}, ScalarFunc IRows={})",
                            this->vectorfunc.input_rows(), this->scalarfunc.input_rows()));
        }
    }

    static constexpr bool vectorfunc_is_segment =
        Is_Segment<VecFunc>::value || Is_ScaledSegment<VecFunc>::value;
    static constexpr bool scalarfunc_is_segment =
        Is_Segment<ScalFunc>::value || Is_ScaledSegment<ScalFunc>::value;
    static constexpr bool is_prod_of_segments = vectorfunc_is_segment && scalarfunc_is_segment;

    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();

        Vector1<Scalar> fxs;
        this->vectorfunc.compute(x, fx);
        this->scalarfunc.compute(x, fxs);

        Scalar ifxs = 1.0 / fxs[0];

        fx *= ifxs;

        // f/g
    }
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();

        auto Impl = [&](auto &fxt, auto &jxs) {
            // f'/g - fg'/g^2

            Vector1<Scalar> fxs;
            this->vectorfunc.compute_jacobian(x, fx, jx);
            this->scalarfunc.compute_jacobian(x, fxs, jxs);

            Scalar ifxs = 1.0 / fxs[0];

            this->vectorfunc.scale_jacobian(jx, ifxs);

            fxt = -fx * (ifxs * ifxs);
            fx *= ifxs;

            this->scalarfunc.right_jacobian_product(jx, fxt, jxs, PlusEqualsAssignment(),
                                                    std::bool_constant<false>());
        };

        const int irows = this->scalarfunc.input_rows();
        const int orows = this->output_rows();

        using FType = Output<Scalar>;
        using JType = typename ScalFunc::template Jacobian<Scalar>;
        tycho::utils::BumpAllocator::allocate_run(Impl, tycho::utils::TempSpec<FType>(orows, 1),
                                                  tycho::utils::TempSpec<JType>(1, irows));
    }
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        CVecRef<InType> x, CVecRef<OutType> fx_, CMatRef<JacType> jx_,
        CVecRef<AdjGradType> adjgrad_, CMatRef<AdjHessType> adjhess_,
        CVecRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();
        VecRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
        MatRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

        //////////////////////////////////////////

        auto Impl = [&](auto &fxt, auto &jxs, auto &gxs, auto &gtmp, auto &hxs) {
            this->vectorfunc.compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, adjgrad,
                                                                             adjhess, adjvars);

            Vector1<Scalar> fxs;
            Vector1<Scalar> ls;
            ls[0] = -fx.dot(adjvars);

            this->scalarfunc.compute_jacobian_adjointgradient_adjointhessian(x, fxs, jxs, gxs, hxs,
                                                                             ls);

            Scalar ifxs = 1.0 / fxs[0];

            this->scalarfunc.scale_hessian(hxs, Scalar(ifxs * ifxs));
            this->scalarfunc.scale_gradient(gxs, Scalar(ifxs * ifxs));

            this->vectorfunc.scale_hessian(adjhess, ifxs);
            this->vectorfunc.scale_gradient(adjgrad, ifxs);

            this->scalarfunc.accumulate_hessian(adjhess, hxs, PlusEqualsAssignment());

            if constexpr (ScalFunc::InputIsDynamic) {
                const int sds = this->scalarfunc.sub_domains.cols();

                if (sds == 0) {
                    hxs.setZero();
                } else {
                    for (int i = 0; i < sds; i++) {
                        int start = this->scalarfunc.sub_domains(0, i);
                        int size = this->scalarfunc.sub_domains(1, i);
                        hxs.middleCols(start, size).setZero();
                    }
                }
            } else {
                constexpr int sds = ScalFunc::INPUT_DOMAIN::sub_domains.size();
                tycho::utils::constexpr_for_loop(
                    std::integral_constant<int, 0>(), std::integral_constant<int, sds>(),
                    [&](auto i) {
                        constexpr int start = ScalFunc::INPUT_DOMAIN::sub_domains[i.value][0];
                        constexpr int size = ScalFunc::INPUT_DOMAIN::sub_domains[i.value][1];
                        hxs.template middleCols<size>(start, size).setZero();
                    });
            }

            gtmp = (gxs + adjgrad) * (-ifxs);

            this->scalarfunc.right_jacobian_product(hxs, gtmp, jxs, DirectAssignment(),
                                                    std::bool_constant<false>());

            if constexpr (ScalFunc::InputIsDynamic) {
                const int sds = this->scalarfunc.sub_domains.cols();

                if (sds == 0) {
                    adjhess += hxs + hxs.transpose();

                } else {
                    for (int i = 0; i < sds; i++) {
                        int start = this->scalarfunc.sub_domains(0, i);
                        int size = this->scalarfunc.sub_domains(1, i);
                        adjhess.middleCols(start, size) += hxs.middleCols(start, size);
                        adjhess.middleRows(start, size) += hxs.middleCols(start, size).transpose();
                    }
                }

            } else {
                constexpr int sds = ScalFunc::INPUT_DOMAIN::sub_domains.size();

                tycho::utils::constexpr_for_loop(
                    std::integral_constant<int, 0>(), std::integral_constant<int, sds>(),
                    [&](auto i) {
                        constexpr int start = ScalFunc::INPUT_DOMAIN::sub_domains[i.value][0];
                        constexpr int size = ScalFunc::INPUT_DOMAIN::sub_domains[i.value][1];
                        adjhess.template middleCols<size>(start, size) +=
                            hxs.template middleCols<size>(start, size);
                        adjhess.template middleRows<size>(start, size) +=
                            hxs.template middleCols<size>(start, size).transpose();
                    });
            }

            this->vectorfunc.scale_jacobian(jx, ifxs);
            this->scalarfunc.accumulate_gradient(adjgrad, gxs, PlusEqualsAssignment());

            fxt = -fx * (ifxs * ifxs);
            fx *= ifxs;

            this->scalarfunc.right_jacobian_product(jx, fxt, jxs, PlusEqualsAssignment(),
                                                    std::bool_constant<false>());
        };

        const int irows = this->scalarfunc.input_rows();
        const int orows = this->output_rows();

        using FType = Output<Scalar>;
        using JType = typename ScalFunc::template Jacobian<Scalar>;
        using GType = typename ScalFunc::template Gradient<Scalar>;
        using HType = typename ScalFunc::template Hessian<Scalar>;

        tycho::utils::BumpAllocator::allocate_run(
            Impl, tycho::utils::TempSpec<FType>(orows, 1), tycho::utils::TempSpec<JType>(1, irows),
            tycho::utils::TempSpec<GType>(irows, 1), tycho::utils::TempSpec<GType>(irows, 1),
            tycho::utils::TempSpec<HType>(irows, irows));
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
};

} // namespace tycho::vf
