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

template <class OuterFunc, class InnerFunc1, class... InnerFuncs>
struct NestedCallAndAppendChain2
    : VectorFunction<NestedCallAndAppendChain2<OuterFunc, InnerFunc1, InnerFuncs...>,
                     InnerFunc1::IRC, OuterFunc::ORC> {
    using Base = VectorFunction<NestedCallAndAppendChain2<OuterFunc, InnerFunc1, InnerFuncs...>,
                                InnerFunc1::IRC, OuterFunc::ORC>;
    using Base::compute;

    VF_TYPE_ALIASES(Base);

    OuterFunc outer_func_;
    InnerFunc1 inner_func1_;

    std::tuple<InnerFuncs...> inner_funcs_;

    using INPUT_DOMAIN = typename InnerFunc1::INPUT_DOMAIN;

    static constexpr bool is_vectorizable = InnerFunc1::is_vectorizable && OuterFunc::is_vectorizable;

    static constexpr int SizeInnerFuncs = sizeof...(InnerFuncs);

    NestedCallAndAppendChain2() {}

    NestedCallAndAppendChain2(OuterFunc outer_func_,
                              std::tuple<InnerFunc1, InnerFuncs...> inner_funct)
        : outer_func_(std::move(outer_func_)) {
        this->inner_func1_ = std::get<0>(inner_funct);
        tycho::utils::constexpr_for_loop(
            std::integral_constant<int, 0>(), std::integral_constant<int, sizeof...(InnerFuncs)>(),
            [&](auto i) {
                std::get<i.value>(this->inner_funcs_) = std::get<i.value + 1>(inner_funct);
            });

        this->set_io_rows(this->inner_func1_.input_rows(), this->outer_func_.output_rows());
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;

        auto Impl = [&](auto &xchain) {
            xchain.template head<Base::IRC>(this->input_rows()) = x;
            this->inner_func1_.compute(
                x, xchain.template segment<InnerFunc1::ORC>(this->input_rows(),
                                                            this->inner_func1_.output_rows()));
            tycho::utils::tuple_for_each(this->inner_funcs_, [&](const auto &func_i) {
                using FTtype = typename std::remove_const<
                    typename std::remove_reference<decltype(func_i)>::type>::type;
                func_i.compute(xchain.template head<FTtype::IRC>(func_i.input_rows()),
                               xchain.template segment<FTtype::ORC>(func_i.input_rows(),
                                                                    func_i.output_rows()));
            });
            this->outer_func_.compute(xchain, fx_);
        };
        tycho::utils::BumpAllocator::allocate_run(
            Impl,
            tycho::utils::TempSpec<typename OuterFunc::template Input<Scalar>>(this->outer_func_.input_rows(), 1));
    }
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        // VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();

        // typename OuterFunc::template Input<Scalar> xchain(this->outer_func_.input_rows());
        // typename InnerFunc1::template Jacobian<Scalar> jx1;
        // std::tuple<typename InnerFuncs::template Jacobian<Scalar>...> jxi;
        // typename OuterFunc::template Jacobian<Scalar> jx_o;

        auto Impl = [&](auto &xchain, auto &jx1, auto &jxi, auto &jx_o) {
            xchain.template head<Base::IRC>(this->input_rows()) = x;
            this->inner_func1_.compute_jacobian(
                x,
                xchain.template segment<InnerFunc1::ORC>(this->input_rows(),
                                                         this->inner_func1_.output_rows()),
                jx1);

            tycho::utils::tuple_for_loop(this->inner_funcs_, [&](const auto &func_i, auto i) {
                using FTtype = typename std::remove_const<
                    typename std::remove_reference<decltype(func_i)>::type>::type;
                func_i.compute_jacobian(
                    xchain.template head<FTtype::IRC>(func_i.input_rows()),
                    xchain.template segment<FTtype::ORC>(func_i.input_rows(), func_i.output_rows()),
                    std::get<i.value>(jxi));
            });

            this->outer_func_.compute_jacobian(xchain, fx_, jx_o);

            tycho::utils::reverse_tuple_for_loop(this->inner_funcs_, [&](const auto &func_i,
                                                                         auto i) {
                using FTtype = typename std::remove_const<
                    typename std::remove_reference<decltype(func_i)>::type>::type;
                func_i.right_jacobian_product(
                    jx_o.template leftCols<FTtype::IRC>(func_i.input_rows()),
                    jx_o.template middleCols<FTtype::ORC>(func_i.input_rows(),
                                                          func_i.output_rows()),
                    std::get<i.value>(jxi), PlusEqualsAssignment(), std::bool_constant<false>());
            });

            this->inner_func1_.right_jacobian_product(
                jx_o.template leftCols<InnerFunc1::IRC>(this->inner_func1_.input_rows()),
                jx_o.template middleCols<InnerFunc1::ORC>(this->inner_func1_.input_rows(),
                                                          this->inner_func1_.output_rows()),
                jx1, PlusEqualsAssignment(), std::bool_constant<false>());

            jx.template leftCols<Base::IRC>(this->input_rows()) =
                jx_o.template leftCols<Base::IRC>(this->input_rows());
        };

        auto make_temp_tuple = [&](auto f) {
            auto app = [&](const auto &...func_i) { return std::tuple{f(func_i)...}; };
            return std::apply(app, this->inner_funcs_);
        };
        auto jis = [&](const auto &func_i) {
            using FTtype = typename std::remove_const<
                typename std::remove_reference<decltype(func_i)>::type>::type;
            using JType = typename FTtype::template Jacobian<Scalar>;
            return tycho::utils::TempSpec<JType>(func_i.output_rows(), func_i.input_rows());
        };
        auto JITemps =
            tycho::utils::TupleOfTempSpecs<typename InnerFuncs::template Jacobian<Scalar>...>{
                make_temp_tuple(jis)};

        tycho::utils::BumpAllocator::allocate_run(
            Impl,
            tycho::utils::TempSpec<typename OuterFunc::template Input<Scalar>>(this->outer_func_.input_rows(), 1),
            tycho::utils::TempSpec<typename InnerFunc1::template Jacobian<Scalar>>(this->inner_func1_.output_rows(),
                                                                this->inner_func1_.input_rows()),
            JITemps,
            tycho::utils::TempSpec<typename OuterFunc::template Jacobian<Scalar>>(this->outer_func_.output_rows(),
                                                               this->outer_func_.input_rows())

        );
    }

    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        CVecRef<InType> x, CVecRef<OutType> fx_,
        CMatRef<JacType> jx_, CVecRef<AdjGradType> adjgrad_,
        CMatRef<AdjHessType> adjhess_, CVecRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        // VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();
        VecRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
        MatRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

        // typename OuterFunc::template Input<Scalar> xchain(this->outer_func_.input_rows());

        // typename OuterFunc::template Jacobian<Scalar> jx_o;  // = typename OuterFunc::template Jacobian<Scalar>::Zero();
        // typename OuterFunc::template Hessian<Scalar> hx_o;   // = OuterFunc_hessian <Scalar>::Zero();
        // typename OuterFunc::template Gradient<Scalar> gx_o;  // = typename OuterFunc::template Gradient<Scalar>::Zero();

        // typename InnerFunc1::template Jacobian<Scalar> jx1;  // = typename InnerFunc1::template Jacobian<Scalar>::Zero();
        // typename InnerFunc1::template Gradient<Scalar> gx1;  // = typename InnerFunc1::template Gradient<Scalar>::Zero();
        // typename InnerFunc1::template Hessian<Scalar> hx1;   // = typename InnerFunc1::template Hessian<Scalar>::Zero();

        // std::tuple<typename InnerFuncs::template Jacobian<Scalar>...> jxi;
        // std::tuple<typename InnerFuncs::template Hessian<Scalar>...> hxi;
        // std::tuple<typename InnerFuncs::template Gradient<Scalar>...> gxi;

        // Eigen::Matrix<Scalar, OuterFunc::IRC, Base::IRC> j0s;

        // xchain.setZero();
        // xchain.template head<Base::IRC>() = x;

        auto Impl = [&](auto &xchain, auto &jx1, auto &gx1, auto &hx1, auto &jxi, auto &gxi,
                        auto &hxi, auto &jx_o, auto &gx_o, auto &hx_o, auto &j0s) {
            xchain.template head<Base::IRC>(this->input_rows()) = x;
            this->inner_func1_.compute(
                x, xchain.template segment<InnerFunc1::ORC>(this->input_rows(),
                                                            this->inner_func1_.output_rows()));

            tycho::utils::tuple_for_each(this->inner_funcs_, [&](const auto &func_i) {
                using FTtype = typename std::remove_const<
                    typename std::remove_reference<decltype(func_i)>::type>::type;
                func_i.compute(xchain.template head<FTtype::IRC>(func_i.input_rows()),
                               xchain.template segment<FTtype::ORC>(func_i.input_rows(),
                                                                    func_i.output_rows()));
            });

            this->outer_func_.compute_jacobian_adjointgradient_adjointhessian(xchain, fx_, jx_o,
                                                                              gx_o, hx_o, adjvars);

            tycho::utils::reverse_tuple_for_loop(this->inner_funcs_, [&](const auto &func_i,
                                                                         auto i) {
                using FTtype = typename std::remove_const<
                    typename std::remove_reference<decltype(func_i)>::type>::type;

                func_i.compute_jacobian_adjointgradient_adjointhessian(
                    xchain.template head<FTtype::IRC>(func_i.input_rows()),
                    xchain.template segment<FTtype::ORC>(func_i.input_rows(), func_i.output_rows()),
                    std::get<i.value>(jxi), std::get<i.value>(gxi), std::get<i.value>(hxi),
                    gx_o.template segment<FTtype::ORC>(func_i.input_rows(), func_i.output_rows()));

                func_i.accumulate_gradient(gx_o.template head<FTtype::IRC>(func_i.input_rows()),
                                           std::get<i.value>(gxi), PlusEqualsAssignment());

                func_i.right_jacobian_product(
                    jx_o.template leftCols<FTtype::IRC>(func_i.input_rows()),
                    jx_o.template middleCols<FTtype::ORC>(func_i.input_rows(),
                                                          func_i.output_rows()),
                    std::get<i.value>(jxi), PlusEqualsAssignment(), std::bool_constant<false>());
            });

            ////////////////////////////////////////////////////////////////////

            this->inner_func1_.compute_jacobian_adjointgradient_adjointhessian(
                x,
                xchain.template segment<InnerFunc1::ORC>(this->inner_func1_.input_rows(),
                                                         this->inner_func1_.output_rows()),
                jx1, adjgrad, adjhess,
                gx_o.template segment<InnerFunc1::ORC>(this->inner_func1_.input_rows()));

            this->inner_func1_.right_jacobian_product(
                jx_o.template leftCols<InnerFunc1::IRC>(this->inner_func1_.input_rows()),
                jx_o.template middleCols<InnerFunc1::ORC>(this->inner_func1_.input_rows(),
                                                          this->inner_func1_.output_rows()),
                jx1, PlusEqualsAssignment(), std::bool_constant<false>());

            jx.template leftCols<Base::IRC>(this->input_rows()) =
                jx_o.template leftCols<Base::IRC>(this->input_rows());
            adjgrad += gx_o.template head<InnerFunc1::IRC>(this->inner_func1_.input_rows());

            /////////////////////

            //////////////////////

            j0s.template topRows<Base::IRC>(this->input_rows()).setIdentity();
            j0s.template middleRows<InnerFunc1::ORC>(this->inner_func1_.input_rows(),
                                                     this->inner_func1_.output_rows()) = jx1;

            tycho::utils::tuple_for_loop(this->inner_funcs_, [&](const auto &func_i, auto i) {
                using FTtype = typename std::remove_const<
                    typename std::remove_reference<decltype(func_i)>::type>::type;

                constexpr int Ev =
                    tycho::utils::SZ_DIFF<FTtype::IRC, Base::IRC>::value; // FTtype::IRC - Base::IRC
                const int ev = func_i.input_rows() - this->input_rows();

                j0s.template middleRows<FTtype::ORC>(func_i.input_rows(), func_i.output_rows()) =
                    std::get<i.value>(jxi).template leftCols<Base::IRC>(this->input_rows()) +
                    std::get<i.value>(jxi).template rightCols<Ev>(func_i.input_rows() -
                                                                  this->input_rows()) *
                        j0s.template middleRows<Ev>(this->input_rows(),
                                                    func_i.input_rows() - this->input_rows());
            });

            tycho::utils::tuple_for_loop(this->inner_funcs_, [&](const auto &func_i, auto i) {
                using FTtype = typename std::remove_const<
                    typename std::remove_reference<decltype(func_i)>::type>::type;

                func_i.accumulate_hessian(hx_o.template topLeftCorner<FTtype::IRC, FTtype::IRC>(
                                              func_i.input_rows(), func_i.input_rows()),
                                          std::get<i.value>(hxi), PlusEqualsAssignment());
            });

            adjhess.template topLeftCorner<Base::IRC, Base::IRC>(this->input_rows(),
                                                                 this->input_rows()) +=
                hx_o.template topLeftCorner<Base::IRC, Base::IRC>(this->input_rows(),
                                                                  this->input_rows());

            constexpr int Ev =
                tycho::utils::SZ_DIFF<OuterFunc::IRC,
                                      Base::IRC>::value; // OuterFunc::IRC - Base::IRC;
            const int ev = this->outer_func_.input_rows() - this->input_rows();
            hx_o.template leftCols<Base::IRC>(this->input_rows()).noalias() =
                hx_o.template rightCols<Ev>(ev) * j0s.template bottomRows<Ev>(ev);

            adjhess.template topLeftCorner<Base::IRC, Base::IRC>(this->input_rows(),
                                                                 this->input_rows()) +=
                hx_o.template topLeftCorner<Base::IRC, Base::IRC>(this->input_rows(),
                                                                  this->input_rows()) +
                hx_o.template topLeftCorner<Base::IRC, Base::IRC>(this->input_rows(),
                                                                  this->input_rows())
                    .transpose();

            adjhess.noalias() +=
                j0s.template bottomRows<Ev>(ev).transpose() *
                hx_o.template leftCols<Base::IRC>(this->input_rows()).template bottomRows<Ev>(ev);
        };

        auto make_temp_tuple = [&](auto f) {
            auto app = [&](const auto &...func_i) { return std::tuple{f(func_i)...}; };
            return std::apply(app, this->inner_funcs_);
        };

        auto jis = [&](const auto &func_i) {
            using FTtype = typename std::remove_const<
                typename std::remove_reference<decltype(func_i)>::type>::type;
            using JType = typename FTtype::template Jacobian<Scalar>;
            return tycho::utils::TempSpec<JType>(func_i.output_rows(), func_i.input_rows());
        };
        auto gis = [&](const auto &func_i) {
            using FTtype = typename std::remove_const<
                typename std::remove_reference<decltype(func_i)>::type>::type;
            using GType = typename FTtype::template Gradient<Scalar>;
            return tycho::utils::TempSpec<GType>(func_i.input_rows(), 1);
        };
        auto his = [&](const auto &func_i) {
            using FTtype = typename std::remove_const<
                typename std::remove_reference<decltype(func_i)>::type>::type;
            using HType = typename FTtype::template Hessian<Scalar>;
            return tycho::utils::TempSpec<HType>(func_i.input_rows(), func_i.input_rows());
        };

        auto JITemps =
            tycho::utils::TupleOfTempSpecs<typename InnerFuncs::template Jacobian<Scalar>...>{
                make_temp_tuple(jis)};
        auto GITemps =
            tycho::utils::TupleOfTempSpecs<typename InnerFuncs::template Gradient<Scalar>...>{
                make_temp_tuple(gis)};
        auto HITemps =
            tycho::utils::TupleOfTempSpecs<typename InnerFuncs::template Hessian<Scalar>...>{
                make_temp_tuple(his)};

        tycho::utils::BumpAllocator::allocate_run(
            Impl,
            tycho::utils::TempSpec<typename OuterFunc::template Input<Scalar>>(this->outer_func_.input_rows(), 1),
            tycho::utils::TempSpec<typename InnerFunc1::template Jacobian<Scalar>>(this->inner_func1_.output_rows(),
                                                                this->inner_func1_.input_rows()),
            tycho::utils::TempSpec<typename InnerFunc1::template Gradient<Scalar>>(this->inner_func1_.input_rows(), 1),
            tycho::utils::TempSpec<typename InnerFunc1::template Hessian<Scalar>>(this->inner_func1_.input_rows(),
                                                               this->inner_func1_.input_rows()),
            JITemps, GITemps, HITemps,
            tycho::utils::TempSpec<typename OuterFunc::template Jacobian<Scalar>>(this->outer_func_.output_rows(),
                                                               this->outer_func_.input_rows()),
            tycho::utils::TempSpec<typename OuterFunc::template Gradient<Scalar>>(this->outer_func_.input_rows(), 1),
            tycho::utils::TempSpec<typename OuterFunc::template Hessian<Scalar>>(this->outer_func_.input_rows(),
                                                              this->outer_func_.input_rows()),
            tycho::utils::TempSpec<Eigen::Matrix<Scalar, OuterFunc::IRC, Base::IRC>>(
                this->outer_func_.input_rows(), this->input_rows()));
    }
};

template <class OuterFunc, class InnerFunc1, class... InnerFuncs>
struct NestedCallAndAppendChain
    : VectorFunction<NestedCallAndAppendChain<OuterFunc, InnerFunc1, InnerFuncs...>,
                     InnerFunc1::IRC, OuterFunc::ORC> {
    using Base = VectorFunction<NestedCallAndAppendChain<OuterFunc, InnerFunc1, InnerFuncs...>,
                                InnerFunc1::IRC, OuterFunc::ORC>;
    using Base::compute;

    VF_TYPE_ALIASES(Base);

    OuterFunc outer_func_;
    InnerFunc1 inner_func1_;

    std::tuple<InnerFuncs...> inner_funcs_;

    using INPUT_DOMAIN = typename InnerFunc1::INPUT_DOMAIN;

    static constexpr bool ReverseAlg = false;
    static constexpr int SizeInnerFuncs = sizeof...(InnerFuncs);

    NestedCallAndAppendChain() {}
    NestedCallAndAppendChain(OuterFunc outer_func_, InnerFunc1 inner_func1_,
                             InnerFuncs... inner_funcs_)
        : outer_func_(std::move(outer_func_)), inner_func1_(std::move(inner_func1_)),
          inner_funcs_(inner_funcs_...) {}
    NestedCallAndAppendChain(OuterFunc outer_func_, InnerFunc1 inner_func1_,
                             std::tuple<InnerFuncs...> inner_funcs_)
        : outer_func_(std::move(outer_func_)), inner_func1_(std::move(inner_func1_)),
          inner_funcs_(inner_funcs_) {}

    NestedCallAndAppendChain(OuterFunc outer_func_,
                             std::tuple<InnerFunc1, InnerFuncs...> inner_funct)
        : outer_func_(std::move(outer_func_)) {
        this->inner_func1_ = std::get<0>(inner_funct);
        tycho::utils::constexpr_for_loop(
            std::integral_constant<int, 0>(), std::integral_constant<int, sizeof...(InnerFuncs)>(),
            [&](auto i) {
                std::get<i.value>(this->inner_funcs_) = std::get<i.value + 1>(inner_funct);
            });
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        // VecRef<OutType> fx = fx_.const_cast_derived();

        typename OuterFunc::template Input<Scalar> xchain(this->outer_func_.input_rows());
        xchain.setZero();
        xchain.template head<Base::IRC>() = x;
        // int start = this->input_rows();

        this->inner_func1_.compute(x, xchain.template segment<InnerFunc1::ORC>(
                                          this->input_rows(), this->inner_func1_.output_rows()));

        tycho::utils::tuple_for_each(this->inner_funcs_, [&](const auto &func_i) {
            using FTtype = typename std::remove_const<
                typename std::remove_reference<decltype(func_i)>::type>::type;
            func_i.compute(
                xchain.template head<FTtype::IRC>(func_i.input_rows()),
                xchain.template segment<FTtype::ORC>(func_i.input_rows(), func_i.output_rows()));
        });

        this->outer_func_.compute(xchain, fx_);
    }
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        // VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();

        typename OuterFunc::template Input<Scalar> xchain(this->outer_func_.input_rows());

        xchain.setZero();
        xchain.template head<Base::IRC>() = x;

        typename InnerFunc1::template Jacobian<Scalar> jx1; // jx1.setZero();

        this->inner_func1_.compute_jacobian(
            x,
            xchain.template segment<InnerFunc1::ORC>(this->input_rows(),
                                                     this->inner_func1_.output_rows()),
            jx1);

        std::tuple<typename InnerFuncs::template Jacobian<Scalar>...> jxi;

        tycho::utils::tuple_for_loop(this->inner_funcs_, [&](const auto &func_i, auto i) {
            using FTtype = typename std::remove_const<
                typename std::remove_reference<decltype(func_i)>::type>::type;
            // std::get<i.value>(jxi).setZero();
            func_i.compute_jacobian(
                xchain.template head<FTtype::IRC>(func_i.input_rows()),
                xchain.template segment<FTtype::ORC>(func_i.input_rows(), func_i.output_rows()),
                std::get<i.value>(jxi));
        });

        typename OuterFunc::template Jacobian<Scalar> jx_o;

        // Eigen::Matrix<Scalar, -1, -1> jx_o;
        // jx_o.resize(this->outer_func_.output_rows(), this->outer_func_.input_rows());
        // std::vector<typename OuterFunc::template Jacobian<Scalar>> jxtt(1);
        // Eigen::Ref< Eigen::Matrix<Scalar,-1,-1>> jx_o(jx_ot);
        // Eigen::Ref< typename OuterFunc::template Jacobian<Scalar>> jx_o(jx_ot);
        // Eigen::Map<typename OuterFunc::template Jacobian<Scalar>> jx_o(jx_ot.data());
        // Eigen::Map<typename OuterFunc::template Jacobian<Scalar>> jx_o(jxtt[0].data(),
        // this->outer_func_.output_rows(), this->outer_func_.input_rows());

        this->outer_func_.compute_jacobian(xchain, fx_, jx_o);

        tycho::utils::reverse_tuple_for_loop(this->inner_funcs_, [&](const auto &func_i, auto i) {
            using FTtype = typename std::remove_const<
                typename std::remove_reference<decltype(func_i)>::type>::type;
            func_i.right_jacobian_product(
                jx_o.template leftCols<FTtype::IRC>(func_i.input_rows()),
                jx_o.template middleCols<FTtype::ORC>(func_i.input_rows(), func_i.output_rows()),
                std::get<i.value>(jxi), PlusEqualsAssignment(), std::bool_constant<false>());
        });

        this->inner_func1_.right_jacobian_product(
            jx_o.template leftCols<InnerFunc1::IRC>(this->inner_func1_.input_rows()),
            jx_o.template middleCols<InnerFunc1::ORC>(this->inner_func1_.input_rows(),
                                                      this->inner_func1_.output_rows()),
            jx1, PlusEqualsAssignment(), std::bool_constant<false>());

        jx.template leftCols<Base::IRC>(this->input_rows()) =
            jx_o.template leftCols<Base::IRC>(this->input_rows());
    }

    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        CVecRef<InType> x, CVecRef<OutType> fx_,
        CMatRef<JacType> jx_, CVecRef<AdjGradType> adjgrad_,
        CMatRef<AdjHessType> adjhess_, CVecRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        // VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();
        VecRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
        MatRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

        typename OuterFunc::template Input<Scalar> xchain(this->outer_func_.input_rows());
        xchain.setZero();
        xchain.template head<Base::IRC>() = x;
        int start = this->input_rows();

        this->inner_func1_.compute(x, xchain.template segment<InnerFunc1::ORC>(
                                          this->input_rows(), this->inner_func1_.output_rows()));

        tycho::utils::tuple_for_each(this->inner_funcs_, [&](const auto &func_i) {
            using FTtype = typename std::remove_const<
                typename std::remove_reference<decltype(func_i)>::type>::type;
            func_i.compute(
                xchain.template head<FTtype::IRC>(func_i.input_rows()),
                xchain.template segment<FTtype::ORC>(func_i.input_rows(), func_i.output_rows()));
        });

        typename OuterFunc::template Jacobian<Scalar> jx_o; // = typename OuterFunc::template Jacobian<Scalar>::Zero();
        typename OuterFunc::template Hessian<Scalar> hx_o;  // = OuterFunc_hessian <Scalar>::Zero();
        typename OuterFunc::template Gradient<Scalar> gx_o; // = typename OuterFunc::template Gradient<Scalar>::Zero();

        typename InnerFunc1::template Jacobian<Scalar> jx1; // = typename InnerFunc1::template Jacobian<Scalar>::Zero();
        typename InnerFunc1::template Gradient<Scalar> gx1; // = typename InnerFunc1::template Gradient<Scalar>::Zero();
        typename InnerFunc1::template Hessian<Scalar> hx1;  // = typename InnerFunc1::template Hessian<Scalar>::Zero();

        std::tuple<typename InnerFuncs::template Jacobian<Scalar>...> jxi;
        std::tuple<typename InnerFuncs::template Hessian<Scalar>...> hxi;
        std::tuple<typename InnerFuncs::template Gradient<Scalar>...> gxi;

        this->outer_func_.compute_jacobian_adjointgradient_adjointhessian(xchain, fx_, jx_o, gx_o,
                                                                          hx_o, adjvars);

        tycho::utils::reverse_tuple_for_loop(this->inner_funcs_, [&](const auto &func_i, auto i) {
            using FTtype = typename std::remove_const<
                typename std::remove_reference<decltype(func_i)>::type>::type;

            // std::get<i.value>(gxi).setZero();
            // std::get<i.value>(hxi).setZero();
            // std::get<i.value>(jxi).setZero();

            func_i.compute_jacobian_adjointgradient_adjointhessian(
                xchain.template head<FTtype::IRC>(func_i.input_rows()),
                xchain.template segment<FTtype::ORC>(func_i.input_rows(), func_i.output_rows()),
                std::get<i.value>(jxi), std::get<i.value>(gxi), std::get<i.value>(hxi),
                gx_o.template segment<FTtype::ORC>(func_i.input_rows(), func_i.output_rows()));

            func_i.accumulate_gradient(gx_o.template head<FTtype::IRC>(func_i.input_rows()),
                                       std::get<i.value>(gxi), PlusEqualsAssignment());

            func_i.right_jacobian_product(
                jx_o.template leftCols<FTtype::IRC>(func_i.input_rows()),
                jx_o.template middleCols<FTtype::ORC>(func_i.input_rows(), func_i.output_rows()),
                std::get<i.value>(jxi), PlusEqualsAssignment(), std::bool_constant<false>());

            ///////////
            // hx_o.template topLeftCorner<FTtype::IRC, FTtype::IRC>(func_i.input_rows(),
            // func_i.input_rows())
            //	+= std::get<i.value>(jxi).transpose()*hx_o.template
            // block<FTtype::ORC,FTtype::ORC>(func_i.input_rows(), func_i.input_rows()) *
            // std::get<i.value>(jxi);

            if constexpr (ReverseAlg) {
                typename FTtype::template Jacobian<Scalar> jt;

                func_i.right_jacobian_product(jt,
                                              hx_o.template block<FTtype::ORC, FTtype::ORC>(
                                                  func_i.input_rows(), func_i.input_rows()),
                                              std::get<i.value>(jxi), DirectAssignment(),
                                              std::bool_constant<false>());

                func_i.right_jacobian_product(hx_o.template topLeftCorner<FTtype::IRC, FTtype::IRC>(
                                                  func_i.input_rows(), func_i.input_rows()),
                                              jt.transpose(), std::get<i.value>(jxi),
                                              PlusEqualsAssignment(), std::bool_constant<false>());

                func_i.accumulate_hessian(hx_o.template topLeftCorner<FTtype::IRC, FTtype::IRC>(
                                              func_i.input_rows(), func_i.input_rows()),
                                          std::get<i.value>(hxi), PlusEqualsAssignment());

                std::get<i.value>(hxi).setZero();

                func_i.right_jacobian_product(
                    std::get<i.value>(hxi),
                    hx_o.template block<FTtype::IRC, FTtype::ORC>(0, func_i.input_rows()),
                    std::get<i.value>(jxi), DirectAssignment(), std::bool_constant<false>());

                hx_o.template topLeftCorner<FTtype::IRC, FTtype::IRC>(func_i.input_rows(),
                                                                      func_i.input_rows()) +=
                    std::get<i.value>(hxi) + std::get<i.value>(hxi).transpose();
            }

            /////////////////
        });

        ////////////////////////////////////////////////////////////////////

        this->inner_func1_.compute_jacobian_adjointgradient_adjointhessian(
            x,
            xchain.template segment<InnerFunc1::ORC>(this->inner_func1_.input_rows(),
                                                     this->inner_func1_.output_rows()),
            jx1, adjgrad, adjhess,
            gx_o.template segment<InnerFunc1::ORC>(this->inner_func1_.input_rows()));

        this->inner_func1_.right_jacobian_product(
            jx_o.template leftCols<InnerFunc1::IRC>(this->inner_func1_.input_rows()),
            jx_o.template middleCols<InnerFunc1::ORC>(this->inner_func1_.input_rows(),
                                                      this->inner_func1_.output_rows()),
            jx1, PlusEqualsAssignment(), std::bool_constant<false>());

        jx.template leftCols<Base::IRC>(this->input_rows()) =
            jx_o.template leftCols<Base::IRC>(this->input_rows());
        adjgrad += gx_o.template head<InnerFunc1::IRC>(this->inner_func1_.input_rows());

        ///////////////
        if constexpr (ReverseAlg) {
            //  hx_o.template topLeftCorner<InnerFunc1::IRC, InnerFunc1::IRC>(
            //      inner_func1_.input_rows(), inner_func1_.input_rows()) +=
            //     jx1.transpose() *
            //     hx_o.template block<InnerFunc1::ORC, InnerFunc1::ORC>(
            //         inner_func1_.input_rows(), inner_func1_.input_rows()) *
            //     jx1;

            using FTtype = InnerFunc1;
            typename FTtype::template Jacobian<Scalar> jt;

            inner_func1_.right_jacobian_product(
                jt,
                hx_o.template block<FTtype::ORC, FTtype::ORC>(inner_func1_.input_rows(),
                                                              inner_func1_.input_rows()),
                jx1, DirectAssignment(), std::bool_constant<false>());

            inner_func1_.right_jacobian_product(
                hx_o.template topLeftCorner<FTtype::IRC, FTtype::IRC>(inner_func1_.input_rows(),
                                                                      inner_func1_.input_rows()),
                jt.transpose(), jx1, PlusEqualsAssignment(), std::bool_constant<false>());

            this->inner_func1_.right_jacobian_product(
                hx1,
                hx_o.template block<InnerFunc1::IRC, InnerFunc1::ORC>(0, inner_func1_.input_rows()),
                jx1, DirectAssignment(), std::bool_constant<false>());

            adjhess += hx_o.template topLeftCorner<InnerFunc1::IRC, InnerFunc1::IRC>(
                           inner_func1_.input_rows(), inner_func1_.input_rows()) +
                       hx1 + hx1.transpose();

        } else {
            //////////////////////
            Eigen::Matrix<Scalar, OuterFunc::IRC, Base::IRC> j0s;

            j0s.template topRows<Base::IRC>(this->input_rows()).setIdentity();

            j0s.template middleRows<InnerFunc1::ORC>(this->inner_func1_.input_rows(),
                                                     this->inner_func1_.output_rows()) = jx1;

            tycho::utils::tuple_for_loop(this->inner_funcs_, [&](const auto &func_i, auto i) {
                using FTtype = typename std::remove_const<
                    typename std::remove_reference<decltype(func_i)>::type>::type;

                j0s.template middleRows<FTtype::ORC>(func_i.input_rows(), func_i.output_rows()) =
                    std::get<i.value>(jxi).template leftCols<Base::IRC>(this->input_rows()) +
                    std::get<i.value>(jxi).template rightCols<FTtype::IRC - Base::IRC>(
                        func_i.input_rows() - this->input_rows()) *
                        j0s.template middleRows<FTtype::IRC - Base::IRC>(
                            this->input_rows(), func_i.input_rows() - this->input_rows());
            });

            tycho::utils::tuple_for_loop(this->inner_funcs_, [&](const auto &func_i, auto i) {
                using FTtype = typename std::remove_const<
                    typename std::remove_reference<decltype(func_i)>::type>::type;
                // if constexpr (!FTtype::is_linear_function) adjhess.noalias() +=
                // j0s.template topRows<FTtype::IRC>(func_i.input_rows()).transpose() *
                // std::get<i.value>(hxi) * j0s.template
                // topRows<FTtype::IRC>(func_i.input_rows());
                func_i.accumulate_hessian(hx_o.template topLeftCorner<FTtype::IRC, FTtype::IRC>(
                                              func_i.input_rows(), func_i.input_rows()),
                                          std::get<i.value>(hxi), PlusEqualsAssignment());
            });

            // std::cout << std::setprecision(4)<< hx_o << std::endl << std::endl;

            // adjhess.noalias() += j0s.transpose() * hx_o * j0s;

            adjhess.template topLeftCorner<Base::IRC, Base::IRC>() +=
                hx_o.template topLeftCorner<Base::IRC, Base::IRC>();

            constexpr int Ev = OuterFunc::IRC - Base::IRC;
            // Eigen::Matrix<Scalar, Ev, Base::IRC> j0s2 = j0s.template
            // bottomRows<Ev>();

            hx_o.template leftCols<Base::IRC>().noalias() =
                hx_o.template rightCols<Ev>() * j0s.template bottomRows<Ev>();

            adjhess.template topLeftCorner<Base::IRC, Base::IRC>() +=
                hx_o.template topLeftCorner<Base::IRC, Base::IRC>() +
                hx_o.template topLeftCorner<Base::IRC, Base::IRC>().transpose();

            adjhess.noalias() += j0s.template bottomRows<Ev>().transpose() *
                                 hx_o.template leftCols<Base::IRC>().template bottomRows<Ev>();
        }
    }
};

} // namespace tycho::vf
