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

#include "tycho/detail/integrators/rk_coeffs.h"
#include "tycho/vector_functions.h"

namespace tycho::integrators {

// Import cross-namespace types from vf, utils, and solvers.
using solvers::SolverIndexingData;
using utils::ArrayOfTempSpecs;
using utils::BumpAllocator;
using utils::SZ_DIFF;
using utils::SZ_PROD;
using utils::SZ_SUM;
using utils::TempSpec;
using vf::Arguments;
using vf::DenseDerivativeMode;
using vf::NestedCallAndAppendChain;
using vf::NestedCallAndAppendChain2;
using vf::StackedOutputs;
using vf::StaticScaleBase;
using vf::VectorExpression;
using vf::VectorFunction;

template <class DODE, RKOptions RKOp>
struct RKStepper : VectorFunction<RKStepper<DODE, RKOp>, SZ_SUM<DODE::IRC, 1>::value, DODE::IRC> {
    using Base = VectorFunction<RKStepper<DODE, RKOp>, SZ_SUM<DODE::IRC, 1>::value, DODE::IRC>;
    DENSE_FUNCTION_BASE_TYPES(Base);

    template <class Scalar> using ODEDeriv = typename DODE::template Output<Scalar>;
    template <class Scalar> using ODEState = typename DODE::template Input<Scalar>;
    template <class Scalar> using ODEJacobian = typename DODE::template Jacobian<Scalar>;
    template <class Scalar> using ODEHessian = typename DODE::template Hessian<Scalar>;

    static const bool is_vectorizable = true;

    using RKData = RKCoeffs<RKOp>;
    static const int Stages = RKData::Stages;
    static const int Stgsm1 = RKData::Stages - 1;
    static const bool is_diag_ = RKData::is_diag_;

    DODE ode_;

    RKStepper() {}
    RKStepper(DODE ode) : ode_(ode) {
        this->set_io_rows(this->ode_.input_rows() + 1, this->ode_.input_rows());
    }

    template <class InType, class OutType>
    inline void compute_impl(const Eigen::MatrixBase<InType> &x,
                             Eigen::MatrixBase<OutType> const &fx_) const {
        typedef typename InType::Scalar Scalar;
        Eigen::MatrixBase<OutType> &fx = fx_.const_cast_derived();

        auto Impl = [&](auto &k_vals, auto &xtup) {
            xtup = x.template segment<DODE::IRC>(0, this->ode_.input_rows());
            Scalar t0 = xtup[this->ode_.t_var()];
            Scalar tf = x[this->ode_.input_rows()];
            Scalar h = tf - t0;

            this->ode_.compute(xtup, k_vals[0]);
            k_vals[0] *= h;

            for (int i = 0; i < Stgsm1; i++) {
                Scalar ti = t0 + RKData::Times[i] * h;
                xtup = x.template segment<DODE::IRC>(0, this->ode_.input_rows());
                xtup[this->ode_.t_var()] = ti;
                const int ip1 = i + 1;
                const int js = is_diag_ ? i : 0;
                for (int j = js; j < ip1; j++) {
                    xtup.template segment<DODE::XV>(0, this->ode_.x_vars()) +=
                        Scalar(RKData::ACoeffs[i][j]) * k_vals[j];
                }

                this->ode_.compute(xtup, k_vals[ip1]);

                k_vals[ip1] *= h;
            }
            xtup = x.template segment<DODE::IRC>(0, this->ode_.input_rows());
            xtup[this->ode_.t_var()] = tf;
            for (int i = 0; i < Stages; i++) {
                xtup.template segment<DODE::XV>(0, this->ode_.x_vars()) +=
                    Scalar(RKData::BCoeffs[i]) * k_vals[i];
            }
            fx = xtup; // Next State
        };

        BumpAllocator::allocate_run(
            Impl, ArrayOfTempSpecs<ODEDeriv<Scalar>, Stages>(this->ode_.output_rows(), 1),
            TempSpec<ODEState<Scalar>>(this->ode_.input_rows(), 1));
    }

    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(const Eigen::MatrixBase<InType> &x,
                                      Eigen::MatrixBase<OutType> const &fx_,
                                      Eigen::MatrixBase<JacType> const &jx_) const {
        typedef typename InType::Scalar Scalar;
        Eigen::MatrixBase<JacType> &jx = jx_.const_cast_derived();
        Eigen::MatrixBase<OutType> &fx = fx_.const_cast_derived();

        auto Impl = [&](auto &k_vals, auto &xtup, auto &k_jac, auto &xi_jac, auto &kx_jacs) {
            xtup = x.template segment<DODE::IRC>(0, this->ode_.input_rows());
            Scalar t0 = xtup[this->ode_.t_var()];
            Scalar tf = x[this->ode_.input_rows()];
            Scalar h = tf - t0;

            xi_jac.setIdentity();
            xi_jac(this->ode_.t_var(), this->input_rows() - 1) = 0;

            this->ode_.compute_jacobian(xtup, k_vals[0], k_jac);

            k_jac *= h;
            kx_jacs[0].noalias() = k_jac * xi_jac;
            kx_jacs[0].col(this->ode_.t_var()).template segment<DODE::XV>(0, this->ode_.x_vars()) -=
                k_vals[0];
            kx_jacs[0]
                .col(this->input_rows() - 1)
                .template segment<DODE::XV>(0, this->ode_.x_vars()) += k_vals[0];

            k_vals[0] *= h;

            for (int i = 0; i < Stgsm1; i++) {
                Scalar ti = t0 + RKData::Times[i] * h;
                xtup = x.template segment<DODE::IRC>(0, this->ode_.input_rows());
                xi_jac.setIdentity();

                xtup[this->ode_.t_var()] = ti;

                xi_jac(this->ode_.t_var(), this->ode_.t_var()) =
                    Scalar(1.0) - Scalar(RKData::Times[i]);
                xi_jac(this->ode_.t_var(), this->input_rows() - 1) = Scalar(RKData::Times[i]);

                const int ip1 = i + 1;
                const int js = is_diag_ ? i : 0;
                for (int j = js; j < ip1; j++) {
                    xtup.template segment<DODE::XV>(0, this->ode_.x_vars()) +=
                        Scalar(RKData::ACoeffs[i][j]) * k_vals[j];
                    xi_jac.template topRows<DODE::XV>(this->ode_.x_vars()) +=
                        Scalar(RKData::ACoeffs[i][j]) * kx_jacs[j];
                }
                k_jac.setZero();

                this->ode_.compute_jacobian(xtup, k_vals[ip1], k_jac);

                kx_jacs[ip1].noalias() = h * k_jac * xi_jac;
                kx_jacs[ip1].col(this->ode_.t_var()).template head<DODE::XV>(this->ode_.x_vars()) -=
                    k_vals[ip1];
                kx_jacs[ip1]
                    .col(this->input_rows() - 1)
                    .template head<DODE::XV>(this->ode_.x_vars()) += k_vals[ip1];

                k_vals[ip1] *= h;
            }
            xtup = x.template segment<DODE::IRC>(0, this->ode_.input_rows());
            xtup[this->ode_.t_var()] = tf;

            xi_jac.setIdentity();

            xi_jac(this->ode_.t_var(), this->ode_.t_var()) = Scalar(0);
            xi_jac(this->ode_.t_var(), (this->input_rows() - 1)) = Scalar(1);

            for (int i = 0; i < Stages; i++) {
                xtup.template segment<DODE::XV>(0, this->ode_.x_vars()) +=
                    Scalar(RKData::BCoeffs[i]) * k_vals[i];
                xi_jac.template topRows<DODE::XV>(this->ode_.x_vars()) +=
                    Scalar(RKData::BCoeffs[i]) * kx_jacs[i];
            }
            fx = xtup; // Next State
            jx = xi_jac;
        };

        using KXjacType = Eigen::Matrix<Scalar, DODE::XV, Base::IRC>;

        BumpAllocator::allocate_run(
            Impl, ArrayOfTempSpecs<ODEDeriv<Scalar>, Stages>(this->ode_.output_rows(), 1),
            TempSpec<ODEState<Scalar>>(this->ode_.input_rows(), 1),
            TempSpec<ODEJacobian<Scalar>>(this->ode_.output_rows(), this->ode_.input_rows()),
            TempSpec<Jacobian<Scalar>>(this->output_rows(), this->input_rows()),
            ArrayOfTempSpecs<KXjacType, Stages>(this->ode_.output_rows(), this->input_rows()));
    }

    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
        ConstMatrixBaseRef<JacType> jx_, ConstVectorBaseRef<AdjGradType> adjgrad_,
        ConstMatrixBaseRef<AdjHessType> adjhess_, ConstVectorBaseRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
        VectorBaseRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
        MatrixBaseRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

        auto Impl = [&](auto &k_vals, auto &xtup, auto &k_jacs, auto &xi_jac, auto &kx_jacs,
                        auto &xs, auto &k_grads, auto &k_hesses, auto &kx_mults, auto &ht_par) {
            xtup = x.template segment<DODE::IRC>(0, this->ode_.input_rows());

            Scalar t0 = xtup[this->ode_.t_var()];
            Scalar tf = x[this->ode_.input_rows()];
            Scalar h = tf - t0;

            xs[0] = xtup;
            this->ode_.compute(xtup, k_vals[0]);
            k_vals[0] *= h;

            for (int i = 0; i < Stgsm1; i++) {
                Scalar ti = t0 + RKData::Times[i] * h;
                xtup = x.template segment<DODE::IRC>(0, this->ode_.input_rows());
                xtup[this->ode_.t_var()] = ti;
                const int ip1 = i + 1;
                const int js = is_diag_ ? i : 0;
                for (int j = js; j < ip1; j++) {
                    xtup.template segment<DODE::XV>(0, this->ode_.x_vars()) +=
                        Scalar(RKData::ACoeffs[i][j]) * k_vals[j];
                }
                xs[ip1] = xtup;
                this->ode_.compute(xtup, k_vals[ip1]);
                k_vals[ip1] *= h;
            }

            xtup = x.template segment<DODE::IRC>(0, this->ode_.input_rows());
            xtup[this->ode_.t_var()] = tf;
            for (int i = 0; i < Stages; i++) {
                xtup.template segment<DODE::XV>(0, this->ode_.x_vars()) +=
                    Scalar(RKData::BCoeffs[i]) * k_vals[i];
                kx_mults[i] = adjvars * Scalar(RKData::BCoeffs[i]) * h;
            }

            fx = xtup; // Next State

            for (int i = Stgsm1 - 1; i >= 0; i--) {

                const int ip1 = i + 1;
                const int js = is_diag_ ? i : 0;
                k_vals[ip1].setZero();
                this->ode_.compute_jacobian_adjointgradient_adjointhessian(
                    xs[ip1], k_vals[ip1], k_jacs[ip1], k_grads[ip1], k_hesses[ip1],
                    kx_mults[ip1].head(this->ode_.output_rows()));

                for (int j = js; j < ip1; j++) {
                    kx_mults[j] += k_grads[ip1] * Scalar(RKData::ACoeffs[i][j]) * h;
                }
            }

            k_vals[0].setZero();
            this->ode_.compute_jacobian_adjointgradient_adjointhessian(
                xs[0], k_vals[0], k_jacs[0], k_grads[0], k_hesses[0],
                kx_mults[0].head(this->ode_.output_rows()));

            adjhess.topLeftCorner(this->ode_.input_rows(), this->ode_.input_rows()) += k_hesses[0];

            xi_jac.setIdentity();
            xi_jac(this->ode_.t_var(), this->input_rows() - 1) = Scalar(0.0);

            k_jacs[0] *= h;
            kx_jacs[0].noalias() = k_jacs[0] * xi_jac;
            kx_jacs[0].col(this->ode_.t_var()).template segment<DODE::XV>(0, this->ode_.x_vars()) -=
                k_vals[0];
            kx_jacs[0]
                .col(this->input_rows() - 1)
                .template segment<DODE::XV>(0, this->ode_.x_vars()) += k_vals[0];

            k_vals[0] *= h;

            ht_par = (xi_jac.transpose() * k_grads[0]) * (1.0 / h);

            //

            for (int i = 0; i < Stgsm1; i++) {
                Scalar ti = t0 + RKData::Times[i] * h;
                xi_jac.setIdentity();

                xi_jac(this->ode_.t_var(), this->ode_.t_var()) =
                    Scalar(1.0) - Scalar(RKData::Times[i]);
                xi_jac(this->ode_.t_var(), this->input_rows() - 1) = Scalar(RKData::Times[i]);

                const int ip1 = i + 1;
                const int js = is_diag_ ? i : 0;
                for (int j = js; j < ip1; j++) {

                    xi_jac.template topRows<DODE::XV>(this->ode_.x_vars()) +=
                        Scalar(RKData::ACoeffs[i][j]) * kx_jacs[j];
                }

                kx_jacs[ip1].noalias() = h * k_jacs[ip1] * xi_jac;
                kx_jacs[ip1].col(this->ode_.t_var()).template head<DODE::XV>(this->ode_.x_vars()) -=
                    k_vals[ip1];
                kx_jacs[ip1]
                    .col(this->input_rows() - 1)
                    .template head<DODE::XV>(this->ode_.x_vars()) += k_vals[ip1];

                k_vals[ip1] *= h;

                adjhess += xi_jac.transpose() * k_hesses[ip1] * xi_jac;

                ht_par += (xi_jac.transpose() * k_grads[ip1]) * (1.0 / h);
            }
            xtup = x.template segment<DODE::IRC>(0, this->ode_.input_rows());
            xtup[this->ode_.t_var()] = tf;

            xi_jac.setIdentity();

            xi_jac(this->ode_.t_var(), this->ode_.t_var()) = Scalar(0.0);
            xi_jac(this->ode_.t_var(), (this->input_rows() - 1)) = Scalar(1.0);

            for (int i = 0; i < Stages; i++) {

                xi_jac.template topRows<DODE::XV>(this->ode_.x_vars()) +=
                    Scalar(RKData::BCoeffs[i]) * kx_jacs[i];
            }

            adjhess.col(this->ode_.t_var()) -= ht_par;
            adjhess.col(this->input_rows() - 1) += ht_par;

            adjhess.row(this->ode_.t_var()) -= ht_par.transpose();
            adjhess.row(this->input_rows() - 1) += ht_par.transpose();

            jx = xi_jac;
            adjgrad = jx.transpose() * adjvars;
        };

        using KXjacType = Eigen::Matrix<Scalar, DODE::XV, Base::IRC>;

        BumpAllocator::allocate_run(
            Impl, ArrayOfTempSpecs<ODEDeriv<Scalar>, Stages>(this->ode_.output_rows(), 1),
            TempSpec<ODEState<Scalar>>(this->ode_.input_rows(), 1),
            ArrayOfTempSpecs<ODEJacobian<Scalar>, Stages>(this->ode_.output_rows(),
                                                          this->ode_.input_rows()),
            TempSpec<Jacobian<Scalar>>(this->output_rows(), this->input_rows()),
            ArrayOfTempSpecs<KXjacType, Stages>(this->ode_.output_rows(), this->input_rows()),
            ArrayOfTempSpecs<ODEState<Scalar>, Stages>(this->ode_.input_rows(), 1),
            ArrayOfTempSpecs<ODEState<Scalar>, Stages>(this->ode_.input_rows(), 1),
            ArrayOfTempSpecs<ODEHessian<Scalar>, Stages>(this->ode_.input_rows(),
                                                         this->ode_.input_rows()),
            ArrayOfTempSpecs<ODEState<Scalar>, Stages>(this->ode_.input_rows(), 1),
            TempSpec<Input<Scalar>>(this->input_rows(), 1));
    }

    /// These Methods are being nulled because it is not
    /// possible for them to be called

    void constraints(ConstEigenRef<Eigen::VectorXd> X, EigenRef<Eigen::VectorXd> FX,
                     const SolverIndexingData &data) const {};
    void constraints_adjointgradient(ConstEigenRef<Eigen::VectorXd> X,
                                     ConstEigenRef<Eigen::VectorXd> L, EigenRef<Eigen::VectorXd> FX,
                                     EigenRef<Eigen::VectorXd> AGX,
                                     const SolverIndexingData &data) const {};
    void constraints_jacobian(ConstEigenRef<Eigen::VectorXd> X, Eigen::Ref<Eigen::VectorXd> FX,
                              Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                              Eigen::Ref<Eigen::VectorXi> KKTLocations,
                              Eigen::Ref<Eigen::VectorXi> KKTClashes,
                              std::vector<std::mutex> &KKTLocks,
                              const SolverIndexingData &data) const {}
    void constraints_jacobian_adjointgradient(
        ConstEigenRef<Eigen::VectorXd> X, ConstEigenRef<Eigen::VectorXd> L,
        Eigen::Ref<Eigen::VectorXd> FX, Eigen::Ref<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
        EigenRef<Eigen::VectorXi> KKTLocations, EigenRef<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex> &KKTLocks, const SolverIndexingData &data) const {}
    void constraints_jacobian_adjointgradient_adjointhessian(
        ConstEigenRef<Eigen::VectorXd> X, ConstEigenRef<Eigen::VectorXd> L,
        EigenRef<Eigen::VectorXd> FX, EigenRef<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
        EigenRef<Eigen::VectorXi> KKTLocations, EigenRef<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex> &KKTLocks, const SolverIndexingData &data) const {}
};

template <class DODE, RKOptions RKOp> struct RKStepper_Impl {
    static auto Definition(const DODE &ode) {
        auto ks0 = std::tuple{};
        return compute_xf<-1, decltype(ks0)>(ode, ks0);
    }

    template <int Stg, int Elem> struct ACoeff : StaticScaleBase<ACoeff<Stg, Elem>> {
        static constexpr double value = RKCoeffs<RKOp>::ACoeffs[Stg][Elem];
    };
    template <int Elem> struct BCoeff : StaticScaleBase<BCoeff<Elem>> {
        static constexpr double value = RKCoeffs<RKOp>::BCoeffs[Elem];
    };

    template <int Stg, class Ks> static auto compute_xf(const DODE &ode, const Ks &ks) {
        constexpr int XV = DODE::XV;
        constexpr int UV = DODE::UV;
        constexpr int PV = DODE::PV;
        constexpr int IRC = SZ_SUM<DODE::IRC, 1>::value;

        auto args = Arguments<DODE::IRC + 1 + (Stg + 1) * DODE::XV>();
        auto empty = std::tuple{};

        // auto xi = kth_stage_sum<Stg, 0, decltype(args)>(ode, args);
        auto xi = kth_stage_sum2<Stg, 0, decltype(args), decltype(empty)>(ode, args, empty);

        auto t0 = args.template coeff<XV>();
        auto tf = args.template coeff<IRC - 1>();
        auto u = args.template segment<UV, XV + 1>();
        auto p = args.template segment<PV, XV + 1 + UV>();
        auto h = tf - t0;

        auto tlam = [&]() {
            if constexpr (Stg == -1)
                return t0;
            else
                return t0 + h * RKCoeffs<RKOp>::Times[Stg];
        };
        auto ti = tlam();
        auto xti = make_state(ode, xi, ti, u, p);

        auto ki = ode.eval(xti) * h;

        if constexpr (Stg == RKCoeffs<RKOp>::Stages - 2) {
            auto xxf = final_state_sum<0, decltype(args), decltype(ki)>(ode, args, ki);
            auto xfun = make_state(ode, xxf, tf, u, p);
            return NestedCallAndAppendChain{xfun, ks};
        } else {
            auto knew = std::tuple_cat(ks, std::make_tuple(ki));

            return compute_xf<Stg + 1, decltype(knew)>(ode, knew);
        }
    }

    template <class Xtype, class Titype, class Utype, class Ptype>
    static auto make_state(const DODE &ode, const Xtype &x0, const Titype &ti, const Utype &u,
                           const Ptype &p) {
        if constexpr (DODE::UV > 0) {
            if constexpr (DODE::PV > 0) {
            } else if constexpr (DODE::PV == 0) {
                return StackedOutputs{x0, ti, u};
            }
        } else if constexpr (DODE::UV == 0) {
            if constexpr (DODE::PV > 0) {
            } else if constexpr (DODE::PV == 0) {
                return StackedOutputs{x0, ti};
            }
        } else if constexpr (DODE::UV == -1) {
            if constexpr (DODE::PV > 0) {
            } else if constexpr (DODE::PV == 0) {
            }
        }
    }

    template <int Stg, int Elem, class Args>
    static auto kth_stage_sum(const DODE &ode, const Args &args) {
        if constexpr (Elem == Stg + 1) {
            return args.template head<DODE::XV>();
        } else {
            if constexpr (RKCoeffs<RKOp>::ACoeffs[Stg][Elem] == 0.0) {
                return kth_stage_sum<Stg, Elem + 1, Args>(ode, args);
            } else {
                return args.template tail<DODE::XV *(Stg + 1)>()
                               .template segment<DODE::XV, DODE::XV * Elem>() *
                           ACoeff<Stg, Elem>().value +
                       kth_stage_sum<Stg, Elem + 1, Args>(ode, args);
            }
        };
    }

    template <int Stg, int Elem, class Args, class Ks>
    static auto kth_stage_sum2(const DODE &ode, const Args &args, const Ks &ks) {
        if constexpr (Elem == Stg + 1) {
            auto next = args.template head<DODE::XV>();
            auto knew = std::tuple_cat(ks, std::make_tuple(next));

            return make_sum_tuple(knew);
        } else {
            if constexpr (RKCoeffs<RKOp>::ACoeffs[Stg][Elem] == 0.0) {
                return kth_stage_sum2<Stg, Elem + 1, Args, Ks>(ode, args, ks);
            } else {
                auto next = args.template tail<DODE::XV *(Stg + 1)>()
                                .template segment<DODE::XV, DODE::XV * Elem>() *
                            ACoeff<Stg, Elem>().value;

                auto knew = std::tuple_cat(ks, std::make_tuple(next));

                return kth_stage_sum2<Stg, Elem + 1, Args, decltype(knew)>(ode, args, knew);
            }
        };
    }

    template <int Elem, class Args, class KF>
    static auto final_state_sum(const DODE &ode, const Args &args, const KF &kf) {
        if constexpr (RKOp == RKOptions::RK4Classic) {
            return make_sum(kf * BCoeff<3>().value, args.template head<DODE::XV>(),
                            args.template tail<DODE::XV *(RKCoeffs<RKOp>::Stages - 1)>()
                                    .template segment<DODE::XV, DODE::XV * 0>() *
                                BCoeff<0>().value,
                            args.template tail<DODE::XV *(RKCoeffs<RKOp>::Stages - 1)>()
                                    .template segment<DODE::XV, DODE::XV * 1>() *
                                BCoeff<1>().value,
                            args.template tail<DODE::XV *(RKCoeffs<RKOp>::Stages - 1)>()
                                    .template segment<DODE::XV, DODE::XV * 2>() *
                                BCoeff<2>().value);
        } else if constexpr (RKOp == RKOptions::DOPRI5) {
            return make_sum(kf * BCoeff<5>(), args.template head<DODE::XV>(),
                            args.template tail<DODE::XV *(RKCoeffs<RKOp>::Stages - 1)>()
                                    .template segment<DODE::XV, DODE::XV * 0>() *
                                BCoeff<0>(),
                            args.template tail<DODE::XV *(RKCoeffs<RKOp>::Stages - 1)>()
                                    .template segment<DODE::XV, DODE::XV * 2>() *
                                BCoeff<2>(),
                            args.template tail<DODE::XV *(RKCoeffs<RKOp>::Stages - 1)>()
                                    .template segment<DODE::XV, DODE::XV * 3>() *
                                BCoeff<3>(),
                            args.template tail<DODE::XV *(RKCoeffs<RKOp>::Stages - 1)>()
                                    .template segment<DODE::XV, DODE::XV * 4>() *
                                BCoeff<4>());
        } else if constexpr (Elem == RKCoeffs<RKOp>::Stages - 1) {
            if constexpr (RKCoeffs<RKOp>::BCoeffs[Elem] == 0.0)
                return args.template head<DODE::XV>();
            else
                return kf * BCoeff<Elem>() + args.template head<DODE::XV>();
        } else {
            if constexpr (RKCoeffs<RKOp>::BCoeffs[Elem] == 0.0)
                return final_state_sum<Elem + 1, Args, KF>(ode, args, kf);
            else
                return args.template tail<DODE::XV *(RKCoeffs<RKOp>::Stages - 1)>()
                               .template segment<DODE::XV, DODE::XV * Elem>() *
                           BCoeff<Elem>() +
                       final_state_sum<Elem + 1, Args, KF>(ode, args, kf);
        };
    }
};

template <class DODE, RKOptions RKOp> struct RKStepper_Impl_NEW {

    template <int Stg, int Elem> struct ACoeff : StaticScaleBase<ACoeff<Stg, Elem>> {
        static constexpr double value = RKCoeffs<RKOp>::ACoeffs[Stg][Elem];
    };
    template <int Elem> struct BCoeff : StaticScaleBase<BCoeff<Elem>> {
        static constexpr double value = RKCoeffs<RKOp>::BCoeffs[Elem];
    };

    static auto Definition(const DODE &ode) {
        auto ks0 = std::tuple{};
        return compute_xf<-1, decltype(ks0)>(ode, ks0);
    }

    template <int Stg, class Ks> static auto compute_xf(const DODE &ode, const Ks &ks) {
        constexpr int XV = DODE::XV;
        constexpr int UV = DODE::UV;
        constexpr int PV = DODE::PV;
        constexpr int IRC = SZ_SUM<DODE::IRC, 1>::value;
        constexpr int ARGSIZE = SZ_SUM<IRC, SZ_PROD<(Stg + 1), XV>::value>::value;

        int xv = ode.x_vars();
        int uv = ode.u_vars();
        int pv = ode.p_vars();
        int irows = ode.input_rows() + 1;
        int argsize = irows + (Stg + 1) * xv;

        auto args = Arguments<ARGSIZE>(argsize);
        auto empty = std::tuple{};

        auto xi = kth_stage_sum<Stg, 0, decltype(args), decltype(empty)>(ode, args, empty);

        auto t0 = args.template coeff<XV>(xv);
        auto up =
            args.template segment<SZ_SUM<UV, PV>::value, SZ_SUM<XV, 1>::value>(xv + 1, uv + pv);
        auto tf = args.template coeff<SZ_DIFF<IRC, 1>::value>(irows - 1);
        auto h = tf - t0;

        auto tlam = [&]() {
            if constexpr (Stg == -1)
                return t0;
            else
                return t0 + h * RKCoeffs<RKOp>::Times[Stg];
        };
        auto make_state = [&](const auto &xii, const auto &tii, const auto &upii) {
            if constexpr (SZ_SUM<DODE::UV, DODE::PV>::value == 0)
                return StackedOutputs{xii, tii};
            else
                return StackedOutputs{xii, tii, upii};
        };

        auto ti = tlam();
        auto xti = make_state(xi, ti, up);
        auto ki = ode.eval(xti) * h;

        if constexpr (Stg == RKCoeffs<RKOp>::Stages - 2) {
            auto xxf = final_state_sum<0, decltype(args), decltype(ki)>(ode, args, ki);
            auto xfun = make_state(xxf, tf, up);
            return NestedCallAndAppendChain2{xfun, ks};
        } else {
            auto knew = std::tuple_cat(ks, std::make_tuple(ki));
            return compute_xf<Stg + 1, decltype(knew)>(ode, knew);
        }
    }

    template <int Stg, int Elem, class Args, class Ks>
    static auto kth_stage_sum(const DODE &ode, const Args &args, const Ks &ks) {
        if constexpr (Elem == Stg + 1) {
            auto next = args.template head<DODE::XV>(ode.x_vars());
            auto knew = std::tuple_cat(ks, std::make_tuple(next));
            return make_sum_tuple(knew);
        } else {
            if constexpr (RKCoeffs<RKOp>::ACoeffs[Stg][Elem] == 0.0) {
                return kth_stage_sum<Stg, Elem + 1, Args, Ks>(ode, args, ks);
            } else {
                auto next = args.template tail<SZ_PROD<(Stg + 1), DODE::XV>::value>((Stg + 1) *
                                                                                    ode.x_vars())
                                .template segment<DODE::XV, SZ_PROD<Elem, DODE::XV>::value>(
                                    ode.x_vars() * Elem, ode.x_vars()) *
                            ACoeff<Stg, Elem>().value;
                auto knew = std::tuple_cat(ks, std::make_tuple(next));
                return kth_stage_sum<Stg, Elem + 1, Args, decltype(knew)>(ode, args, knew);
            }
        };
    }

    template <int Elem, class Args, class KF>
    static auto final_state_sum(const DODE &ode, const Args &args, const KF &kf) {
        //// Finish this
        if constexpr (RKOp == RKOptions::RK4Classic) {

            // constexpr int XV       = DODE::XV;
            constexpr int TAILSIZE = SZ_PROD<DODE::XV, (RKCoeffs<RKOp>::Stages - 1)>::value;
            int xv = ode.x_vars();
            int tailsize = xv * (RKCoeffs<RKOp>::Stages - 1);
            return make_sum(
                kf * BCoeff<3>().value, args.template head<DODE::XV>(xv),

                args.template tail<TAILSIZE>(tailsize)
                        .template segment<DODE::XV, SZ_PROD<DODE::XV, 0>::value>(0, xv) *
                    BCoeff<0>().value,

                args.template tail<TAILSIZE>(tailsize)
                        .template segment<DODE::XV, SZ_PROD<DODE::XV, 1>::value>(xv, xv) *
                    BCoeff<1>().value,

                args.template tail<TAILSIZE>(tailsize)
                        .template segment<DODE::XV, SZ_PROD<DODE::XV, 2>::value>(2 * xv, xv) *
                    BCoeff<2>().value);
        } else if constexpr (RKOp == RKOptions::DOPRI5) {

            // constexpr int XV       = DODE::XV;
            constexpr int TAILSIZE = SZ_PROD<DODE::XV, (RKCoeffs<RKOp>::Stages - 1)>::value;
            int xv = ode.x_vars();
            int tailsize = xv * (RKCoeffs<RKOp>::Stages - 1);
            return make_sum(
                kf * BCoeff<5>().value, args.template head<DODE::XV>(xv),

                args.template tail<TAILSIZE>(tailsize)
                        .template segment<DODE::XV, SZ_PROD<DODE::XV, 0>::value>(0, xv) *
                    BCoeff<0>().value,

                // args.template tail<TAILSIZE>(tailsize)
                // .template segment<DODE::XV, SZ_PROD<DODE::XV, 1>::value>(xv, xv) *
                // BCoeff<1>().value,

                args.template tail<TAILSIZE>(tailsize)
                        .template segment<DODE::XV, SZ_PROD<DODE::XV, 2>::value>(2 * xv, xv) *
                    BCoeff<2>().value,

                args.template tail<TAILSIZE>(tailsize)
                        .template segment<DODE::XV, SZ_PROD<DODE::XV, 3>::value>(3 * xv, xv) *
                    BCoeff<3>().value,

                args.template tail<TAILSIZE>(tailsize)
                        .template segment<DODE::XV, SZ_PROD<DODE::XV, 4>::value>(4 * xv, xv) *
                    BCoeff<4>().value);
        } else if constexpr (RKOp == RKOptions::DOPRI87) {

            // constexpr int XV       = DODE::XV;
            constexpr int TAILSIZE = SZ_PROD<DODE::XV, (RKCoeffs<RKOp>::Stages - 1)>::value;
            int xv = ode.x_vars();
            int tailsize = xv * (RKCoeffs<RKOp>::Stages - 1);
            return make_sum(
                kf * BCoeff<12>().value, args.template head<DODE::XV>(xv),

                args.template tail<TAILSIZE>(tailsize)
                        .template segment<DODE::XV, SZ_PROD<DODE::XV, 0>::value>(0, xv) *
                    BCoeff<0>().value,

                // args.template tail<TAILSIZE>(tailsize)
                // .template segment<DODE::XV, SZ_PROD<DODE::XV, 1>::value>(xv, xv) *
                // BCoeff<1>().value,

                args.template tail<TAILSIZE>(tailsize)
                        .template segment<DODE::XV, SZ_PROD<DODE::XV, 5>::value>(5 * xv, xv) *
                    BCoeff<5>().value,

                args.template tail<TAILSIZE>(tailsize)
                        .template segment<DODE::XV, SZ_PROD<DODE::XV, 6>::value>(6 * xv, xv) *
                    BCoeff<6>().value,

                args.template tail<TAILSIZE>(tailsize)
                        .template segment<DODE::XV, SZ_PROD<DODE::XV, 7>::value>(7 * xv, xv) *
                    BCoeff<7>().value,

                args.template tail<TAILSIZE>(tailsize)
                        .template segment<DODE::XV, SZ_PROD<DODE::XV, 8>::value>(8 * xv, xv) *
                    BCoeff<8>().value,

                args.template tail<TAILSIZE>(tailsize)
                        .template segment<DODE::XV, SZ_PROD<DODE::XV, 9>::value>(9 * xv, xv) *
                    BCoeff<9>().value,

                args.template tail<TAILSIZE>(tailsize)
                        .template segment<DODE::XV, SZ_PROD<DODE::XV, 10>::value>(10 * xv, xv) *
                    BCoeff<10>().value,

                args.template tail<TAILSIZE>(tailsize)
                        .template segment<DODE::XV, SZ_PROD<DODE::XV, 11>::value>(11 * xv, xv) *
                    BCoeff<11>().value);
        }

        else if constexpr (Elem == RKCoeffs<RKOp>::Stages - 1) {
            if constexpr (RKCoeffs<RKOp>::BCoeffs[Elem] == 0.0)
                return args.template head<DODE::XV>();
            else
                return kf * BCoeff<Elem>() + args.template head<DODE::XV>();
        } else {
            if constexpr (RKCoeffs<RKOp>::BCoeffs[Elem] == 0.0)
                return final_state_sum<Elem + 1, Args, KF>(ode, args, kf);
            else
                return args.template tail<DODE::XV *(RKCoeffs<RKOp>::Stages - 1)>()
                               .template segment<DODE::XV, DODE::XV * Elem>() *
                           BCoeff<Elem>() +
                       final_state_sum<Elem + 1, Args, KF>(ode, args, kf);
        };
    }
};

template <class DODE, RKOptions RKOp>
struct RKStepper_NEW
    : VectorExpression<RKStepper_NEW<DODE, RKOp>, RKStepper_Impl_NEW<DODE, RKOp>, const DODE &> {
    using Base =
        VectorExpression<RKStepper_NEW<DODE, RKOp>, RKStepper_Impl_NEW<DODE, RKOp>, const DODE &>;
    // using Base::Base;
    // static const bool is_vectorizable = false;

    RKStepper_NEW(const DODE &ode) : Base(ode) {}

    /// These Methods are being nulled because it is not
    /// possible for them to be called

    void constraints(ConstEigenRef<Eigen::VectorXd> X, EigenRef<Eigen::VectorXd> FX,
                     const SolverIndexingData &data) const {};
    void constraints_adjointgradient(ConstEigenRef<Eigen::VectorXd> X,
                                     ConstEigenRef<Eigen::VectorXd> L, EigenRef<Eigen::VectorXd> FX,
                                     EigenRef<Eigen::VectorXd> AGX,
                                     const SolverIndexingData &data) const {};
    void constraints_jacobian(ConstEigenRef<Eigen::VectorXd> X, Eigen::Ref<Eigen::VectorXd> FX,
                              Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                              Eigen::Ref<Eigen::VectorXi> KKTLocations,
                              Eigen::Ref<Eigen::VectorXi> KKTClashes,
                              std::vector<std::mutex> &KKTLocks,
                              const SolverIndexingData &data) const {}
    void constraints_jacobian_adjointgradient(
        ConstEigenRef<Eigen::VectorXd> X, ConstEigenRef<Eigen::VectorXd> L,
        Eigen::Ref<Eigen::VectorXd> FX, Eigen::Ref<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
        EigenRef<Eigen::VectorXi> KKTLocations, EigenRef<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex> &KKTLocks, const SolverIndexingData &data) const {}
    void constraints_jacobian_adjointgradient_adjointhessian(
        ConstEigenRef<Eigen::VectorXd> X, ConstEigenRef<Eigen::VectorXd> L,
        EigenRef<Eigen::VectorXd> FX, EigenRef<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
        EigenRef<Eigen::VectorXi> KKTLocations, EigenRef<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex> &KKTLocks, const SolverIndexingData &data) const {}
};

template <class DODE, RKOptions RKOp>
struct RKStepper2 : VectorFunction<RKStepper2<DODE, RKOp>, SZ_SUM<DODE::IRC, 1>::value, DODE::IRC,
                                   DenseDerivativeMode::FDiffFwd, DenseDerivativeMode::FDiffFwd> {
    using Base = VectorFunction<RKStepper2<DODE, RKOp>, SZ_SUM<DODE::IRC, 1>::value, DODE::IRC,
                                DenseDerivativeMode::FDiffFwd, DenseDerivativeMode::FDiffFwd>;
    DENSE_FUNCTION_BASE_TYPES(Base);

    template <class Scalar> using ODEDeriv = typename DODE::template Output<Scalar>;
    template <class Scalar> using ODEState = typename DODE::template Input<Scalar>;
    template <class Scalar> using ODEJacobian = typename DODE::template Jacobian<Scalar>;

    using RKData = RKCoeffs<RKOp>;
    static const int Stages = RKData::Stages;
    static const int Stgsm1 = RKData::Stages - 1;
    static const bool is_diag_ = RKData::is_diag_;

    DODE ode_;

    template <class T, int SZ> using STDarray = std::array<T, SZ>;

    RKStepper2() {}
    RKStepper2(DODE ode) : ode_(ode) {
        this->set_io_rows(this->ode_.input_rows() + 1, this->ode_.input_rows());
        this->set_jac_fd_steps(1.0e-6);
        this->set_hess_fd_steps(1.0e-6);
    }

    template <class InType, class OutType>
    inline void compute_impl(const Eigen::MatrixBase<InType> &x,
                             Eigen::MatrixBase<OutType> const &fx_) const {
        typedef typename InType::Scalar Scalar;
        Eigen::MatrixBase<OutType> &fx = fx_.const_cast_derived();

        STDarray<ODEDeriv<Scalar>, Stages> k_vals;
        for (auto &K : k_vals)
            K = ODEDeriv<Scalar>::Zero(this->ode_.output_rows());
        ODEState<Scalar> xtup = x.template segment<DODE::IRC>(0, this->ode_.input_rows());
        Scalar t0 = xtup[this->ode_.t_var()];
        Scalar tf = x[this->input_rows() - 1];
        Scalar h = tf - t0;

        this->ode_.compute(xtup, k_vals[0]);
        k_vals[0] *= h;

        for (int i = 0; i < Stgsm1; i++) {
            Scalar ti = t0 + RKData::Times[i] * h;
            xtup = x.template segment<DODE::IRC>(0, this->ode_.input_rows());
            xtup[this->ode_.t_var()] = ti;
            const int ip1 = i + 1;
            const int js = is_diag_ ? i : 0;
            for (int j = js; j < ip1; j++) {
                xtup.template segment<DODE::XV>(0, this->ode_.x_vars()) +=
                    RKData::ACoeffs[i][j] * k_vals[j];
            }

            this->ode_.compute(xtup, k_vals[ip1]);

            k_vals[ip1] *= h;
        }
        xtup = x.template segment<DODE::IRC>(0, this->ode_.input_rows());
        xtup[this->ode_.t_var()] = tf;
        for (int i = 0; i < Stages; i++) {
            xtup.template segment<DODE::XV>(0, this->ode_.x_vars()) +=
                RKData::BCoeffs[i] * k_vals[i];
        }
        fx = xtup;
    }

    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl1(const Eigen::MatrixBase<InType> &x,
                                       Eigen::MatrixBase<OutType> const &fx_,
                                       Eigen::MatrixBase<JacType> const &jx_) const {
        typedef typename InType::Scalar Scalar;
        Eigen::MatrixBase<JacType> &jx = jx_.const_cast_derived();
        Eigen::MatrixBase<OutType> &fx = fx_.const_cast_derived();

        STDarray<ODEDeriv<Scalar>, Stages> k_vals;
        ODEJacobian<Scalar> k_jacs;
        Jacobian<Scalar> xi_jac = Jacobian<Scalar>::Zero(this->output_rows(), this->input_rows());
        ;

        STDarray<Eigen::Matrix<Scalar, DODE::XV, Base::IRC>, Stages> kx_jacs;

        for (auto &K : k_vals)
            K = ODEDeriv<Scalar>::Zero(this->ode_.output_rows());

        ODEState<Scalar> xtup = x.template head<DODE::IRC>(this->ode_.input_rows());

        Scalar t0 = xtup[this->ode_.t_var()];
        Scalar tf = x[this->input_rows() - 1];
        Scalar h = tf - t0;
        xi_jac.setIdentity();
        xi_jac(this->ode_.t_var(), this->input_rows() - 1) = 0;
        k_jacs.setZero();
        this->ode_.compute_jacobian(xtup, k_vals[0], k_jacs);

        k_jacs *= h;
        kx_jacs[0] = k_jacs * xi_jac;
        kx_jacs[0].col(this->ode_.t_var()).template head<DODE::XV>(this->ode_.x_vars()) -=
            k_vals[0];
        kx_jacs[0].col(this->input_rows() - 1).template head<DODE::XV>(this->ode_.x_vars()) +=
            k_vals[0];
        k_vals[0] *= h;

        for (int i = 0; i < Stgsm1; i++) {
            const int ip1 = i + 1;
            Scalar ti = t0 + RKData::Times[i] * h;
            xtup = x.template head<DODE::IRC>(this->ode_.input_rows());
            xi_jac.setIdentity();

            xtup[this->ode_.t_var()] = ti;
            xi_jac(this->ode_.t_var(), this->ode_.t_var()) = 1.0 - RKData::Times[i];
            xi_jac(this->ode_.t_var(), this->input_rows() - 1) = RKData::Times[i];

            const int js = is_diag_ ? i : 0;
            for (int j = js; j < ip1; j++) {
                xtup.template head<DODE::XV>(this->ode_.x_vars()) +=
                    RKData::ACoeffs[i][j] * k_vals[j];
                xi_jac.template topRows<DODE::XV>(this->ode_.x_vars()) +=
                    RKData::ACoeffs[i][j] * kx_jacs[j];
            }

            k_jacs.setZero();
            this->ode_.compute_jacobian(xtup, k_vals[ip1], k_jacs);
            kx_jacs[ip1].noalias() = h * k_jacs * xi_jac;
            kx_jacs[ip1].col(this->ode_.t_var()).template head<DODE::XV>(this->ode_.x_vars()) -=
                k_vals[ip1];
            kx_jacs[ip1].col(this->input_rows() - 1).template head<DODE::XV>(this->ode_.x_vars()) +=
                k_vals[ip1];
            k_vals[ip1] *= h;
        }
        xtup = x.template head<DODE::IRC>(this->ode_.input_rows());
        xtup[this->ode_.t_var()] = tf;
        xi_jac.setIdentity();

        xi_jac(this->ode_.t_var(), this->ode_.t_var()) = 0;
        xi_jac(this->ode_.t_var(), (this->input_rows() - 1)) = 1;

        for (int i = 0; i < Stages; i++) {
            xtup.template head<DODE::XV>(this->ode_.x_vars()) += RKData::BCoeffs[i] * k_vals[i];
            xi_jac.template topRows<DODE::XV>(this->ode_.x_vars()) +=
                RKData::BCoeffs[i] * kx_jacs[i];
        }

        fx = xtup;
        jx = xi_jac;
    }
};

} // namespace tycho::integrators
