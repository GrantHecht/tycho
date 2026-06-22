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
using vf::CDiagRef;
using vf::CEigRef;
using vf::CMatRef;
using vf::CVecRef;
using vf::DenseDerivativeMode;
using vf::DiagRef;
using vf::EigRef;
using vf::MatRef;
using vf::NestedCallAndAppendChain;
using vf::NestedCallAndAppendChain2;
using vf::StackedOutputs;
using vf::StaticScaleBase;
using vf::VecRef;
using vf::VectorExpression;
using vf::VectorFunction;

/// @brief One-step Runge-Kutta integrator implemented as a VectorFunction.
///
/// Maps a packed input `[x0; tf]` (ODE state at t0 concatenated with final time)
/// to the propagated ODE state at tf using the Butcher tableau specified by RKOp.
/// Used by AdaptiveDriver and Integrator to propagate a single step.
///
/// @tparam DODE ODE VectorFunction type (must expose IRC, XV, UV, PV, t_var(), etc.).
/// @tparam RKOp IVPAlg selector choosing the Butcher tableau via RKCoeffs<RKOp>.
template <class DODE, IVPAlg RKOp>
struct RKStepper : VectorFunction<RKStepper<DODE, RKOp>, SZ_SUM<DODE::IRC, 1>::value, DODE::IRC> {
    /// CRTP base type.
    using Base = VectorFunction<RKStepper<DODE, RKOp>, SZ_SUM<DODE::IRC, 1>::value, DODE::IRC>;
    VF_TYPE_ALIASES(Base);

    /// @internal
    /// @brief Scalar-typed ODE output.
    template <class Scalar> using ODEDeriv = typename DODE::template Output<Scalar>;
    /// @internal
    /// @brief Scalar-typed ODE input.
    template <class Scalar> using ODEState = typename DODE::template Input<Scalar>;
    /// @internal
    /// @brief Scalar-typed ODE Jacobian.
    template <class Scalar> using ODEJacobian = typename DODE::template Jacobian<Scalar>;
    /// @internal
    /// @brief Scalar-typed ODE Hessian.
    template <class Scalar> using ODEHessian = typename DODE::template Hessian<Scalar>;

    /// @internal
    /// @brief Enables SuperScalar dispatch for batch trajectories.
    static constexpr bool is_vectorizable = true;

    using RKData = RKCoeffs<RKOp>;                 ///< Butcher tableau data type for RKOp.
    static constexpr int Stages = RKData::Stages;  ///< Number of stages in the selected tableau.
    static constexpr int Stgsm1 = RKData::Stages - 1; ///< Stages minus one (used in stage loops).

    DODE ode_; ///< @internal The underlying ODE VectorFunction.

    /// @brief Default constructor; leaves ode_ uninitialized.
    RKStepper() {}
    /// @brief Constructs from an ODE VectorFunction; sets I/O dimensions automatically.
    RKStepper(DODE ode) : ode_(ode) {
        this->set_io_rows(this->ode_.input_rows() + 1, this->ode_.input_rows());
    }

    /// @internal
    /// @brief Propagate ODE from x0 to tf via RKOp; writes final state into fx_.
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
                Scalar ti = t0 + RKData::C[i] * h;
                xtup = x.template segment<DODE::IRC>(0, this->ode_.input_rows());
                xtup[this->ode_.t_var()] = ti;
                const int ip1 = i + 1;
                const int js = 0;
                for (int j = js; j < ip1; j++) {
                    xtup.template segment<DODE::XV>(0, this->ode_.x_vars()) +=
                        Scalar(RKData::A[i][j]) * k_vals[j];
                }

                this->ode_.compute(xtup, k_vals[ip1]);

                k_vals[ip1] *= h;
            }
            xtup = x.template segment<DODE::IRC>(0, this->ode_.input_rows());
            xtup[this->ode_.t_var()] = tf;
            for (int i = 0; i < Stages; i++) {
                xtup.template segment<DODE::XV>(0, this->ode_.x_vars()) +=
                    Scalar(RKData::B[i]) * k_vals[i];
            }
            fx = xtup; // Next State
        };

        BumpAllocator::allocate_run(
            Impl, ArrayOfTempSpecs<ODEDeriv<Scalar>, Stages>(this->ode_.output_rows(), 1),
            TempSpec<ODEState<Scalar>>(this->ode_.input_rows(), 1));
    }

    /// @internal
    /// @brief Propagate ODE and compute the state-transition Jacobian via dense output.
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
                Scalar ti = t0 + RKData::C[i] * h;
                xtup = x.template segment<DODE::IRC>(0, this->ode_.input_rows());
                xi_jac.setIdentity();

                xtup[this->ode_.t_var()] = ti;

                xi_jac(this->ode_.t_var(), this->ode_.t_var()) = Scalar(1.0) - Scalar(RKData::C[i]);
                xi_jac(this->ode_.t_var(), this->input_rows() - 1) = Scalar(RKData::C[i]);

                const int ip1 = i + 1;
                const int js = 0;
                for (int j = js; j < ip1; j++) {
                    xtup.template segment<DODE::XV>(0, this->ode_.x_vars()) +=
                        Scalar(RKData::A[i][j]) * k_vals[j];
                    xi_jac.template topRows<DODE::XV>(this->ode_.x_vars()) +=
                        Scalar(RKData::A[i][j]) * kx_jacs[j];
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
                    Scalar(RKData::B[i]) * k_vals[i];
                xi_jac.template topRows<DODE::XV>(this->ode_.x_vars()) +=
                    Scalar(RKData::B[i]) * kx_jacs[i];
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

    /// @internal
    /// @brief Propagate ODE, compute Jacobian, adjoint gradient, and adjoint Hessian.
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
                Scalar ti = t0 + RKData::C[i] * h;
                xtup = x.template segment<DODE::IRC>(0, this->ode_.input_rows());
                xtup[this->ode_.t_var()] = ti;
                const int ip1 = i + 1;
                const int js = 0;
                for (int j = js; j < ip1; j++) {
                    xtup.template segment<DODE::XV>(0, this->ode_.x_vars()) +=
                        Scalar(RKData::A[i][j]) * k_vals[j];
                }
                xs[ip1] = xtup;
                this->ode_.compute(xtup, k_vals[ip1]);
                k_vals[ip1] *= h;
            }

            xtup = x.template segment<DODE::IRC>(0, this->ode_.input_rows());
            xtup[this->ode_.t_var()] = tf;
            for (int i = 0; i < Stages; i++) {
                xtup.template segment<DODE::XV>(0, this->ode_.x_vars()) +=
                    Scalar(RKData::B[i]) * k_vals[i];
                kx_mults[i] = adjvars * Scalar(RKData::B[i]) * h;
            }

            fx = xtup; // Next State

            for (int i = Stgsm1 - 1; i >= 0; i--) {

                const int ip1 = i + 1;
                const int js = 0;
                k_vals[ip1].setZero();
                this->ode_.compute_jacobian_adjointgradient_adjointhessian(
                    xs[ip1], k_vals[ip1], k_jacs[ip1], k_grads[ip1], k_hesses[ip1],
                    kx_mults[ip1].head(this->ode_.output_rows()));

                for (int j = js; j < ip1; j++) {
                    kx_mults[j] += k_grads[ip1] * Scalar(RKData::A[i][j]) * h;
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

            for (int i = 0; i < Stgsm1; i++) {
                Scalar ti = t0 + RKData::C[i] * h;
                xi_jac.setIdentity();

                xi_jac(this->ode_.t_var(), this->ode_.t_var()) = Scalar(1.0) - Scalar(RKData::C[i]);
                xi_jac(this->ode_.t_var(), this->input_rows() - 1) = Scalar(RKData::C[i]);

                const int ip1 = i + 1;
                const int js = 0;
                for (int j = js; j < ip1; j++) {

                    xi_jac.template topRows<DODE::XV>(this->ode_.x_vars()) +=
                        Scalar(RKData::A[i][j]) * kx_jacs[j];
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
                    Scalar(RKData::B[i]) * kx_jacs[i];
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

    /// @brief Required by SolverBase; unreachable because RKStepper is never registered as a
    /// constraint.
    void constraints(ConstEigenRef<Eigen::VectorXd> X, EigenRef<Eigen::VectorXd> FX,
                     const SolverIndexingData &data) const {};
    /// @internal
    /// @brief No-op; satisfies SolverBase interface — RKStepper is never registered as a
    /// constraint.
    void constraints_adjointgradient(ConstEigenRef<Eigen::VectorXd> X,
                                     ConstEigenRef<Eigen::VectorXd> L, EigenRef<Eigen::VectorXd> FX,
                                     EigenRef<Eigen::VectorXd> AGX,
                                     const SolverIndexingData &data) const {};
    /// @internal
    /// @brief No-op; satisfies SolverBase interface — RKStepper is never registered as a
    /// constraint.
    void constraints_jacobian(ConstEigenRef<Eigen::VectorXd> X, Eigen::Ref<Eigen::VectorXd> FX,
                              Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                              Eigen::Ref<Eigen::VectorXi> KKTLocations,
                              Eigen::Ref<Eigen::VectorXi> KKTClashes,
                              std::vector<std::mutex> &KKTLocks,
                              const SolverIndexingData &data) const {}
    /// @internal
    /// @brief No-op; satisfies SolverBase interface — RKStepper is never registered as a
    /// constraint.
    void constraints_jacobian_adjointgradient(
        ConstEigenRef<Eigen::VectorXd> X, ConstEigenRef<Eigen::VectorXd> L,
        Eigen::Ref<Eigen::VectorXd> FX, Eigen::Ref<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
        EigenRef<Eigen::VectorXi> KKTLocations, EigenRef<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex> &KKTLocks, const SolverIndexingData &data) const {}
    /// @internal
    /// @brief No-op; satisfies SolverBase interface — RKStepper is never registered as a
    /// constraint.
    void constraints_jacobian_adjointgradient_adjointhessian(
        ConstEigenRef<Eigen::VectorXd> X, ConstEigenRef<Eigen::VectorXd> L,
        EigenRef<Eigen::VectorXd> FX, EigenRef<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
        EigenRef<Eigen::VectorXi> KKTLocations, EigenRef<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex> &KKTLocks, const SolverIndexingData &data) const {}
};

/// @internal
/// @brief Expression-template helper that builds the static (symbolic) RKStepper definition
/// used by the collocation transcription layer (Phase / transcription). Not used in adaptive
/// integration; AdaptiveDriver and Integrator use RKStepper directly.
template <class DODE, IVPAlg RKOp> struct RKStepper_Impl {
    /// @internal
    /// @brief Builds the symbolic Runge-Kutta expression graph for the given ODE.
    static auto Definition(const DODE &ode) {
        auto ks0 = std::tuple{};
        return compute_xf<-1, decltype(ks0)>(ode, ks0);
    }

    /// @internal
    /// @brief Compile-time Butcher A[Stg][Elem] coefficient wrapped as a StaticScaleBase.
    template <int Stg, int Elem> struct ACoeff : StaticScaleBase<ACoeff<Stg, Elem>> {
        static constexpr double value = RKCoeffs<RKOp>::A[Stg][Elem]; ///< @internal A[Stg][Elem] coefficient.
    };
    /// @internal
    /// @brief Compile-time Butcher B[Elem] coefficient wrapped as a StaticScaleBase.
    template <int Elem> struct BCoeff : StaticScaleBase<BCoeff<Elem>> {
        static constexpr double value = RKCoeffs<RKOp>::B[Elem]; ///< @internal B[Elem] weight.
    };

    /// @internal
    /// @brief Recursively builds the stage chain up to and including stage Stg, then assembles
    /// the final state sum and wraps it in a NestedCallAndAppendChain ready for Phase
    /// transcription.
    template <int Stg, class Ks> static auto compute_xf(const DODE &ode, const Ks &ks) {
        constexpr int XV = DODE::XV;
        constexpr int UV = DODE::UV;
        constexpr int PV = DODE::PV;
        constexpr int IRC = SZ_SUM<DODE::IRC, 1>::value;

        auto args = Arguments<DODE::IRC + 1 + (Stg + 1) * DODE::XV>();
        auto empty = std::tuple{};

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
                return t0 + h * RKCoeffs<RKOp>::C[Stg];
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

    /// @internal
    /// @brief Assembles a full ODE input vector [x0, ti, u, p] from separate components.
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

    /// @internal
    /// @brief Accumulates the weighted k-sum for stage Stg using a compile-time Elem recursion.
    template <int Stg, int Elem, class Args>
    static auto kth_stage_sum(const DODE &ode, const Args &args) {
        if constexpr (Elem == Stg + 1) {
            return args.template head<DODE::XV>();
        } else {
            if constexpr (RKCoeffs<RKOp>::A[Stg][Elem] == 0.0) {
                return kth_stage_sum<Stg, Elem + 1, Args>(ode, args);
            } else {
                return args.template tail<DODE::XV *(Stg + 1)>()
                               .template segment<DODE::XV, DODE::XV * Elem>() *
                           ACoeff<Stg, Elem>().value +
                       kth_stage_sum<Stg, Elem + 1, Args>(ode, args);
            }
        };
    }

    /// @internal
    /// @brief Tuple-accumulating variant of kth_stage_sum; collects sub-expressions into a tuple
    /// for make_sum_tuple rather than returning a single sum expression directly.
    template <int Stg, int Elem, class Args, class Ks>
    static auto kth_stage_sum2(const DODE &ode, const Args &args, const Ks &ks) {
        if constexpr (Elem == Stg + 1) {
            auto next = args.template head<DODE::XV>();
            auto knew = std::tuple_cat(ks, std::make_tuple(next));

            return make_sum_tuple(knew);
        } else {
            if constexpr (RKCoeffs<RKOp>::A[Stg][Elem] == 0.0) {
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

    /// @internal
    /// @brief Accumulates the B-weighted final state sum xf = x0 + h * Σ B[i]*k[i] as an
    /// expression tree using a compile-time Elem recursion.
    template <int Elem, class Args, class KF>
    static auto final_state_sum(const DODE &ode, const Args &args, const KF &kf) {
        if constexpr (Elem == RKCoeffs<RKOp>::Stages - 1) {
            if constexpr (RKCoeffs<RKOp>::B[Elem] == 0.0)
                return args.template head<DODE::XV>();
            else
                return kf * BCoeff<Elem>() + args.template head<DODE::XV>();
        } else {
            if constexpr (RKCoeffs<RKOp>::B[Elem] == 0.0)
                return final_state_sum<Elem + 1, Args, KF>(ode, args, kf);
            else
                return args.template tail<DODE::XV *(RKCoeffs<RKOp>::Stages - 1)>()
                               .template segment<DODE::XV, DODE::XV * Elem>() *
                           BCoeff<Elem>() +
                       final_state_sum<Elem + 1, Args, KF>(ode, args, kf);
        };
    }
};

} // namespace tycho::integrators
