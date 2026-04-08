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

#include "tycho/detail/optimal_control/core/optimal_control_flags.h"
#include "tycho/vector_functions.h"
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
#include "tycho/detail/utils/flat_map.h"
#include "tycho/detail/utils/function_return_type.h"
#include "tycho/detail/utils/get_core_count.h"
#include "tycho/detail/utils/math_functions.h"
#include "tycho/detail/utils/sizing_helpers.h"
#include "tycho/detail/utils/std_extensions.h"
#include "tycho/detail/utils/thread_pool.h"
#include "tycho/detail/utils/type_name.h"
#include "tycho/detail/utils/type_storage.h"

namespace tycho::oc {

// Import cross-namespace types from vf and utils.
using utils::SZ_MAX;
using utils::SZ_PROD;
using utils::SZ_SUM;
using vf::Arguments;
using vf::DenseDerivativeMode;
using vf::GenericFunction;
using vf::Is_SuperScalar;
using vf::StackedOutputs;
using vf::ThreadingFlags;
using vf::CMatRef;
using vf::CVecRef;
using vf::MatRef;
using vf::VecRef;
using vf::VectorExpression;
using vf::VectorFunction;

// Solvers types
using tycho::solvers::SolverIndexingData;

template <class DODE, class Integrator> struct ShootingDefect_Impl {
    static auto Definition(const DODE &ode, const Integrator &integ) {
        constexpr int IRC = SZ_SUM<SZ_PROD<DODE::XtUV, 2>::value, DODE::PV>::value;
        int input_rows = ode.xtu_vars() * 2 + ode.p_vars();

        auto args = Arguments<IRC>(input_rows);
        // Input[x1,t1,u1,x2,t2,u2,pv]

        auto x1 = args.template head<DODE::XtUV>(ode.xtu_vars());
        auto t1 = x1.template coeff<DODE::XV>(ode.x_vars());
        auto x2 = args.template segment<DODE::XtUV, DODE::XtUV>(ode.xtu_vars(), ode.xtu_vars());
        auto t2 = x2.template coeff<DODE::XV>(ode.x_vars());

        auto tm = 0.5 * (t1 + t2);

        auto pvars = args.template tail<DODE::PV>(ode.p_vars());

        auto make_state = [&](auto xx) {
            if constexpr (DODE::PV == 0) {
                return StackedOutputs{xx, tm};
            } else {
                return StackedOutputs{xx, pvars, tm};
            }
        };

        auto Arc1Input = make_state(x1);
        auto Arc2Input = make_state(x2);

        auto defect = integ.eval(Arc1Input).template head<DODE::XV>(ode.x_vars()) -
                      integ.eval(Arc2Input).template head<DODE::XV>(ode.x_vars());

        return defect;
    }
};

template <class DODE, class Integrator>
struct ShootingDefect
    : VectorExpression<ShootingDefect<DODE, Integrator>, ShootingDefect_Impl<DODE, Integrator>,
                       const DODE &, const Integrator &> {
    using Base =
        VectorExpression<ShootingDefect<DODE, Integrator>, ShootingDefect_Impl<DODE, Integrator>,
                         const DODE &, const Integrator &>;
    // using Base::Base;
    ShootingDefect() {}
    ShootingDefect(const DODE &ode, const Integrator &integ) : Base(ode, integ) {}
    bool enable_hessian_sparsity_ = false;
};

template <class DODE, class Integrator>
struct CentralShootingDefect
    : VectorFunction<CentralShootingDefect<DODE, Integrator>,
                     SZ_SUM<SZ_PROD<DODE::XtUV, 2>::value, DODE::PV>::value, DODE::XV> {

    using Base = VectorFunction<CentralShootingDefect<DODE, Integrator>,
                                SZ_SUM<SZ_PROD<DODE::XtUV, 2>::value, DODE::PV>::value, DODE::XV>;

    VF_TYPE_ALIASES(Base);

    template <class Scalar> using ODEState = typename DODE::template Input<Scalar>;
    template <class Scalar> using ODEDeriv = typename DODE::template Output<Scalar>;
    template <class Scalar> using IntegJac = typename Integrator::template Jacobian<Scalar>;

    static constexpr bool is_vectorizable = true;
    bool enable_hessian_sparsity_ = false;

    DODE ode_;
    Integrator integ_;

    CentralShootingDefect(const DODE &ode, const Integrator &integ) : ode_(ode), integ_(integ) {
        this->set_io_rows(2 * this->ode_.xtu_vars() + this->ode_.p_vars(), this->ode_.x_vars());
    }

    CentralShootingDefect() {}

    template <class InType>
    void extract_scalar_inputs(CVecRef<InType> X1X2,
                               std::vector<Input<double>> &X1X2s) const {

        typedef typename InType::Scalar Scalar;

        X1X2s.resize(Scalar::SizeAtCompileTime);
        for (int v = 0; v < Scalar::SizeAtCompileTime; v++) {
            X1X2s[v].resize(this->input_rows());
            for (int i = 0; i < this->input_rows(); i++) {
                X1X2s[v][i] = X1X2[i][v];
            }
        }
    }

    template <class InType>
    void extract_scalar_lmults(CVecRef<InType> Lf,
                               std::vector<Output<double>> &Lfs) const {

        typedef typename InType::Scalar Scalar;

        Lfs.resize(Scalar::SizeAtCompileTime);
        for (int v = 0; v < Scalar::SizeAtCompileTime; v++) {
            Lfs[v].resize(this->output_rows());
            for (int i = 0; i < this->output_rows(); i++) {
                Lfs[v][i] = Lf[i][v];
            }
        }
    }

    void get_input_states_tfs(const std::vector<Input<double>> &X1X2s,
                              std::vector<ODEState<double>> &Xs, Eigen::VectorXd &tfs) const {

        Xs.resize(2 * X1X2s.size());
        tfs.resize(2 * X1X2s.size());

        for (int i = 0; i < X1X2s.size(); i++) {

            Xs[2 * i].resize(this->ode_.input_rows());
            Xs[2 * i + 1].resize(this->ode_.input_rows());

            Xs[2 * i].head(this->ode_.xtu_vars()) = X1X2s[i].head(this->ode_.xtu_vars());
            Xs[2 * i + 1].head(this->ode_.xtu_vars()) =
                X1X2s[i].segment(this->ode_.xtu_vars(), this->ode_.xtu_vars());

            double tm = (Xs[2 * i][this->ode_.t_var()] + Xs[2 * i + 1][this->ode_.t_var()]) / 2.0;

            tfs[2 * i] = tm;
            tfs[2 * i + 1] = tm;

            if constexpr (DODE::PV != 0) {

                Xs[2 * i].tail(this->ode_.p_vars()) = X1X2s[i].tail(this->ode_.p_vars());
                Xs[2 * i + 1].tail(this->ode_.p_vars()) = X1X2s[i].tail(this->ode_.p_vars());
            }
        }
    }
    void get_lmults(const std::vector<Output<double>> &Ls,
                    std::vector<ODEState<double>> &Lfs) const {

        Lfs.resize(2 * Ls.size());

        for (int i = 0; i < Ls.size(); i++) {

            Lfs[2 * i].resize(this->ode_.input_rows());
            Lfs[2 * i + 1].resize(this->ode_.input_rows());
            Lfs[2 * i].setZero();
            Lfs[2 * i + 1].setZero();

            Lfs[2 * i].head(this->ode_.x_vars()) = Ls[i];
            Lfs[2 * i + 1].head(this->ode_.x_vars()) = Ls[i];
        }
    }

    std::vector<Output<double>> compute_impl_v(const std::vector<Input<double>> &X1X2s) const {

        std::vector<ODEState<double>> Xs;
        Eigen::VectorXd tfs;
        std::vector<ODEState<double>> Xfs;

        this->get_input_states_tfs(X1X2s, Xs, tfs);

        Xfs = this->integ_.integrate(Xs, tfs);

        std::vector<Output<double>> fxs(X1X2s.size());

        for (int i = 0; i < X1X2s.size(); i++) {
            fxs[i] =
                Xfs[2 * i].head(this->ode_.x_vars()) - Xfs[2 * i + 1].head(this->ode_.x_vars());
        }
        return fxs;
    }

    std::tuple<std::vector<Output<double>>, std::vector<Jacobian<double>>>
    compute_jacobian_impl_v(const std::vector<Input<double>> &X1X2s) const {

        std::vector<ODEState<double>> Xs;
        Eigen::VectorXd tfs;

        this->get_input_states_tfs(X1X2s, Xs, tfs);
        auto Xfs_Jfs = this->integ_.integrate_stm(Xs, tfs);

        std::vector<Output<double>> fxs(X1X2s.size());
        std::vector<Jacobian<double>> jxs(X1X2s.size());

        Eigen::Matrix<double, DODE::XV, SZ_PROD<Integrator::IRC, 2>::value> IJac(
            ode_.output_rows(), integ_.input_rows() * 2);
        Eigen::Matrix<double, SZ_PROD<Integrator::IRC, 2>::value, Base::IRC> XJac(
            integ_.input_rows() * 2, this->input_rows());

        XJac.setZero();

        XJac.topLeftCorner(ode_.xtu_vars(), ode_.xtu_vars()).setIdentity();
        XJac.block(ode_.xtu_vars(), 2 * ode_.xtu_vars(), ode_.p_vars(), ode_.p_vars())
            .setIdentity();
        XJac(ode_.input_rows(), ode_.t_var()) = .5;
        XJac(ode_.input_rows(), ode_.xtu_vars() + ode_.t_var()) = .5;

        XJac.block(integ_.input_rows(), ode_.xtu_vars(), ode_.xtu_vars(), ode_.xtu_vars())
            .setIdentity();
        XJac.block(integ_.input_rows() + ode_.xtu_vars(), 2 * ode_.xtu_vars(), ode_.p_vars(),
                   ode_.p_vars())
            .setIdentity();

        XJac(integ_.input_rows() + ode_.input_rows(), ode_.t_var()) = .5;
        XJac(integ_.input_rows() + ode_.input_rows(), ode_.xtu_vars() + ode_.t_var()) = .5;

        for (int i = 0; i < X1X2s.size(); i++) {

            auto &[Xf1, Jf1] = Xfs_Jfs[2 * i];
            auto &[Xf2, Jf2] = Xfs_Jfs[2 * i + 1];

            Jf2 *= -1.0;

            fxs[i] = Xf1.head(ode_.x_vars()) - Xf2.head(ode_.x_vars());

            IJac.leftCols(integ_.input_rows()) = Jf1.topRows(ode_.x_vars());
            IJac.rightCols(integ_.input_rows()) = Jf2.topRows(ode_.x_vars());

            jxs[i].noalias() = IJac * XJac;
        }
        return std::tuple{fxs, jxs};
    }

    std::tuple<std::vector<Output<double>>, std::vector<Jacobian<double>>,
               std::vector<Hessian<double>>>
    compute_all_impl_v(const std::vector<Input<double>> &X1X2s,
                       const std::vector<Output<double>> &Ls) const {

        std::vector<ODEState<double>> Xs;
        Eigen::VectorXd tfs;
        std::vector<ODEState<double>> Lfs;

        this->get_input_states_tfs(X1X2s, Xs, tfs);
        this->get_lmults(Ls, Lfs);

        auto Xfs_Jfs_Hfs = this->integ_.integrate_stm2(Xs, tfs, Lfs);

        std::vector<Output<double>> fxs(X1X2s.size());
        std::vector<Jacobian<double>> jxs(X1X2s.size());
        std::vector<Hessian<double>> hxs(X1X2s.size());

        Eigen::Matrix<double, DODE::XV, SZ_PROD<Integrator::IRC, 2>::value> IJac(
            ode_.output_rows(), integ_.input_rows() * 2);
        IJac.setZero();

        Eigen::Matrix<double, SZ_PROD<Integrator::IRC, 2>::value,
                      SZ_PROD<Integrator::IRC, 2>::value>
            IHess(integ_.input_rows() * 2, integ_.input_rows() * 2);

        IHess.setZero();

        Eigen::Matrix<double, SZ_PROD<Integrator::IRC, 2>::value, Base::IRC> XJac(
            integ_.input_rows() * 2, this->input_rows());
        XJac.setZero();

        XJac.topLeftCorner(ode_.xtu_vars(), ode_.xtu_vars()).setIdentity();
        XJac.block(ode_.xtu_vars(), 2 * ode_.xtu_vars(), ode_.p_vars(), ode_.p_vars())
            .setIdentity();
        XJac(ode_.input_rows(), ode_.t_var()) = .5;
        XJac(ode_.input_rows(), ode_.xtu_vars() + ode_.t_var()) = .5;

        XJac.block(integ_.input_rows(), ode_.xtu_vars(), ode_.xtu_vars(), ode_.xtu_vars())
            .setIdentity();
        XJac.block(integ_.input_rows() + ode_.xtu_vars(), 2 * ode_.xtu_vars(), ode_.p_vars(),
                   ode_.p_vars())
            .setIdentity();

        XJac(integ_.input_rows() + ode_.input_rows(), ode_.t_var()) = .5;
        XJac(integ_.input_rows() + ode_.input_rows(), ode_.xtu_vars() + ode_.t_var()) = .5;

        for (int i = 0; i < X1X2s.size(); i++) {

            auto &[Xf1, Jf1, Hf1] = Xfs_Jfs_Hfs[2 * i];
            auto &[Xf2, Jf2, Hf2] = Xfs_Jfs_Hfs[2 * i + 1];

            Jf2 *= -1.0;
            Hf2 *= -1.0;

            fxs[i] = Xf1.head(this->ode_.x_vars()) - Xf2.head(this->ode_.x_vars());

            IJac.leftCols(integ_.input_rows()) = Jf1.topRows(ode_.x_vars());
            IJac.rightCols(integ_.input_rows()) = Jf2.topRows(ode_.x_vars());

            IHess.topLeftCorner(integ_.input_rows(), integ_.input_rows()) = Hf1;
            IHess.bottomRightCorner(integ_.input_rows(), integ_.input_rows()) = Hf2;

            jxs[i].noalias() = IJac * XJac;
            hxs[i].noalias() = XJac.transpose() * IHess * XJac;
        }
        return std::tuple{fxs, jxs, hxs};
    }

    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();

        std::vector<Input<double>> X1X2s;

        if constexpr (!Is_SuperScalar<Scalar>::value) {
            X1X2s.push_back(x);
        } else {
            this->extract_scalar_inputs(x, X1X2s);
        }

        auto fxs = this->compute_impl_v(X1X2s);

        if constexpr (!Is_SuperScalar<Scalar>::value) {
            fx = fxs.front();
        } else {
            for (int v = 0; v < Scalar::SizeAtCompileTime; v++) {
                for (int i = 0; i < this->output_rows(); i++) {
                    fx[i][v] = fxs[v][i];
                }
            }
        }
    }
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();

        std::vector<Input<double>> X1X2s;

        if constexpr (!Is_SuperScalar<Scalar>::value) {
            X1X2s.push_back(x);
        } else {
            this->extract_scalar_inputs(x, X1X2s);
        }

        auto [fxs, jxs] = this->compute_jacobian_impl_v(X1X2s);

        if constexpr (!Is_SuperScalar<Scalar>::value) {
            fx = fxs.front();
            jx = jxs.front();

        } else {
            for (int v = 0; v < Scalar::SizeAtCompileTime; v++) {
                for (int i = 0; i < this->output_rows(); i++) {
                    fx[i][v] = fxs[v][i];
                }

                for (int j = 0; j < this->input_rows(); j++) {
                    for (int i = 0; i < this->output_rows(); i++) {
                        jx(i, j)[v] = jxs[v](i, j);
                    }
                }
            }
        }
    }
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        CVecRef<InType> x, CVecRef<OutType> fx_,
        CMatRef<JacType> jx_, CVecRef<AdjGradType> adjgrad_,
        CMatRef<AdjHessType> adjhess_, CVecRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();
        VecRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
        MatRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

        std::vector<Input<double>> X1X2s;
        std::vector<Output<double>> Lfs;

        if constexpr (!Is_SuperScalar<Scalar>::value) {
            X1X2s.push_back(x);
            Lfs.push_back(adjvars);
        } else {
            this->extract_scalar_inputs(x, X1X2s);
            this->extract_scalar_lmults(adjvars, Lfs);
        }

        auto [fxs, jxs, hxs] = this->compute_all_impl_v(X1X2s, Lfs);

        if constexpr (!Is_SuperScalar<Scalar>::value) {
            fx = fxs.front();
            jx = jxs.front();
            adjhess = hxs.front();
        } else {
            for (int v = 0; v < Scalar::SizeAtCompileTime; v++) {

                for (int i = 0; i < this->output_rows(); i++) {
                    fx[i][v] = fxs[v][i];
                }

                for (int j = 0; j < this->input_rows(); j++) {
                    for (int i = 0; i < this->output_rows(); i++) {
                        jx(i, j)[v] = jxs[v](i, j);
                    }
                }

                for (int j = 0; j < this->input_rows(); j++) {
                    for (int i = 0; i < this->input_rows(); i++) {
                        adjhess(i, j)[v] = hxs[v](i, j);
                    }
                }
            }
        }

        adjgrad = jx.transpose() * adjvars;
    }

    void constraints_jacobian_adjointgradient_adjointhessian_test(
        ConstEigenRef<Eigen::VectorXd> X, ConstEigenRef<Eigen::VectorXd> L,
        EigenRef<Eigen::VectorXd> FX, EigenRef<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
        EigenRef<Eigen::VectorXi> KKTLocations, EigenRef<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex> &KKTLocks, const SolverIndexingData &data) const {

        Input<double> x(this->input_rows());
        Output<double> l(this->output_rows());

        Eigen::Map<Output<double>> fx(NULL, this->output_rows());
        Eigen::Map<Input<double>> agx(NULL, this->input_rows());

        std::vector<Input<double>> X1X2s;
        std::vector<Output<double>> Lfs;

        for (int V = 0; V < data.num_appl(); V++) {
            this->gather_input(X, x, V, data);
            this->gather_mult(L, l, V, data);

            X1X2s.push_back(x);
            Lfs.push_back(l);
        }

        auto [fxs, jxs, hxs] = this->compute_all_impl_v(X1X2s, Lfs);

        for (int V = 0; V < data.num_appl(); V++) {

            new (&fx) Eigen::Map<Output<double>>(FX.data() + data.inner_constraint_starts_[V],
                                                 this->output_rows());
            new (&agx) Eigen::Map<Input<double>>(AGX.data() + data.inner_gradient_starts_[V],
                                                 this->input_rows());

            fx = fxs[V];
            agx = jxs[V].transpose() * Lfs[V];
            this->derived().kkt_fill_all(V, jxs[V], hxs[V], KKTmat, KKTLocations, KKTClashes,
                                         KKTLocks, data);
        }
    }
};

} // namespace tycho::oc
