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
#include "tycho/detail/utils/std_extensions.h"
#include "tycho/detail/utils/math_functions.h"
#include "tycho/detail/utils/type_name.h"
#include "tycho/detail/utils/type_storage.h"
#include "tycho/detail/utils/sizing_helpers.h"
#include "tycho/detail/utils/thread_pool.h"
#include "tycho/detail/utils/flat_map.h"
#include "tycho/detail/utils/function_return_type.h"
#include "tycho/detail/utils/get_core_count.h"
#include "tycho/detail/utils/crtp_base.h"

namespace tycho::oc {

// Import cross-namespace types from vf and utils.
using utils::SZ_SUM;
using utils::SZ_MAX;
using utils::SZ_PROD;
using vf::DenseDerivativeMode;
using vf::GenericFunction;
using vf::VectorExpression;
using vf::VectorFunction;
using vf::ThreadingFlags;
using vf::Arguments;
using vf::Is_SuperScalar;
using vf::StackedOutputs;

// Solvers types
using tycho::solvers::SolverIndexingData;

template <class DODE, class Integrator> struct ShootingDefect_Impl {
    static auto Definition(const DODE &ode, const Integrator &integ) {
        constexpr int IRC = SZ_SUM<SZ_PROD<DODE::XtUV, 2>::value, DODE::PV>::value;
        int input_rows = ode.XtUVars() * 2 + ode.PVars();

        auto args = Arguments<IRC>(input_rows);
        // Input[x1,t1,u1,x2,t2,u2,pv]

        auto x1 = args.template head<DODE::XtUV>(ode.XtUVars());
        auto t1 = x1.template coeff<DODE::XV>(ode.XVars());
        auto x2 = args.template segment<DODE::XtUV, DODE::XtUV>(ode.XtUVars(), ode.XtUVars());
        auto t2 = x2.template coeff<DODE::XV>(ode.XVars());

        auto tm = 0.5 * (t1 + t2);

        auto pvars = args.template tail<DODE::PV>(ode.PVars());

        auto make_state = [&](auto xx) {
            if constexpr (DODE::PV == 0) {
                return StackedOutputs{xx, tm};
            } else {
                return StackedOutputs{xx, pvars, tm};
            }
        };

        auto Arc1Input = make_state(x1);
        auto Arc2Input = make_state(x2);

        auto defect = integ.eval(Arc1Input).template head<DODE::XV>(ode.XVars()) -
                      integ.eval(Arc2Input).template head<DODE::XV>(ode.XVars());

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
    bool EnableHessianSparsity = false;
};

template <class DODE, class Integrator>
struct CentralShootingDefect
    : VectorFunction<CentralShootingDefect<DODE, Integrator>,
                     SZ_SUM<SZ_PROD<DODE::XtUV, 2>::value, DODE::PV>::value, DODE::XV> {

    using Base = VectorFunction<CentralShootingDefect<DODE, Integrator>,
                                SZ_SUM<SZ_PROD<DODE::XtUV, 2>::value, DODE::PV>::value, DODE::XV>;

    DENSE_FUNCTION_BASE_TYPES(Base);

    template <class Scalar> using ODEState = typename DODE::template Input<Scalar>;
    template <class Scalar> using ODEDeriv = typename DODE::template Output<Scalar>;
    template <class Scalar> using IntegJac = typename Integrator::template Jacobian<Scalar>;

    static const bool is_vectorizable = true;
    bool EnableHessianSparsity = false;

    DODE ode;
    Integrator integ;

    CentralShootingDefect(const DODE &ode, const Integrator &integ) : ode(ode), integ(integ) {
        this->set_io_rows(2 * this->ode.XtUVars() + this->ode.PVars(), this->ode.XVars());
    }

    CentralShootingDefect() {}

    template <class InType>
    void extract_scalar_inputs(ConstVectorBaseRef<InType> X1X2,
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
    void extract_scalar_lmults(ConstVectorBaseRef<InType> Lf,
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

            Xs[2 * i].resize(this->ode.input_rows());
            Xs[2 * i + 1].resize(this->ode.input_rows());

            Xs[2 * i].head(this->ode.XtUVars()) = X1X2s[i].head(this->ode.XtUVars());
            Xs[2 * i + 1].head(this->ode.XtUVars()) =
                X1X2s[i].segment(this->ode.XtUVars(), this->ode.XtUVars());

            double tm = (Xs[2 * i][this->ode.TVar()] + Xs[2 * i + 1][this->ode.TVar()]) / 2.0;

            tfs[2 * i] = tm;
            tfs[2 * i + 1] = tm;

            if constexpr (DODE::PV != 0) {

                Xs[2 * i].tail(this->ode.PVars()) = X1X2s[i].tail(this->ode.PVars());
                Xs[2 * i + 1].tail(this->ode.PVars()) = X1X2s[i].tail(this->ode.PVars());
            }
        }
    }
    void get_lmults(const std::vector<Output<double>> &Ls,
                    std::vector<ODEState<double>> &Lfs) const {

        Lfs.resize(2 * Ls.size());

        for (int i = 0; i < Ls.size(); i++) {

            Lfs[2 * i].resize(this->ode.input_rows());
            Lfs[2 * i + 1].resize(this->ode.input_rows());
            Lfs[2 * i].setZero();
            Lfs[2 * i + 1].setZero();

            Lfs[2 * i].head(this->ode.XVars()) = Ls[i];
            Lfs[2 * i + 1].head(this->ode.XVars()) = Ls[i];
        }
    }

    std::vector<Output<double>> compute_impl_v(const std::vector<Input<double>> &X1X2s) const {

        std::vector<ODEState<double>> Xs;
        Eigen::VectorXd tfs;
        std::vector<ODEState<double>> Xfs;

        this->get_input_states_tfs(X1X2s, Xs, tfs);

        Xfs = this->integ.integrate(Xs, tfs);

        std::vector<Output<double>> fxs(X1X2s.size());

        for (int i = 0; i < X1X2s.size(); i++) {
            fxs[i] = Xfs[2 * i].head(this->ode.XVars()) - Xfs[2 * i + 1].head(this->ode.XVars());
        }
        return fxs;
    }

    std::tuple<std::vector<Output<double>>, std::vector<Jacobian<double>>>
    compute_jacobian_impl_v(const std::vector<Input<double>> &X1X2s) const {

        std::vector<ODEState<double>> Xs;
        Eigen::VectorXd tfs;

        this->get_input_states_tfs(X1X2s, Xs, tfs);
        auto Xfs_Jfs = this->integ.integrate_stm(Xs, tfs);

        std::vector<Output<double>> fxs(X1X2s.size());
        std::vector<Jacobian<double>> jxs(X1X2s.size());

        Eigen::Matrix<double, DODE::XV, SZ_PROD<Integrator::IRC, 2>::value> IJac(ode.output_rows(),
                                                                                 integ.input_rows() * 2);
        Eigen::Matrix<double, SZ_PROD<Integrator::IRC, 2>::value, Base::IRC> XJac(integ.input_rows() * 2,
                                                                                  this->input_rows());

        XJac.setZero();

        XJac.topLeftCorner(ode.XtUVars(), ode.XtUVars()).setIdentity();
        XJac.block(ode.XtUVars(), 2 * ode.XtUVars(), ode.PVars(), ode.PVars()).setIdentity();
        XJac(ode.input_rows(), ode.TVar()) = .5;
        XJac(ode.input_rows(), ode.XtUVars() + ode.TVar()) = .5;

        XJac.block(integ.input_rows(), ode.XtUVars(), ode.XtUVars(), ode.XtUVars()).setIdentity();
        XJac.block(integ.input_rows() + ode.XtUVars(), 2 * ode.XtUVars(), ode.PVars(), ode.PVars())
            .setIdentity();

        XJac(integ.input_rows() + ode.input_rows(), ode.TVar()) = .5;
        XJac(integ.input_rows() + ode.input_rows(), ode.XtUVars() + ode.TVar()) = .5;

        for (int i = 0; i < X1X2s.size(); i++) {

            auto &[Xf1, Jf1] = Xfs_Jfs[2 * i];
            auto &[Xf2, Jf2] = Xfs_Jfs[2 * i + 1];

            Jf2 *= -1.0;

            fxs[i] = Xf1.head(ode.XVars()) - Xf2.head(ode.XVars());

            IJac.leftCols(integ.input_rows()) = Jf1.topRows(ode.XVars());
            IJac.rightCols(integ.input_rows()) = Jf2.topRows(ode.XVars());

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

        auto Xfs_Jfs_Hfs = this->integ.integrate_stm2(Xs, tfs, Lfs);

        std::vector<Output<double>> fxs(X1X2s.size());
        std::vector<Jacobian<double>> jxs(X1X2s.size());
        std::vector<Hessian<double>> hxs(X1X2s.size());

        Eigen::Matrix<double, DODE::XV, SZ_PROD<Integrator::IRC, 2>::value> IJac(ode.output_rows(),
                                                                                 integ.input_rows() * 2);
        IJac.setZero();

        Eigen::Matrix<double, SZ_PROD<Integrator::IRC, 2>::value,
                      SZ_PROD<Integrator::IRC, 2>::value>
            IHess(integ.input_rows() * 2, integ.input_rows() * 2);

        IHess.setZero();

        Eigen::Matrix<double, SZ_PROD<Integrator::IRC, 2>::value, Base::IRC> XJac(integ.input_rows() * 2,
                                                                                  this->input_rows());
        XJac.setZero();

        XJac.topLeftCorner(ode.XtUVars(), ode.XtUVars()).setIdentity();
        XJac.block(ode.XtUVars(), 2 * ode.XtUVars(), ode.PVars(), ode.PVars()).setIdentity();
        XJac(ode.input_rows(), ode.TVar()) = .5;
        XJac(ode.input_rows(), ode.XtUVars() + ode.TVar()) = .5;

        XJac.block(integ.input_rows(), ode.XtUVars(), ode.XtUVars(), ode.XtUVars()).setIdentity();
        XJac.block(integ.input_rows() + ode.XtUVars(), 2 * ode.XtUVars(), ode.PVars(), ode.PVars())
            .setIdentity();

        XJac(integ.input_rows() + ode.input_rows(), ode.TVar()) = .5;
        XJac(integ.input_rows() + ode.input_rows(), ode.XtUVars() + ode.TVar()) = .5;

        for (int i = 0; i < X1X2s.size(); i++) {

            auto &[Xf1, Jf1, Hf1] = Xfs_Jfs_Hfs[2 * i];
            auto &[Xf2, Jf2, Hf2] = Xfs_Jfs_Hfs[2 * i + 1];

            Jf2 *= -1.0;
            Hf2 *= -1.0;

            fxs[i] = Xf1.head(this->ode.XVars()) - Xf2.head(this->ode.XVars());

            IJac.leftCols(integ.input_rows()) = Jf1.topRows(ode.XVars());
            IJac.rightCols(integ.input_rows()) = Jf2.topRows(ode.XVars());

            IHess.topLeftCorner(integ.input_rows(), integ.input_rows()) = Hf1;
            IHess.bottomRightCorner(integ.input_rows(), integ.input_rows()) = Hf2;

            jxs[i].noalias() = IJac * XJac;
            hxs[i].noalias() = XJac.transpose() * IHess * XJac;
        }
        return std::tuple{fxs, jxs, hxs};
    }

    template <class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();

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
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

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
        ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
        ConstMatrixBaseRef<JacType> jx_, ConstVectorBaseRef<AdjGradType> adjgrad_,
        ConstMatrixBaseRef<AdjHessType> adjhess_, ConstVectorBaseRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
        VectorBaseRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
        MatrixBaseRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

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

        for (int V = 0; V < data.NumAppl(); V++) {
            this->gather_input(X, x, V, data);
            this->gather_mult(L, l, V, data);

            X1X2s.push_back(x);
            Lfs.push_back(l);
        }

        auto [fxs, jxs, hxs] = this->compute_all_impl_v(X1X2s, Lfs);

        for (int V = 0; V < data.NumAppl(); V++) {

            new (&fx) Eigen::Map<Output<double>>(FX.data() + data.InnerConstraintStarts[V],
                                                 this->output_rows());
            new (&agx)
                Eigen::Map<Input<double>>(AGX.data() + data.InnerGradientStarts[V], this->input_rows());

            fx = fxs[V];
            agx = jxs[V].transpose() * Lfs[V];
            this->derived().kkt_fill_all(V, jxs[V], hxs[V], KKTmat, KKTLocations, KKTClashes,
                                       KKTLocks, data);
        }
    }
};

} // namespace tycho::oc

