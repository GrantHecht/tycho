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

#include "tycho/detail/optimal_control/transcription/transcription_sizing.h"
#include "tycho/detail/vf/core/vector_function.h"

namespace tycho::oc {

// Import cross-namespace types from vf and utils.
using utils::BumpAllocator;
using utils::SZ_MAX;
using utils::SZ_PROD;
using utils::SZ_SUM;
using utils::TempSpec;
using vf::DenseDerivativeMode;
using vf::GenericFunction;
using vf::ThreadingFlags;
using vf::VectorExpression;
using vf::VectorFunction;

template <class DODE>
struct TrapezoidalDefects
    : VectorFunction<TrapezoidalDefects<DODE>,
                     DefectConstSizes<2, DODE::XV, DODE::UV, DODE::PV>::DefIRC,
                     DefectConstSizes<2, DODE::XV, DODE::UV, DODE::PV>::DefORC> {
    static const int CS = 2;
    using Base = VectorFunction<TrapezoidalDefects<DODE>,
                                DefectConstSizes<CS, DODE::XV, DODE::UV, DODE::PV>::DefIRC,
                                DefectConstSizes<CS, DODE::XV, DODE::UV, DODE::PV>::DefORC>;

    /////////////////////////////////////////////////////////////////////////////////
    template <class Scalar> using Output = typename Base::template Output<Scalar>;
    template <class Scalar> using Input = typename Base::template Input<Scalar>;
    template <class Scalar> using Jacobian = typename Base::template Jacobian<Scalar>;
    template <class Scalar> using Hessian = typename Base::template Hessian<Scalar>;
    ///////////////////////////////////////////////////////////////////////////////////
    template <class Scalar> using ODEOutput = typename DODE::template Output<Scalar>;
    template <class Scalar> using ODEInput = typename DODE::template Input<Scalar>;
    template <class Scalar> using ODEGrad = typename DODE::template Gradient<Scalar>;
    template <class Scalar> using ODEJacobian = typename DODE::template Jacobian<Scalar>;
    template <class Scalar> using ODEHessian = typename DODE::template Hessian<Scalar>;
    DODE ode_;
    bool enable_hessian_sparsity_ = false;
    Eigen::MatrixXi nz_locs_;
    static const bool is_vectorizable = DODE::is_vectorizable;

    void exact_hessian_sparsity(Eigen::VectorXd xtup1, Eigen::VectorXd xtup2) {

        Input<double> xin(this->input_rows());
        xin.head(this->ode_.xtu_vars()) = xtup1.head(this->ode_.xtu_vars());
        xin.segment(this->ode_.xtu_vars(), this->ode_.xtu_vars()) =
            xtup2.head(this->ode_.xtu_vars());
        xin.tail(this->ode_.p_vars()) = xtup2.tail(this->ode_.p_vars());

        Eigen::VectorXd ran(this->input_rows());
        ran.setRandom();
        ran *= 1.0e-10;

        xin += ran;

        Output<double> lm(this->output_rows());
        lm.setRandom();

        Hessian<double> hess = this->adjointhessian(xin, lm);

        for (int i = 0; i < this->input_rows(); i++) {
            for (int j = 0; j < this->input_rows(); j++) {
                if (nz_locs_(i, j) == 1) {
                    if (abs(hess(i, j)) == 0.0) {
                        nz_locs_(i, j) = 0;
                    }
                }
            }
        }
    }

    TrapezoidalDefects(const DODE &od) { this->set_ode(od); }
    void set_ode(const DODE &od) {
        this->ode_ = od;
        this->set_output_rows(this->ode_.output_rows() * (CS - 1));
        this->set_input_rows(CS * this->ode_.xtu_vars() + this->ode_.p_vars());

        nz_locs_.resize(this->input_rows(), this->input_rows());
        nz_locs_.setZero();

        int xtu = this->ode_.xtu_vars();
        nz_locs_.topLeftCorner(xtu, xtu).setOnes();
        nz_locs_.block(xtu, xtu, xtu, xtu).setOnes();
        nz_locs_.bottomRightCorner(this->ode_.p_vars(), this->ode_.p_vars()).setOnes();

        int j = 0;
        int cardinals = 2;
        nz_locs_
            .block(j * this->ode_.xtu_vars(), cardinals * this->ode_.xtu_vars(),
                   this->ode_.xtu_vars(), this->ode_.p_vars())
            .setOnes();

        nz_locs_
            .block(cardinals * this->ode_.xtu_vars(), j * this->ode_.xtu_vars(),
                   this->ode_.p_vars(), this->ode_.xtu_vars())
            .setOnes();
        j = 1;
        nz_locs_
            .block(j * this->ode_.xtu_vars(), cardinals * this->ode_.xtu_vars(),
                   this->ode_.xtu_vars(), this->ode_.p_vars())
            .setOnes();

        nz_locs_
            .block(cardinals * this->ode_.xtu_vars(), j * this->ode_.xtu_vars(),
                   this->ode_.p_vars(), this->ode_.xtu_vars())
            .setOnes();

        nz_locs_.col(this->ode_.t_var()).setOnes();
        nz_locs_.col(this->ode_.t_var() + this->ode_.xtu_vars() * (cardinals - 1)).setOnes();
        nz_locs_.row(this->ode_.t_var()).setOnes();
        nz_locs_.row(this->ode_.t_var() + this->ode_.xtu_vars() * (cardinals - 1)).setOnes();
    }

    inline bool hessian_elem_is_non_zero(int row, int col) const {
        if (this->enable_hessian_sparsity_) {
            return bool(this->nz_locs_(row, col));
        } else {
            return true;
        }
    }
    inline void add_hessian_elem(double v, int row, int col, double *mpt, const int *lpt,
                                 int &freeloc) const {
        if (this->enable_hessian_sparsity_) {
            if (bool(this->nz_locs_(row, col))) {
                mpt[lpt[freeloc]] += v;
                freeloc++;
            }
        } else {
            mpt[lpt[freeloc]] += v;
            freeloc++;
        }
    }

    template <class InType, class OutType>
    inline void compute_impl(const Eigen::MatrixBase<InType> &x,
                             Eigen::MatrixBase<OutType> const &fx_) const {
        typedef typename InType::Scalar Scalar;
        Eigen::MatrixBase<OutType> &fx = fx_.const_cast_derived();

        auto impl = [&](auto &X0, auto &X1, auto &FX0, auto &FX1) {
            X0.template segment<DODE::XtUV>(0, this->ode_.xtu_vars()) =
                x.template segment<DODE::XtUV>(0, this->ode_.xtu_vars());
            X0.tail(this->ode_.p_vars()) = x.tail(this->ode_.p_vars());

            X1.template segment<DODE::XtUV>(0, this->ode_.xtu_vars()) =
                x.template segment<DODE::XtUV>(this->ode_.xtu_vars(), this->ode_.xtu_vars());
            X1.tail(this->ode_.p_vars()) = x.tail(this->ode_.p_vars());
            Scalar h = X1[this->ode_.t_var()] - X0[this->ode_.t_var()];

            this->ode_.compute(X0, FX0);
            this->ode_.compute(X1, FX1);

            fx = (X1.template segment<DODE::XV>(0, this->ode_.x_vars()) -
                  X0.template segment<DODE::XV>(0, this->ode_.x_vars())) -
                 (h / 2.0) * (FX0 + FX1);
        };
        const int irows = this->ode_.input_rows();
        const int orows = this->ode_.output_rows();

        using IType = ODEInput<Scalar>;
        using OType = ODEOutput<Scalar>;

        BumpAllocator::allocate_run(impl, TempSpec<IType>(irows, 1), TempSpec<IType>(irows, 1),
                                    TempSpec<OType>(orows, 1), TempSpec<OType>(orows, 1));
        fx *= Scalar(-1.0);
    }

    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(const Eigen::MatrixBase<InType> &x,
                                      Eigen::MatrixBase<OutType> const &fx_,
                                      Eigen::MatrixBase<JacType> const &jx_) const {
        typedef typename InType::Scalar Scalar;
        Eigen::MatrixBase<JacType> &jx = jx_.const_cast_derived();
        Eigen::MatrixBase<OutType> &fx = fx_.const_cast_derived();

        Scalar one_half(0.5);
        Scalar one(1.0);
        auto impl = [&](auto &X0, auto &X1, auto &FX0, auto &FX1, auto &JX0, auto &JX1) {
            X0.template segment<DODE::XtUV>(0, this->ode_.xtu_vars()) =
                x.template segment<DODE::XtUV>(0, this->ode_.xtu_vars());
            X0.tail(this->ode_.p_vars()) = x.tail(this->ode_.p_vars());

            X1.template segment<DODE::XtUV>(0, this->ode_.xtu_vars()) =
                x.template segment<DODE::XtUV>(this->ode_.xtu_vars(), this->ode_.xtu_vars());
            X1.tail(this->ode_.p_vars()) = x.tail(this->ode_.p_vars());
            Scalar h = X1[this->ode_.t_var()] - X0[this->ode_.t_var()];

            this->ode_.compute_jacobian(X0, FX0, JX0);
            this->ode_.compute_jacobian(X1, FX1, JX1);

            fx = (X1.template segment<DODE::XV>(0, this->ode_.x_vars()) -
                  X0.template segment<DODE::XV>(0, this->ode_.x_vars())) -
                 (h / 2.0) * (FX0 + FX1);

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            jx.template block<DODE::XV, DODE::XV>(0, 0, this->ode_.x_vars(), this->ode_.x_vars())
                .diagonal()
                .setConstant(-one);
            jx.template block<DODE::XV, DODE::XV>(0, this->ode_.xtu_vars(), this->ode_.x_vars(),
                                                  this->ode_.x_vars())
                .diagonal()
                .setConstant(one);

            ODEOutput<Scalar> tds = -one_half * (FX0 + FX1);
            jx.col(this->ode_.t_var()) -= tds;
            jx.col(this->ode_.xtu_vars() + this->ode_.t_var()) += tds;

            jx.template block<DODE::XV, DODE::XtUV>(0, 0, this->ode_.x_vars(),
                                                    this->ode_.xtu_vars()) +=
                (-h / 2.0) * JX0.template block<DODE::XV, DODE::XtUV>(0, 0, this->ode_.x_vars(),
                                                                      this->ode_.xtu_vars());

            jx.template block<DODE::XV, DODE::XtUV>(0, this->ode_.xtu_vars(), this->ode_.x_vars(),
                                                    this->ode_.xtu_vars()) +=
                (-h / 2.0) * JX1.template block<DODE::XV, DODE::XtUV>(0, 0, this->ode_.x_vars(),
                                                                      this->ode_.xtu_vars());

            jx.template rightCols<DODE::PV>(this->ode_.p_vars()) =
                (-h / 2.0) * (JX0.template rightCols<DODE::PV>(this->ode_.p_vars()) +
                              JX1.template rightCols<DODE::PV>(this->ode_.p_vars()));

            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        };

        const int irows = this->ode_.input_rows();
        const int orows = this->ode_.output_rows();

        using IType = ODEInput<Scalar>;
        using OType = ODEOutput<Scalar>;
        using JType = ODEJacobian<Scalar>;

        BumpAllocator::allocate_run(impl, TempSpec<IType>(irows, 1), TempSpec<IType>(irows, 1),
                                    TempSpec<OType>(orows, 1), TempSpec<OType>(orows, 1),
                                    TempSpec<JType>(orows, irows), TempSpec<JType>(orows, irows));

        fx *= Scalar(-1.0);
        jx *= Scalar(-1.0);
    }

    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        const Eigen::MatrixBase<InType> &x, Eigen::MatrixBase<OutType> const &fx_,
        Eigen::MatrixBase<JacType> const &jx_, Eigen::MatrixBase<AdjGradType> const &adjgrad_,
        Eigen::MatrixBase<AdjHessType> const &adjhess_,
        const Eigen::MatrixBase<AdjVarType> &adjvars) const {
        typedef typename InType::Scalar Scalar;
        Eigen::MatrixBase<JacType> &jx = jx_.const_cast_derived();
        Eigen::MatrixBase<OutType> &fx = fx_.const_cast_derived();
        Eigen::MatrixBase<AdjGradType> &adjgrad = adjgrad_.const_cast_derived();
        Eigen::MatrixBase<AdjHessType> &adjhess = adjhess_.const_cast_derived();

        Scalar one_half(0.5);
        Scalar one(1.0);
        auto impl = [&](auto &X0, auto &X1, auto &FX0, auto &FX1, auto &JX0, auto &JX1, auto &AGX0,
                        auto &AGX1, auto &HX0, auto &HX1) {
            X0.template segment<DODE::XtUV>(0, this->ode_.xtu_vars()) =
                x.template segment<DODE::XtUV>(0, this->ode_.xtu_vars());
            X0.tail(this->ode_.p_vars()) = x.tail(this->ode_.p_vars());

            X1.template segment<DODE::XtUV>(0, this->ode_.xtu_vars()) =
                x.template segment<DODE::XtUV>(this->ode_.xtu_vars(), this->ode_.xtu_vars());
            X1.tail(this->ode_.p_vars()) = x.tail(this->ode_.p_vars());
            Scalar h = X1[this->ode_.t_var()] - X0[this->ode_.t_var()];

            this->ode_.compute_jacobian_adjointgradient_adjointhessian(X0, FX0, JX0, AGX0, HX0,
                                                                       adjvars);
            this->ode_.compute_jacobian_adjointgradient_adjointhessian(X1, FX1, JX1, AGX1, HX1,
                                                                       adjvars);
            fx = (X1.template segment<DODE::XV>(0, this->ode_.x_vars()) -
                  X0.template segment<DODE::XV>(0, this->ode_.x_vars())) -
                 (h / 2.0) * (FX0 + FX1);

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            jx.template block<DODE::XV, DODE::XV>(0, 0, this->ode_.x_vars(), this->ode_.x_vars())
                .diagonal()
                .setConstant(-one);
            jx.template block<DODE::XV, DODE::XV>(0, this->ode_.xtu_vars(), this->ode_.x_vars(),
                                                  this->ode_.x_vars())
                .diagonal()
                .setConstant(one);

            ODEOutput<Scalar> tds = -one_half * (FX0 + FX1);
            jx.col(this->ode_.t_var()) -= tds;
            jx.col(this->ode_.xtu_vars() + this->ode_.t_var()) += tds;

            jx.template block<DODE::XV, DODE::XtUV>(0, 0, this->ode_.x_vars(),
                                                    this->ode_.xtu_vars()) +=
                (-h / 2.0) * JX0.template block<DODE::XV, DODE::XtUV>(0, 0, this->ode_.x_vars(),
                                                                      this->ode_.xtu_vars());

            jx.template block<DODE::XV, DODE::XtUV>(0, this->ode_.xtu_vars(), this->ode_.x_vars(),
                                                    this->ode_.xtu_vars()) +=
                (-h / 2.0) * JX1.template block<DODE::XV, DODE::XtUV>(0, 0, this->ode_.x_vars(),
                                                                      this->ode_.xtu_vars());

            jx.template rightCols<DODE::PV>(this->ode_.p_vars()) =
                (-h / 2.0) * (JX0.template rightCols<DODE::PV>(this->ode_.p_vars()) +
                              JX1.template rightCols<DODE::PV>(this->ode_.p_vars()));

            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            adjgrad = (adjvars.transpose() * jx).transpose();

            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            adjhess.template block<DODE::XtUV, DODE::XtUV>(0, 0, this->ode_.xtu_vars(),
                                                           this->ode_.xtu_vars()) =
                (-h / 2.0) * HX0.template block<DODE::XtUV, DODE::XtUV>(0, 0, this->ode_.xtu_vars(),
                                                                        this->ode_.xtu_vars());
            adjhess.template block<DODE::XtUV, DODE::XtUV>(
                this->ode_.xtu_vars(), this->ode_.xtu_vars(), this->ode_.xtu_vars(),
                this->ode_.xtu_vars()) =
                (-h / 2.0) * HX1.template block<DODE::XtUV, DODE::XtUV>(0, 0, this->ode_.xtu_vars(),
                                                                        this->ode_.xtu_vars());

            adjhess.template bottomRightCorner<DODE::PV, DODE::PV>(this->ode_.p_vars(),
                                                                   this->ode_.p_vars()) =
                (-h / 2.0) * (HX0.template bottomRightCorner<DODE::PV, DODE::PV>(
                                  this->ode_.p_vars(), this->ode_.p_vars()) +
                              HX1.template bottomRightCorner<DODE::PV, DODE::PV>(
                                  this->ode_.p_vars(), this->ode_.p_vars()));

            int j = 0;
            constexpr int cardinals = 2;
            Input<Scalar> ht_par(this->input_rows());
            ht_par.setZero();

            adjhess.template block<DODE::XtUV, DODE::PV>(
                j * this->ode_.xtu_vars(), cardinals * this->ode_.xtu_vars(), this->ode_.xtu_vars(),
                this->ode_.p_vars()) +=
                (-h / 2.0) * HX0.template block<DODE::XtUV, DODE::PV>(0, this->ode_.xtu_vars(),
                                                                      this->ode_.xtu_vars(),
                                                                      this->ode_.p_vars());
            adjhess.template block<DODE::PV, DODE::XtUV>(
                cardinals * this->ode_.xtu_vars(), j * this->ode_.xtu_vars(), this->ode_.p_vars(),
                this->ode_.xtu_vars()) +=
                (-h / 2.0) * HX0.template block<DODE::PV, DODE::XtUV>(this->ode_.xtu_vars(), 0,
                                                                      this->ode_.p_vars(),
                                                                      this->ode_.xtu_vars());
            ht_par.template segment<DODE::XtUV>(j * this->ode_.xtu_vars(), this->ode_.xtu_vars()) +=
                -AGX0.template segment<DODE::XtUV>(0, this->ode_.xtu_vars()) * one_half;
            ht_par.tail(this->ode_.p_vars()) += -AGX0.tail(this->ode_.p_vars()) * one_half;

            j = 1;
            adjhess.template block<DODE::XtUV, DODE::PV>(
                j * this->ode_.xtu_vars(), cardinals * this->ode_.xtu_vars(), this->ode_.xtu_vars(),
                this->ode_.p_vars()) +=
                (-h / 2.0) * HX1.template block<DODE::XtUV, DODE::PV>(0, this->ode_.xtu_vars(),
                                                                      this->ode_.xtu_vars(),
                                                                      this->ode_.p_vars());
            adjhess.template block<DODE::PV, DODE::XtUV>(
                cardinals * this->ode_.xtu_vars(), j * this->ode_.xtu_vars(), this->ode_.p_vars(),
                this->ode_.xtu_vars()) +=
                (-h / 2.0) * HX1.template block<DODE::PV, DODE::XtUV>(this->ode_.xtu_vars(), 0,
                                                                      this->ode_.p_vars(),
                                                                      this->ode_.xtu_vars());
            ht_par.template segment<DODE::XtUV>(j * this->ode_.xtu_vars(), this->ode_.xtu_vars()) +=
                -AGX1.template segment<DODE::XtUV>(0, this->ode_.xtu_vars()) * one_half;
            ht_par.tail(this->ode_.p_vars()) += -AGX1.tail(this->ode_.p_vars()) * one_half;

            adjhess.col(this->ode_.t_var()) -= ht_par;
            adjhess.col(this->ode_.t_var() + this->ode_.xtu_vars() * (cardinals - 1)) += ht_par;
            adjhess.row(this->ode_.t_var()) -= ht_par;
            adjhess.row(this->ode_.t_var() + this->ode_.xtu_vars() * (cardinals - 1)) += ht_par;
        };

        const int irows = this->ode_.input_rows();
        const int orows = this->ode_.output_rows();

        using IType = ODEInput<Scalar>;
        using OType = ODEOutput<Scalar>;
        using JType = ODEJacobian<Scalar>;
        using GType = ODEGrad<Scalar>;
        using HType = ODEHessian<Scalar>;

        BumpAllocator::allocate_run(impl, TempSpec<IType>(irows, 1), TempSpec<IType>(irows, 1),

                                    TempSpec<OType>(orows, 1), TempSpec<OType>(orows, 1),

                                    TempSpec<JType>(orows, irows), TempSpec<JType>(orows, irows),

                                    TempSpec<GType>(irows, 1), TempSpec<GType>(irows, 1),

                                    TempSpec<HType>(irows, irows), TempSpec<HType>(irows, irows));

        fx *= Scalar(-1.0);
        jx *= Scalar(-1.0);
        adjgrad *= Scalar(-1.0);
        adjhess *= Scalar(-1.0);
    }
};

} // namespace tycho::oc
