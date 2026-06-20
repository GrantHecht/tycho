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

#include "tycho/detail/optimal_control/transcription/lgl_coeffs.h"
#include "tycho/detail/vf/core/vector_function.h"

namespace tycho::oc {

// Import cross-namespace types from vf and utils.
using utils::SZ_MAX;
using utils::SZ_PROD;
using utils::SZ_SUM;
using vf::DenseDerivativeMode;
using vf::GenericFunction;
using vf::ThreadingFlags;
using vf::VectorExpression;
using vf::VectorFunction;

/// @internal
/// @brief Sizing layer of the LGL control-spline constraint (static-control case).
/// @tparam Derived  The concrete spline type (CRTP host).
/// @tparam CSC      Number of cardinal states per interval.
/// @tparam USZ      Control-vector dimension.
/// @tparam Order    Spline continuity order.
/// @endinternal
template <class Derived, int CSC, int USZ, int Order>
struct LGLControlSplineSize : VectorFunction<Derived, (2 * CSC - 1) * (USZ + 1), USZ * Order> {
    /// @brief Convenience alias for the VectorFunction CRTP base class.
    using Base = VectorFunction<Derived, (2 * CSC - 1) * (USZ + 1), USZ * Order>;
    using Base::Base;

    static constexpr int Usz = USZ;       ///< Control-vector dimension.
    static constexpr int t_usz = USZ + 1; ///< Time + control dimension per node.

    const int t_usize() const { return t_usz; } ///< @brief Time + control size per node.
    const int Usize() const { return Usz; }     ///< @brief Control size per node.

    /// @brief Time + control node-vector type. @tparam Scalar Arithmetic scalar type.
    template <class Scalar> using TUVec = Eigen::Matrix<Scalar, USZ + 1, 1>;
    /// @brief Control node-vector type. @tparam Scalar Arithmetic scalar type.
    template <class Scalar> using UVec = Eigen::Matrix<Scalar, USZ, 1>;

    /// @brief No-op setter (control size is fixed at compile time).
    /// @param u  Ignored.
    void set_usize(int u) {}
};

/// @internal
/// @brief Runtime-control specialization of @ref LGLControlSplineSize (`USZ == -1`).
/// @tparam Derived  The concrete spline type (CRTP host).
/// @tparam CSC      Number of cardinal states per interval.
/// @tparam Order    Spline continuity order.
/// @endinternal
template <class Derived, int CSC, int Order>
struct LGLControlSplineSize<Derived, CSC, -1, Order> : VectorFunction<Derived, -1, -1> {
    /// @brief Convenience alias for the VectorFunction CRTP base class.
    using Base = VectorFunction<Derived, -1, -1>;
    using Base::Base;

    static constexpr int Usz = -1;   ///< Control dimension (runtime).
    static constexpr int t_usz = -1; ///< Time + control dimension (runtime).

    int usz_d_ = -1;   ///< Runtime control dimension.
    int t_usz_d_ = -1; ///< Runtime time + control dimension.

    int t_usize() const { return t_usz_d_; } ///< @brief Time + control size per node.
    int Usize() const { return usz_d_; }     ///< @brief Control size per node.

    /// @brief Time + control node-vector type. @tparam Scalar Arithmetic scalar type.
    template <class Scalar> using TUVec = Eigen::Matrix<Scalar, -1, 1>;
    /// @brief Control node-vector type. @tparam Scalar Arithmetic scalar type.
    template <class Scalar> using UVec = Eigen::Matrix<Scalar, -1, 1>;

    /// @brief Set the runtime control size and derive the IO row counts.
    /// @param u  Number of control variables.
    void set_usize(int u) {
        this->usz_d_ = u;
        this->t_usz_d_ = u + 1;
        int irr = (2 * CSC - 1) * (u + 1);
        int orr = u * Order;
        this->set_io_rows(irr, orr);
    }
};

/// @ingroup optimal_control
/// @brief Control-continuity constraint joining two LGL intervals at a shared node.
///
/// Equality constraint VectorFunction enforcing continuity (up to @p Order
/// derivatives) of the polynomial control spline across the boundary between
/// two consecutive LGL intervals.
/// @tparam CSC    Number of cardinal states per interval.
/// @tparam USZ    Control-vector dimension (`Eigen::Dynamic` for runtime size).
/// @tparam Order  Number of derivative-continuity conditions enforced.
template <int CSC, int USZ, int Order = CSC - 2>
struct LGLControlSpline : LGLControlSplineSize<LGLControlSpline<CSC, USZ, Order>, CSC, USZ, Order> {
    /// @brief Convenience alias for the sizing CRTP base class.
    using SplineBase = LGLControlSplineSize<LGLControlSpline<CSC, USZ, Order>, CSC, USZ, Order>;

    /// @brief Time + control node-vector type. @tparam Scalar Arithmetic scalar type.
    template <class Scalar> using TUVec = typename SplineBase::template TUVec<Scalar>;
    /// @brief Control node-vector type. @tparam Scalar Arithmetic scalar type.
    template <class Scalar> using UVec = typename SplineBase::template UVec<Scalar>;

    /// @brief Convenience alias for a fixed-size std::array.
    /// @tparam T   Element type.
    /// @tparam SZ  Array length.
    template <class T, int SZ> using STDarray = std::array<T, SZ>;
    using Coeffs = LGLCoeffs<CSC>; ///< @brief The LGL coefficient table for this order.

    static constexpr int t_u_num = (2 * CSC - 1); ///< Number of time/control nodes across the join.
    static constexpr int UeqNum = Order;          ///< Number of continuity conditions.

    /// @brief Default constructor; control size must be set before use.
    LGLControlSpline() {}

    /// @brief Construct with a runtime control size.
    /// @param usize  Number of control variables.
    LGLControlSpline(int usize) { this->set_usize(usize); }

    /// @internal
    /// @brief Evaluate the control-continuity residuals.
    /// @tparam InType   Eigen input-vector type.
    /// @tparam OutType  Eigen output-vector type.
    /// @param x    Packed time/control node values across the two intervals.
    /// @param fx_  Output continuity-residual vector to write.
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(const Eigen::MatrixBase<InType> &x,
                             Eigen::MatrixBase<OutType> const &fx_) const {
        typedef typename InType::Scalar Scalar;
        Eigen::MatrixBase<OutType> &fx = fx_.const_cast_derived();

        STDarray<TUVec<Scalar>, t_u_num> t_u_vs;

        for (int i = 0; i < t_u_num; i++) {
            t_u_vs[i] = x.template segment<SplineBase::t_usz>(i * this->t_usize(), this->t_usize());
        }
        Scalar h0 = t_u_vs[CSC - 1][0] - t_u_vs[0][0];
        Scalar h1 = t_u_vs.back()[0] - t_u_vs[CSC - 1][0];

        for (int j = 0; j < UeqNum;
             j++) { // j = 0-> 1st order continuity, j=1 2nd order continuity ...
            for (int i = 0; i < CSC; i++) {
                fx.template segment<SplineBase::Usz>(j * this->Usize(), this->Usize()) +=
                    (Coeffs::UOneSpline_Weights[j][i] * t_u_vs[i].tail(this->Usize()) /
                         (pow(h0, Scalar(j + 1))) -
                     Coeffs::UZeroSpline_Weights[j][i] * t_u_vs[i + CSC - 1].tail(this->Usize()) /
                         (pow(h1, Scalar(j + 1))));
            }
        }
    }

    /// @internal
    /// @brief Evaluate the control-continuity residuals and their Jacobian.
    /// @tparam InType   Eigen input-vector type.
    /// @tparam OutType  Eigen output-vector type.
    /// @tparam JacType  Eigen Jacobian-matrix type.
    /// @param x    Packed time/control node values across the two intervals.
    /// @param fx_  Output continuity-residual vector to write.
    /// @param jx_  Output Jacobian to write.
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(const Eigen::MatrixBase<InType> &x,
                                      Eigen::MatrixBase<OutType> const &fx_,
                                      Eigen::MatrixBase<JacType> const &jx_) const {
        typedef typename InType::Scalar Scalar;
        Eigen::MatrixBase<JacType> &jx = jx_.const_cast_derived();
        Eigen::MatrixBase<OutType> &fx = fx_.const_cast_derived();

        STDarray<TUVec<Scalar>, t_u_num> t_u_vs;

        for (int i = 0; i < t_u_num; i++) {
            t_u_vs[i] = x.template segment<SplineBase::t_usz>(i * this->t_usize(), this->t_usize());
        }

        Scalar h0 = t_u_vs[CSC - 1][0] - t_u_vs[0][0];
        Scalar h1 = t_u_vs.back()[0] - t_u_vs[CSC - 1][0];

        for (int j = 0; j < UeqNum;
             j++) { // j = 0-> 1st order continuity, j=1 2nd order continuity ...
            Scalar h0pow = 1.0 / pow(h0, Scalar(j + 1));
            Scalar h1pow = 1.0 / pow(h1, Scalar(j + 1));
            Scalar hdt = Scalar(j + 1);

            for (int i = 0; i < CSC; i++) {
                fx.template segment<SplineBase::Usz>(j * this->Usize(), this->Usize()) +=
                    ((Coeffs::UOneSpline_Weights[j][i] * h0pow) * t_u_vs[i].tail(this->Usize()) -
                     (Coeffs::UZeroSpline_Weights[j][i] * h1pow) *
                         t_u_vs[i + CSC - 1].tail(this->Usize()));

                jx.col(0).template segment<SplineBase::Usz>(j * this->Usize(), this->Usize()) +=
                    (Coeffs::UOneSpline_Weights[j][i] * hdt * h0pow / h0) *
                    t_u_vs[i].tail(this->Usize());

                jx.col((CSC - 1) * this->t_usize())
                    .template segment<SplineBase::Usz>(j * this->Usize(), this->Usize()) +=
                    -(Coeffs::UOneSpline_Weights[j][i] * hdt * h0pow / h0) *
                        t_u_vs[i].tail(this->Usize()) -
                    (Coeffs::UZeroSpline_Weights[j][i] * hdt * h1pow / h1) *
                        t_u_vs[i + CSC - 1].tail(this->Usize());

                jx.col((2 * CSC - 2) * this->t_usize())
                    .template segment<SplineBase::Usz>(j * this->Usize(), this->Usize()) +=
                    (Coeffs::UZeroSpline_Weights[j][i] * hdt * h1pow / h1) *
                    t_u_vs[i + CSC - 1].tail(this->Usize());

                jx.template block<SplineBase::Usz, SplineBase::Usz>(
                      j * this->Usize(), i * this->t_usize() + 1, this->Usize(), this->Usize())
                    .diagonal() +=
                    UVec<Scalar>::Constant(this->Usize(), Coeffs::UOneSpline_Weights[j][i] * h0pow);

                jx.template block<SplineBase::Usz, SplineBase::Usz>(
                      j * this->Usize(), (i + CSC - 1) * this->t_usize() + 1, this->Usize(),
                      this->Usize())
                    .diagonal() += UVec<Scalar>::Constant(
                    this->Usize(), -Coeffs::UZeroSpline_Weights[j][i] * h1pow);
            }
        }
    }

    /// @internal
    /// @brief Evaluate the continuity residuals, Jacobian, and adjoint gradient.
    /// @tparam InType       Eigen input-vector type.
    /// @tparam OutType      Eigen output-vector type.
    /// @tparam JacType      Eigen Jacobian-matrix type.
    /// @tparam AdjGradType  Eigen adjoint-gradient vector type.
    /// @tparam AdjVarType   Eigen adjoint-variable vector type.
    /// @param x        Packed time/control node values across the two intervals.
    /// @param fx_      Output continuity-residual vector to write.
    /// @param jx_      Output Jacobian to write.
    /// @param adjgrad_ Output adjoint gradient to write.
    /// @param adjvars  Adjoint (Lagrange-multiplier) seed vector.
    /// @endinternal
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjVarType>
    inline void compute_jacobian_adjointgradient(
        const Eigen::MatrixBase<InType> &x, Eigen::MatrixBase<OutType> const &fx_,
        Eigen::MatrixBase<JacType> const &jx_, Eigen::MatrixBase<AdjGradType> const &adjgrad_,
        const Eigen::MatrixBase<AdjVarType> &adjvars) const {
        this->compute_jacobian(x, fx_, jx_);
        Eigen::MatrixBase<AdjGradType> &adjgrad = adjgrad_.const_cast_derived();
        adjgrad = (adjvars.transpose() * jx_).transpose();
    }

    /// @internal
    /// @brief Evaluate the continuity residuals, Jacobian, adjoint gradient, and adjoint Hessian.
    /// @tparam InType       Eigen input-vector type.
    /// @tparam OutType      Eigen output-vector type.
    /// @tparam JacType      Eigen Jacobian-matrix type.
    /// @tparam AdjGradType  Eigen adjoint-gradient vector type.
    /// @tparam AdjHessType  Eigen adjoint-Hessian matrix type.
    /// @tparam AdjVarType   Eigen adjoint-variable vector type.
    /// @param x        Packed time/control node values across the two intervals.
    /// @param fx_      Output continuity-residual vector to write.
    /// @param jx_      Output Jacobian to write.
    /// @param adjgrad_ Output adjoint gradient to write.
    /// @param adjhess_ Output adjoint Hessian to write.
    /// @param adjvars  Adjoint (Lagrange-multiplier) seed vector.
    /// @endinternal
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

        STDarray<TUVec<Scalar>, t_u_num> t_u_vs;

        for (int i = 0; i < t_u_num; i++) {
            t_u_vs[i] = x.template segment<SplineBase::t_usz>(i * this->t_usize(), this->t_usize());
        }

        Scalar h0 = t_u_vs[CSC - 1][0] - t_u_vs[0][0];
        Scalar h1 = t_u_vs.back()[0] - t_u_vs[CSC - 1][0];

        for (int j = 0; j < UeqNum;
             j++) { // j = 0-> 1st order continuity, j=1 2nd order continuity ...
            Scalar h0pow = 1.0 / pow(h0, Scalar(j + 1));
            Scalar h1pow = 1.0 / pow(h1, Scalar(j + 1));
            Scalar hdt = Scalar(j + 1);
            Scalar h2dt = Scalar(j + 2);
            Scalar OTH = 0.0;
            Scalar ZTH = 0.0;

            for (int i = 0; i < CSC; i++) {
                fx.template segment<SplineBase::Usz>(j * this->Usize(), this->Usize()) +=
                    ((Coeffs::UOneSpline_Weights[j][i] * h0pow) * t_u_vs[i].tail(this->Usize()) -
                     (Coeffs::UZeroSpline_Weights[j][i] * h1pow) *
                         t_u_vs[i + CSC - 1].tail(this->Usize()));

                OTH += (Coeffs::UOneSpline_Weights[j][i] * hdt * h2dt * h0pow / (h0 * h0)) *
                       t_u_vs[i]
                           .tail(this->Usize())
                           .dot(adjvars.template segment<SplineBase::Usz>(j * this->Usize(),
                                                                          this->Usize()));
                ZTH += (Coeffs::UZeroSpline_Weights[j][i] * hdt * h2dt * h1pow / (h1 * h1)) *
                       t_u_vs[i + CSC - 1]
                           .tail(this->Usize())
                           .dot(adjvars.template segment<SplineBase::Usz>(j * this->Usize(),
                                                                          this->Usize()));

                jx.col(0).template segment<SplineBase::Usz>(j * this->Usize(), this->Usize()) +=
                    (Coeffs::UOneSpline_Weights[j][i] * hdt * h0pow / h0) *
                    t_u_vs[i].tail(this->Usize());

                jx.col((CSC - 1) * this->t_usize())
                    .template segment<SplineBase::Usz>(j * this->Usize(), this->Usize()) +=
                    -(Coeffs::UOneSpline_Weights[j][i] * hdt * h0pow / h0) *
                        t_u_vs[i].tail(this->Usize()) -
                    (Coeffs::UZeroSpline_Weights[j][i] * hdt * h1pow / h1) *
                        t_u_vs[i + CSC - 1].tail(this->Usize());

                jx.col((2 * CSC - 2) * this->t_usize())
                    .template segment<SplineBase::Usz>(j * this->Usize(), this->Usize()) +=
                    (Coeffs::UZeroSpline_Weights[j][i] * hdt * h1pow / h1) *
                    t_u_vs[i + CSC - 1].tail(this->Usize());

                jx.template block<SplineBase::Usz, SplineBase::Usz>(
                      j * this->Usize(), i * this->t_usize() + 1, this->Usize(), this->Usize())
                    .diagonal() +=
                    UVec<Scalar>::Constant(this->Usize(), Coeffs::UOneSpline_Weights[j][i] * h0pow);

                jx.template block<SplineBase::Usz, SplineBase::Usz>(
                      j * this->Usize(), (i + CSC - 1) * this->t_usize() + 1, this->Usize(),
                      this->Usize())
                    .diagonal() += UVec<Scalar>::Constant(
                    this->Usize(), -Coeffs::UZeroSpline_Weights[j][i] * h1pow);
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                adjhess.row(0).template segment<SplineBase::Usz>(i * this->t_usize() + 1,
                                                                 this->Usize()) +=
                    adjvars.template segment<SplineBase::Usz>(j * this->Usize(), this->Usize())
                        .cwiseProduct(UVec<Scalar>::Constant(
                            this->Usize(), Coeffs::UOneSpline_Weights[j][i] * hdt * h0pow / h0))
                        .transpose();

                adjhess.row((CSC - 1) * this->t_usize())
                    .template segment<SplineBase::Usz>(i * this->t_usize() + 1, this->Usize()) -=
                    adjvars.template segment<SplineBase::Usz>(j * this->Usize(), this->Usize())
                        .cwiseProduct(UVec<Scalar>::Constant(
                            this->Usize(), Coeffs::UOneSpline_Weights[j][i] * hdt * h0pow / h0))
                        .transpose();

                adjhess.col(0).template segment<SplineBase::Usz>(i * this->t_usize() + 1,
                                                                 this->Usize()) +=
                    adjvars.template segment<SplineBase::Usz>(j * this->Usize(), this->Usize())
                        .cwiseProduct(UVec<Scalar>::Constant(
                            this->Usize(), Coeffs::UOneSpline_Weights[j][i] * hdt * h0pow / h0));

                adjhess.col((CSC - 1) * this->t_usize())
                    .template segment<SplineBase::Usz>(i * this->t_usize() + 1, this->Usize()) -=
                    adjvars.template segment<SplineBase::Usz>(j * this->Usize(), this->Usize())
                        .cwiseProduct(UVec<Scalar>::Constant(
                            this->Usize(), Coeffs::UOneSpline_Weights[j][i] * hdt * h0pow / h0));
                /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                adjhess.row((CSC - 1) * this->t_usize())
                    .template segment<SplineBase::Usz>((i + CSC - 1) * this->t_usize() + 1,
                                                       this->Usize()) -=
                    adjvars.template segment<SplineBase::Usz>(j * this->Usize(), this->Usize())
                        .cwiseProduct(UVec<Scalar>::Constant(
                            this->Usize(), Coeffs::UZeroSpline_Weights[j][i] * hdt * h1pow / h1))
                        .transpose();

                adjhess.row((2 * CSC - 2) * this->t_usize())
                    .template segment<SplineBase::Usz>((i + CSC - 1) * this->t_usize() + 1,
                                                       this->Usize()) +=
                    adjvars.template segment<SplineBase::Usz>(j * this->Usize(), this->Usize())
                        .cwiseProduct(UVec<Scalar>::Constant(
                            this->Usize(), Coeffs::UZeroSpline_Weights[j][i] * hdt * h1pow / h1))
                        .transpose();

                adjhess.col((CSC - 1) * this->t_usize())
                    .template segment<SplineBase::Usz>((i + CSC - 1) * this->t_usize() + 1,
                                                       this->Usize()) -=
                    adjvars.template segment<SplineBase::Usz>(j * this->Usize(), this->Usize())
                        .cwiseProduct(UVec<Scalar>::Constant(
                            this->Usize(), Coeffs::UZeroSpline_Weights[j][i] * hdt * h1pow / h1));

                adjhess.col((2 * CSC - 2) * this->t_usize())
                    .template segment<SplineBase::Usz>((i + CSC - 1) * this->t_usize() + 1,
                                                       this->Usize()) +=
                    adjvars.template segment<SplineBase::Usz>(j * this->Usize(), this->Usize())
                        .cwiseProduct(UVec<Scalar>::Constant(
                            this->Usize(), Coeffs::UZeroSpline_Weights[j][i] * hdt * h1pow / h1));
            }

            adjhess(0, 0) += OTH;
            adjhess((CSC - 1) * this->t_usize(), (CSC - 1) * this->t_usize()) += OTH;
            adjhess(0, (CSC - 1) * this->t_usize()) += -OTH;
            adjhess((CSC - 1) * this->t_usize(), 0) += -OTH;

            adjhess((CSC - 1) * this->t_usize(), (CSC - 1) * this->t_usize()) -= ZTH;
            adjhess((2 * CSC - 2) * this->t_usize(), (2 * CSC - 2) * this->t_usize()) -= ZTH;
            adjhess((2 * CSC - 2) * this->t_usize(), (CSC - 1) * this->t_usize()) += ZTH;
            adjhess((CSC - 1) * this->t_usize(), (2 * CSC - 2) * this->t_usize()) += ZTH;
        }

        adjgrad = (adjvars.transpose() * jx).transpose();

        // QED
    }
};

} // namespace tycho::oc
