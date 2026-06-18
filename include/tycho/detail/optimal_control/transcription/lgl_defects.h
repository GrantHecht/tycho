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
#include "tycho/detail/optimal_control/transcription/transcription_sizing.h"
#include "tycho/detail/vf/core/vector_function.h"

namespace tycho::oc {

// Import cross-namespace types from vf and utils.
using utils::ArrayOfTempSpecs;
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
/// @ingroup optimal_control
/// @brief Legendre-Gauss-Lobatto collocation defect constraint for an ODE.
///
/// The equality constraint VectorFunction enforcing that an ODE's discretized
/// trajectory satisfies the LGL collocation defect equations over each interval
/// (the dynamics constraint of the transcription). Provides analytic Jacobian
/// and adjoint-Hessian derivatives.
/// @tparam DODE  The concrete ODE type whose dynamics are collocated.
/// @tparam CS    Number of cardinal states per collocation interval.
template <class DODE, int CS>
struct LGLDefects : VectorFunction<LGLDefects<DODE, CS>,
                                   DefectConstSizes<CS, DODE::XV, DODE::UV, DODE::PV>::DefIRC,
                                   DefectConstSizes<CS, DODE::XV, DODE::UV, DODE::PV>::DefORC> {
    static constexpr int Cardinals = CS;     ///< Number of cardinal states per interval.
    static constexpr int Interiors = CS - 1; ///< Number of interior (defect) points per interval.

    /// @brief Convenience alias for the VectorFunction CRTP base class.
    using Base = VectorFunction<LGLDefects<DODE, CS>,
                                DefectConstSizes<CS, DODE::XV, DODE::UV, DODE::PV>::DefIRC,
                                DefectConstSizes<CS, DODE::XV, DODE::UV, DODE::PV>::DefORC>;

    /////////////////////////////////////////////////////////////////////////////////
    /// @brief Defect output-vector type. @tparam Scalar Arithmetic scalar type.
    template <class Scalar> using Output = typename Base::template Output<Scalar>;
    /// @brief Defect input-vector type. @tparam Scalar Arithmetic scalar type.
    template <class Scalar> using Input = typename Base::template Input<Scalar>;
    /// @brief Defect Jacobian type. @tparam Scalar Arithmetic scalar type.
    template <class Scalar> using Jacobian = typename Base::template Jacobian<Scalar>;
    /// @brief Defect Hessian type. @tparam Scalar Arithmetic scalar type.
    template <class Scalar> using Hessian = typename Base::template Hessian<Scalar>;
    //////////////////////////////////////////////////////////////////////////////////
    /// @brief ODE output-vector type. @tparam Scalar Arithmetic scalar type.
    template <class Scalar> using ODEOutput = typename DODE::template Output<Scalar>;
    /// @brief ODE input-vector type. @tparam Scalar Arithmetic scalar type.
    template <class Scalar> using ODEInput = typename DODE::template Input<Scalar>;
    /// @brief ODE gradient type. @tparam Scalar Arithmetic scalar type.
    template <class Scalar> using ODEGrad = typename DODE::template Gradient<Scalar>;
    /// @brief ODE Jacobian type. @tparam Scalar Arithmetic scalar type.
    template <class Scalar> using ODEJacobian = typename DODE::template Jacobian<Scalar>;
    /// @brief ODE Hessian type. @tparam Scalar Arithmetic scalar type.
    template <class Scalar> using ODEHessian = typename DODE::template Hessian<Scalar>;

    /// @brief Convenience alias for a fixed-size std::array.
    /// @tparam T   Element type.
    /// @tparam SZ  Array length.
    template <class T, int SZ> using STDarray = std::array<T, SZ>;
    using Coeffs = LGLCoeffs<CS>; ///< @brief The LGL coefficient table for this order.
    /////////////////////////////////////////////////////////////////////////////
    DODE ode_; ///< The ODE whose dynamics are collocated.
    static constexpr bool is_vectorizable =
        DODE::is_vectorizable; ///< Inherits SuperScalar support from the ODE.

    /// @brief Construct from an ODE and size the defect function.
    /// @param od  The ODE to collocate.
    LGLDefects(const DODE &od) { this->set_ode(od); }
    /// @brief Set the ODE and derive the defect IO row counts.
    /// @param od  The ODE to collocate.
    void set_ode(const DODE &od) {
        this->ode_ = od;
        this->set_output_rows(this->ode_.output_rows() * (CS - 1));
        this->set_input_rows(CS * this->ode_.xtu_vars() + this->ode_.p_vars());
    }

    /// @internal
    /// @brief Evaluate the LGL defect residuals for one interval block.
    /// @tparam InType   Eigen input-vector type.
    /// @tparam OutType  Eigen output-vector type.
    /// @param x    Packed cardinal states and parameters of the interval.
    /// @param fx_  Output defect-residual vector to write.
    /// @endinternal
    template <class InType, class OutType>
    inline void compute_impl(const Eigen::MatrixBase<InType> &x,
                             Eigen::MatrixBase<OutType> const &fx_) const {
        typedef typename InType::Scalar Scalar;
        Eigen::MatrixBase<OutType> &fx = fx_.const_cast_derived();

        auto impl = [&](auto &C_XS, auto &C_DXS, auto &I_XS, auto &I_DXS) {
            for (int i = 0; i < Cardinals; i++) {
                C_XS[i].template segment<DODE::XtUV>(0, this->ode_.xtu_vars()) =
                    x.template segment<DODE::XtUV>(i * this->ode_.xtu_vars(),
                                                   this->ode_.xtu_vars());
                if constexpr (DODE::PV >= 0) {
                    C_XS[i].template tail<DODE::PV>(this->ode_.p_vars()) =
                        x.template tail<DODE::PV>(this->ode_.p_vars());
                } else {
                    C_XS[i].tail(this->ode_.p_vars()) = x.tail(this->ode_.p_vars());
                }

                this->ode_.compute(C_XS[i], C_DXS[i]);
            }

            Scalar h = C_XS.back()[this->ode_.t_var()] - C_XS[0][this->ode_.t_var()];

            for (int i = 0; i < Interiors; i++) {
                I_XS[i][this->ode_.t_var()] =
                    C_XS[0][this->ode_.t_var()] + h * Coeffs::InteriorSpacings[i];

                if constexpr (DODE::PV >= 0) {
                    I_XS[i].template tail<DODE::PV>(this->ode_.p_vars()) =
                        x.template tail<DODE::PV>(this->ode_.p_vars());
                } else {
                    I_XS[i].tail(this->ode_.p_vars()) = x.tail(this->ode_.p_vars());
                }
                for (int j = 0; j < Cardinals; j++) {
                    I_XS[i].template segment<DODE::XV>(0, this->ode_.x_vars()) +=
                        (Scalar(Coeffs::Cardinal_XInterp_Weights[i][j]) *
                             C_XS[j].template segment<DODE::XV>(0, this->ode_.x_vars()) +
                         (Coeffs::Cardinal_DXInterp_Weights[i][j] * h) * C_DXS[j]);
                    I_XS[i].template segment<DODE::UV>(this->ode_.xt_vars(), this->ode_.u_vars()) +=
                        Scalar(Coeffs::Cardinal_UPoly_Weights[i][j]) *
                        C_XS[j].template segment<DODE::UV>(this->ode_.xt_vars(),
                                                           this->ode_.u_vars());

                    fx.template segment<DODE::XV>(i * this->ode_.x_vars(), this->ode_.x_vars()) +=
                        (Scalar(Coeffs::Cardinal_XDef_Weights[i][j]) *
                             C_XS[j].template segment<DODE::XV>(0, this->ode_.x_vars()) +
                         (Coeffs::Cardinal_DXDef_Weights[i][j] * h) * C_DXS[j]);
                }
                this->ode_.compute(I_XS[i], I_DXS[i]);
                fx.template segment<DODE::XV>(i * this->ode_.x_vars(), this->ode_.x_vars()) +=
                    (h * Coeffs::Interior_DXDef_Weights[i] * I_DXS[i]);
            }
        };

        using XType = ODEInput<Scalar>;
        using FXType = ODEOutput<Scalar>;

        const int irowsode = this->ode_.input_rows();
        const int orowsode = this->ode_.output_rows();

        BumpAllocator::allocate_run(impl, ArrayOfTempSpecs<XType, Cardinals>(irowsode, 1),
                                    ArrayOfTempSpecs<FXType, Cardinals>(orowsode, 1),
                                    ArrayOfTempSpecs<XType, Interiors>(irowsode, 1),
                                    ArrayOfTempSpecs<FXType, Interiors>(orowsode, 1));
    }
    /// @internal
    /// @brief Evaluate the LGL defect residuals and their Jacobian.
    /// @tparam InType   Eigen input-vector type.
    /// @tparam OutType  Eigen output-vector type.
    /// @tparam JacType  Eigen Jacobian-matrix type.
    /// @param x    Packed cardinal states and parameters of the interval.
    /// @param fx_  Output defect-residual vector to write.
    /// @param jx_  Output Jacobian to write.
    /// @endinternal
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(const Eigen::MatrixBase<InType> &x,
                                      Eigen::MatrixBase<OutType> const &fx_,
                                      Eigen::MatrixBase<JacType> const &jx_) const {
        typedef typename InType::Scalar Scalar;
        Eigen::MatrixBase<JacType> &jx = jx_.const_cast_derived();
        Eigen::MatrixBase<OutType> &fx = fx_.const_cast_derived();

        auto impl = [&](auto &C_XS, auto &C_DXS, auto &C_JDXS, auto &I_XS, auto &I_DXS,
                        auto &I_JDXS, auto &DI_DCS) {
            for (int i = 0; i < Cardinals; i++) {
                C_XS[i].template segment<DODE::XtUV>(0, this->ode_.xtu_vars()) =
                    x.template segment<DODE::XtUV>(i * this->ode_.xtu_vars(),
                                                   this->ode_.xtu_vars());
                if constexpr (DODE::PV >= 0) {
                    C_XS[i].template tail<DODE::PV>(this->ode_.p_vars()) =
                        x.template tail<DODE::PV>(this->ode_.p_vars());
                } else {
                    C_XS[i].tail(this->ode_.p_vars()) = x.tail(this->ode_.p_vars());
                }

                this->ode_.compute_jacobian(C_XS[i], C_DXS[i], C_JDXS[i]);
            }

            Scalar h = C_XS.back()[this->ode_.t_var()] - C_XS[0][this->ode_.t_var()];

            for (int i = 0; i < Interiors; i++) {

                I_XS[i][this->ode_.t_var()] =
                    C_XS[0][this->ode_.t_var()] + h * Coeffs::InteriorSpacings[i];
                if constexpr (DODE::PV >= 0) {
                    I_XS[i].template tail<DODE::PV>(this->ode_.p_vars()) =
                        x.template tail<DODE::PV>(this->ode_.p_vars());
                } else {
                    I_XS[i].tail(this->ode_.p_vars()) = x.tail(this->ode_.p_vars());
                }

                if (i > 0) {
                    DI_DCS.setZero();
                }

                DI_DCS(this->ode_.t_var(), this->ode_.t_var()) =
                    Scalar(1.0 - Coeffs::InteriorSpacings[i]);
                DI_DCS(this->ode_.t_var(), this->ode_.xtu_vars() * (Cardinals - 1) +
                                               this->ode_.t_var()) = Coeffs::InteriorSpacings[i];
                DI_DCS
                    .template block<DODE::PV, DODE::PV>(this->ode_.xtu_vars(),
                                                        Cardinals * this->ode_.xtu_vars(),
                                                        this->ode_.p_vars(), this->ode_.p_vars())
                    .diagonal()
                    .setConstant(Scalar(1.0));

                for (int j = 0; j < Cardinals; j++) {
                    ////////////////////////////// Sum up Cardinal Interpolation
                    /// Terms/////////////////////////////////////////
                    I_XS[i].template segment<DODE::XV>(0, this->ode_.x_vars()) +=
                        (Scalar(Coeffs::Cardinal_XInterp_Weights[i][j]) *
                             C_XS[j].template segment<DODE::XV>(0, this->ode_.x_vars()) +
                         (Coeffs::Cardinal_DXInterp_Weights[i][j] * h) * C_DXS[j]);

                    DI_DCS
                        .template block<DODE::XV, DODE::XV>(
                            0, j * this->ode_.xtu_vars(), this->ode_.x_vars(), this->ode_.x_vars())
                        .diagonal()
                        .setConstant(Scalar(Coeffs::Cardinal_XInterp_Weights[i][j]));

                    DI_DCS.template block<DODE::XV, DODE::XtUV>(
                        0, j * this->ode_.xtu_vars(), this->ode_.x_vars(), this->ode_.xtu_vars()) +=
                        (Coeffs::Cardinal_DXInterp_Weights[i][j] * h) *
                        C_JDXS[j].template leftCols<DODE::XtUV>(this->ode_.xtu_vars());

                    DI_DCS.template block<DODE::XV, DODE::PV>(0, Cardinals * this->ode_.xtu_vars(),
                                                              this->ode_.x_vars(),
                                                              this->ode_.p_vars()) +=
                        (Coeffs::Cardinal_DXInterp_Weights[i][j] * h) *
                        C_JDXS[j].template rightCols<DODE::PV>(this->ode_.p_vars());

                    DI_DCS.col(this->ode_.t_var())
                        .template segment<DODE::XV>(0, this->ode_.x_vars()) -=
                        Scalar(Coeffs::Cardinal_DXInterp_Weights[i][j]) * C_DXS[j];
                    DI_DCS.col(this->ode_.xtu_vars() * (Cardinals - 1) + this->ode_.t_var())
                        .template segment<DODE::XV>(0, this->ode_.x_vars()) +=
                        Scalar(Coeffs::Cardinal_DXInterp_Weights[i][j]) * C_DXS[j];

                    I_XS[i].template segment<DODE::UV>(this->ode_.xt_vars(), this->ode_.u_vars()) +=
                        Scalar(Coeffs::Cardinal_UPoly_Weights[i][j]) *
                        C_XS[j].template segment<DODE::UV>(this->ode_.xt_vars(),
                                                           this->ode_.u_vars());

                    DI_DCS
                        .template block<DODE::UV, DODE::UV>(
                            this->ode_.xt_vars(), j * this->ode_.xtu_vars() + this->ode_.xt_vars(),
                            this->ode_.u_vars(), this->ode_.u_vars())
                        .diagonal()
                        .setConstant(Scalar(Coeffs::Cardinal_UPoly_Weights[i][j]));

                    ////////////////////////////// Sum up Cardinal Output
                    /// Terms/////////////////////////////////////////
                    fx.template segment<DODE::XV>(i * this->ode_.x_vars(), this->ode_.x_vars()) +=
                        (Scalar(Coeffs::Cardinal_XDef_Weights[i][j]) *
                             C_XS[j].template head<DODE::XV>(this->ode_.x_vars()) +
                         (Coeffs::Cardinal_DXDef_Weights[i][j] * h) * C_DXS[j]);

                    jx.template block<DODE::XV, DODE::XV>(i * this->ode_.x_vars(),
                                                          j * this->ode_.xtu_vars(),
                                                          this->ode_.x_vars(), this->ode_.x_vars())
                        .diagonal()
                        .setConstant(Scalar(Coeffs::Cardinal_XDef_Weights[i][j]));

                    jx.template block<DODE::XV, DODE::XtUV>(
                        i * this->ode_.x_vars(), j * this->ode_.xtu_vars(), this->ode_.x_vars(),
                        this->ode_.xtu_vars()) +=
                        (Coeffs::Cardinal_DXDef_Weights[i][j] * h) *
                        C_JDXS[j].template leftCols<DODE::XtUV>(this->ode_.xtu_vars());

                    jx.template block<DODE::XV, DODE::PV>(
                        i * this->ode_.x_vars(), Cardinals * this->ode_.xtu_vars(),
                        this->ode_.x_vars(), this->ode_.p_vars()) +=
                        (Coeffs::Cardinal_DXDef_Weights[i][j] * h) *
                        C_JDXS[j].template rightCols<DODE::PV>(this->ode_.p_vars());

                    jx.col(this->ode_.t_var())
                        .template segment<DODE::XV>(i * this->ode_.x_vars(), this->ode_.x_vars()) -=
                        Scalar(Coeffs::Cardinal_DXDef_Weights[i][j]) * C_DXS[j];
                    jx.col(this->ode_.xtu_vars() * (Cardinals - 1) + this->ode_.t_var())
                        .template segment<DODE::XV>(i * this->ode_.x_vars(), this->ode_.x_vars()) +=
                        Scalar(Coeffs::Cardinal_DXDef_Weights[i][j]) * C_DXS[j];
                }

                this->ode_.compute_jacobian(I_XS[i], I_DXS[i], I_JDXS[i]);

                fx.template segment<DODE::XV>(i * this->ode_.x_vars(), this->ode_.x_vars()) +=
                    (h * Coeffs::Interior_DXDef_Weights[i] * I_DXS[i]);
                jx.template middleRows<DODE::XV>(i * this->ode_.x_vars(), this->ode_.x_vars()) +=
                    ((h * Coeffs::Interior_DXDef_Weights[i]) * I_JDXS[i]) * (DI_DCS);

                jx.col(this->ode_.t_var())
                    .template segment<DODE::XV>(i * this->ode_.x_vars(), this->ode_.x_vars()) -=
                    Scalar(Coeffs::Interior_DXDef_Weights[i]) * I_DXS[i];
                jx.col(this->ode_.xtu_vars() * (Cardinals - 1) + this->ode_.t_var())
                    .template segment<DODE::XV>(i * this->ode_.x_vars(), this->ode_.x_vars()) +=
                    Scalar(Coeffs::Interior_DXDef_Weights[i]) * I_DXS[i];
            }
            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        };

        using XType = ODEInput<Scalar>;
        using FXType = ODEOutput<Scalar>;
        using JXType = ODEJacobian<Scalar>;

        using DIType = Eigen::Matrix<Scalar, DODE::IRC, Base::IRC>;

        const int irowsode = this->ode_.input_rows();
        const int orowsode = this->ode_.output_rows();

        BumpAllocator::allocate_run(impl, ArrayOfTempSpecs<XType, Cardinals>(irowsode, 1),
                                    ArrayOfTempSpecs<FXType, Cardinals>(orowsode, 1),
                                    ArrayOfTempSpecs<JXType, Cardinals>(orowsode, irowsode),
                                    ArrayOfTempSpecs<XType, Interiors>(irowsode, 1),
                                    ArrayOfTempSpecs<FXType, Interiors>(orowsode, 1),
                                    ArrayOfTempSpecs<JXType, Interiors>(orowsode, irowsode),
                                    TempSpec<DIType>(irowsode, this->input_rows()));
    }

    /// @internal
    /// @brief Evaluate the LGL defect residuals, Jacobian, adjoint gradient, and adjoint Hessian.
    /// @tparam InType       Eigen input-vector type.
    /// @tparam OutType      Eigen output-vector type.
    /// @tparam JacType      Eigen Jacobian-matrix type.
    /// @tparam AdjGradType  Eigen adjoint-gradient vector type.
    /// @tparam AdjHessType  Eigen adjoint-Hessian matrix type.
    /// @tparam AdjVarType   Eigen adjoint-variable vector type.
    /// @param x        Packed cardinal states and parameters of the interval.
    /// @param fx_      Output defect-residual vector to write.
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

        auto impl = [&](auto &C_XS, auto &C_DXS, auto &C_JDXS,
                        auto &C_AGXS, // Not an array
                        auto &C_AVS,
                        auto &C_HDXS, // Not an array
                        auto &I_XS, auto &I_DXS, auto &I_JDXS, auto &I_AGXS, auto &I_AVS,
                        auto &I_HDXS, auto &DI_DCS, auto &HTpar) {
            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            for (int i = 0; i < Cardinals; i++) {

                C_XS[i].template segment<DODE::XtUV>(0, this->ode_.xtu_vars()) =
                    x.template segment<DODE::XtUV>(i * this->ode_.xtu_vars(),
                                                   this->ode_.xtu_vars());
                if constexpr (DODE::PV >= 0) {
                    C_XS[i].template tail<DODE::PV>(this->ode_.p_vars()) =
                        x.template tail<DODE::PV>(this->ode_.p_vars());
                } else {
                    C_XS[i].tail(this->ode_.p_vars()) = x.tail(this->ode_.p_vars());
                }

                this->ode_.compute(C_XS[i], C_DXS[i]);
            }
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            Scalar h = C_XS.back()[this->ode_.t_var()] - C_XS[0][this->ode_.t_var()];

            for (int i = 0; i < Interiors; i++) {

                I_XS[i][this->ode_.t_var()] =
                    C_XS[0][this->ode_.t_var()] + h * Coeffs::InteriorSpacings[i];
                if constexpr (DODE::PV >= 0) {
                    I_XS[i].template tail<DODE::PV>(this->ode_.p_vars()) =
                        x.template tail<DODE::PV>(this->ode_.p_vars());
                } else {
                    I_XS[i].tail(this->ode_.p_vars()) = x.tail(this->ode_.p_vars());
                }

                I_AVS[i] = adjvars.template segment<DODE::XV>(i * this->ode_.x_vars(),
                                                              this->ode_.x_vars());

                for (int j = 0; j < Cardinals; j++) {
                    ////////////////////////////// Sum up Cardinal Interpolation
                    /// Terms/////////////////////////////////////////
                    I_XS[i].template segment<DODE::XV>(0, this->ode_.x_vars()) +=
                        (Scalar(Coeffs::Cardinal_XInterp_Weights[i][j]) *
                             C_XS[j].template segment<DODE::XV>(0, this->ode_.x_vars()) +
                         (Coeffs::Cardinal_DXInterp_Weights[i][j] * h) * C_DXS[j]);

                    I_XS[i].template segment<DODE::UV>(this->ode_.xt_vars(), this->ode_.u_vars()) +=
                        Scalar(Coeffs::Cardinal_UPoly_Weights[i][j]) *
                        C_XS[j].template segment<DODE::UV>(this->ode_.xt_vars(),
                                                           this->ode_.u_vars());
                }

                this->ode_.compute_jacobian_adjointgradient_adjointhessian(
                    I_XS[i], I_DXS[i], I_JDXS[i], I_AGXS[i], I_HDXS[i], I_AVS[i]);

                for (int j = 0; j < Cardinals; j++) {
                    Scalar scale = Scalar(Coeffs::Interior_DXDef_Weights[i] *
                                          Coeffs::Cardinal_DXInterp_Weights[i][j]);
                    C_AVS[j] +=
                        I_AGXS[i].template head<DODE::XV>(this->ode_.x_vars()) * (scale * h * h);
                    C_AVS[j] += I_AVS[i] * (Coeffs::Cardinal_DXDef_Weights[i][j] * h);
                }
            }

            for (int j = 0; j < Cardinals; j++) {

                if (j > 0) {
                    C_AGXS.setZero();
                    C_HDXS.setZero();
                }
                this->ode_.compute_jacobian_adjointgradient_adjointhessian(
                    C_XS[j], C_DXS[j], C_JDXS[j], C_AGXS, C_HDXS, C_AVS[j]);

                adjhess.template block<DODE::XtUV, DODE::XtUV>(
                    j * this->ode_.xtu_vars(), j * this->ode_.xtu_vars(), this->ode_.xtu_vars(),
                    this->ode_.xtu_vars()) +=
                    C_HDXS.template block<DODE::XtUV, DODE::XtUV>(0, 0, this->ode_.xtu_vars(),
                                                                  this->ode_.xtu_vars());
                adjhess.template block<DODE::XtUV, DODE::PV>(
                    j * this->ode_.xtu_vars(), Cardinals * this->ode_.xtu_vars(),
                    this->ode_.xtu_vars(), this->ode_.p_vars()) +=
                    C_HDXS.template block<DODE::XtUV, DODE::PV>(
                        0, this->ode_.xtu_vars(), this->ode_.xtu_vars(), this->ode_.p_vars());
                adjhess.template block<DODE::PV, DODE::XtUV>(
                    Cardinals * this->ode_.xtu_vars(), j * this->ode_.xtu_vars(),
                    this->ode_.p_vars(), this->ode_.xtu_vars()) +=
                    C_HDXS.template block<DODE::PV, DODE::XtUV>(
                        this->ode_.xtu_vars(), 0, this->ode_.p_vars(), this->ode_.xtu_vars());
                adjhess.template bottomRightCorner<DODE::PV, DODE::PV>(this->ode_.p_vars(),
                                                                       this->ode_.p_vars()) +=
                    C_HDXS.template bottomRightCorner<DODE::PV, DODE::PV>(this->ode_.p_vars(),
                                                                          this->ode_.p_vars());
                HTpar.template segment<DODE::XtUV>(j * this->ode_.xtu_vars(),
                                                   this->ode_.xtu_vars()) +=
                    C_AGXS.template segment<DODE::XtUV>(0, this->ode_.xtu_vars()) * (1.0 / h);

                if constexpr (DODE::PV >= 0) {
                    HTpar.template tail<DODE::PV>(this->ode_.p_vars()) +=
                        C_AGXS.template tail<DODE::PV>(this->ode_.p_vars()) * (1.0 / h);
                } else {
                    HTpar.tail(this->ode_.p_vars()) += C_AGXS.tail(this->ode_.p_vars()) * (1.0 / h);
                }
            }

            for (int i = 0; i < Interiors; i++) {
                if (i > 0)
                    DI_DCS.setZero();
                DI_DCS(this->ode_.t_var(), this->ode_.t_var()) =
                    Scalar(1.0 - Coeffs::InteriorSpacings[i]);
                DI_DCS(this->ode_.t_var(),
                       this->ode_.xtu_vars() * (Cardinals - 1) + this->ode_.t_var()) =
                    Scalar(Coeffs::InteriorSpacings[i]);
                DI_DCS
                    .template block<DODE::PV, DODE::PV>(this->ode_.xtu_vars(),
                                                        Cardinals * this->ode_.xtu_vars(),
                                                        this->ode_.p_vars(), this->ode_.p_vars())
                    .diagonal()
                    .setConstant(Scalar(1.0));

                for (int j = 0; j < Cardinals; j++) {
                    ////////////////////////////// Sum up Cardinal Interpolation
                    /// Terms/////////////////////////////////////////

                    DI_DCS
                        .template block<DODE::XV, DODE::XV>(
                            0, j * this->ode_.xtu_vars(), this->ode_.x_vars(), this->ode_.x_vars())
                        .diagonal()
                        .setConstant(Scalar(Coeffs::Cardinal_XInterp_Weights[i][j]));

                    DI_DCS.template block<DODE::XV, DODE::XtUV>(
                        0, j * this->ode_.xtu_vars(), this->ode_.x_vars(), this->ode_.xtu_vars()) +=
                        (Coeffs::Cardinal_DXInterp_Weights[i][j] * h) *
                        C_JDXS[j].template leftCols<DODE::XtUV>(this->ode_.xtu_vars());

                    DI_DCS.template block<DODE::XV, DODE::PV>(0, Cardinals * this->ode_.xtu_vars(),
                                                              this->ode_.x_vars(),
                                                              this->ode_.p_vars()) +=
                        (Coeffs::Cardinal_DXInterp_Weights[i][j] * h) *
                        C_JDXS[j].template rightCols<DODE::PV>(this->ode_.p_vars());

                    DI_DCS.col(this->ode_.t_var()).template head<DODE::XV>(this->ode_.x_vars()) -=
                        Scalar(Coeffs::Cardinal_DXInterp_Weights[i][j]) * C_DXS[j];
                    DI_DCS.col(this->ode_.xtu_vars() * (Cardinals - 1) + this->ode_.t_var())
                        .template head<DODE::XV>(this->ode_.x_vars()) +=
                        Scalar(Coeffs::Cardinal_DXInterp_Weights[i][j]) * C_DXS[j];

                    DI_DCS
                        .template block<DODE::UV, DODE::UV>(
                            this->ode_.xt_vars(), j * this->ode_.xtu_vars() + this->ode_.xt_vars(),
                            this->ode_.u_vars(), this->ode_.u_vars())
                        .diagonal()
                        .setConstant(Scalar(Coeffs::Cardinal_UPoly_Weights[i][j]));

                    ////////////////////////////// Sum up Cardinal Output
                    /// Terms/////////////////////////////////////////
                    fx.template segment<DODE::XV>(i * this->ode_.x_vars(), this->ode_.x_vars()) +=
                        (Scalar(Coeffs::Cardinal_XDef_Weights[i][j]) *
                             C_XS[j].template head<DODE::XV>(this->ode_.x_vars()) +
                         (Coeffs::Cardinal_DXDef_Weights[i][j] * h) * C_DXS[j]);

                    jx.template block<DODE::XV, DODE::XV>(i * this->ode_.x_vars(),
                                                          j * this->ode_.xtu_vars(),
                                                          this->ode_.x_vars(), this->ode_.x_vars())
                        .diagonal()
                        .setConstant(Scalar(Coeffs::Cardinal_XDef_Weights[i][j]));

                    jx.template block<DODE::XV, DODE::XtUV>(
                        i * this->ode_.x_vars(), j * this->ode_.xtu_vars(), this->ode_.x_vars(),
                        this->ode_.xtu_vars()) +=
                        (Coeffs::Cardinal_DXDef_Weights[i][j] * h) *
                        C_JDXS[j].template leftCols<DODE::XtUV>(this->ode_.xtu_vars());

                    jx.template block<DODE::XV, DODE::PV>(
                        i * this->ode_.x_vars(), Cardinals * this->ode_.xtu_vars(),
                        this->ode_.x_vars(), this->ode_.p_vars()) +=
                        (Coeffs::Cardinal_DXDef_Weights[i][j] * h) *
                        C_JDXS[j].template rightCols<DODE::PV>(this->ode_.p_vars());

                    jx.col(this->ode_.t_var())
                        .template segment<DODE::XV>(i * this->ode_.x_vars(), this->ode_.x_vars()) -=
                        Scalar(Coeffs::Cardinal_DXDef_Weights[i][j]) * C_DXS[j];
                    jx.col(this->ode_.xtu_vars() * (Cardinals - 1) + this->ode_.t_var())
                        .template segment<DODE::XV>(i * this->ode_.x_vars(), this->ode_.x_vars()) +=
                        Scalar(Coeffs::Cardinal_DXDef_Weights[i][j]) * C_DXS[j];
                }

                fx.template segment<DODE::XV>(i * this->ode_.x_vars(), this->ode_.x_vars()) +=
                    (h * Coeffs::Interior_DXDef_Weights[i] * I_DXS[i]);
                jx.template middleRows<DODE::XV>(i * this->ode_.x_vars(), this->ode_.x_vars())
                    .noalias() += ((h * Coeffs::Interior_DXDef_Weights[i]) * I_JDXS[i]) * (DI_DCS);

                jx.col(this->ode_.t_var())
                    .template segment<DODE::XV>(i * this->ode_.x_vars(), this->ode_.x_vars()) -=
                    Scalar(Coeffs::Interior_DXDef_Weights[i]) * I_DXS[i];
                jx.col(this->ode_.xtu_vars() * (Cardinals - 1) + this->ode_.t_var())
                    .template segment<DODE::XV>(i * this->ode_.x_vars(), this->ode_.x_vars()) +=
                    Scalar(Coeffs::Interior_DXDef_Weights[i]) * I_DXS[i];

                adjhess.noalias() += DI_DCS.transpose() *
                                     (I_HDXS[i] * (h * Coeffs::Interior_DXDef_Weights[i])) * DI_DCS;
                HTpar.noalias() +=
                    ((I_AGXS[i].transpose() * Scalar(Coeffs::Interior_DXDef_Weights[i])) * DI_DCS)
                        .transpose();
            }

            adjhess.col(this->ode_.t_var()) -= HTpar;
            adjhess.col(this->ode_.t_var() + this->ode_.xtu_vars() * (Cardinals - 1)) += HTpar;
            adjhess.row(this->ode_.t_var()) -= HTpar;
            adjhess.row(this->ode_.t_var() + this->ode_.xtu_vars() * (Cardinals - 1)) += HTpar;
            adjgrad.noalias() = (adjvars.transpose() * jx).transpose();
            // QED
        };

        using XType = ODEInput<Scalar>;
        using FXType = ODEOutput<Scalar>;
        using JXType = ODEJacobian<Scalar>;
        using AGXType = ODEInput<Scalar>;
        using AVType = ODEOutput<Scalar>;
        using HType = ODEHessian<Scalar>;

        using DIType = Eigen::Matrix<Scalar, DODE::IRC, Base::IRC>;
        using HTParType = Input<Scalar>;

        const int irowsode = this->ode_.input_rows();
        const int orowsode = this->ode_.output_rows();

        BumpAllocator::allocate_run(
            impl, ArrayOfTempSpecs<XType, Cardinals>(irowsode, 1),
            ArrayOfTempSpecs<FXType, Cardinals>(orowsode, 1),
            ArrayOfTempSpecs<JXType, Cardinals>(orowsode, irowsode), TempSpec<AGXType>(irowsode, 1),
            ArrayOfTempSpecs<AVType, Cardinals>(orowsode, 1), TempSpec<HType>(irowsode, irowsode),

            ArrayOfTempSpecs<XType, Interiors>(irowsode, 1),
            ArrayOfTempSpecs<FXType, Interiors>(orowsode, 1),
            ArrayOfTempSpecs<JXType, Interiors>(orowsode, irowsode),
            ArrayOfTempSpecs<AGXType, Interiors>(irowsode, 1),
            ArrayOfTempSpecs<AVType, Interiors>(orowsode, 1),
            ArrayOfTempSpecs<HType, Interiors>(irowsode, irowsode),

            TempSpec<DIType>(irowsode, this->input_rows()),
            TempSpec<HTParType>(this->input_rows(), 1));
    }
};

} // namespace tycho::oc
