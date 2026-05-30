// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Defines the type erasure spec (DenseFunctionSpec) for the methods of Dense
// Tycho vector functions. This encompasses the primary compute and derivative
// methods with both double and super-scalar arguments as well as selected
// jacobian operations on double valued matrices that have been shown to improve
// performance under certain circumstances. This Spec is used to define the
// type erased GenericFunction type.
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
//   - PR 9: Removed dead Model<>/ExternalInterface<> boilerplate
// =============================================================================

#pragma once

#include "tycho/detail/vf/core/assignment_types.h"
#include "tycho/detail/vf/core/eigen_ref_aliases.h"
#include "tycho/detail/vf/derivatives/detect_super_scalar.h"
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
namespace tycho::vf {

/// @brief Type-erasure spec describing the dense VectorFunction interface.
/// @ingroup vf
///
/// Bundles the matrix type aliases (`double` and SuperScalar variants) and the
/// abstract @ref DenseFunctionSpec::Concept "Concept" virtual interface used to
/// build the type-erased `GenericFunction`. See the file header for rationale.
/// @tparam IR  Input row count (`-1` if dynamic).
/// @tparam OR  Output row count (`-1` if dynamic).
template <int IR, int OR> struct DenseFunctionSpec {
    /// @brief Output column-vector type for scalar @p Scalar.
    /// @tparam Scalar  Element scalar type.
    template <class Scalar> using Output = Eigen::Matrix<Scalar, OR, 1>;
    /// @brief Input column-vector type for scalar @p Scalar.
    /// @tparam Scalar  Element scalar type.
    template <class Scalar> using Input = Eigen::Matrix<Scalar, IR, 1>;
    /// @brief Jacobian matrix type for scalar @p Scalar.
    /// @tparam Scalar  Element scalar type.
    template <class Scalar> using Jacobian = Eigen::Matrix<Scalar, OR, IR>;
    /// @brief Hessian matrix type for scalar @p Scalar.
    /// @tparam Scalar  Element scalar type.
    template <class Scalar> using Hessian = Eigen::Matrix<Scalar, IR, IR>;

    using InType = Eigen::Ref<const Input<double>>; ///< @brief Const input reference (`double`).
    using OutType = Eigen::Ref<Output<double>>;     ///< @brief Mutable output reference (`double`).

    /// @brief Jacobian reference (`double`); strided row-vector form when `OR == 1`.
    using JacType = typename std::conditional<
        OR == 1, Eigen::Ref<Eigen::Matrix<double, -1, IR>, 0, Eigen::Stride<-1, -1>>,
        Eigen::Ref<Jacobian<double>>>::type;

    using AdjGradType =
        Eigen::Ref<Input<double>>; ///< @brief Adjoint-gradient reference (`double`).
    using AdjVarType =
        Eigen::Ref<const Output<double>>; ///< @brief Adjoint-variable reference (`double`).
    using AdjHessType =
        Eigen::Ref<Hessian<double>>; ///< @brief Adjoint-Hessian reference (`double`).

    using SuperInType =
        Eigen::Ref<const Input<tycho::DefaultSuperScalar>>; ///< @brief Const input reference
                                                            ///< (SuperScalar).
    using SuperOutType =
        Eigen::Ref<Output<tycho::DefaultSuperScalar>>; ///< @brief Mutable output reference
                                                       ///< (SuperScalar).

    /// @brief Jacobian reference (SuperScalar); strided row-vector form when `OR == 1`.
    using SuperJacType = typename std::conditional<
        OR == 1,
        Eigen::Ref<Eigen::Matrix<tycho::DefaultSuperScalar, -1, IR>, 0, Eigen::Stride<-1, -1>>,
        Eigen::Ref<Jacobian<tycho::DefaultSuperScalar>>>::type;

    using SuperAdjGradType =
        Eigen::Ref<Input<tycho::DefaultSuperScalar>>; ///< @brief Adjoint-gradient reference
                                                      ///< (SuperScalar).
    using SuperAdjVarType =
        Eigen::Ref<const Output<tycho::DefaultSuperScalar>>; ///< @brief Adjoint-variable reference
                                                             ///< (SuperScalar).
    using SuperAdjHessType =
        Eigen::Ref<Hessian<tycho::DefaultSuperScalar>>; ///< @brief Adjoint-Hessian reference
                                                        ///< (SuperScalar).

    using RightJacTarget =
        Eigen::Ref<Eigen::Matrix<double, -1, IR>>; ///< @brief Target of a right Jacobian product
                                                   ///< (`double`).
    using LeftJacMatrix =
        Eigen::Ref<const Eigen::Matrix<double, -1, OR>>;      ///< @brief Left operand of a Jacobian
                                                              ///< product (`double`).
    using LeftDiagMatrix = Eigen::DiagonalMatrix<double, OR>; ///< @brief Diagonal left operand of a
                                                              ///< Jacobian product (`double`).

    using SuperLeftJacMatrix =
        Eigen::Ref<const Eigen::Matrix<tycho::DefaultSuperScalar, -1,
                                       OR>>; ///< @brief Left operand of a Jacobian product
                                             ///< (SuperScalar).
    using SuperRightJacTarget =
        Eigen::Ref<Eigen::Matrix<tycho::DefaultSuperScalar, -1,
                                 IR>>; ///< @brief Target of a right Jacobian product (SuperScalar).

    /// @brief Abstract base class (the type-erasure "concept") of the dense VF interface.
    struct Concept { // abstract base class for model.
        /// @brief Virtual destructor for safe polymorphic deletion.
        virtual ~Concept() = default;
        // Your (internal) interface goes here.

        /// @brief Returns the active input sub-domain layout.
        /// @return The domain matrix describing contiguous input ranges.
        virtual DomainMatrix input_domain() const = 0;
        /// @brief Returns whether the wrapped function is linear.
        /// @return `true` if linear.
        virtual bool is_linear() const = 0;
        /// @brief Enables or disables vectorized evaluation.
        /// @param b  `true` to enable vectorization.
        virtual void enable_vectorization(bool b) const = 0;

        /// @brief Evaluates the function (`double`).
        /// @param x  Input vector.
        /// @param fx_  Output vector to write.
        virtual void compute(CVecRef<InType> x, CVecRef<OutType> fx_) const = 0;

        /// @brief Evaluates the function (SuperScalar).
        /// @param x  Input vector.
        /// @param fx_  Output vector to write.
        virtual void compute(CVecRef<SuperInType> x, CVecRef<SuperOutType> fx_) const = 0;

        /// @brief Evaluates the function and its Jacobian (`double`).
        /// @param x  Input vector.
        /// @param fx_  Output vector to write.
        /// @param jx_  Jacobian to write.
        virtual void compute_jacobian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const = 0;

        /// @brief Evaluates the function and its Jacobian (SuperScalar).
        /// @param x  Input vector.
        /// @param fx_  Output vector to write.
        /// @param jx_  Jacobian to write.
        virtual void compute_jacobian(CVecRef<SuperInType> x, CVecRef<SuperOutType> fx_,
                                      CMatRef<SuperJacType> jx_) const = 0;

        /// @brief Evaluates value, Jacobian, adjoint gradient, and adjoint Hessian (`double`).
        /// @param x  Input vector.
        /// @param fx_  Output vector to write.
        /// @param jx_  Jacobian to write.
        /// @param adjgrad_  Adjoint gradient to write.
        /// @param adjhess_  Adjoint Hessian to write.
        /// @param adjvars  Adjoint (output) multipliers.
        virtual void compute_jacobian_adjointgradient_adjointhessian(
            CVecRef<InType> x, CVecRef<OutType> fx_, CMatRef<JacType> jx_,
            CVecRef<AdjGradType> adjgrad_, CMatRef<AdjHessType> adjhess_,
            CVecRef<AdjVarType> adjvars) const = 0;

        /// @brief Evaluates value, Jacobian, adjoint gradient, and adjoint Hessian (SuperScalar).
        /// @param x  Input vector.
        /// @param fx_  Output vector to write.
        /// @param jx_  Jacobian to write.
        /// @param adjgrad_  Adjoint gradient to write.
        /// @param adjhess_  Adjoint Hessian to write.
        /// @param adjvars  Adjoint (output) multipliers.
        virtual void compute_jacobian_adjointgradient_adjointhessian(
            CVecRef<SuperInType> x, CVecRef<SuperOutType> fx_, CMatRef<SuperJacType> jx_,
            CVecRef<SuperAdjGradType> adjgrad_, CMatRef<SuperAdjHessType> adjhess_,
            CVecRef<SuperAdjVarType> adjvars) const = 0;

        /// @brief Scales a Jacobian in place.
        /// @param target_  Jacobian to scale.
        /// @param s  Scale factor.
        virtual void scale_jacobian(CMatRef<JacType> target_, double s) const = 0;
        /*virtual void scale_gradient(CMatRef<AdjGradType> target_,
                                    double s) const = 0;
        virtual void scale_hessian(CMatRef<AdjHessType> target_,
                                   double s) const = 0;*/

        /// @brief Accumulates a Jacobian via direct assignment.
        /// @param target_  Destination Jacobian.
        /// @param right  Source Jacobian.
        /// @param assign  Direct-assignment tag.
        virtual void accumulate_jacobian(CMatRef<JacType> target_, CMatRef<JacType> right,
                                         DirectAssignment assign) const = 0;
        /// @brief Accumulates a Jacobian via `+=`.
        /// @param target_  Destination Jacobian.
        /// @param right  Source Jacobian.
        /// @param assign  Plus-equals assignment tag.
        virtual void accumulate_jacobian(CMatRef<JacType> target_, CMatRef<JacType> right,
                                         PlusEqualsAssignment assign) const = 0;
        /// @brief Accumulates a Jacobian via `-=`.
        /// @param target_  Destination Jacobian.
        /// @param right  Source Jacobian.
        /// @param assign  Minus-equals assignment tag.
        virtual void accumulate_jacobian(CMatRef<JacType> target_, CMatRef<JacType> right,
                                         MinusEqualsAssignment assign) const = 0;

        /* virtual void accumulate_gradient(CMatRef<AdjGradType> target_,
                                          CMatRef<AdjGradType> right,
                                          DirectAssignment assign) const = 0;
         virtual void accumulate_gradient(CMatRef<AdjGradType> target_,
                                          CMatRef<AdjGradType> right,
                                          PlusEqualsAssignment assign) const = 0;
         virtual void accumulate_gradient(CMatRef<AdjGradType> target_,
                                          CMatRef<AdjGradType> right,
                                          MinusEqualsAssignment assign) const = 0;

         virtual void accumulate_hessian(CMatRef<AdjHessType> target_,
                                         CMatRef<AdjHessType> right,
                                         DirectAssignment assign) const = 0;
         virtual void accumulate_hessian(CMatRef<AdjHessType> target_,
                                         CMatRef<AdjHessType> right,
                                         PlusEqualsAssignment assign) const = 0;
         virtual void accumulate_hessian(CMatRef<AdjHessType> target_,
                                         CMatRef<AdjHessType> right,
                                         MinusEqualsAssignment assign) const = 0;*/

        /// @brief Computes `left * right` into the target (aliased, direct assign).
        /// @param target_  Destination matrix.
        /// @param left  Left operand.
        /// @param right  Right (Jacobian) operand.
        /// @param assign  Direct-assignment tag.
        /// @param aliased  Aliasing tag (`true`).
        virtual void right_jacobian_product(CMatRef<RightJacTarget> target_,
                                            CEigRef<LeftJacMatrix> left, CEigRef<JacType> right,
                                            DirectAssignment assign,
                                            std::bool_constant<true> aliased) const = 0;
        /// @brief Computes `left * right` into the target (aliased, `+=`).
        /// @param target_  Destination matrix.
        /// @param left  Left operand.
        /// @param right  Right (Jacobian) operand.
        /// @param assign  Plus-equals assignment tag.
        /// @param aliased  Aliasing tag (`true`).
        virtual void right_jacobian_product(CMatRef<RightJacTarget> target_,
                                            CEigRef<LeftJacMatrix> left, CEigRef<JacType> right,
                                            PlusEqualsAssignment assign,
                                            std::bool_constant<true> aliased) const = 0;
        /// @brief Computes `left * right` into the target (aliased, `-=`).
        /// @param target_  Destination matrix.
        /// @param left  Left operand.
        /// @param right  Right (Jacobian) operand.
        /// @param assign  Minus-equals assignment tag.
        /// @param aliased  Aliasing tag (`true`).
        virtual void right_jacobian_product(CMatRef<RightJacTarget> target_,
                                            CEigRef<LeftJacMatrix> left, CEigRef<JacType> right,
                                            MinusEqualsAssignment assign,
                                            std::bool_constant<true> aliased) const = 0;

        /// @brief Computes `left * right` into the target (non-aliased, direct assign).
        /// @param target_  Destination matrix.
        /// @param left  Left operand.
        /// @param right  Right (Jacobian) operand.
        /// @param assign  Direct-assignment tag.
        /// @param aliased  Aliasing tag (`false`).
        virtual void right_jacobian_product(CMatRef<RightJacTarget> target_,
                                            CEigRef<LeftJacMatrix> left, CEigRef<JacType> right,
                                            DirectAssignment assign,
                                            std::bool_constant<false> aliased) const = 0;
        /// @brief Computes `left * right` into the target (non-aliased, `+=`).
        /// @param target_  Destination matrix.
        /// @param left  Left operand.
        /// @param right  Right (Jacobian) operand.
        /// @param assign  Plus-equals assignment tag.
        /// @param aliased  Aliasing tag (`false`).
        virtual void right_jacobian_product(CMatRef<RightJacTarget> target_,
                                            CEigRef<LeftJacMatrix> left, CEigRef<JacType> right,
                                            PlusEqualsAssignment assign,
                                            std::bool_constant<false> aliased) const = 0;
        /// @brief Computes `left * right` into the target (non-aliased, `-=`).
        /// @param target_  Destination matrix.
        /// @param left  Left operand.
        /// @param right  Right (Jacobian) operand.
        /// @param assign  Minus-equals assignment tag.
        /// @param aliased  Aliasing tag (`false`).
        virtual void right_jacobian_product(CMatRef<RightJacTarget> target_,
                                            CEigRef<LeftJacMatrix> left, CEigRef<JacType> right,
                                            MinusEqualsAssignment assign,
                                            std::bool_constant<false> aliased) const = 0;

        /*virtual void right_jacobian_product(
            CMatRef<RightJacTarget> target_,
            CEigRef<LeftDiagMatrix> left,
            CEigRef<JacType> right, DirectAssignment assign,
            std::bool_constant<true> aliased) const = 0;
        virtual void right_jacobian_product(
            CMatRef<RightJacTarget> target_,
            CEigRef<LeftDiagMatrix> left,
            CEigRef<JacType> right, PlusEqualsAssignment assign,
            std::bool_constant<true> aliased) const = 0;
        virtual void right_jacobian_product(
            CMatRef<RightJacTarget> target_,
            CEigRef<LeftDiagMatrix> left,
            CEigRef<JacType> right, MinusEqualsAssignment assign,
            std::bool_constant<true> aliased) const = 0;*/
    };
};

} // namespace tycho::vf
