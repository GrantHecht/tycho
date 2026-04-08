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
#include "tycho/detail/utils/crtp_base.h"
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

template <int IR, int OR> struct DenseFunctionSpec {
    template <class Scalar> using Output = Eigen::Matrix<Scalar, OR, 1>;
    template <class Scalar> using Input = Eigen::Matrix<Scalar, IR, 1>;
    template <class Scalar> using Jacobian = Eigen::Matrix<Scalar, OR, IR>;
    template <class Scalar> using Hessian = Eigen::Matrix<Scalar, IR, IR>;

    using InType = Eigen::Ref<const Input<double>>;
    using OutType = Eigen::Ref<Output<double>>;

    using JacType = typename std::conditional<
        OR == 1, Eigen::Ref<Eigen::Matrix<double, -1, IR>, 0, Eigen::Stride<-1, -1>>,
        Eigen::Ref<Jacobian<double>>>::type;

    using AdjGradType = Eigen::Ref<Input<double>>;
    using AdjVarType = Eigen::Ref<const Output<double>>;
    using AdjHessType = Eigen::Ref<Hessian<double>>;

    using SuperInType = Eigen::Ref<const Input<tycho::DefaultSuperScalar>>;
    using SuperOutType = Eigen::Ref<Output<tycho::DefaultSuperScalar>>;

    using SuperJacType = typename std::conditional<
        OR == 1,
        Eigen::Ref<Eigen::Matrix<tycho::DefaultSuperScalar, -1, IR>, 0, Eigen::Stride<-1, -1>>,
        Eigen::Ref<Jacobian<tycho::DefaultSuperScalar>>>::type;

    using SuperAdjGradType = Eigen::Ref<Input<tycho::DefaultSuperScalar>>;
    using SuperAdjVarType = Eigen::Ref<const Output<tycho::DefaultSuperScalar>>;
    using SuperAdjHessType = Eigen::Ref<Hessian<tycho::DefaultSuperScalar>>;

    using RightJacTarget = Eigen::Ref<Eigen::Matrix<double, -1, IR>>;
    using LeftJacMatrix = Eigen::Ref<const Eigen::Matrix<double, -1, OR>>;
    using LeftDiagMatrix = Eigen::DiagonalMatrix<double, OR>;

    using SuperLeftJacMatrix = Eigen::Ref<const Eigen::Matrix<tycho::DefaultSuperScalar, -1, OR>>;
    using SuperRightJacTarget = Eigen::Ref<Eigen::Matrix<tycho::DefaultSuperScalar, -1, IR>>;

    struct Concept { // abstract base class for model.
        virtual ~Concept() = default;
        // Your (internal) interface goes here.

        virtual DomainMatrix input_domain() const = 0;
        virtual bool is_linear() const = 0;
        virtual void enable_vectorization(bool b) const = 0;

        virtual void compute(CVecRef<InType> x,
                             CVecRef<OutType> fx_) const = 0;

        virtual void compute(CVecRef<SuperInType> x,
                             CVecRef<SuperOutType> fx_) const = 0;

        virtual void compute_jacobian(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const = 0;

        virtual void compute_jacobian(CVecRef<SuperInType> x,
                                      CVecRef<SuperOutType> fx_,
                                      CMatRef<SuperJacType> jx_) const = 0;

        virtual void compute_jacobian_adjointgradient_adjointhessian(
            CVecRef<InType> x, CVecRef<OutType> fx_,
            CMatRef<JacType> jx_, CVecRef<AdjGradType> adjgrad_,
            CMatRef<AdjHessType> adjhess_,
            CVecRef<AdjVarType> adjvars) const = 0;

        virtual void compute_jacobian_adjointgradient_adjointhessian(
            CVecRef<SuperInType> x, CVecRef<SuperOutType> fx_,
            CMatRef<SuperJacType> jx_, CVecRef<SuperAdjGradType> adjgrad_,
            CMatRef<SuperAdjHessType> adjhess_,
            CVecRef<SuperAdjVarType> adjvars) const = 0;

        virtual void scale_jacobian(CMatRef<JacType> target_, double s) const = 0;
        /*virtual void scale_gradient(CMatRef<AdjGradType> target_,
                                    double s) const = 0;
        virtual void scale_hessian(CMatRef<AdjHessType> target_,
                                   double s) const = 0;*/

        virtual void accumulate_jacobian(CMatRef<JacType> target_,
                                         CMatRef<JacType> right,
                                         DirectAssignment assign) const = 0;
        virtual void accumulate_jacobian(CMatRef<JacType> target_,
                                         CMatRef<JacType> right,
                                         PlusEqualsAssignment assign) const = 0;
        virtual void accumulate_jacobian(CMatRef<JacType> target_,
                                         CMatRef<JacType> right,
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

        virtual void right_jacobian_product(CMatRef<RightJacTarget> target_,
                                            CEigRef<LeftJacMatrix> left,
                                            CEigRef<JacType> right,
                                            DirectAssignment assign,
                                            std::bool_constant<true> aliased) const = 0;
        virtual void right_jacobian_product(CMatRef<RightJacTarget> target_,
                                            CEigRef<LeftJacMatrix> left,
                                            CEigRef<JacType> right,
                                            PlusEqualsAssignment assign,
                                            std::bool_constant<true> aliased) const = 0;
        virtual void right_jacobian_product(CMatRef<RightJacTarget> target_,
                                            CEigRef<LeftJacMatrix> left,
                                            CEigRef<JacType> right,
                                            MinusEqualsAssignment assign,
                                            std::bool_constant<true> aliased) const = 0;

        virtual void right_jacobian_product(CMatRef<RightJacTarget> target_,
                                            CEigRef<LeftJacMatrix> left,
                                            CEigRef<JacType> right,
                                            DirectAssignment assign,
                                            std::bool_constant<false> aliased) const = 0;
        virtual void right_jacobian_product(CMatRef<RightJacTarget> target_,
                                            CEigRef<LeftJacMatrix> left,
                                            CEigRef<JacType> right,
                                            PlusEqualsAssignment assign,
                                            std::bool_constant<false> aliased) const = 0;
        virtual void right_jacobian_product(CMatRef<RightJacTarget> target_,
                                            CEigRef<LeftJacMatrix> left,
                                            CEigRef<JacType> right,
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
