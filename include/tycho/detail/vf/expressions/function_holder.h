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

/// @brief CRTP adapter that wraps a single VectorFunction as a named subtype.
///
/// FunctionHolder forwards every evaluation and derivative-assembly call to the
/// held function @ref func_, while presenting a distinct concrete type to the
/// rest of the DSL. It is the base used to give a composed expression a stable
/// identity (e.g. a user-defined dynamics or constraint type) and to allow the
/// wrapped function to be swapped at runtime via @ref set_function.
///
/// @tparam Derived  CRTP most-derived type built on this holder.
/// @tparam Func     Wrapped VectorFunction type whose calls are forwarded.
/// @tparam IR       Compile-time input-row count (Eigen::Dynamic if runtime).
/// @tparam OR       Compile-time output-row count (Eigen::Dynamic if runtime).
/// @ingroup vf
template <class Derived, class Func, int IR, int OR>
struct FunctionHolder : VectorFunction<Derived, IR, OR, DenseDerivativeMode::Analytic> {
    /// @brief VectorFunction CRTP base specialized for analytic derivatives.
    using Base = VectorFunction<Derived, IR, OR, DenseDerivativeMode::Analytic>;
    using Base::compute;
    VF_TYPE_ALIASES(Base);
    Func func_; ///< @brief The wrapped function all calls are forwarded to.
    /// @brief Input-domain descriptor inherited from the wrapped function.
    using INPUT_DOMAIN = typename Func::INPUT_DOMAIN;
    static constexpr bool is_linear_function =
        Func::is_linear_function; ///< @brief True if the wrapped function is linear.
    static constexpr bool is_vectorizable =
        Func::is_vectorizable; ///< @brief True if the wrapped function supports SuperScalar packs.

    /// @brief Default-construct an empty holder (no wrapped function set).
    FunctionHolder() {}
    /// @brief Construct from a function, adopting its I/O sizes and domain.
    /// @param f  The VectorFunction to wrap and forward to.
    FunctionHolder(Func f) : func_(std::move(f)) {
        this->set_io_rows(this->func_.input_rows(), this->func_.output_rows());
        this->set_input_domain(this->input_rows(), {func_.input_domain()});
    }
    /// @brief Replace the wrapped function, re-deriving I/O sizes and domain.
    /// @param f  The new VectorFunction to wrap.
    void set_function(Func f) {
        this->func_ = f;
        this->set_io_rows(this->func_.input_rows(), this->func_.output_rows());
        this->set_input_domain(this->input_rows(), {func_.input_domain()});
    }
    /// @brief Evaluate the wrapped function: `fx = func_(x)`.
    /// @tparam InType   Concrete Eigen input expression type.
    /// @tparam OutType  Concrete Eigen output buffer type.
    /// @param  x        Input vector (size = `input_rows()`).
    /// @param  fx_      Output buffer (size = `output_rows()`).
    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        this->func_.compute(x, fx_);
    }
    /// @brief Forward the value-and-Jacobian evaluation to the wrapped function.
    /// @tparam InType   Concrete Eigen input expression type.
    /// @tparam OutType  Concrete Eigen output buffer type.
    /// @tparam JacType  Concrete Eigen Jacobian buffer type.
    /// @param  x        Input vector (size = `input_rows()`).
    /// @param  fx_      Output buffer (size = `output_rows()`).
    /// @param  jx_      Jacobian buffer (`output_rows()` x `input_rows()`).
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        this->func_.compute_jacobian(x, fx_, jx_);
    }
    /// @brief Forward the value/Jacobian/adjoint-gradient/adjoint-Hessian call.
    /// @tparam InType       Concrete Eigen input expression type.
    /// @tparam OutType      Concrete Eigen output buffer type.
    /// @tparam JacType      Concrete Eigen Jacobian buffer type.
    /// @tparam AdjGradType  Concrete Eigen adjoint-gradient buffer type.
    /// @tparam AdjHessType  Concrete Eigen adjoint-Hessian buffer type.
    /// @tparam AdjVarType   Concrete Eigen adjoint (Lagrange-multiplier) type.
    /// @param  x        Input vector (size = `input_rows()`).
    /// @param  fx_      Output buffer (size = `output_rows()`).
    /// @param  jx_      Jacobian buffer (`output_rows()` x `input_rows()`).
    /// @param  adjgrad_ Adjoint-gradient accumulator (size = `input_rows()`).
    /// @param  adjhess_ Adjoint-Hessian accumulator (`input_rows()` square).
    /// @param  adjvars  Adjoint variables seeding the reverse pass.
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        CVecRef<InType> x, CVecRef<OutType> fx_, CMatRef<JacType> jx_,
        CVecRef<AdjGradType> adjgrad_, CMatRef<AdjHessType> adjhess_,
        CVecRef<AdjVarType> adjvars) const {
        this->func_.compute_jacobian_adjointgradient_adjointhessian(x, fx_, jx_, adjgrad_, adjhess_,
                                                                    adjvars);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// @brief Forward the chain-rule right-Jacobian product to the wrapped function.
    /// @tparam Target      Concrete Eigen target/output block type.
    /// @tparam Left        Concrete Eigen left-operand block type.
    /// @tparam Right       Concrete Eigen right-operand (this Jacobian) type.
    /// @tparam Assignment  Assignment policy (direct or plus-equals).
    /// @tparam Aliased     Whether target and operands may alias.
    /// @param  target_  Destination block accumulating the product.
    /// @param  left     Upstream Jacobian factor.
    /// @param  right    This function's local Jacobian factor.
    /// @param  assign   Assignment-policy tag instance.
    /// @param  aliased  Aliasing-flag tag instance.
    template <class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void right_jacobian_product(CMatRef<Target> target_, CEigRef<Left> left,
                                       CEigRef<Right> right, Assignment assign,
                                       std::bool_constant<Aliased> aliased) const {
        this->func_.right_jacobian_product(target_, left, right, assign, aliased);
    }
    /// @brief Forward the symmetric (Hessian-style) Jacobian product.
    /// @tparam Target      Concrete Eigen target/output block type.
    /// @tparam Left        Concrete Eigen left-operand block type.
    /// @tparam Right       Concrete Eigen right-operand block type.
    /// @tparam Assignment  Assignment policy (direct or plus-equals).
    /// @tparam Aliased     Whether target and operands may alias.
    /// @param  target_  Destination block accumulating the product.
    /// @param  left     Left operand of the symmetric product.
    /// @param  right    Right operand of the symmetric product.
    /// @param  assign   Assignment-policy tag instance.
    /// @param  aliased  Aliasing-flag tag instance.
    template <class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void symetric_jacobian_product(CMatRef<Target> target_, CEigRef<Left> left,
                                          CEigRef<Right> right, Assignment assign,
                                          std::bool_constant<Aliased> aliased) const {
        this->func_.symetric_jacobian_product(target_, left, right, assign, aliased);
    }
    /// @brief Forward Jacobian accumulation into a target block.
    /// @tparam Target      Concrete Eigen target/output block type.
    /// @tparam JacType     Concrete Eigen source-Jacobian type.
    /// @tparam Assignment  Assignment policy (direct or plus-equals).
    /// @param  target_  Destination block.
    /// @param  right    Source Jacobian to accumulate.
    /// @param  assign   Assignment-policy tag instance.
    template <class Target, class JacType, class Assignment>
    inline void accumulate_jacobian(CMatRef<Target> target_, CMatRef<JacType> right,
                                    Assignment assign) const {
        this->func_.accumulate_jacobian(target_, right, assign);
    }
    /// @brief Forward gradient accumulation into a target block.
    /// @tparam Target      Concrete Eigen target/output block type.
    /// @tparam JacType     Concrete Eigen source-gradient type.
    /// @tparam Assignment  Assignment policy (direct or plus-equals).
    /// @param  target_  Destination block.
    /// @param  right    Source gradient to accumulate.
    /// @param  assign   Assignment-policy tag instance.
    template <class Target, class JacType, class Assignment>
    inline void accumulate_gradient(CMatRef<Target> target_, CMatRef<JacType> right,
                                    Assignment assign) const {
        this->func_.accumulate_gradient(target_, right, assign);
    }
    /// @brief Forward Hessian accumulation into a target block.
    /// @tparam Target      Concrete Eigen target/output block type.
    /// @tparam JacType     Concrete Eigen source-Hessian type.
    /// @tparam Assignment  Assignment policy (direct or plus-equals).
    /// @param  target_  Destination block.
    /// @param  right    Source Hessian to accumulate.
    /// @param  assign   Assignment-policy tag instance.
    template <class Target, class JacType, class Assignment>
    inline void accumulate_hessian(CMatRef<Target> target_, CMatRef<JacType> right,
                                   Assignment assign) const {
        this->func_.accumulate_hessian(target_, right, assign);
    }
    /// @brief Forward in-place scaling of a Jacobian block by a scalar.
    /// @tparam Target  Concrete Eigen target/output block type.
    /// @tparam Scalar  Scalar multiplier type.
    /// @param  target_ Jacobian block to scale in place.
    /// @param  s       Scalar factor.
    template <class Target, class Scalar>
    inline void scale_jacobian(CMatRef<Target> target_, Scalar s) const {
        this->func_.scale_jacobian(target_, s);
    }
    /// @brief Forward in-place scaling of a gradient block by a scalar.
    /// @tparam Target  Concrete Eigen target/output block type.
    /// @tparam Scalar  Scalar multiplier type.
    /// @param  target_ Gradient block to scale in place.
    /// @param  s       Scalar factor.
    template <class Target, class Scalar>
    inline void scale_gradient(CMatRef<Target> target_, Scalar s) const {
        this->func_.scale_gradient(target_, s);
    }
    /// @brief Forward in-place scaling of a Hessian block by a scalar.
    /// @tparam Target  Concrete Eigen target/output block type.
    /// @tparam Scalar  Scalar multiplier type.
    /// @param  target_ Hessian block to scale in place.
    /// @param  s       Scalar factor.
    template <class Target, class Scalar>
    inline void scale_hessian(CMatRef<Target> target_, Scalar s) const {
        this->func_.scale_hessian(target_, s);
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
};

} // namespace tycho::vf
