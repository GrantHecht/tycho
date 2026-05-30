// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Defines the class GenericFunction which is a vector function that can be
// constructed form ANY other dense vector function with compatible input and output sizes. This is
// probably the most important class in the vector functions part of the library as it allows us to
// type erase arbitrarily compilicated compile time or run time defined vector functions. It holds
// type easure object that constructable from any compatible object and forwards its compute calls
// as well as selected product and accumulation operations to this type erased function.
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
//   - PR 6: replaced rubber_types GFTE with flat GFStorage (SBO + value semantics)
// =============================================================================

#pragma once

#include "tycho/detail/vf/common/common_functions.h"
#include "tycho/detail/vf/core/dense_function_base.h"
#include "tycho/detail/vf/type_erasure/gf_type_erasure.h"

namespace tycho::vf {

/// @brief Type-erased VectorFunction holding any compatible function of the same size.
///
/// GenericFunction is the central type-erasure facility of the VectorFunction
/// subsystem and the type most user code holds. It can be constructed from *any*
/// dense VectorFunction with matching input/output sizes, hiding the (often
/// deeply nested, compile-time-built) expression type behind a single runtime
/// interface. It forwards compute, Jacobian, and adjoint-gradient/Hessian calls
/// — plus selected product and accumulation operations — to the erased function
/// via a flat vtable (GFConcept). Value semantics are preserved: copies share
/// ownership of the underlying function (O(1) refcount copy) and the dynamic
/// `GenericFunction<-1,-1>` form can absorb any fixed-size function.
/// @tparam IR  Compile-time input-row count (-1 for dynamic).
/// @tparam OR  Compile-time output-row count (-1 for dynamic).
/// @ingroup vf
template <int IR, int OR> struct GenericFunction : VectorFunction<GenericFunction<IR, OR>, IR, OR> {
    /// @brief CRTP VectorFunction base type.
    using Base = VectorFunction<GenericFunction<IR, OR>, IR, OR>;
    VF_TYPE_ALIASES(Base);

    static constexpr bool is_generic_function =
        true; ///< @brief Trait flag: this is a type-erased function.
    static constexpr bool is_vectorizable =
        true; ///< @brief SuperScalar (SIMD) compute is supported.

    /// @brief Dense function spec providing the scalar and SuperScalar IO/Jacobian types.
    using Dspec = DenseFunctionSpec<IR, OR>;
    /// @brief Target matrix type for right-Jacobian products.
    using RightJacTarget = Eigen::Ref<Eigen::Matrix<double, -1, IR>>;
    /// @brief This concrete GenericFunction type (CRTP self type).
    using Derived = GenericFunction<IR, OR>;
    /// @brief Abstract type-erasure interface backing the stored function.
    using Concept = GFConcept<IR, OR>;

    GFStorage<IR, OR> func_; ///< @brief Shared-ownership storage holding the erased function.
    bool is_linear_ = false; ///< @brief Cached linearity flag of the stored function.

    /// @brief Construct an empty GenericFunction (no stored function).
    GenericFunction() {}

    /// @brief Construct by type-erasing a compatible dense VectorFunction.
    ///
    /// Excludes raw Eigen types (which would otherwise match) so only genuine
    /// VectorFunctions are accepted.
    /// @tparam T  A dense VectorFunction that is not an Eigen expression type.
    /// @param t   Function to wrap and store.
    template <DenseVectorFunction T>
        requires(!std::derived_from<std::decay_t<T>, Eigen::EigenBase<std::decay_t<T>>>)
    GenericFunction(const T &t) {
        func_.emplace(t);
        this->cachedata();
    }

    /// @brief Deep-clone copy constructor from the same GenericFunction type.
    /// @param obj  Source function to copy (must be non-empty).
    GenericFunction(const GenericFunction<IR, OR> &obj) {
        if (!obj.func_.empty()) {
            this->func_ = obj.func_;
            this->cachedata();
        } else {
            throw std::invalid_argument("Attempting to copy null function");
        }
    }

    /// @brief Cross-size copy into a dynamic GenericFunction.
    ///
    /// Only enabled when this type is fully dynamic (IR==-1, OR==-1); absorbs a
    /// source of any fixed size by cloning it into dynamic storage.
    /// @tparam IR1  Source input-row count.
    /// @tparam OR1  Source output-row count.
    /// @param obj   Source function to copy (must be non-empty).
    template <int IR1, int OR1>
        requires(IR == -1 && OR == -1)
    GenericFunction(const GenericFunction<IR1, OR1> &obj) {
        if (!obj.func_.empty()) {
            obj.func_.get().clone_into_dynamic(this->func_);
            cachedata();
        } else {
            throw std::invalid_argument("Attempting to copy null function");
        }
    }

    /// @brief Refresh cached IO sizes, input domain, and linearity from the stored function.
    void cachedata() {
        this->set_io_rows(this->func_.get().input_rows(), this->func_.get().output_rows());
        this->set_input_domain(this->input_rows(), {this->func_.get().input_domain()});
        this->is_linear_ = this->func_.get().is_linear();
    }

    /// @brief Name of the stored function.
    /// @return The wrapped function's name string.
    std::string name() const { return this->func_.get().name(); }
    /// @brief Whether the stored function is linear.
    /// @return True if the wrapped function is linear.
    inline bool is_linear() const { return this->is_linear_; }
    /// @brief Enable or disable SuperScalar (SIMD) vectorization on the stored function.
    /// @param b  True to enable vectorization.
    void enable_vectorization(bool b) const {
        this->func_.get().enable_vectorization(b);
        this->enable_vectorization_ = b;
    }

    // The virtual dispatch on GFConcept requires explicit Eigen::Ref arguments.
    // The _impl methods convert from the caller's Eigen expression type to the
    // concrete Ref type expected by the virtual signature before dispatching.

    /// @internal
    /// @brief Forward value evaluation to the erased function (CRTP compute hook).
    /// @tparam InTypeTT   Input vector expression type.
    /// @tparam OutTypeTT  Output vector expression type.
    /// @param x    Input vector.
    /// @param fx_  Output vector to fill.
    /// @endinternal
    template <class InTypeTT, class OutTypeTT>
    inline void compute_impl(CVecRef<InTypeTT> x, CVecRef<OutTypeTT> fx_) const {
        using Scalar = typename InTypeTT::Scalar;
        if constexpr (std::is_same_v<Scalar, double>) {
            typename Dspec::InType xt(x.derived());
            typename Dspec::OutType fxt(fx_.const_cast_derived());
            this->func_.get().compute(xt, fxt);
        } else {
            typename Dspec::SuperInType xt(x.derived());
            typename Dspec::SuperOutType fxt(fx_.const_cast_derived());
            this->func_.get().compute(xt, fxt);
        }
    }

    /// @internal
    /// @brief Forward value+Jacobian evaluation to the erased function (CRTP hook).
    /// @tparam InTypeTT   Input vector expression type.
    /// @tparam OutTypeTT  Output vector expression type.
    /// @tparam JacTypeTT  Jacobian matrix expression type.
    /// @param x    Input vector.
    /// @param fx_  Output vector to fill.
    /// @param jx_  Jacobian matrix to fill.
    /// @endinternal
    template <class InTypeTT, class OutTypeTT, class JacTypeTT>
    inline void compute_jacobian_impl(CVecRef<InTypeTT> x, CVecRef<OutTypeTT> fx_,
                                      CMatRef<JacTypeTT> jx_) const {
        using Scalar = typename InTypeTT::Scalar;
        if constexpr (std::is_same_v<Scalar, double>) {
            typename Dspec::InType xt(x.derived());
            typename Dspec::OutType fxt(fx_.const_cast_derived());
            typename Dspec::JacType jxt(jx_.const_cast_derived());
            this->func_.get().compute_jacobian(xt, fxt, jxt);
        } else {
            typename Dspec::SuperInType xt(x.derived());
            typename Dspec::SuperOutType fxt(fx_.const_cast_derived());
            typename Dspec::SuperJacType jxt(jx_.const_cast_derived());
            this->func_.get().compute_jacobian(xt, fxt, jxt);
        }
    }

    /// @internal
    /// @brief Forward value+Jacobian+adjoint gradient+Hessian to the erased function.
    /// @tparam InTypeTT       Input vector expression type.
    /// @tparam OutTypeTT      Output vector expression type.
    /// @tparam JacTypeTT      Jacobian matrix expression type.
    /// @tparam AdjGradTypeTT  Adjoint-gradient vector expression type.
    /// @tparam AdjHessTypeTT  Adjoint-Hessian matrix expression type.
    /// @tparam AdjVarTypeTT   Adjoint (multiplier) vector expression type.
    /// @param x        Input vector.
    /// @param fx_      Output vector to fill.
    /// @param jx_      Jacobian matrix to fill.
    /// @param adjgrad_ Adjoint gradient to fill.
    /// @param adjhess_ Adjoint Hessian to fill.
    /// @param adjvars  Adjoint variables (input).
    /// @endinternal
    template <class InTypeTT, class OutTypeTT, class JacTypeTT, class AdjGradTypeTT,
              class AdjHessTypeTT, class AdjVarTypeTT>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        CVecRef<InTypeTT> x, CVecRef<OutTypeTT> fx_, CMatRef<JacTypeTT> jx_,
        CVecRef<AdjGradTypeTT> adjgrad_, CMatRef<AdjHessTypeTT> adjhess_,
        CVecRef<AdjVarTypeTT> adjvars) const {
        using Scalar = typename InTypeTT::Scalar;
        if constexpr (std::is_same_v<Scalar, double>) {
            typename Dspec::InType xt(x.derived());
            typename Dspec::OutType fxt(fx_.const_cast_derived());
            typename Dspec::JacType jxt(jx_.const_cast_derived());
            typename Dspec::AdjGradType adjgradt(adjgrad_.const_cast_derived());
            typename Dspec::AdjHessType adjhesst(adjhess_.const_cast_derived());
            typename Dspec::AdjVarType adjvarst(adjvars.derived());
            this->func_.get().compute_jacobian_adjointgradient_adjointhessian(
                xt, fxt, jxt, adjgradt, adjhesst, adjvarst);
        } else {
            typename Dspec::SuperInType xt(x.derived());
            typename Dspec::SuperOutType fxt(fx_.const_cast_derived());
            typename Dspec::SuperJacType jxt(jx_.const_cast_derived());
            typename Dspec::SuperAdjGradType adjgradt(adjgrad_.const_cast_derived());
            typename Dspec::SuperAdjHessType adjhesst(adjhess_.const_cast_derived());
            typename Dspec::SuperAdjVarType adjvarst(adjvars.derived());
            this->func_.get().compute_jacobian_adjointgradient_adjointhessian(
                xt, fxt, jxt, adjgradt, adjhesst, adjvarst);
        }
    }

    /////////////////////////////////////////////////////////////////////////////////

    /// @internal
    /// @brief Compute `target = left * right` (left times the stored Jacobian).
    ///
    /// Dispatches to the erased function's specialised product when the target,
    /// left, and right types are convertible to the concrete Ref forms; otherwise
    /// falls back to the generic base implementation.
    /// @tparam Target      Target matrix expression type.
    /// @tparam Left        Left-factor expression type.
    /// @tparam Right       Right-factor (Jacobian) expression type.
    /// @tparam Assignment  Assignment policy (direct / plus-equals / minus-equals).
    /// @tparam Aliased     Whether target may alias an operand.
    /// @param target_  Output matrix.
    /// @param left     Left factor.
    /// @param right    Right factor (the Jacobian).
    /// @param assign   Assignment policy instance.
    /// @param aliased  Aliasing flag instance.
    /// @endinternal
    template <class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void right_jacobian_product(CMatRef<Target> target_, CEigRef<Left> left,
                                       CEigRef<Right> right, Assignment assign,
                                       std::bool_constant<Aliased> aliased) const {
        typedef typename Target::Scalar Scalar;

        if constexpr (std::is_same_v<Scalar, double>) {
            constexpr bool TargConv =
                std::is_convertible_v<decltype(target_.const_cast_derived()), RightJacTarget>;
            if constexpr (TargConv) {
                CMatRef<Right> right_ref = right.derived();
                if constexpr (Is_EigenDiagonalMatrix<Left>::value) {
                    Base::right_jacobian_product(target_, left, right, assign, aliased);
                } else {
                    CMatRef<Left> left_ref = left.derived();
                    typename Dspec::RightJacTarget tgt(target_.const_cast_derived());
                    typename Dspec::LeftJacMatrix lft(left_ref.derived());
                    typename Dspec::JacType rht(right_ref.const_cast_derived());
                    this->func_.get().right_jacobian_product(tgt, lft, rht, assign, aliased);
                }
            } else {
                Base::right_jacobian_product(target_, left, right, assign, aliased);
            }
        } else {
            Base::right_jacobian_product(target_, left, right, assign, aliased);
        }
    }

    /// @internal
    /// @brief Accumulate the adjoint Hessian into @p target_ (skipped when linear).
    /// @tparam Target         Target matrix expression type.
    /// @tparam AdjHessTypeTT  Source Hessian expression type.
    /// @tparam Assignment     Assignment policy.
    /// @param target_  Output matrix.
    /// @param right    Source Hessian.
    /// @param assign   Assignment policy instance.
    /// @endinternal
    template <class Target, class AdjHessTypeTT, class Assignment>
    void accumulate_hessian(CMatRef<Target> target_, CMatRef<AdjHessTypeTT> right,
                            Assignment assign) const {
        if (!this->is_linear())
            Base::accumulate_hessian(target_, right, assign);
    }

    /// @internal
    /// @brief Accumulate the Jacobian into @p target_.
    ///
    /// Uses the erased function's fast linear path when the function is linear and
    /// the scalar is double; otherwise defers to the base implementation.
    /// @tparam Target      Target matrix expression type.
    /// @tparam JacTypeTT   Source Jacobian expression type.
    /// @tparam Assignment  Assignment policy.
    /// @param target_  Output matrix.
    /// @param right    Source Jacobian.
    /// @param assign   Assignment policy instance.
    /// @endinternal
    template <class Target, class JacTypeTT, class Assignment>
    void accumulate_jacobian(CMatRef<Target> target_, CMatRef<JacTypeTT> right,
                             Assignment assign) const {
        typedef typename Target::Scalar Scalar;

        if constexpr (std::is_same_v<Scalar, double>) {
            if (this->is_linear()) {
                typename Dspec::JacType jtarg(target_.const_cast_derived());
                typename Dspec::JacType jright(right.const_cast_derived());
                this->func_.get().accumulate_jacobian(jtarg, jright, assign);
            } else {
                Base::accumulate_jacobian(target_, right, assign);
            }
        } else {
            Base::accumulate_jacobian(target_, right, assign);
        }
    }

    ////////////////////////////////////////////////////////////////////////////////
    /// @internal
    /// @brief Scale the Jacobian in @p target_ by scalar @p s.
    /// @tparam JacTypeTT  Jacobian expression type.
    /// @tparam Scalar     Scalar type of @p s.
    /// @param target_  Jacobian to scale in place.
    /// @param s        Scale factor.
    /// @endinternal
    template <class JacTypeTT, class Scalar>
    void scale_jacobian(CMatRef<JacTypeTT> target_, Scalar s) const {
        if constexpr (std::is_same_v<Scalar, tycho::DefaultSuperScalar>) {
            Base::scale_jacobian(target_, s);
        } else {
            typename Dspec::JacType jxt(target_.const_cast_derived());
            this->func_.get().scale_jacobian(jxt, s);
        }
    }

    /// @internal
    /// @brief Scale the adjoint Hessian in @p target_ by scalar @p s (skipped when linear).
    /// @tparam AdjHessTypeTT  Hessian expression type.
    /// @tparam Scalar         Scalar type of @p s.
    /// @param target_  Hessian to scale in place.
    /// @param s        Scale factor.
    /// @endinternal
    template <class AdjHessTypeTT, class Scalar>
    void scale_hessian(CMatRef<AdjHessTypeTT> target_, Scalar s) const {
        if (!this->is_linear())
            Base::scale_hessian(target_, s);
    }
    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////
    /// @brief Construct a GenericFunction copy of @p obj (Python-binding helper).
    /// @tparam T  Source function type.
    /// @param obj  Function to copy.
    /// @return A new GenericFunction wrapping a copy of @p obj.
    template <class T> static GenericFunction<IR, OR> PyCopy(const T &obj) {
        return GenericFunction<IR, OR>(obj);
    }
};

} // namespace tycho::vf
