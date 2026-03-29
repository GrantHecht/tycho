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

template <int IR, int OR> struct GenericFunction : VectorFunction<GenericFunction<IR, OR>, IR, OR> {
    using Base = VectorFunction<GenericFunction<IR, OR>, IR, OR>;
    DENSE_FUNCTION_BASE_TYPES(Base);

    static const bool is_generic_function = true;
    static const bool is_vectorizable = true;

    using Dspec = DenseFunctionSpec<IR, OR>;
    using RightJacTarget = Eigen::Ref<Eigen::Matrix<double, -1, IR>>;
    using Derived = GenericFunction<IR, OR>;
    using Concept = GFConcept<IR, OR>;

    GFStorage<IR, OR> func;
    bool islinear = false;

    GenericFunction() {}

    template <class T, std::enable_if_t<
                           !std::is_base_of_v<Eigen::EigenBase<std::decay_t<T>>, std::decay_t<T>>,
                           bool> = true>
    GenericFunction(const T &t) {
        func.emplace(t);
        this->cachedata();
    }

    // Same-type copy: deep-clone via GFStorage copy semantics
    GenericFunction(const GenericFunction<IR, OR> &obj) {
        if (!obj.func.empty()) {
            this->func = obj.func;
            this->cachedata();
        } else {
            throw std::invalid_argument("Attempting to copy null function");
        }
    }

    // Cross-type copy: only supported when the target is dynamic (IR==-1, OR==-1)
    template <int IR1, int OR1>
        requires(IR == -1 && OR == -1)
    GenericFunction(const GenericFunction<IR1, OR1> &obj) {
        if (!obj.func.empty()) {
            obj.func.get().clone_into_dynamic(this->func);
            cachedata();
        } else {
            throw std::invalid_argument("Attempting to copy null function");
        }
    }

    void cachedata() {
        this->set_io_rows(this->func.get().input_rows(), this->func.get().output_rows());
        this->set_input_domain(this->input_rows(), {this->func.get().input_domain()});
        this->islinear = this->func.get().is_linear();
    }

    std::string name() const { return this->func.get().name(); }
    inline bool is_linear() const { return this->islinear; }
    void enable_vectorization(bool b) const {
        this->func.get().enable_vectorization(b);
        this->enable_vectorization_ = b;
    }

    // The virtual dispatch on GFConcept requires explicit Eigen::Ref arguments.
    // The _impl methods convert from the caller's Eigen expression type to the
    // concrete Ref type expected by the virtual signature before dispatching.

    template <class InTypeTT, class OutTypeTT>
    inline void compute_impl(ConstVectorBaseRef<InTypeTT> x,
                             ConstVectorBaseRef<OutTypeTT> fx_) const {
        using Scalar = typename InTypeTT::Scalar;
        if constexpr (std::is_same_v<Scalar, double>) {
            typename Dspec::InType xt(x.derived());
            typename Dspec::OutType fxt(fx_.const_cast_derived());
            this->func.get().compute(xt, fxt);
        } else {
            typename Dspec::SuperInType xt(x.derived());
            typename Dspec::SuperOutType fxt(fx_.const_cast_derived());
            this->func.get().compute(xt, fxt);
        }
    }

    template <class InTypeTT, class OutTypeTT, class JacTypeTT>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InTypeTT> x,
                                      ConstVectorBaseRef<OutTypeTT> fx_,
                                      ConstMatrixBaseRef<JacTypeTT> jx_) const {
        using Scalar = typename InTypeTT::Scalar;
        if constexpr (std::is_same_v<Scalar, double>) {
            typename Dspec::InType xt(x.derived());
            typename Dspec::OutType fxt(fx_.const_cast_derived());
            typename Dspec::JacType jxt(jx_.const_cast_derived());
            this->func.get().compute_jacobian(xt, fxt, jxt);
        } else {
            typename Dspec::SuperInType xt(x.derived());
            typename Dspec::SuperOutType fxt(fx_.const_cast_derived());
            typename Dspec::SuperJacType jxt(jx_.const_cast_derived());
            this->func.get().compute_jacobian(xt, fxt, jxt);
        }
    }

    template <class InTypeTT, class OutTypeTT, class JacTypeTT, class AdjGradTypeTT,
              class AdjHessTypeTT, class AdjVarTypeTT>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        ConstVectorBaseRef<InTypeTT> x, ConstVectorBaseRef<OutTypeTT> fx_,
        ConstMatrixBaseRef<JacTypeTT> jx_, ConstVectorBaseRef<AdjGradTypeTT> adjgrad_,
        ConstMatrixBaseRef<AdjHessTypeTT> adjhess_,
        ConstVectorBaseRef<AdjVarTypeTT> adjvars) const {
        using Scalar = typename InTypeTT::Scalar;
        if constexpr (std::is_same_v<Scalar, double>) {
            typename Dspec::InType xt(x.derived());
            typename Dspec::OutType fxt(fx_.const_cast_derived());
            typename Dspec::JacType jxt(jx_.const_cast_derived());
            typename Dspec::AdjGradType adjgradt(adjgrad_.const_cast_derived());
            typename Dspec::AdjHessType adjhesst(adjhess_.const_cast_derived());
            typename Dspec::AdjVarType adjvarst(adjvars.derived());
            this->func.get().compute_jacobian_adjointgradient_adjointhessian(xt, fxt, jxt, adjgradt,
                                                                             adjhesst, adjvarst);
        } else {
            typename Dspec::SuperInType xt(x.derived());
            typename Dspec::SuperOutType fxt(fx_.const_cast_derived());
            typename Dspec::SuperJacType jxt(jx_.const_cast_derived());
            typename Dspec::SuperAdjGradType adjgradt(adjgrad_.const_cast_derived());
            typename Dspec::SuperAdjHessType adjhesst(adjhess_.const_cast_derived());
            typename Dspec::SuperAdjVarType adjvarst(adjvars.derived());
            this->func.get().compute_jacobian_adjointgradient_adjointhessian(xt, fxt, jxt, adjgradt,
                                                                             adjhesst, adjvarst);
        }
    }

    /////////////////////////////////////////////////////////////////////////////////

    template <class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void right_jacobian_product(ConstMatrixBaseRef<Target> target_,
                                       ConstEigenBaseRef<Left> left, ConstEigenBaseRef<Right> right,
                                       Assignment assign,
                                       std::bool_constant<Aliased> aliased) const {
        typedef typename Target::Scalar Scalar;

        if constexpr (std::is_same_v<Scalar, double>) {
            constexpr bool TargConv =
                std::is_convertible_v<decltype(target_.const_cast_derived()), RightJacTarget>;
            if constexpr (TargConv) {
                ConstMatrixBaseRef<Right> right_ref = right.derived();
                if constexpr (Is_EigenDiagonalMatrix<Left>::value) {
                    Base::right_jacobian_product(target_, left, right, assign, aliased);
                } else {
                    ConstMatrixBaseRef<Left> left_ref = left.derived();
                    typename Dspec::RightJacTarget tgt(target_.const_cast_derived());
                    typename Dspec::LeftJacMatrix lft(left_ref.derived());
                    typename Dspec::JacType rht(right_ref.const_cast_derived());
                    this->func.get().right_jacobian_product(tgt, lft, rht, assign, aliased);
                }
            } else {
                Base::right_jacobian_product(target_, left, right, assign, aliased);
            }
        } else {
            Base::right_jacobian_product(target_, left, right, assign, aliased);
        }
    }

    template <class Target, class AdjHessTypeTT, class Assignment>
    void accumulate_hessian(ConstMatrixBaseRef<Target> target_,
                            ConstMatrixBaseRef<AdjHessTypeTT> right, Assignment assign) const {
        if (!this->is_linear())
            Base::accumulate_hessian(target_, right, assign);
    }

    template <class Target, class JacTypeTT, class Assignment>
    void accumulate_jacobian(ConstMatrixBaseRef<Target> target_,
                             ConstMatrixBaseRef<JacTypeTT> right, Assignment assign) const {
        typedef typename Target::Scalar Scalar;

        if constexpr (std::is_same_v<Scalar, double>) {
            if (this->is_linear()) {
                typename Dspec::JacType jtarg(target_.const_cast_derived());
                typename Dspec::JacType jright(right.const_cast_derived());
                this->func.get().accumulate_jacobian(jtarg, jright, assign);
            } else {
                Base::accumulate_jacobian(target_, right, assign);
            }
        } else {
            Base::accumulate_jacobian(target_, right, assign);
        }
    }

    ////////////////////////////////////////////////////////////////////////////////
    template <class JacTypeTT, class Scalar>
    void scale_jacobian(ConstMatrixBaseRef<JacTypeTT> target_, Scalar s) const {
        if constexpr (std::is_same_v<Scalar, tycho::DefaultSuperScalar>) {
            Base::scale_jacobian(target_, s);
        } else {
            typename Dspec::JacType jxt(target_.const_cast_derived());
            this->func.get().scale_jacobian(jxt, s);
        }
    }

    template <class AdjHessTypeTT, class Scalar>
    void scale_hessian(ConstMatrixBaseRef<AdjHessTypeTT> target_, Scalar s) const {
        if (!this->is_linear())
            Base::scale_hessian(target_, s);
    }
    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////
    template <class T> static GenericFunction<IR, OR> PyCopy(const T &obj) {
        return GenericFunction<IR, OR>(obj);
    }
};

} // namespace tycho::vf
