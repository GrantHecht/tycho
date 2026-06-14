// =============================================================================
// New file in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt)
//
// Replaces the rubber_types::TypeErasure composition used for GenericFunction's
// internal type erasure (GFTE) with a purpose-built, flat implementation.
//
// Design goals:
//   1. Flat vtable — GFConcept inherits Spec::Concepts non-virtually, so there
//      are no vbase offset tables or extra indirection per vtable lookup.
//   2. Shared ownership — GFStorage uses shared_ptr for O(1) copy (reference
//      count increment), matching the old rubber_types performance model.
//   3. Single virtual dispatch for solver calls — ConstraintInterface and
//      ObjectiveInterface store T directly via pack_into, not the GFTE wrapper.
//      PR 9: pack_into methods now call ci.storage_.emplace / oi.storage_.emplace
//      directly, bypassing any ConstraintInterface/ObjectiveInterface constructor
//      overhead.
// =============================================================================

#pragma once

#include "tycho/detail/solvers/sizing_specs.h"
#include "tycho/detail/solvers/solver_interface_specs.h"
#include "tycho/detail/vf/core/dense_function_specs.h"

#include <concepts>
#include <memory>
#include <type_traits>
#include <utility>

namespace tycho::vf {

// Import solver types used by GFConcept / GFModel.
using solvers::ConstraintBase;
using solvers::ConstraintInterface;
using solvers::ConstraintModel;
using solvers::ObjectiveBase;
using solvers::ObjectiveInterface;
using solvers::ObjectiveModel;
using solvers::SizableSpec;
using solvers::SolverConstraintSpec;
using solvers::SolverObjectiveSpec;

/// @internal
/// @brief Forward declaration of the shared-ownership storage container.
/// @tparam IR  Input-row count.
/// @tparam OR  Output-row count.
/// @endinternal
template <int IR, int OR> class GFStorage;

// ==========================================================================
// Section 1: GFHolder<T> — owns one T, nothing else
// ==========================================================================

/// @internal
/// @brief Trivial owner of a single value @p T used as a base of GFModel.
/// @tparam T  Owned value type.
/// @endinternal
template <class T> struct GFHolder {
    T data_; ///< @internal Owned value. @endinternal
    /// @internal
    /// @brief Construct by moving @p t into the holder.
    /// @param t  Value to own.
    /// @endinternal
    explicit GFHolder(T t) : data_(std::move(t)) {}
};

// ==========================================================================
// Section 2: GFStorable concept — constrains what can be stored in GFStorage
//
// Checks the most common interface requirements at the call site before
// full GFModel instantiation, giving cleaner compiler diagnostics.
// ==========================================================================

/// @internal
/// @brief Concept constraining types storable in GFStorage.
///
/// Checks the most common interface requirements (sizing, linearity, domain,
/// name, vectorizability) at the call site, before the full GFModel is
/// instantiated, for cleaner compiler diagnostics on bad types.
/// @tparam T   Candidate stored type.
/// @tparam IR  Input-row count.
/// @tparam OR  Output-row count.
/// @endinternal
template <typename T, int IR, int OR>
concept GFStorable = requires(const T &t) {
    { t.input_rows() } -> std::same_as<int>;
    { t.output_rows() } -> std::same_as<int>;
    { t.is_linear() } -> std::same_as<bool>;
    { t.input_domain() };
    { t.name() } -> std::same_as<std::string>;
    { T::is_vectorizable } -> std::convertible_to<bool>;
};

// ==========================================================================
// Section 3: GFConcept<IR, OR> — flat abstract class via non-virtual MI
//
// GFConcept inherits the Spec::Concept classes *non-virtually*. They share
// no common base, so the inheritance is diamond-free and produces a single
// flat vtable with no vbase offset overhead.
//
// Three new virtuals cover copy and solver-interface pack operations that
// have no counterpart in the existing Spec files.
// ==========================================================================

/// @internal
/// @brief Flat abstract interface backing GenericFunction (non-objective, OR != 1).
///
/// Inherits the per-Spec Concept classes non-virtually so the combined vtable is
/// diamond-free and carries no vbase-offset overhead. Adds the copy and
/// solver-pack virtuals that no existing Spec provides.
/// @tparam IR  Input-row count.
/// @tparam OR  Output-row count.
/// @endinternal
// Primary template: constraint functions (OR != 1)
template <int IR, int OR>
struct GFConcept : DenseFunctionSpec<IR, OR>::Concept, // non-virtual — no diamond
                   SizableSpec::Concept,
                   SolverConstraintSpec::Concept {

    /// @internal @brief Virtual destructor. @endinternal
    virtual ~GFConcept() = default;

    /// @internal @brief Deep-clone into same-size storage. @param s Destination. @endinternal
    virtual void clone_into(GFStorage<IR, OR> &s) const = 0;
    /// @internal @brief Deep-clone into dynamic storage. @param s Destination. @endinternal
    virtual void clone_into_dynamic(GFStorage<-1, -1> &s) const = 0;

    /// @internal
    /// @brief Pack the stored function directly into a solver constraint interface.
    /// @param ci  Constraint interface to populate.
    /// @endinternal
    virtual void pack_into_constraint_interface(ConstraintInterface &ci) const = 0;
};

/// @internal
/// @brief Objective-capable specialisation of GFConcept for scalar output (OR == 1).
///
/// Additionally inherits SolverObjectiveSpec and adds the objective pack virtual.
/// @tparam IR  Input-row count.
/// @endinternal
// Partial specialisation for OR==1: additionally inherits SolverObjectiveSpec
template <int IR>
struct GFConcept<IR, 1> : DenseFunctionSpec<IR, 1>::Concept,
                          SizableSpec::Concept,
                          SolverConstraintSpec::Concept,
                          SolverObjectiveSpec::Concept {

    /// @internal @brief Virtual destructor. @endinternal
    virtual ~GFConcept() = default;

    /// @internal @brief Deep-clone into same-size storage. @param s Destination. @endinternal
    virtual void clone_into(GFStorage<IR, 1> &s) const = 0;
    /// @internal @brief Deep-clone into dynamic storage. @param s Destination. @endinternal
    virtual void clone_into_dynamic(GFStorage<-1, -1> &s) const = 0;
    /// @internal @brief Pack into a solver constraint interface. @param ci Interface. @endinternal
    virtual void pack_into_constraint_interface(ConstraintInterface &ci) const = 0;
    /// @internal @brief Pack into a solver objective interface. @param oi Interface. @endinternal
    virtual void pack_into_objective_interface(ObjectiveInterface &oi) const = 0;
};

// ==========================================================================
// Section 4: GFModelCommon<IR, OR, T> — implements all overrides except
//            pack_into_objective_interface (added by GFModel<IR,1,T>)
//
// Override bodies are grouped with section comments matching the source Spec
// files. If a new pure virtual is added to any Spec, the compiler fails to
// instantiate GFModelCommon (abstract class), pointing to the right section.
// ==========================================================================

/// @internal
/// @brief Concrete GFConcept implementation forwarding every virtual to a stored @p T.
///
/// Implements all overrides except pack_into_objective_interface (added by the
/// objective-capable GFModel<IR,1,T> specialisation). Override bodies are grouped
/// by their originating Spec; adding a new pure virtual to any Spec makes this
/// class fail to instantiate, pointing at the missing section.
/// @tparam IR  Input-row count.
/// @tparam OR  Output-row count.
/// @tparam T   Stored function type (must satisfy GFStorable).
/// @endinternal
template <int IR, int OR, GFStorable<IR, OR> T>
struct GFModelCommon : GFHolder<T>, GFConcept<IR, OR> {
    using GFHolder<T>::GFHolder;

    // Convenient type aliases (mirrors DenseFunctionSpec<IR,OR> outer struct)
    using Dspec = DenseFunctionSpec<IR, OR>; ///< @internal Dense function spec. @endinternal
    using InType = typename Dspec::InType;   ///< @internal Scalar input type. @endinternal
    using OutType = typename Dspec::OutType; ///< @internal Scalar output type. @endinternal
    using JacType = typename Dspec::JacType; ///< @internal Scalar Jacobian type. @endinternal
    using AdjGradType =
        typename Dspec::AdjGradType; ///< @internal Scalar adjoint-gradient type. @endinternal
    using AdjVarType =
        typename Dspec::AdjVarType; ///< @internal Scalar adjoint-variable type. @endinternal
    using AdjHessType =
        typename Dspec::AdjHessType; ///< @internal Scalar adjoint-Hessian type. @endinternal
    using SuperInType =
        typename Dspec::SuperInType; ///< @internal SuperScalar input type. @endinternal
    using SuperOutType =
        typename Dspec::SuperOutType; ///< @internal SuperScalar output type. @endinternal
    using SuperJacType =
        typename Dspec::SuperJacType; ///< @internal SuperScalar Jacobian type. @endinternal
    using SuperAdjGradType =
        typename Dspec::SuperAdjGradType; ///< @internal SuperScalar adjoint-gradient type.
                                          ///< @endinternal
    using SuperAdjVarType =
        typename Dspec::SuperAdjVarType; ///< @internal SuperScalar adjoint-variable type.
                                         ///< @endinternal
    using SuperAdjHessType =
        typename Dspec::SuperAdjHessType; ///< @internal SuperScalar adjoint-Hessian type.
                                          ///< @endinternal
    using RightJacTarget = typename Dspec::RightJacTarget; ///< @internal Right-Jacobian-product
                                                           ///< target type. @endinternal
    using LeftJacMatrix =
        typename Dspec::LeftJacMatrix; ///< @internal Left-factor matrix type. @endinternal

    /// @internal @brief Const MatrixBase ref alias (CVecRef). @tparam S Expression type.
    /// @endinternal
    template <class S> using CVR = const Eigen::MatrixBase<S> &; // CVecRef
    /// @internal @brief Const MatrixBase ref alias (CMatRef). @tparam S Expression type.
    /// @endinternal
    template <class S> using CMR = const Eigen::MatrixBase<S> &; // CMatRef
    /// @internal @brief Const EigenBase ref alias (CEigRef). @tparam S Expression type.
    /// @endinternal
    template <class S> using CER = const Eigen::EigenBase<S> &; // CEigRef

    // ---- DenseFunctionSpec overrides (see DenseFunctionSpecs.h) ----

    /// @internal @brief Forward input_domain() to the stored function. @return Domain matrix.
    /// @endinternal
    DomainMatrix input_domain() const override { return this->data_.input_domain(); }
    /// @internal @brief Forward is_linear() to the stored function. @return Linearity flag.
    /// @endinternal
    bool is_linear() const override { return this->data_.is_linear(); }
    /// @internal @brief Forward enable_vectorization() to the stored function. @param b Enable
    /// flag. @endinternal
    void enable_vectorization(bool b) const override { this->data_.enable_vectorization(b); }

    /// @internal @brief Forward scalar value evaluation to the stored function. @param x Input.
    /// @param fx_ Output. @endinternal
    void compute(CVR<InType> x, CVR<OutType> fx_) const override { this->data_.compute(x, fx_); }
    /// @internal @brief Forward SuperScalar value evaluation to the stored function. @param x
    /// Input. @param fx_ Output. @endinternal
    void compute(CVR<SuperInType> x, CVR<SuperOutType> fx_) const override {
        this->data_.compute(x, fx_);
    }

    /// @internal @brief Forward scalar value+Jacobian to the stored function. @param x Input.
    /// @param fx_ Output. @param jx_ Jacobian. @endinternal
    void compute_jacobian(CVR<InType> x, CVR<OutType> fx_, CMR<JacType> jx_) const override {
        this->data_.compute_jacobian(x, fx_, jx_);
    }
    /// @internal @brief Forward SuperScalar value+Jacobian to the stored function. @param x Input.
    /// @param fx_ Output. @param jx_ Jacobian. @endinternal
    void compute_jacobian(CVR<SuperInType> x, CVR<SuperOutType> fx_,
                          CMR<SuperJacType> jx_) const override {
        this->data_.compute_jacobian(x, fx_, jx_);
    }

    /// @internal
    /// @brief Forward the full scalar value/Jacobian/adjoint-gradient/Hessian evaluation.
    /// @param x Input. @param fx_ Output. @param jx_ Jacobian. @param adjgrad_ Adjoint gradient.
    /// @param adjhess_ Adjoint Hessian. @param adjvars Adjoint variables.
    /// @endinternal
    void compute_jacobian_adjointgradient_adjointhessian(CVR<InType> x, CVR<OutType> fx_,
                                                         CMR<JacType> jx_,
                                                         CVR<AdjGradType> adjgrad_,
                                                         CMR<AdjHessType> adjhess_,
                                                         CVR<AdjVarType> adjvars) const override {
        this->data_.compute_jacobian_adjointgradient_adjointhessian(x, fx_, jx_, adjgrad_, adjhess_,
                                                                    adjvars);
    }
    /// @internal
    /// @brief Forward the full SuperScalar value/Jacobian/adjoint-gradient/Hessian evaluation.
    /// @param x Input. @param fx_ Output. @param jx_ Jacobian. @param adjgrad_ Adjoint gradient.
    /// @param adjhess_ Adjoint Hessian. @param adjvars Adjoint variables.
    /// @endinternal
    void compute_jacobian_adjointgradient_adjointhessian(
        CVR<SuperInType> x, CVR<SuperOutType> fx_, CMR<SuperJacType> jx_,
        CVR<SuperAdjGradType> adjgrad_, CMR<SuperAdjHessType> adjhess_,
        CVR<SuperAdjVarType> adjvars) const override {
        this->data_.compute_jacobian_adjointgradient_adjointhessian(x, fx_, jx_, adjgrad_, adjhess_,
                                                                    adjvars);
    }

    /// @internal @brief Forward Jacobian scaling to the stored function. @param target_ Jacobian.
    /// @param s Scale. @endinternal
    void scale_jacobian(CMR<JacType> target_, double s) const override {
        this->data_.scale_jacobian(target_, s);
    }

    /// @internal @brief Forward Jacobian accumulation (direct assign). @param target_ Target.
    /// @param right Source. @param assign Policy. @endinternal
    void accumulate_jacobian(CMR<JacType> target_, CMR<JacType> right,
                             DirectAssignment assign) const override {
        this->data_.accumulate_jacobian(target_, right, assign);
    }
    /// @internal @brief Forward Jacobian accumulation (plus-equals). @param target_ Target. @param
    /// right Source. @param assign Policy. @endinternal
    void accumulate_jacobian(CMR<JacType> target_, CMR<JacType> right,
                             PlusEqualsAssignment assign) const override {
        this->data_.accumulate_jacobian(target_, right, assign);
    }
    /// @internal @brief Forward Jacobian accumulation (minus-equals). @param target_ Target. @param
    /// right Source. @param assign Policy. @endinternal
    void accumulate_jacobian(CMR<JacType> target_, CMR<JacType> right,
                             MinusEqualsAssignment assign) const override {
        this->data_.accumulate_jacobian(target_, right, assign);
    }

    /// @internal @brief Forward right-Jacobian product (direct, aliased). @param target_ Target.
    /// @param left Left. @param right Jacobian. @param assign Policy. @param aliased Aliasing flag.
    /// @endinternal
    void right_jacobian_product(CMR<RightJacTarget> target_, CER<LeftJacMatrix> left,
                                CER<JacType> right, DirectAssignment assign,
                                std::bool_constant<true> aliased) const override {
        this->data_.right_jacobian_product(target_, left, right, assign, aliased);
    }
    /// @internal @brief Forward right-Jacobian product (plus-equals, aliased). @param target_
    /// Target. @param left Left. @param right Jacobian. @param assign Policy. @param aliased
    /// Aliasing flag. @endinternal
    void right_jacobian_product(CMR<RightJacTarget> target_, CER<LeftJacMatrix> left,
                                CER<JacType> right, PlusEqualsAssignment assign,
                                std::bool_constant<true> aliased) const override {
        this->data_.right_jacobian_product(target_, left, right, assign, aliased);
    }
    /// @internal @brief Forward right-Jacobian product (minus-equals, aliased). @param target_
    /// Target. @param left Left. @param right Jacobian. @param assign Policy. @param aliased
    /// Aliasing flag. @endinternal
    void right_jacobian_product(CMR<RightJacTarget> target_, CER<LeftJacMatrix> left,
                                CER<JacType> right, MinusEqualsAssignment assign,
                                std::bool_constant<true> aliased) const override {
        this->data_.right_jacobian_product(target_, left, right, assign, aliased);
    }
    /// @internal @brief Forward right-Jacobian product (direct, non-aliased). @param target_
    /// Target. @param left Left. @param right Jacobian. @param assign Policy. @param aliased
    /// Aliasing flag. @endinternal
    void right_jacobian_product(CMR<RightJacTarget> target_, CER<LeftJacMatrix> left,
                                CER<JacType> right, DirectAssignment assign,
                                std::bool_constant<false> aliased) const override {
        this->data_.right_jacobian_product(target_, left, right, assign, aliased);
    }
    /// @internal @brief Forward right-Jacobian product (plus-equals, non-aliased). @param target_
    /// Target. @param left Left. @param right Jacobian. @param assign Policy. @param aliased
    /// Aliasing flag. @endinternal
    void right_jacobian_product(CMR<RightJacTarget> target_, CER<LeftJacMatrix> left,
                                CER<JacType> right, PlusEqualsAssignment assign,
                                std::bool_constant<false> aliased) const override {
        this->data_.right_jacobian_product(target_, left, right, assign, aliased);
    }
    /// @internal @brief Forward right-Jacobian product (minus-equals, non-aliased). @param target_
    /// Target. @param left Left. @param right Jacobian. @param assign Policy. @param aliased
    /// Aliasing flag. @endinternal
    void right_jacobian_product(CMR<RightJacTarget> target_, CER<LeftJacMatrix> left,
                                CER<JacType> right, MinusEqualsAssignment assign,
                                std::bool_constant<false> aliased) const override {
        this->data_.right_jacobian_product(target_, left, right, assign, aliased);
    }

    // ---- SizableSpec overrides (see SizingSpecs.h) ----

    /// @internal @brief Forward name() to the stored function. @return Name. @endinternal
    std::string name() const override { return this->data_.name(); }
    /// @internal @brief Forward input_rows() to the stored function. @return Input rows.
    /// @endinternal
    int input_rows() const override { return this->data_.input_rows(); }
    /// @internal @brief Forward output_rows() to the stored function. @return Output rows.
    /// @endinternal
    int output_rows() const override { return this->data_.output_rows(); }
    /// @internal @brief Forward thread_safe() to the stored function. @return Thread-safety flag.
    /// @endinternal
    bool thread_safe() const override { return this->data_.thread_safe(); }

    // ---- SolverConstraintSpec overrides (see SolverInterfaceSpecs.h) ----

    /// @internal
    /// @brief Forward constraint evaluation to the stored function.
    /// @param X Decision vector. @param FX Constraint output. @param data Solver indexing data.
    /// @endinternal
    void constraints(const Eigen::Ref<const Eigen::VectorXd> &X, Eigen::Ref<Eigen::VectorXd> FX,
                     const tycho::solvers::SolverIndexingData &data) const override {
        this->data_.constraints(X, FX, data);
    }
    /// @internal
    /// @brief Forward constraint value + adjoint gradient evaluation.
    /// @param X Decision vector. @param L Multipliers. @param FX Constraint output.
    /// @param AGX Adjoint gradient output. @param data Solver indexing data.
    /// @endinternal
    void
    constraints_adjointgradient(const Eigen::Ref<const Eigen::VectorXd> &X,
                                const Eigen::Ref<const Eigen::VectorXd> &L,
                                Eigen::Ref<Eigen::VectorXd> FX, Eigen::Ref<Eigen::VectorXd> AGX,
                                const tycho::solvers::SolverIndexingData &data) const override {
        this->data_.constraints_adjointgradient(X, L, FX, AGX, data);
    }
    /// @internal
    /// @brief Forward constraint value + sparse Jacobian assembly into the KKT matrix.
    /// @param X Decision vector. @param FX Constraint output. @param KKTmat KKT matrix.
    /// @param KKTLocations Nonzero locations. @param KKTClashes Clash counts.
    /// @param KKTLocks Per-block mutexes. @param data Solver indexing data.
    /// @endinternal
    void constraints_jacobian(const Eigen::Ref<const Eigen::VectorXd> &X,
                              Eigen::Ref<Eigen::VectorXd> FX,
                              Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                              Eigen::Ref<Eigen::VectorXi> KKTLocations,
                              Eigen::Ref<Eigen::VectorXi> KKTClashes,
                              std::vector<std::mutex> &KKTLocks,
                              const tycho::solvers::SolverIndexingData &data) const override {
        this->data_.constraints_jacobian(X, FX, KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
    }
    /// @internal
    /// @brief Forward constraint value + adjoint gradient + sparse Jacobian assembly.
    /// @param X Decision vector. @param L Multipliers. @param FX Constraint output.
    /// @param AGX Adjoint gradient output. @param KKTmat KKT matrix.
    /// @param KKTLocations Nonzero locations. @param KKTClashes Clash counts.
    /// @param KKTLocks Per-block mutexes. @param data Solver indexing data.
    /// @endinternal
    void constraints_jacobian_adjointgradient(
        const Eigen::Ref<const Eigen::VectorXd> &X, const Eigen::Ref<const Eigen::VectorXd> &L,
        Eigen::Ref<Eigen::VectorXd> FX, Eigen::Ref<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
        Eigen::Ref<Eigen::VectorXi> KKTLocations, Eigen::Ref<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex> &KKTLocks,
        const tycho::solvers::SolverIndexingData &data) const override {
        this->data_.constraints_jacobian_adjointgradient(X, L, FX, AGX, KKTmat, KKTLocations,
                                                         KKTClashes, KKTLocks, data);
    }
    /// @internal
    /// @brief Forward constraint value + adjoint gradient + Jacobian + Hessian assembly.
    /// @param X Decision vector. @param L Multipliers. @param FX Constraint output.
    /// @param AGX Adjoint gradient output. @param KKTmat KKT matrix.
    /// @param KKTLocations Nonzero locations. @param KKTClashes Clash counts.
    /// @param KKTLocks Per-block mutexes. @param data Solver indexing data.
    /// @endinternal
    void constraints_jacobian_adjointgradient_adjointhessian(
        const Eigen::Ref<const Eigen::VectorXd> &X, const Eigen::Ref<const Eigen::VectorXd> &L,
        Eigen::Ref<Eigen::VectorXd> FX, Eigen::Ref<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
        Eigen::Ref<Eigen::VectorXi> KKTLocations, Eigen::Ref<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex> &KKTLocks,
        const tycho::solvers::SolverIndexingData &data) const override {
        this->data_.constraints_jacobian_adjointgradient_adjointhessian(
            X, L, FX, AGX, KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
    }
    /// @internal
    /// @brief Forward KKT sparsity-space reservation to the stored function.
    /// @param KKTrows Row indices output. @param KKTcols Column indices output.
    /// @param freeloc Running free-slot cursor. @param conoffset Constraint row offset.
    /// @param dojac Reserve Jacobian space. @param dohess Reserve Hessian space.
    /// @param data Solver indexing data.
    /// @endinternal
    void get_kkt_space(Eigen::Ref<Eigen::VectorXi> KKTrows, Eigen::Ref<Eigen::VectorXi> KKTcols,
                       int &freeloc, int conoffset, bool dojac, bool dohess,
                       tycho::solvers::SolverIndexingData &data) override {
        this->data_.get_kkt_space(KKTrows, KKTcols, freeloc, conoffset, dojac, dohess, data);
    }
    /// @internal
    /// @brief Forward the KKT nonzero count query to the stored function.
    /// @param dojac Count Jacobian entries. @param dohess Count Hessian entries.
    /// @return Number of KKT nonzero elements.
    /// @endinternal
    int num_kkt_elements(bool dojac, bool dohess) const override {
        return this->data_.num_kkt_elements(dojac, dohess);
    }

    // ---- GFConcept: copy and pack operations ----

    // Bodies defined out-of-line after GFStorage (emplace() must be visible)
    /// @internal @brief Deep-clone the stored function into same-size storage. @param s
    /// Destination. @endinternal
    void clone_into(GFStorage<IR, OR> &s) const override;
    /// @internal @brief Deep-clone the stored function into dynamic storage. @param s Destination.
    /// @endinternal
    void clone_into_dynamic(GFStorage<-1, -1> &s) const override;
    /// @internal @brief Store the function directly into a solver constraint interface. @param ci
    /// Interface. @endinternal
    void pack_into_constraint_interface(ConstraintInterface &ci) const override {
        ci.storage_.emplace<ConstraintModel<T>>(this->data_);
    }
};

// ==========================================================================
// GFModel<IR, OR, T> — concrete final type, inherits GFModelCommon
//
// Primary template covers OR != 1. GFModel<IR,1,T> partial specialisation
// adds pack_into_objective_interface required by GFConcept<IR,1>.
//
// sizeof(GFModel<IR,OR,T>) = sizeof(vptr) + sizeof(T). No extra wrappers.
// ==========================================================================

/// @internal
/// @brief Final concrete model for non-objective functions (OR != 1).
/// @tparam IR  Input-row count.
/// @tparam OR  Output-row count.
/// @tparam T   Stored function type.
/// @endinternal
// Primary template (non-objective)
template <int IR, int OR, GFStorable<IR, OR> T> struct GFModel final : GFModelCommon<IR, OR, T> {
    using GFModelCommon<IR, OR, T>::GFModelCommon;
};

/// @internal
/// @brief Final concrete model for scalar-output (objective-capable) functions.
///
/// Adds the SolverObjectiveSpec overrides and the objective pack operation
/// required by GFConcept<IR,1>.
/// @tparam IR  Input-row count.
/// @tparam T   Stored function type.
/// @endinternal
// Partial specialisation for scalar output (objective-capable)
template <int IR, GFStorable<IR, 1> T> struct GFModel<IR, 1, T> final : GFModelCommon<IR, 1, T> {
    using GFModelCommon<IR, 1, T>::GFModelCommon;

    // ---- SolverObjectiveSpec overrides (see SolverInterfaceSpecs.h) ----

    /// @internal
    /// @brief Forward objective value evaluation to the stored function.
    /// @param ObjScale Objective scale factor. @param X Decision vector. @param Val Objective
    /// output.
    /// @param data Solver indexing data.
    /// @endinternal
    void objective(double ObjScale, const Eigen::Ref<const Eigen::VectorXd> &X, double &Val,
                   const tycho::solvers::SolverIndexingData &data) const override {
        this->data_.objective(ObjScale, X, Val, data);
    }
    /// @internal
    /// @brief Forward objective value + gradient evaluation to the stored function.
    /// @param ObjScale Objective scale factor. @param X Decision vector. @param Val Objective
    /// output.
    /// @param GX Gradient output. @param data Solver indexing data.
    /// @endinternal
    void objective_gradient(double ObjScale, const Eigen::Ref<const Eigen::VectorXd> &X,
                            double &Val, Eigen::Ref<Eigen::VectorXd> GX,
                            const tycho::solvers::SolverIndexingData &data) const override {
        this->data_.objective_gradient(ObjScale, X, Val, GX, data);
    }
    /// @internal
    /// @brief Forward objective value + gradient + sparse Hessian assembly.
    /// @param ObjScale Objective scale factor. @param X Decision vector. @param Val Objective
    /// output.
    /// @param GX Gradient output. @param KKTmat KKT matrix. @param KKTLocations Nonzero locations.
    /// @param KKTClashes Clash counts. @param KKTLocks Per-block mutexes. @param data Solver
    /// indexing data.
    /// @endinternal
    void objective_gradient_hessian(double ObjScale, const Eigen::Ref<const Eigen::VectorXd> &X,
                                    double &Val, Eigen::Ref<Eigen::VectorXd> GX,
                                    Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                                    Eigen::Ref<Eigen::VectorXi> KKTLocations,
                                    Eigen::Ref<Eigen::VectorXi> KKTClashes,
                                    std::vector<std::mutex> &KKTLocks,
                                    const tycho::solvers::SolverIndexingData &data) const override {
        this->data_.objective_gradient_hessian(ObjScale, X, Val, GX, KKTmat, KKTLocations,
                                               KKTClashes, KKTLocks, data);
    }

    // ---- GFConcept<IR,1>: objective pack operation ----

    /// @internal @brief Store the function directly into a solver objective interface. @param oi
    /// Interface. @endinternal
    void pack_into_objective_interface(ObjectiveInterface &oi) const override {
        oi.storage_.emplace<ObjectiveModel<T>>(this->data_);
    }
};

// ==========================================================================
// Section 5: GFStorage<IR, OR> — shared-ownership container
//
// Shared ownership via std::shared_ptr. Copy is O(1) (atomic refcount
// increment), matching the original rubber_types performance model.
// The flat vtable (GFConcept) and concept-constrained emplace() are
// preserved — only the ownership model changes.
// ==========================================================================

/// @internal
/// @brief Shared-ownership container holding a type-erased function (GFConcept).
///
/// Backs GenericFunction. Copies are O(1) atomic refcount increments via
/// std::shared_ptr, matching the original rubber_types performance model while
/// preserving the flat vtable and concept-constrained emplace.
/// @tparam IR  Input-row count.
/// @tparam OR  Output-row count.
/// @endinternal
template <int IR, int OR> class GFStorage {
    using Concept = GFConcept<IR, OR>; ///< @internal Erased interface type. @endinternal
    std::shared_ptr<Concept> ptr_; ///< @internal Shared pointer to the erased model. @endinternal

  public:
    /// @internal @brief Construct an empty storage. @endinternal
    GFStorage() noexcept = default;

    /// @internal
    /// @brief Wrap @p obj in a GFModel and take shared ownership.
    /// @tparam T  Stored function type (must satisfy GFStorable).
    /// @param obj  Function to store.
    /// @endinternal
    // C++20 concept constraint: "T does not satisfy GFStorable<IR,OR>" on bad types
    template <GFStorable<IR, OR> T> void emplace(T obj) {
        using Model = GFModel<IR, OR, std::decay_t<T>>;
        ptr_ = std::make_shared<Model>(std::move(obj));
    }

    // Shared-ownership copy/move — O(1) refcount increment / pointer swap
    /// @internal @brief Shared-ownership copy (refcount increment). @endinternal
    GFStorage(const GFStorage &) = default;
    /// @internal @brief Move constructor (pointer swap). @endinternal
    GFStorage(GFStorage &&) noexcept = default;
    /// @internal @brief Shared-ownership copy assignment. @return Reference to this. @endinternal
    GFStorage &operator=(const GFStorage &) = default;
    /// @internal @brief Move assignment. @return Reference to this. @endinternal
    GFStorage &operator=(GFStorage &&) noexcept = default;
    /// @internal @brief Destructor (releases shared ownership). @endinternal
    ~GFStorage() = default;

    /// @internal @brief Whether no function is currently stored. @return True if empty.
    /// @endinternal
    [[nodiscard]] bool empty() const noexcept { return !ptr_; }

    /// @internal @brief Access the stored erased interface by reference. @return Reference to the
    /// model. @endinternal
    Concept &get() const noexcept { return *ptr_; }
    /// @internal @brief Access the stored erased interface by pointer. @return Pointer to the model
    /// (or null). @endinternal
    Concept *get_ptr() const noexcept { return ptr_.get(); }
};

// ==========================================================================
// Out-of-line definitions for GFModelCommon::clone_into / clone_into_dynamic
// Must appear after GFStorage is fully defined (bodies call s.emplace()).
// ==========================================================================

// Out-of-line bodies for the in-class clone_into / clone_into_dynamic
// declarations documented above; no separate Doxygen block to avoid a
// duplicate @param section.
template <int IR, int OR, GFStorable<IR, OR> T>
void GFModelCommon<IR, OR, T>::clone_into(GFStorage<IR, OR> &s) const {
    s.emplace(this->data_);
}

template <int IR, int OR, GFStorable<IR, OR> T>
void GFModelCommon<IR, OR, T>::clone_into_dynamic(GFStorage<-1, -1> &s) const {
    s.emplace(this->data_);
}

} // namespace tycho::vf
