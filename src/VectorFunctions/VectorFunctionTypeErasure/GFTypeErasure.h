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
//   2. Small-buffer optimisation (SBO) — GFStorage holds objects up to
//      SBO_CAP bytes inline (no heap allocation for common built-in types).
//   3. Value semantics — GFStorage has deep-copy semantics (clone-on-copy),
//      replacing the shared_ptr shared-ownership model.
//   4. Single virtual dispatch for solver calls — ConstraintInterface and
//      ObjectiveInterface store T directly via pack_into, not the GFTE wrapper.
// =============================================================================

#pragma once

#include "DenseFunctionSpecs.h"
#include "SizingSpecs.h"
#include "SolverInterfaceSpecs.h"

#include <concepts>
#include <cstddef>
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace Tycho {

// Forward declarations (full definitions provided by included headers above,
// or lazily at instantiation time for GenericFunction-dependent bodies)
struct ConstraintInterface;
struct ObjectiveInterface;
template <int IR, int OR> class GFStorage;

// ==========================================================================
// Section 1: GFHolder<T> — owns one T, nothing else
// ==========================================================================

template <class T> struct GFHolder {
    T data_;
    explicit GFHolder(T t) : data_(std::move(t)) {}
};

// ==========================================================================
// Section 2: GFStorable concept — constrains what can be stored in GFStorage
//
// Checks the most common interface requirements at the call site before
// full GFModel instantiation, giving cleaner compiler diagnostics.
// ==========================================================================

template <typename T, int IR, int OR>
concept GFStorable = requires(const T &t) {
    { t.IRows() } -> std::same_as<int>;
    { t.ORows() } -> std::same_as<int>;
    { t.is_linear() } -> std::same_as<bool>;
    { t.input_domain() };
    { t.name() } -> std::same_as<std::string>;
    { T::IsVectorizable } -> std::convertible_to<bool>;
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

// Primary template: constraint functions (OR != 1)
template <int IR, int OR>
struct GFConcept : DenseFunctionSpec<IR, OR>::Concept, // non-virtual — no diamond
                   SizableSpec::Concept,
                   SolverConstraintSpec::Concept {

    virtual ~GFConcept() = default;

    // Copy operations (no existing Spec covers these)
    virtual void clone_into(GFStorage<IR, OR> &) const = 0;
    virtual void clone_into_dynamic(GFStorage<-1, -1> &) const = 0;

    // Solver-interface pack (direct store, no double-erasure)
    virtual void pack_into_constraint_interface(ConstraintInterface &) const = 0;
};

// Partial specialisation for OR==1: additionally inherits SolverObjectiveSpec
template <int IR>
struct GFConcept<IR, 1> : DenseFunctionSpec<IR, 1>::Concept,
                          SizableSpec::Concept,
                          SolverConstraintSpec::Concept,
                          SolverObjectiveSpec::Concept {

    virtual ~GFConcept() = default;

    virtual void clone_into(GFStorage<IR, 1> &) const = 0;
    virtual void clone_into_dynamic(GFStorage<-1, -1> &) const = 0;
    virtual void pack_into_constraint_interface(ConstraintInterface &) const = 0;
    virtual void pack_into_objective_interface(ObjectiveInterface &) const = 0;
};

// ==========================================================================
// Section 4: GFModelCommon<IR, OR, T> — implements all overrides except
//            pack_into_objective_interface (added by GFModel<IR,1,T>)
//
// Override bodies are grouped with section comments matching the source Spec
// files. If a new pure virtual is added to any Spec, the compiler fails to
// instantiate GFModelCommon (abstract class), pointing to the right section.
// ==========================================================================

template <int IR, int OR, GFStorable<IR, OR> T>
struct GFModelCommon : GFHolder<T>, GFConcept<IR, OR> {
    using GFHolder<T>::GFHolder;

    // Convenient type aliases (mirrors DenseFunctionSpec<IR,OR> outer struct)
    using Dspec = DenseFunctionSpec<IR, OR>;
    using InType = typename Dspec::InType;
    using OutType = typename Dspec::OutType;
    using JacType = typename Dspec::JacType;
    using AdjGradType = typename Dspec::AdjGradType;
    using AdjVarType = typename Dspec::AdjVarType;
    using AdjHessType = typename Dspec::AdjHessType;
    using SuperInType = typename Dspec::SuperInType;
    using SuperOutType = typename Dspec::SuperOutType;
    using SuperJacType = typename Dspec::SuperJacType;
    using SuperAdjGradType = typename Dspec::SuperAdjGradType;
    using SuperAdjVarType = typename Dspec::SuperAdjVarType;
    using SuperAdjHessType = typename Dspec::SuperAdjHessType;
    using RightJacTarget = typename Dspec::RightJacTarget;
    using LeftJacMatrix = typename Dspec::LeftJacMatrix;

    template <class S> using CVR = const Eigen::MatrixBase<S> &; // ConstVectorBaseRef
    template <class S> using CMR = const Eigen::MatrixBase<S> &; // ConstMatrixBaseRef
    template <class S> using CER = const Eigen::EigenBase<S> &;  // ConstEigenBaseRef

    // ---- DenseFunctionSpec overrides (see DenseFunctionSpecs.h) ----

    DomainMatrix input_domain() const override { return this->data_.input_domain(); }
    bool is_linear() const override { return this->data_.is_linear(); }
    void enable_vectorization(bool b) const override { this->data_.enable_vectorization(b); }

    void compute(CVR<InType> x, CVR<OutType> fx_) const override { this->data_.compute(x, fx_); }
    void compute(CVR<SuperInType> x, CVR<SuperOutType> fx_) const override {
        this->data_.compute(x, fx_);
    }

    void compute_jacobian(CVR<InType> x, CVR<OutType> fx_, CMR<JacType> jx_) const override {
        this->data_.compute_jacobian(x, fx_, jx_);
    }
    void compute_jacobian(CVR<SuperInType> x, CVR<SuperOutType> fx_,
                          CMR<SuperJacType> jx_) const override {
        this->data_.compute_jacobian(x, fx_, jx_);
    }

    void compute_jacobian_adjointgradient_adjointhessian(CVR<InType> x, CVR<OutType> fx_,
                                                         CMR<JacType> jx_,
                                                         CVR<AdjGradType> adjgrad_,
                                                         CMR<AdjHessType> adjhess_,
                                                         CVR<AdjVarType> adjvars) const override {
        this->data_.compute_jacobian_adjointgradient_adjointhessian(x, fx_, jx_, adjgrad_, adjhess_,
                                                                    adjvars);
    }
    void compute_jacobian_adjointgradient_adjointhessian(
        CVR<SuperInType> x, CVR<SuperOutType> fx_, CMR<SuperJacType> jx_,
        CVR<SuperAdjGradType> adjgrad_, CMR<SuperAdjHessType> adjhess_,
        CVR<SuperAdjVarType> adjvars) const override {
        this->data_.compute_jacobian_adjointgradient_adjointhessian(x, fx_, jx_, adjgrad_, adjhess_,
                                                                    adjvars);
    }

    void scale_jacobian(CMR<JacType> target_, double s) const override {
        this->data_.scale_jacobian(target_, s);
    }

    void accumulate_jacobian(CMR<JacType> target_, CMR<JacType> right,
                             DirectAssignment assign) const override {
        this->data_.accumulate_jacobian(target_, right, assign);
    }
    void accumulate_jacobian(CMR<JacType> target_, CMR<JacType> right,
                             PlusEqualsAssignment assign) const override {
        this->data_.accumulate_jacobian(target_, right, assign);
    }
    void accumulate_jacobian(CMR<JacType> target_, CMR<JacType> right,
                             MinusEqualsAssignment assign) const override {
        this->data_.accumulate_jacobian(target_, right, assign);
    }

    void right_jacobian_product(CMR<RightJacTarget> target_, CER<LeftJacMatrix> left,
                                CER<JacType> right, DirectAssignment assign,
                                std::bool_constant<true> aliased) const override {
        this->data_.right_jacobian_product(target_, left, right, assign, aliased);
    }
    void right_jacobian_product(CMR<RightJacTarget> target_, CER<LeftJacMatrix> left,
                                CER<JacType> right, PlusEqualsAssignment assign,
                                std::bool_constant<true> aliased) const override {
        this->data_.right_jacobian_product(target_, left, right, assign, aliased);
    }
    void right_jacobian_product(CMR<RightJacTarget> target_, CER<LeftJacMatrix> left,
                                CER<JacType> right, MinusEqualsAssignment assign,
                                std::bool_constant<true> aliased) const override {
        this->data_.right_jacobian_product(target_, left, right, assign, aliased);
    }
    void right_jacobian_product(CMR<RightJacTarget> target_, CER<LeftJacMatrix> left,
                                CER<JacType> right, DirectAssignment assign,
                                std::bool_constant<false> aliased) const override {
        this->data_.right_jacobian_product(target_, left, right, assign, aliased);
    }
    void right_jacobian_product(CMR<RightJacTarget> target_, CER<LeftJacMatrix> left,
                                CER<JacType> right, PlusEqualsAssignment assign,
                                std::bool_constant<false> aliased) const override {
        this->data_.right_jacobian_product(target_, left, right, assign, aliased);
    }
    void right_jacobian_product(CMR<RightJacTarget> target_, CER<LeftJacMatrix> left,
                                CER<JacType> right, MinusEqualsAssignment assign,
                                std::bool_constant<false> aliased) const override {
        this->data_.right_jacobian_product(target_, left, right, assign, aliased);
    }

    // ---- SizableSpec overrides (see SizingSpecs.h) ----

    std::string name() const override { return this->data_.name(); }
    int IRows() const override { return this->data_.IRows(); }
    int ORows() const override { return this->data_.ORows(); }
    bool thread_safe() const override { return this->data_.thread_safe(); }

    // ---- SolverConstraintSpec overrides (see SolverInterfaceSpecs.h) ----

    void constraints(const Eigen::Ref<const Eigen::VectorXd> &X, Eigen::Ref<Eigen::VectorXd> FX,
                     const SolverIndexingData &data) const override {
        this->data_.constraints(X, FX, data);
    }
    void constraints_adjointgradient(const Eigen::Ref<const Eigen::VectorXd> &X,
                                     const Eigen::Ref<const Eigen::VectorXd> &L,
                                     Eigen::Ref<Eigen::VectorXd> FX,
                                     Eigen::Ref<Eigen::VectorXd> AGX,
                                     const SolverIndexingData &data) const override {
        this->data_.constraints_adjointgradient(X, L, FX, AGX, data);
    }
    void constraints_jacobian(const Eigen::Ref<const Eigen::VectorXd> &X,
                              Eigen::Ref<Eigen::VectorXd> FX,
                              Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                              Eigen::Ref<Eigen::VectorXi> KKTLocations,
                              Eigen::Ref<Eigen::VectorXi> KKTClashes,
                              std::vector<std::mutex> &KKTLocks,
                              const SolverIndexingData &data) const override {
        this->data_.constraints_jacobian(X, FX, KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
    }
    void constraints_jacobian_adjointgradient(
        const Eigen::Ref<const Eigen::VectorXd> &X, const Eigen::Ref<const Eigen::VectorXd> &L,
        Eigen::Ref<Eigen::VectorXd> FX, Eigen::Ref<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
        Eigen::Ref<Eigen::VectorXi> KKTLocations, Eigen::Ref<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex> &KKTLocks, const SolverIndexingData &data) const override {
        this->data_.constraints_jacobian_adjointgradient(X, L, FX, AGX, KKTmat, KKTLocations,
                                                         KKTClashes, KKTLocks, data);
    }
    void constraints_jacobian_adjointgradient_adjointhessian(
        const Eigen::Ref<const Eigen::VectorXd> &X, const Eigen::Ref<const Eigen::VectorXd> &L,
        Eigen::Ref<Eigen::VectorXd> FX, Eigen::Ref<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
        Eigen::Ref<Eigen::VectorXi> KKTLocations, Eigen::Ref<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex> &KKTLocks, const SolverIndexingData &data) const override {
        this->data_.constraints_jacobian_adjointgradient_adjointhessian(
            X, L, FX, AGX, KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
    }
    void getKKTSpace(Eigen::Ref<Eigen::VectorXi> KKTrows, Eigen::Ref<Eigen::VectorXi> KKTcols,
                     int &freeloc, int conoffset, bool dojac, bool dohess,
                     SolverIndexingData &data) override {
        this->data_.getKKTSpace(KKTrows, KKTcols, freeloc, conoffset, dojac, dohess, data);
    }
    int numKKTEles(bool dojac, bool dohess) const override {
        return this->data_.numKKTEles(dojac, dohess);
    }

    // ---- GFConcept: copy and pack operations ----

    // Bodies defined out-of-line after GFStorage (emplace() must be visible)
    void clone_into(GFStorage<IR, OR> &s) const override;
    void clone_into_dynamic(GFStorage<-1, -1> &s) const override;
    void pack_into_constraint_interface(ConstraintInterface &ci) const override {
        ci = ConstraintInterface(this->data_);
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

// Primary template (non-objective)
template <int IR, int OR, GFStorable<IR, OR> T> struct GFModel final : GFModelCommon<IR, OR, T> {
    using GFModelCommon<IR, OR, T>::GFModelCommon;
};

// Partial specialisation for scalar output (objective-capable)
template <int IR, GFStorable<IR, 1> T> struct GFModel<IR, 1, T> final : GFModelCommon<IR, 1, T> {
    using GFModelCommon<IR, 1, T>::GFModelCommon;

    // ---- SolverObjectiveSpec overrides (see SolverInterfaceSpecs.h) ----

    void objective(double ObjScale, const Eigen::Ref<const Eigen::VectorXd> &X, double &Val,
                   const SolverIndexingData &data) const override {
        this->data_.objective(ObjScale, X, Val, data);
    }
    void objective_gradient(double ObjScale, const Eigen::Ref<const Eigen::VectorXd> &X,
                            double &Val, Eigen::Ref<Eigen::VectorXd> GX,
                            const SolverIndexingData &data) const override {
        this->data_.objective_gradient(ObjScale, X, Val, GX, data);
    }
    void objective_gradient_hessian(double ObjScale, const Eigen::Ref<const Eigen::VectorXd> &X,
                                    double &Val, Eigen::Ref<Eigen::VectorXd> GX,
                                    Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                                    Eigen::Ref<Eigen::VectorXi> KKTLocations,
                                    Eigen::Ref<Eigen::VectorXi> KKTClashes,
                                    std::vector<std::mutex> &KKTLocks,
                                    const SolverIndexingData &data) const override {
        this->data_.objective_gradient_hessian(ObjScale, X, Val, GX, KKTmat, KKTLocations,
                                               KKTClashes, KKTLocks, data);
    }

    // ---- GFConcept<IR,1>: objective pack operation ----

    void pack_into_objective_interface(ObjectiveInterface &oi) const override {
        oi = ObjectiveInterface(this->data_);
    }
};

// ==========================================================================
// Section 5: GFStorage<IR, OR> — SBO container, value semantics
//
// Value type with deep-copy semantics (clone-on-copy via GFConcept::clone_into).
// For SBO-eligible types, clone is a placement-new into the inline buffer —
// no heap allocation.
//
// Assumptions on trivial relocatability:
//   Inline-stored objects are moved via std::memcpy. This is valid for all
//   Tycho built-in VF types (plain data + vptr, no internal self-pointers).
//   Large or exotic types fall through to the heap path and are moved by
//   pointer swap (always correct).
// ==========================================================================

template <int IR, int OR> class GFStorage {
    // SBO_CAP sized to cover common built-in VF types without heap fallback.
    // sizeof(GFModel<IR,OR,T>) = sizeof(vptr) + sizeof(T). Most built-ins
    // are 8–72 bytes. 128 bytes gives comfortable headroom.
    static constexpr std::size_t SBO_CAP = 128;
    static constexpr std::size_t SBO_ALIGN = alignof(std::max_align_t);

    using Concept = GFConcept<IR, OR>;

    enum class Kind : uint8_t { Empty, Inline, Heap };

    Kind kind_ = Kind::Empty;
    union {
        alignas(SBO_ALIGN) std::byte buf_[SBO_CAP];
        Concept *ptr_;
    };

    Concept *concept_ptr() const noexcept {
        return kind_ == Kind::Inline
                   ? std::launder(reinterpret_cast<Concept *>(const_cast<std::byte *>(buf_)))
                   : ptr_;
    }

    void destroy() noexcept {
        if (kind_ == Kind::Inline)
            concept_ptr()->~Concept();
        else if (kind_ == Kind::Heap)
            delete ptr_;
        kind_ = Kind::Empty;
    }

  public:
    GFStorage() noexcept = default;

    // C++20 concept constraint: "T does not satisfy GFStorable<IR,OR>" on bad types
    template <GFStorable<IR, OR> T> void emplace(T obj) {
        using Model = GFModel<IR, OR, std::decay_t<T>>;
        destroy();
        if constexpr (sizeof(Model) <= SBO_CAP && alignof(Model) <= SBO_ALIGN) {
            std::construct_at(reinterpret_cast<Model *>(buf_), std::move(obj));
            kind_ = Kind::Inline;
        } else {
            ptr_ = new Model(std::move(obj));
            kind_ = Kind::Heap;
        }
    }

    // Deep-copy (clone_into calls emplace on the destination)
    GFStorage(const GFStorage &o) {
        if (!o.empty())
            o.concept_ptr()->clone_into(*this);
    }

    // Move: pointer swap for heap, memcpy for inline (trivially relocatable
    // for all Tycho built-in VF types), no-op for empty.
    GFStorage(GFStorage &&o) noexcept {
        switch (o.kind_) {
        case Kind::Inline:
            std::memcpy(buf_, o.buf_, SBO_CAP);
            kind_ = Kind::Inline;
            o.kind_ = Kind::Empty;
            break;
        case Kind::Heap:
            ptr_ = o.ptr_;
            kind_ = Kind::Heap;
            o.ptr_ = nullptr;
            o.kind_ = Kind::Empty;
            break;
        case Kind::Empty:
            kind_ = Kind::Empty;
            break;
        }
    }

    GFStorage &operator=(const GFStorage &o) {
        if (this != &o) {
            destroy();
            if (!o.empty())
                o.concept_ptr()->clone_into(*this);
        }
        return *this;
    }

    GFStorage &operator=(GFStorage &&o) noexcept {
        if (this != &o) {
            destroy();
            new (this) GFStorage(std::move(o));
        }
        return *this;
    }

    ~GFStorage() { destroy(); }

    [[nodiscard("Check empty() before calling get()")]]
    bool empty() const noexcept {
        return kind_ == Kind::Empty;
    }

    Concept &get() const noexcept { return *concept_ptr(); }
    Concept *get_ptr() const noexcept { return concept_ptr(); }
};

// ==========================================================================
// Out-of-line definitions for GFModelCommon::clone_into / clone_into_dynamic
// Must appear after GFStorage is fully defined (bodies call s.emplace()).
// ==========================================================================

template <int IR, int OR, GFStorable<IR, OR> T>
void GFModelCommon<IR, OR, T>::clone_into(GFStorage<IR, OR> &s) const {
    s.emplace(this->data_);
}

template <int IR, int OR, GFStorable<IR, OR> T>
void GFModelCommon<IR, OR, T>::clone_into_dynamic(GFStorage<-1, -1> &s) const {
    s.emplace(this->data_);
}

} // namespace Tycho
