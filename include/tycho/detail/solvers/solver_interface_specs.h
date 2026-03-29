// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Defines the type erasure specs (SolverConstraintSpec, SolverObjectiveSpec)
// and concrete type erasure (ConstraintInterface, ObjectiveInterface) that
// enable vector functions to interface with PSIOPT and NonLinearProgram.
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
//   - PR 9: Replaced rubber_types with TypeStorage; deleted dead
//     Model<>/ExternalInterface<> boilerplate and SolverInterfaceSelector.
// =============================================================================

#pragma once

#include "tycho/detail/solvers/indexing_data.h"
#include "tycho/detail/solvers/sizing_specs.h"
#include "tycho/detail/vf/core/expression_fwd_declarations.h"
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

namespace tycho::solvers {

// Import cross-namespace types used throughout the solver layer.
using utils::TypeStorage;
using vf::GenericFunction;

/*
 * Spec for vector function that can be used as a constraint inside of PSIOPT.
 */
struct SolverConstraintSpec {
    struct Concept {
        virtual ~Concept() = default;

        virtual void constraints(const Eigen::Ref<const Eigen::VectorXd> &X,
                                 Eigen::Ref<Eigen::VectorXd> FX,
                                 const SolverIndexingData &data) const = 0;

        virtual void constraints_adjointgradient(const Eigen::Ref<const Eigen::VectorXd> &X,
                                                 const Eigen::Ref<const Eigen::VectorXd> &L,
                                                 Eigen::Ref<Eigen::VectorXd> FX,
                                                 Eigen::Ref<Eigen::VectorXd> AGX,
                                                 const SolverIndexingData &data) const = 0;

        virtual void constraints_jacobian(const Eigen::Ref<const Eigen::VectorXd> &X,
                                          Eigen::Ref<Eigen::VectorXd> FX,
                                          Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                                          Eigen::Ref<Eigen::VectorXi> KKTLocations,
                                          Eigen::Ref<Eigen::VectorXi> KKTClashes,
                                          std::vector<std::mutex> &KKTLocks,
                                          const SolverIndexingData &data) const = 0;

        virtual void constraints_jacobian_adjointgradient(
            const Eigen::Ref<const Eigen::VectorXd> &X, const Eigen::Ref<const Eigen::VectorXd> &L,
            Eigen::Ref<Eigen::VectorXd> FX, Eigen::Ref<Eigen::VectorXd> AGX,
            Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
            Eigen::Ref<Eigen::VectorXi> KKTLocations, Eigen::Ref<Eigen::VectorXi> KKTClashes,
            std::vector<std::mutex> &KKTLocks, const SolverIndexingData &data) const = 0;

        virtual void constraints_jacobian_adjointgradient_adjointhessian(
            const Eigen::Ref<const Eigen::VectorXd> &X, const Eigen::Ref<const Eigen::VectorXd> &L,
            Eigen::Ref<Eigen::VectorXd> FX, Eigen::Ref<Eigen::VectorXd> AGX,
            Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
            Eigen::Ref<Eigen::VectorXi> KKTLocations, Eigen::Ref<Eigen::VectorXi> KKTClashes,
            std::vector<std::mutex> &KKTLocks, const SolverIndexingData &data) const = 0;

        virtual void get_kkt_space(Eigen::Ref<Eigen::VectorXi> KKTrows,
                                   Eigen::Ref<Eigen::VectorXi> KKTcols, int &freeloc, int conoffset,
                                   bool dojac, bool dohess, SolverIndexingData &data) = 0;

        virtual int num_kkt_elements(bool dojac, bool dohess) const = 0;
    };
};

/*
 * Spec for scalar vector functions that can be used as an objective inside of PSIOPT.
 */
struct SolverObjectiveSpec {
    struct Concept {
        virtual ~Concept() = default;

        virtual void objective(double ObjScale, const Eigen::Ref<const Eigen::VectorXd> &X,
                               double &Val, const SolverIndexingData &data) const = 0;

        virtual void objective_gradient(double ObjScale, const Eigen::Ref<const Eigen::VectorXd> &X,
                                        double &Val, Eigen::Ref<Eigen::VectorXd> GX,
                                        const SolverIndexingData &data) const = 0;

        virtual void objective_gradient_hessian(
            double ObjScale, const Eigen::Ref<const Eigen::VectorXd> &X, double &Val,
            Eigen::Ref<Eigen::VectorXd> GX, Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
            Eigen::Ref<Eigen::VectorXi> KKTLocations, Eigen::Ref<Eigen::VectorXi> KKTClashes,
            std::vector<std::mutex> &KKTLocks, const SolverIndexingData &data) const = 0;
    };
};

// ==========================================================================
// ConstraintBase / ConstraintModel<T> / ConstraintInterface
// ==========================================================================

struct ConstraintBase : SolverConstraintSpec::Concept, SizableSpec::Concept {
    virtual void clone_into(TypeStorage<ConstraintBase> &) const = 0;
};

template <typename T> struct ConstraintModel final : ConstraintBase {
    T data_;
    explicit ConstraintModel(T t) : data_(std::move(t)) {}

    // ---- SizableSpec::Concept ----
    std::string name() const override { return data_.name(); }
    int input_rows() const override { return data_.input_rows(); }
    int output_rows() const override { return data_.output_rows(); }
    bool thread_safe() const override { return data_.thread_safe(); }

    // ---- SolverConstraintSpec::Concept ----
    void constraints(const Eigen::Ref<const Eigen::VectorXd> &X, Eigen::Ref<Eigen::VectorXd> FX,
                     const SolverIndexingData &data) const override {
        data_.constraints(X, FX, data);
    }
    void constraints_adjointgradient(const Eigen::Ref<const Eigen::VectorXd> &X,
                                     const Eigen::Ref<const Eigen::VectorXd> &L,
                                     Eigen::Ref<Eigen::VectorXd> FX,
                                     Eigen::Ref<Eigen::VectorXd> AGX,
                                     const SolverIndexingData &data) const override {
        data_.constraints_adjointgradient(X, L, FX, AGX, data);
    }
    void constraints_jacobian(const Eigen::Ref<const Eigen::VectorXd> &X,
                              Eigen::Ref<Eigen::VectorXd> FX,
                              Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                              Eigen::Ref<Eigen::VectorXi> KKTLocations,
                              Eigen::Ref<Eigen::VectorXi> KKTClashes,
                              std::vector<std::mutex> &KKTLocks,
                              const SolverIndexingData &data) const override {
        data_.constraints_jacobian(X, FX, KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
    }
    void constraints_jacobian_adjointgradient(
        const Eigen::Ref<const Eigen::VectorXd> &X, const Eigen::Ref<const Eigen::VectorXd> &L,
        Eigen::Ref<Eigen::VectorXd> FX, Eigen::Ref<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
        Eigen::Ref<Eigen::VectorXi> KKTLocations, Eigen::Ref<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex> &KKTLocks, const SolverIndexingData &data) const override {
        data_.constraints_jacobian_adjointgradient(X, L, FX, AGX, KKTmat, KKTLocations, KKTClashes,
                                                   KKTLocks, data);
    }
    void constraints_jacobian_adjointgradient_adjointhessian(
        const Eigen::Ref<const Eigen::VectorXd> &X, const Eigen::Ref<const Eigen::VectorXd> &L,
        Eigen::Ref<Eigen::VectorXd> FX, Eigen::Ref<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
        Eigen::Ref<Eigen::VectorXi> KKTLocations, Eigen::Ref<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex> &KKTLocks, const SolverIndexingData &data) const override {
        data_.constraints_jacobian_adjointgradient_adjointhessian(
            X, L, FX, AGX, KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
    }
    void get_kkt_space(Eigen::Ref<Eigen::VectorXi> KKTrows, Eigen::Ref<Eigen::VectorXi> KKTcols,
                       int &freeloc, int conoffset, bool dojac, bool dohess,
                       SolverIndexingData &data) override {
        data_.get_kkt_space(KKTrows, KKTcols, freeloc, conoffset, dojac, dohess, data);
    }
    int num_kkt_elements(bool dojac, bool dohess) const override {
        return data_.num_kkt_elements(dojac, dohess);
    }

    void clone_into(TypeStorage<ConstraintBase> &s) const override {
        s.emplace<ConstraintModel<T>>(data_);
    }
};

struct ConstraintInterface;
struct ObjectiveInterface;

struct ConstraintInterface {
    TypeStorage<ConstraintBase> storage;

    ConstraintInterface() = default;

    template <class T, std::enable_if_t<
                           !std::is_base_of_v<Eigen::EigenBase<std::decay_t<T>>, std::decay_t<T>>,
                           bool> = true>
    ConstraintInterface(const T &t) {
        storage.emplace<ConstraintModel<std::decay_t<T>>>(t);
    }

    // Stores T directly (one virtual dispatch per solver call) instead of double-erasure.
    template <int IR, int OR> ConstraintInterface(const GenericFunction<IR, OR> &t) {
        t.func.get().pack_into_constraint_interface(*this);
    }

    // ---- Forwarding methods ----
    std::string name() const { return storage.get().name(); }
    int input_rows() const { return storage.get().input_rows(); }
    int output_rows() const { return storage.get().output_rows(); }
    bool thread_safe() const { return storage.get().thread_safe(); }

    void constraints(const Eigen::Ref<const Eigen::VectorXd> &X, Eigen::Ref<Eigen::VectorXd> FX,
                     const SolverIndexingData &data) const {
        storage.get().constraints(X, FX, data);
    }
    void constraints_adjointgradient(const Eigen::Ref<const Eigen::VectorXd> &X,
                                     const Eigen::Ref<const Eigen::VectorXd> &L,
                                     Eigen::Ref<Eigen::VectorXd> FX,
                                     Eigen::Ref<Eigen::VectorXd> AGX,
                                     const SolverIndexingData &data) const {
        storage.get().constraints_adjointgradient(X, L, FX, AGX, data);
    }
    void constraints_jacobian(const Eigen::Ref<const Eigen::VectorXd> &X,
                              Eigen::Ref<Eigen::VectorXd> FX,
                              Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                              Eigen::Ref<Eigen::VectorXi> KKTLocations,
                              Eigen::Ref<Eigen::VectorXi> KKTClashes,
                              std::vector<std::mutex> &KKTLocks,
                              const SolverIndexingData &data) const {
        storage.get().constraints_jacobian(X, FX, KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
    }
    void constraints_jacobian_adjointgradient(
        const Eigen::Ref<const Eigen::VectorXd> &X, const Eigen::Ref<const Eigen::VectorXd> &L,
        Eigen::Ref<Eigen::VectorXd> FX, Eigen::Ref<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
        Eigen::Ref<Eigen::VectorXi> KKTLocations, Eigen::Ref<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex> &KKTLocks, const SolverIndexingData &data) const {
        storage.get().constraints_jacobian_adjointgradient(X, L, FX, AGX, KKTmat, KKTLocations,
                                                           KKTClashes, KKTLocks, data);
    }
    void constraints_jacobian_adjointgradient_adjointhessian(
        const Eigen::Ref<const Eigen::VectorXd> &X, const Eigen::Ref<const Eigen::VectorXd> &L,
        Eigen::Ref<Eigen::VectorXd> FX, Eigen::Ref<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
        Eigen::Ref<Eigen::VectorXi> KKTLocations, Eigen::Ref<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex> &KKTLocks, const SolverIndexingData &data) const {
        storage.get().constraints_jacobian_adjointgradient_adjointhessian(
            X, L, FX, AGX, KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
    }
    void get_kkt_space(Eigen::Ref<Eigen::VectorXi> KKTrows, Eigen::Ref<Eigen::VectorXi> KKTcols,
                       int &freeloc, int conoffset, bool dojac, bool dohess,
                       SolverIndexingData &data) {
        storage.get().get_kkt_space(KKTrows, KKTcols, freeloc, conoffset, dojac, dohess, data);
    }
    int num_kkt_elements(bool dojac, bool dohess) const {
        return storage.get().num_kkt_elements(dojac, dohess);
    }
};

// ==========================================================================
// ObjectiveBase / ObjectiveModel<T> / ObjectiveInterface
// ==========================================================================

struct ObjectiveBase : SolverConstraintSpec::Concept,
                       SolverObjectiveSpec::Concept,
                       SizableSpec::Concept {
    virtual void clone_into(TypeStorage<ObjectiveBase> &) const = 0;
};

template <typename T> struct ObjectiveModel final : ObjectiveBase {
    T data_;
    explicit ObjectiveModel(T t) : data_(std::move(t)) {}

    // ---- SizableSpec::Concept ----
    std::string name() const override { return data_.name(); }
    int input_rows() const override { return data_.input_rows(); }
    int output_rows() const override { return data_.output_rows(); }
    bool thread_safe() const override { return data_.thread_safe(); }

    // ---- SolverConstraintSpec::Concept ----
    void constraints(const Eigen::Ref<const Eigen::VectorXd> &X, Eigen::Ref<Eigen::VectorXd> FX,
                     const SolverIndexingData &data) const override {
        data_.constraints(X, FX, data);
    }
    void constraints_adjointgradient(const Eigen::Ref<const Eigen::VectorXd> &X,
                                     const Eigen::Ref<const Eigen::VectorXd> &L,
                                     Eigen::Ref<Eigen::VectorXd> FX,
                                     Eigen::Ref<Eigen::VectorXd> AGX,
                                     const SolverIndexingData &data) const override {
        data_.constraints_adjointgradient(X, L, FX, AGX, data);
    }
    void constraints_jacobian(const Eigen::Ref<const Eigen::VectorXd> &X,
                              Eigen::Ref<Eigen::VectorXd> FX,
                              Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                              Eigen::Ref<Eigen::VectorXi> KKTLocations,
                              Eigen::Ref<Eigen::VectorXi> KKTClashes,
                              std::vector<std::mutex> &KKTLocks,
                              const SolverIndexingData &data) const override {
        data_.constraints_jacobian(X, FX, KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
    }
    void constraints_jacobian_adjointgradient(
        const Eigen::Ref<const Eigen::VectorXd> &X, const Eigen::Ref<const Eigen::VectorXd> &L,
        Eigen::Ref<Eigen::VectorXd> FX, Eigen::Ref<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
        Eigen::Ref<Eigen::VectorXi> KKTLocations, Eigen::Ref<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex> &KKTLocks, const SolverIndexingData &data) const override {
        data_.constraints_jacobian_adjointgradient(X, L, FX, AGX, KKTmat, KKTLocations, KKTClashes,
                                                   KKTLocks, data);
    }
    void constraints_jacobian_adjointgradient_adjointhessian(
        const Eigen::Ref<const Eigen::VectorXd> &X, const Eigen::Ref<const Eigen::VectorXd> &L,
        Eigen::Ref<Eigen::VectorXd> FX, Eigen::Ref<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
        Eigen::Ref<Eigen::VectorXi> KKTLocations, Eigen::Ref<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex> &KKTLocks, const SolverIndexingData &data) const override {
        data_.constraints_jacobian_adjointgradient_adjointhessian(
            X, L, FX, AGX, KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
    }
    void get_kkt_space(Eigen::Ref<Eigen::VectorXi> KKTrows, Eigen::Ref<Eigen::VectorXi> KKTcols,
                       int &freeloc, int conoffset, bool dojac, bool dohess,
                       SolverIndexingData &data) override {
        data_.get_kkt_space(KKTrows, KKTcols, freeloc, conoffset, dojac, dohess, data);
    }
    int num_kkt_elements(bool dojac, bool dohess) const override {
        return data_.num_kkt_elements(dojac, dohess);
    }

    // ---- SolverObjectiveSpec::Concept ----
    void objective(double ObjScale, const Eigen::Ref<const Eigen::VectorXd> &X, double &Val,
                   const SolverIndexingData &data) const override {
        data_.objective(ObjScale, X, Val, data);
    }
    void objective_gradient(double ObjScale, const Eigen::Ref<const Eigen::VectorXd> &X,
                            double &Val, Eigen::Ref<Eigen::VectorXd> GX,
                            const SolverIndexingData &data) const override {
        data_.objective_gradient(ObjScale, X, Val, GX, data);
    }
    void objective_gradient_hessian(double ObjScale, const Eigen::Ref<const Eigen::VectorXd> &X,
                                    double &Val, Eigen::Ref<Eigen::VectorXd> GX,
                                    Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                                    Eigen::Ref<Eigen::VectorXi> KKTLocations,
                                    Eigen::Ref<Eigen::VectorXi> KKTClashes,
                                    std::vector<std::mutex> &KKTLocks,
                                    const SolverIndexingData &data) const override {
        data_.objective_gradient_hessian(ObjScale, X, Val, GX, KKTmat, KKTLocations, KKTClashes,
                                         KKTLocks, data);
    }

    void clone_into(TypeStorage<ObjectiveBase> &s) const override {
        s.emplace<ObjectiveModel<T>>(data_);
    }
};

struct ObjectiveInterface {
    TypeStorage<ObjectiveBase> storage;

    ObjectiveInterface() = default;

    template <class T, std::enable_if_t<
                           !std::is_base_of_v<Eigen::EigenBase<std::decay_t<T>>, std::decay_t<T>>,
                           bool> = true>
    ObjectiveInterface(const T &t) {
        storage.emplace<ObjectiveModel<std::decay_t<T>>>(t);
    }

    // Stores T directly (one virtual dispatch per solver call).
    template <int IR> ObjectiveInterface(const GenericFunction<IR, 1> &t) {
        t.func.get().pack_into_objective_interface(*this);
    }

    // ---- Forwarding methods ----
    std::string name() const { return storage.get().name(); }
    int input_rows() const { return storage.get().input_rows(); }
    int output_rows() const { return storage.get().output_rows(); }
    bool thread_safe() const { return storage.get().thread_safe(); }

    void constraints(const Eigen::Ref<const Eigen::VectorXd> &X, Eigen::Ref<Eigen::VectorXd> FX,
                     const SolverIndexingData &data) const {
        storage.get().constraints(X, FX, data);
    }
    void constraints_adjointgradient(const Eigen::Ref<const Eigen::VectorXd> &X,
                                     const Eigen::Ref<const Eigen::VectorXd> &L,
                                     Eigen::Ref<Eigen::VectorXd> FX,
                                     Eigen::Ref<Eigen::VectorXd> AGX,
                                     const SolverIndexingData &data) const {
        storage.get().constraints_adjointgradient(X, L, FX, AGX, data);
    }
    void constraints_jacobian(const Eigen::Ref<const Eigen::VectorXd> &X,
                              Eigen::Ref<Eigen::VectorXd> FX,
                              Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                              Eigen::Ref<Eigen::VectorXi> KKTLocations,
                              Eigen::Ref<Eigen::VectorXi> KKTClashes,
                              std::vector<std::mutex> &KKTLocks,
                              const SolverIndexingData &data) const {
        storage.get().constraints_jacobian(X, FX, KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
    }
    void constraints_jacobian_adjointgradient(
        const Eigen::Ref<const Eigen::VectorXd> &X, const Eigen::Ref<const Eigen::VectorXd> &L,
        Eigen::Ref<Eigen::VectorXd> FX, Eigen::Ref<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
        Eigen::Ref<Eigen::VectorXi> KKTLocations, Eigen::Ref<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex> &KKTLocks, const SolverIndexingData &data) const {
        storage.get().constraints_jacobian_adjointgradient(X, L, FX, AGX, KKTmat, KKTLocations,
                                                           KKTClashes, KKTLocks, data);
    }
    void constraints_jacobian_adjointgradient_adjointhessian(
        const Eigen::Ref<const Eigen::VectorXd> &X, const Eigen::Ref<const Eigen::VectorXd> &L,
        Eigen::Ref<Eigen::VectorXd> FX, Eigen::Ref<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
        Eigen::Ref<Eigen::VectorXi> KKTLocations, Eigen::Ref<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex> &KKTLocks, const SolverIndexingData &data) const {
        storage.get().constraints_jacobian_adjointgradient_adjointhessian(
            X, L, FX, AGX, KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
    }
    void get_kkt_space(Eigen::Ref<Eigen::VectorXi> KKTrows, Eigen::Ref<Eigen::VectorXi> KKTcols,
                       int &freeloc, int conoffset, bool dojac, bool dohess,
                       SolverIndexingData &data) {
        storage.get().get_kkt_space(KKTrows, KKTcols, freeloc, conoffset, dojac, dohess, data);
    }
    int num_kkt_elements(bool dojac, bool dohess) const {
        return storage.get().num_kkt_elements(dojac, dohess);
    }

    void objective(double ObjScale, const Eigen::Ref<const Eigen::VectorXd> &X, double &Val,
                   const SolverIndexingData &data) const {
        storage.get().objective(ObjScale, X, Val, data);
    }
    void objective_gradient(double ObjScale, const Eigen::Ref<const Eigen::VectorXd> &X,
                            double &Val, Eigen::Ref<Eigen::VectorXd> GX,
                            const SolverIndexingData &data) const {
        storage.get().objective_gradient(ObjScale, X, Val, GX, data);
    }
    void objective_gradient_hessian(double ObjScale, const Eigen::Ref<const Eigen::VectorXd> &X,
                                    double &Val, Eigen::Ref<Eigen::VectorXd> GX,
                                    Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                                    Eigen::Ref<Eigen::VectorXi> KKTLocations,
                                    Eigen::Ref<Eigen::VectorXi> KKTClashes,
                                    std::vector<std::mutex> &KKTLocks,
                                    const SolverIndexingData &data) const {
        storage.get().objective_gradient_hessian(ObjScale, X, Val, GX, KKTmat, KKTLocations,
                                                 KKTClashes, KKTLocks, data);
    }
};

} // namespace tycho::solvers
