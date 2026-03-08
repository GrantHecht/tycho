// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Defines the rubber_types compatible type erasure specs
// (SolverConstraintSpec,SolverObjectiveSpec) and objects (ConstraintInterface,ObjectiveInterface) that
// enable vector functions to interface with PSIOPT and NonLinearProgram.
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> Tycho
//   - Python binding methods (Build(py::module)) moved to src/Bindings/ (PR 2)
//   - pybind11 / pybind11 header references removed
// =============================================================================

#pragma once

#include "DeepCopySpecs.h"
#include "IndexingData.h"
#include "SizingSpecs.h"
#include "VectorFunctions/CommonFunctions/ExpressionFwdDeclarations.h"
#include "pch.h"

namespace Tycho {

/*
 * Spec for vector function that can be used as a constraint inside of PSIOPT. Erases all
 * .constraints_xxx methods as well as the function specific methods for requesting and allocating
 * space from the solver.
 */
struct SolverConstraintSpec {
    ////////////////////////////////////////////////////////
    struct Concept { // abstract base class for model.
        virtual ~Concept() = default;
        // Your (internal) interface goes here.
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

        virtual void getKKTSpace(Eigen::Ref<Eigen::VectorXi> KKTrows,
                                 Eigen::Ref<Eigen::VectorXi> KKTcols, int &freeloc, int conoffset,
                                 bool dojac, bool dohess, SolverIndexingData &data) = 0;

        virtual int numKKTEles(bool dojac, bool dohess) const = 0;
    };
    ////////////////////////////////////////////////////////
    template <class Holder> struct Model : public Holder, public virtual Concept {
        using Holder::Holder;
        // Pass through to encapsulated value.

        virtual void constraints(const Eigen::Ref<const Eigen::VectorXd> &X,
                                 Eigen::Ref<Eigen::VectorXd> FX,
                                 const SolverIndexingData &data) const override {
            return rubber_types::model_get(*this).constraints(X, FX, data);
        }

        virtual void constraints_adjointgradient(const Eigen::Ref<const Eigen::VectorXd> &X,
                                                 const Eigen::Ref<const Eigen::VectorXd> &L,
                                                 Eigen::Ref<Eigen::VectorXd> FX,
                                                 Eigen::Ref<Eigen::VectorXd> AGX,
                                                 const SolverIndexingData &data) const override {
            return rubber_types::model_get(*this).constraints_adjointgradient(X, L, FX, AGX, data);
        }

        virtual void constraints_jacobian(const Eigen::Ref<const Eigen::VectorXd> &X,
                                          Eigen::Ref<Eigen::VectorXd> FX,
                                          Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                                          Eigen::Ref<Eigen::VectorXi> KKTLocations,
                                          Eigen::Ref<Eigen::VectorXi> KKTClashes,
                                          std::vector<std::mutex> &KKTLocks,
                                          const SolverIndexingData &data) const override {
            return rubber_types::model_get(*this).constraints_jacobian(X, FX, KKTmat, KKTLocations,
                                                                       KKTClashes, KKTLocks, data);
        }

        virtual void constraints_jacobian_adjointgradient(
            const Eigen::Ref<const Eigen::VectorXd> &X, const Eigen::Ref<const Eigen::VectorXd> &L,
            Eigen::Ref<Eigen::VectorXd> FX, Eigen::Ref<Eigen::VectorXd> AGX,
            Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
            Eigen::Ref<Eigen::VectorXi> KKTLocations, Eigen::Ref<Eigen::VectorXi> KKTClashes,
            std::vector<std::mutex> &KKTLocks, const SolverIndexingData &data) const override {
            return rubber_types::model_get(*this).constraints_jacobian_adjointgradient(
                X, L, FX, AGX, KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
        }

        virtual void constraints_jacobian_adjointgradient_adjointhessian(
            const Eigen::Ref<const Eigen::VectorXd> &X, const Eigen::Ref<const Eigen::VectorXd> &L,
            Eigen::Ref<Eigen::VectorXd> FX, Eigen::Ref<Eigen::VectorXd> AGX,
            Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
            Eigen::Ref<Eigen::VectorXi> KKTLocations, Eigen::Ref<Eigen::VectorXi> KKTClashes,
            std::vector<std::mutex> &KKTLocks, const SolverIndexingData &data) const override {
            return rubber_types::model_get(*this)
                .constraints_jacobian_adjointgradient_adjointhessian(
                    X, L, FX, AGX, KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
        }

        virtual void getKKTSpace(Eigen::Ref<Eigen::VectorXi> KKTrows,
                                 Eigen::Ref<Eigen::VectorXi> KKTcols, int &freeloc, int conoffset,
                                 bool dojac, bool dohess, SolverIndexingData &data) override {
            return rubber_types::model_get(*this).getKKTSpace(KKTrows, KKTcols, freeloc, conoffset,
                                                              dojac, dohess, data);
        }

        virtual int numKKTEles(bool dojac, bool dohess) const override {
            return rubber_types::model_get(*this).numKKTEles(dojac, dohess);
        }
    };
    ////////////////////////////////////////////////////////
    template <class Container> struct ExternalInterface : public Container {
        using Container_ = Container;
        using Container_::Container_;

        // Defines Basic Interface to PSIOPT Solver for VectorFunction Constraints.

        void constraints(const Eigen::Ref<const Eigen::VectorXd> &X, Eigen::Ref<Eigen::VectorXd> FX,
                         const SolverIndexingData &data) const {
            return rubber_types::interface_get(*this).constraints(X, FX, data);
        }

        void constraints_adjointgradient(const Eigen::Ref<const Eigen::VectorXd> &X,
                                         const Eigen::Ref<const Eigen::VectorXd> &L,
                                         Eigen::Ref<Eigen::VectorXd> FX,
                                         Eigen::Ref<Eigen::VectorXd> AGX,
                                         const SolverIndexingData &data) const {
            return rubber_types::interface_get(*this).constraints_adjointgradient(X, L, FX, AGX,
                                                                                  data);
        }

        void constraints_jacobian(const Eigen::Ref<const Eigen::VectorXd> &X,
                                  Eigen::Ref<Eigen::VectorXd> FX,
                                  Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                                  Eigen::Ref<Eigen::VectorXi> KKTLocations,
                                  Eigen::Ref<Eigen::VectorXi> KKTClashes,
                                  std::vector<std::mutex> &KKTLocks,
                                  const SolverIndexingData &data) const {
            return rubber_types::interface_get(*this).constraints_jacobian(
                X, FX, KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
        }

        void constraints_jacobian_adjointgradient(
            const Eigen::Ref<const Eigen::VectorXd> &X, const Eigen::Ref<const Eigen::VectorXd> &L,
            Eigen::Ref<Eigen::VectorXd> FX, Eigen::Ref<Eigen::VectorXd> AGX,
            Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
            Eigen::Ref<Eigen::VectorXi> KKTLocations, Eigen::Ref<Eigen::VectorXi> KKTClashes,
            std::vector<std::mutex> &KKTLocks, const SolverIndexingData &data) const {
            return rubber_types::interface_get(*this).constraints_jacobian_adjointgradient(
                X, L, FX, AGX, KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
        }

        void constraints_jacobian_adjointgradient_adjointhessian(
            const Eigen::Ref<const Eigen::VectorXd> &X, const Eigen::Ref<const Eigen::VectorXd> &L,
            Eigen::Ref<Eigen::VectorXd> FX, Eigen::Ref<Eigen::VectorXd> AGX,
            Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
            Eigen::Ref<Eigen::VectorXi> KKTLocations, Eigen::Ref<Eigen::VectorXi> KKTClashes,
            std::vector<std::mutex> &KKTLocks, const SolverIndexingData &data) const {
            return rubber_types::interface_get(*this)
                .constraints_jacobian_adjointgradient_adjointhessian(
                    X, L, FX, AGX, KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
        }

        void getKKTSpace(Eigen::Ref<Eigen::VectorXi> KKTrows, Eigen::Ref<Eigen::VectorXi> KKTcols,
                         int &freeloc, int conoffset, bool dojac, bool dohess,
                         SolverIndexingData &data) {
            return rubber_types::interface_get(*this).getKKTSpace(KKTrows, KKTcols, freeloc,
                                                                  conoffset, dojac, dohess, data);
        }

        int numKKTEles(bool dojac, bool dohess) const {
            return rubber_types::interface_get(*this).numKKTEles(dojac, dohess);
        }
    };
    /////////////////////////////////////////////////////////
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Spec for scalar vector functions that can be used as an objective inside of PSIOPT. Erases all
 * .objective_xxx methods.
 */
struct SolverObjectiveSpec {
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct Concept { // abstract base class for model.
        virtual ~Concept() = default;
        // Your (internal) interface goes here.
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
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <class Holder> struct Model : public Holder, public virtual Concept {
        using Holder::Holder;
        // Pass through to encapsulated value.
        virtual void objective(double ObjScale, const Eigen::Ref<const Eigen::VectorXd> &X,
                               double &Val, const SolverIndexingData &data) const override {
            return rubber_types::model_get(*this).objective(ObjScale, X, Val, data);
        }

        virtual void objective_gradient(double ObjScale, const Eigen::Ref<const Eigen::VectorXd> &X,
                                        double &Val, Eigen::Ref<Eigen::VectorXd> GX,
                                        const SolverIndexingData &data) const override {
            return rubber_types::model_get(*this).objective_gradient(ObjScale, X, Val, GX, data);
        }

        virtual void objective_gradient_hessian(
            double ObjScale, const Eigen::Ref<const Eigen::VectorXd> &X, double &Val,
            Eigen::Ref<Eigen::VectorXd> GX, Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
            Eigen::Ref<Eigen::VectorXi> KKTLocations, Eigen::Ref<Eigen::VectorXi> KKTClashes,
            std::vector<std::mutex> &KKTLocks, const SolverIndexingData &data) const override {
            return rubber_types::model_get(*this).objective_gradient_hessian(
                ObjScale, X, Val, GX, KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
        }
    };
    //////////////////////////////////////////////////////////////////////////////

    template <class Container> struct ExternalInterface : public Container {
        using Container_ = Container;
        using Container_::Container_;

        // Define the external interface. Should match encapsulated type.
        void objective(double ObjScale, const Eigen::Ref<const Eigen::VectorXd> &X, double &Val,
                       const SolverIndexingData &data) const {
            return rubber_types::interface_get(*this).objective(ObjScale, X, Val, data);
        }

        void objective_gradient(double ObjScale, const Eigen::Ref<const Eigen::VectorXd> &X,
                                double &Val, Eigen::Ref<Eigen::VectorXd> GX,
                                const SolverIndexingData &data) const {
            return rubber_types::interface_get(*this).objective_gradient(ObjScale, X, Val, GX,
                                                                         data);
        }

        void objective_gradient_hessian(double ObjScale, const Eigen::Ref<const Eigen::VectorXd> &X,
                                        double &Val, Eigen::Ref<Eigen::VectorXd> GX,
                                        Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                                        Eigen::Ref<Eigen::VectorXi> KKTLocations,
                                        Eigen::Ref<Eigen::VectorXi> KKTClashes,
                                        std::vector<std::mutex> &KKTLocks,
                                        const SolverIndexingData &data) const {
            return rubber_types::interface_get(*this).objective_gradient_hessian(
                ObjScale, X, Val, GX, KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
        }
    };
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Template tool for selecting correct interface spec based of output rows of vector function.
 */
template <int IR, int OR> struct SolverInterfaceSelector {
    using type = SolverConstraintSpec;
};
template <int IR> struct SolverInterfaceSelector<IR, 1> {
    using type = rubber_types::MergeSpecs<SolverConstraintSpec, SolverObjectiveSpec>;
};

/*
 * Forward declare to define self deepcopy and deepcopy of Objective to Constraint
 */
struct ConstraintInterface;
struct ObjectiveInterface;

/*
* ConstraintInterface.
  Combines SolverConstraintSpec,SizableSpec and the ability to deep copy into itself.
  Has explicit constructor taking GenericFunction so that we copy the function
  type erasred by the GenericFunction instance, and hopefully call a type-specific version of it's
methods.
*/
struct ConstraintInterface : rubber_types::TypeErasure<SolverConstraintSpec, SizableSpec,
                                                       DeepCopySpecs<ConstraintInterface>> {
    using Base = rubber_types::TypeErasure<SolverConstraintSpec, SizableSpec,
                                           DeepCopySpecs<ConstraintInterface>>;
    template <class T,
              std::enable_if_t<
                  !std::is_base_of_v<Eigen::EigenBase<std::decay_t<T>>, std::decay_t<T>>, bool> =
                  true>
    ConstraintInterface(const T &t) : Base(t) {}
    template <int IR, int OR>
    ConstraintInterface(const GenericFunction<IR, OR> &t) : Base(t.func) {}
    ConstraintInterface() {}
};

/*
* ObjectiveInterface.
  Combines SolverConstraintSpec,SolverObjectiveSpec,SizableSpec and the ability to deep copy into
itself and a ConsttaintInterface. Has explicit constructor taking GenericFunction so that we copy
the function type erasred by the GenericFunction instance, and hopefully call a type-specific
version of it's methods.
*/

struct ObjectiveInterface
    : rubber_types::TypeErasure<SolverConstraintSpec, SolverObjectiveSpec, SizableSpec,
                                DeepCopySpecs<ObjectiveInterface, ConstraintInterface>> {
    using Base = rubber_types::TypeErasure<SolverConstraintSpec, SolverObjectiveSpec, SizableSpec,
                                           DeepCopySpecs<ObjectiveInterface, ConstraintInterface>>;

    template <class T,
              std::enable_if_t<
                  !std::is_base_of_v<Eigen::EigenBase<std::decay_t<T>>, std::decay_t<T>>, bool> =
                  true>
    ObjectiveInterface(const T &t) : Base(t) {}
    template <int IR> ObjectiveInterface(const GenericFunction<IR, 1> &t) : Base(t.func) {}
    ObjectiveInterface() {}
};

} // namespace Tycho
