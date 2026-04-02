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

#include "tycho/detail/vf/core/dense_function_base.h"

namespace tycho::vf {

template <class Derived, int IR>
struct DenseScalarFunctionBase : DenseFunctionBase<Derived, IR, 1> {
    using Base = DenseFunctionBase<Derived, IR, 1>;
    DENSE_FUNCTION_BASE_TYPES(Base);

    void objective(double ObjScale, ConstEigenRef<Eigen::VectorXd> X, double &Val,
                   const tycho::solvers::SolverIndexingData &data) const {
        Input<double> x(this->input_rows());
        Output<double> fx(1);

        for (int V = 0; V < data.num_appl(); V++) {
            this->gather_input(X, x, V, data);
            fx.setZero();
            this->derived().compute(x, fx);
            Val += fx[0] * ObjScale;
        }
    }
    void objective_gradient(double ObjScale, ConstEigenRef<Eigen::VectorXd> X, double &Val,
                            EigenRef<Eigen::VectorXd> GX,
                            const tycho::solvers::SolverIndexingData &data) const {
        Input<double> x(this->input_rows());
        Output<double> fx(1);
        Jacobian<double> jx(1, this->input_rows());
        Eigen::Map<Input<double>> gx(NULL, this->input_rows());

        for (int V = 0; V < data.num_appl(); V++) {
            this->gather_input(X, x, V, data);
            new (&gx) Eigen::Map<Input<double>>(GX.data() + data.inner_gradient_starts_[V],
                                                this->input_rows());
            fx.setZero();
            jx.setZero();
            gx.setZero();

            this->derived().compute_jacobian(x, fx, jx);
            Val += fx[0] * ObjScale;
            gx = jx.transpose() * ObjScale;
        }
    }
    void objective_gradient_hessian(double ObjScale, ConstEigenRef<Eigen::VectorXd> X, double &Val,
                                    EigenRef<Eigen::VectorXd> GX,
                                    Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                                    EigenRef<Eigen::VectorXi> KKTLocations,
                                    EigenRef<Eigen::VectorXi> KKTClashes,
                                    std::vector<std::mutex> &KKTLocks,
                                    const tycho::solvers::SolverIndexingData &data) const {
        Input<double> x(this->input_rows());
        Output<double> fx(1);
        Jacobian<double> jx(1, this->input_rows());
        Eigen::Map<Input<double>> gx(NULL, this->input_rows());
        Hessian<double> hx(this->input_rows(), this->input_rows());
        Output<double> lm(1);
        lm[0] = ObjScale;

        for (int V = 0; V < data.num_appl(); V++) {
            this->gather_input(X, x, V, data);
            new (&gx) Eigen::Map<Input<double>>(GX.data() + data.inner_gradient_starts_[V],
                                                this->input_rows());

            fx.setZero();
            jx.setZero();
            gx.setZero();
            hx.setZero();

            this->derived().compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, gx, hx, lm);

            Val += fx[0] * ObjScale;
            this->kkt_fill_hess(V, hx, KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
        }
    }
    //////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////

  protected:
    // double Scale = 1.0;
    void kkt_fill_hess(int Apl, const Hessian<double> &hx,
                       Eigen::SparseMatrix<double, Eigen::RowMajor> &KKTmat,
                       EigenRef<Eigen::VectorXi> KKTLocs, EigenRef<Eigen::VectorXi> VarClashes,
                       std::vector<std::mutex> &ClashLocks,
                       const tycho::solvers::SolverIndexingData &data) const {
        int freeloc = data.inner_kkt_starts_[Apl];
        double *mpt = KKTmat.valuePtr();
        const int *lpt = KKTLocs.data();
        int ActiveVar;

        auto Lock = [&](int var) {
            if (VarClashes[var] == -1) {
                //// uncontested
            } else {
                /// contested lock mutex
                ClashLocks[VarClashes[var]].lock();
            }
        };
        auto UnLock = [&](int var) {
            if (VarClashes[var] == -1) {
                //// uncontested
            } else {
                /// contested unlock mutex
                ClashLocks[VarClashes[var]].unlock();
            }
        };

        const int IRR = (Base::IRC > 0) ? Base::IRC : this->input_rows();

        for (int i = 0; i < IRR; i++) {
            ActiveVar = data.v_loc(i, Apl);
            Lock(ActiveVar);
            ///// insert hessian column symetrically
            for (int j = i; j < IRR; j++) {
                this->derived().add_hessian_elem(hx(j, i), j, i, mpt, lpt, freeloc);
            }
            ///////////////////////////////////////////////////////////////////////////////
            UnLock(ActiveVar);
        }
    }
};
} // namespace tycho::vf
