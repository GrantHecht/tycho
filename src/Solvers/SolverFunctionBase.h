// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Implements the SolverFunctionBase class which is the base class to
// ConstraintFunction and ObjectiveFunction. Holds an Constraint/ObjectiveInterface type erasure class
// and SolverIndexingData struct. Defines methods for the function to request and reserve KKT and RHS
// space from the solver, and passes relevant arguments to the underlying type erased function or index
// data structure. The two Derived classes ( Constraint/ObjectiveInterface) then define the rest of the
// interface to the type-erased functions constraints and objective methods.
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> Tycho
//   - Python binding methods (Build(py::module)) moved to src/Bindings/ (PR 2)
//   - pybind11 / pybind11 header references removed
// =============================================================================

#pragma once

#include "VectorFunctions/FunctionalFlags.h"
#include "VectorFunctions/IndexingData.h"
#include "VectorFunctions/VectorFunctionTypeErasure/SolverInterfaceSpecs.h"
#include "pch.h"

namespace Tycho {

template <class FuncType> struct SolverFunctionBase {
    using MatrixXi = Eigen::MatrixXi;
    using VectorXi = Eigen::VectorXi;

    FuncType function;
    SolverIndexingData index_data;
    int ThreadMode = ThreadingFlags::ByApplication;

    SolverFunctionBase() {}

    void print_data() {
        using std::cout;
        using std::endl;

        cout << "Name: " << this->function.name() << endl << endl;
        cout << "Input  Rows:" << this->function.IRows() << endl << endl;
        cout << "Output Rows:" << this->function.ORows() << endl << endl;
        cout << "Thread Policy:" << ThreadMode << endl << endl;

        cout << "Vindex: " << endl << this->index_data.getVindex() << endl << endl;
        if (this->index_data.cindex_init) {
            cout << "Cindex: " << endl << this->index_data.getCindex() << endl << endl;
        }
    }

    int numKKTEles(bool dojac, bool dohess) {
        return this->function.numKKTEles(dojac, dohess) * this->index_data.NumAppl();
    }
    int numConEles() const { return this->function.ORows() * this->index_data.NumAppl(); }
    int numGradEles() const { return this->function.IRows() * this->index_data.NumAppl(); }
    int getThreadMode() const { return this->ThreadMode; }
    void getKKTSpace(EigenRef<VectorXi> KKTrows, EigenRef<VectorXi> KKTcols, int &freeloc,
                     int conoffset, bool dojac, bool dohess) {
        this->function.getKKTSpace(KKTrows, KKTcols, freeloc, conoffset, dojac, dohess,
                                   this->index_data);
    }
    void getGradientSpace(EigenRef<VectorXi> GXrows, int &freeloc) {
        this->index_data.getGradientSpace(GXrows, freeloc);
    }
    void getConstraintSpace(EigenRef<VectorXi> FXrows, int &freeloc) {
        this->index_data.getConstraintSpace(FXrows, freeloc);
    }
};

} // namespace Tycho
