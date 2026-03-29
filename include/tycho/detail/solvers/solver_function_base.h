// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Implements the SolverFunctionBase class which is the base class to
// ConstraintFunction and ObjectiveFunction. Holds an Constraint/ObjectiveInterface type erasure
// class and SolverIndexingData struct. Defines methods for the function to request and reserve KKT
// and RHS space from the solver, and passes relevant arguments to the underlying type erased
// function or index data structure. The two Derived classes ( Constraint/ObjectiveInterface) then
// define the rest of the interface to the type-erased functions constraints and objective methods.
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
// =============================================================================

#pragma once

#include <iostream>
#include <string>

#include <Eigen/Core>

#include "tycho/detail/solvers/indexing_data.h"
#include "tycho/detail/solvers/solver_interface_specs.h"
#include "tycho/detail/typedefs/eigen_types.h"
#include "tycho/detail/vf/core/functional_flags.h"

namespace tycho::solvers {

// Import cross-namespace types used by the solver function base.
using vf::ThreadingFlags;

template <class FuncType> struct SolverFunctionBase {
    using MatrixXi = Eigen::MatrixXi;
    using VectorXi = Eigen::VectorXi;

    FuncType function;
    SolverIndexingData index_data;
    ThreadingFlags thread_mode_ = ThreadingFlags::ByApplication;

    SolverFunctionBase() {}

    void print_data() {
        using std::cout;
        using std::endl;

        cout << "Name: " << this->function.name() << endl << endl;
        cout << "Input  Rows:" << this->function.input_rows() << endl << endl;
        cout << "Output Rows:" << this->function.output_rows() << endl << endl;
        cout << "Thread Policy:" << static_cast<int>(thread_mode_) << endl << endl;

        cout << "v_index_: " << endl << this->index_data.get_v_index() << endl << endl;
        if (this->index_data.cindex_init) {
            cout << "c_index_: " << endl << this->index_data.get_c_index() << endl << endl;
        }
    }

    int num_kkt_elements(bool dojac, bool dohess) {
        return this->function.num_kkt_elements(dojac, dohess) * this->index_data.num_appl();
    }
    int num_con_eles() const { return this->function.output_rows() * this->index_data.num_appl(); }
    int num_grad_eles() const { return this->function.input_rows() * this->index_data.num_appl(); }
    ThreadingFlags get_thread_mode() const { return this->thread_mode_; }
    void get_kkt_space(EigenRef<VectorXi> KKTrows, EigenRef<VectorXi> KKTcols, int &freeloc,
                       int conoffset, bool dojac, bool dohess) {
        this->function.get_kkt_space(KKTrows, KKTcols, freeloc, conoffset, dojac, dohess,
                                     this->index_data);
    }
    void get_gradient_space(EigenRef<VectorXi> GXrows, int &freeloc) {
        this->index_data.get_gradient_space(GXrows, freeloc);
    }
    void get_constraint_space(EigenRef<VectorXi> FXrows, int &freeloc) {
        this->index_data.get_constraint_space(FXrows, freeloc);
    }
};

} // namespace tycho::solvers
