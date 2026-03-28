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
//   - Namespace renamed: asset -> Tycho
//   - Python binding methods (Build(py::module)) moved to src/Bindings/ (PR 2)
//   - pybind11 header references removed
// =============================================================================

#pragma once

#include "tycho/detail/optimal_control/core/ode_sizes.h"
#include "tycho/detail/optimal_control/core/optimal_control_flags.h"
#include "tycho/detail/solvers/non_linear_program.h"
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
#include "tycho/detail/utils/std_extensions.h"
#include "tycho/detail/utils/math_functions.h"
#include "tycho/detail/utils/type_name.h"
#include "tycho/detail/utils/type_storage.h"
#include "tycho/detail/utils/sizing_helpers.h"
#include "tycho/detail/utils/thread_pool.h"
#include "tycho/detail/utils/flat_map.h"
#include "tycho/detail/utils/function_return_type.h"
#include "tycho/detail/utils/get_core_count.h"
#include "tycho/detail/utils/crtp_base.h"

namespace tycho::oc {

// Solvers types
using tycho::solvers::ConstraintInterface;
using tycho::solvers::ObjectiveInterface;
using tycho::solvers::NonLinearProgram;

struct PhaseIndexer : ODESize<-1, -1, -1> {
    using VectorXi = Eigen::VectorXi;
    using MatrixXi = Eigen::MatrixXi;

    int StaticPVars;
    int StatPVars() const { return this->StaticPVars; }

    std::shared_ptr<NonLinearProgram> nlp;

    VectorXi ODEFirstStateLocs;
    VectorXi ODELastStateLocs;
    VectorXi ODEParamLocs;
    VectorXi StaticParamLocs;

    int num_defects;
    int num_states;
    int num_controls;

    bool BlockedControls = false;
    int BlockedControlStart;

    int DefectCardinalStates;
    int num_nodal_states;

    int num_phase_vars;
    int num_phase_eq_cons = 0;
    int num_phase_iq_cons = 0;
    int next_phase_eq_con = 0;
    int next_phase_iq_con = 0;

    int StartObj = 0;
    int StartEq = 0;
    int StartIq = 0;

    int StartEqCons = 0;
    int StartIqCons = 0;

    int num_obj_funs = 0;
    int num_eq_funs = 0;
    int num_iq_funs = 0;

    PhaseIndexer() {}
    PhaseIndexer(int Xv, int Uv, int OPv, int SPv) {
        this->set_xvars(Xv);
        this->set_uvars(Uv);
        this->set_pvars(OPv);
        this->StaticPVars = SPv;
    }

    void set_dimensions(int DCS, int Dnum, bool BlockCon) {
        this->num_defects = Dnum;
        this->DefectCardinalStates = DCS;
        this->num_states = (this->DefectCardinalStates - 1) * this->num_defects + 1;
        this->num_nodal_states = this->num_defects + 1;

        this->BlockedControls = BlockCon;

        if (this->BlockedControls) {
            this->num_phase_vars = this->num_states * this->XtVars() +
                                 this->num_defects * this->UVars() + this->PVars() +
                                 this->StatPVars();

            ODEFirstStateLocs.setLinSpaced(this->XtUVars(), 0, this->XtUVars() - 1);
            ODEFirstStateLocs.tail(this->UVars()) += VectorXi::Constant(
                this->UVars(), this->XtVars() * (this->num_states) - this->XtVars());

            ODELastStateLocs =
                ODEFirstStateLocs +
                VectorXi::Constant(this->XtUVars(), this->XtVars() * (this->num_states - 1));
            ODELastStateLocs.tail(this->UVars()) =
                ODEFirstStateLocs.tail(this->UVars()) +
                VectorXi::Constant(this->UVars(), this->UVars() * (this->num_defects - 1));

        } else {
            this->num_phase_vars =
                this->num_states * this->XtUVars() + this->PVars() + this->StatPVars();

            ODEFirstStateLocs.setLinSpaced(this->XtUVars(), 0, this->XtUVars() - 1);
            ODELastStateLocs =
                ODEFirstStateLocs +
                VectorXi::Constant(this->XtUVars(), this->XtUVars() * (this->num_states - 1));
        }

        ODEParamLocs.setLinSpaced(this->PVars(), 0, this->PVars() - 1);
        ODEParamLocs += Eigen::VectorXi::Constant(
            this->PVars(), this->num_phase_vars - this->PVars() - this->StatPVars());

        StaticParamLocs.setLinSpaced(this->StatPVars(), 0, this->StatPVars() - 1);
        StaticParamLocs +=
            Eigen::VectorXi::Constant(this->StatPVars(), this->num_phase_vars - this->StatPVars());
    }
    void begin_indexing(std::shared_ptr<NonLinearProgram> np, int n, int ep, int ip) {
        this->nlp = np;

        this->num_phase_eq_cons = 0;
        this->num_phase_iq_cons = 0;

        this->ODEFirstStateLocs += Eigen::VectorXi::Constant(this->ODEFirstStateLocs.size(), n);
        this->ODELastStateLocs += Eigen::VectorXi::Constant(this->ODELastStateLocs.size(), n);
        this->ODEParamLocs += Eigen::VectorXi::Constant(this->ODEParamLocs.size(), n);
        this->StaticParamLocs += Eigen::VectorXi::Constant(this->StaticParamLocs.size(), n);
        this->next_phase_eq_con = ep;
        this->next_phase_iq_con = ip;

        this->StartEqCons = ep;
        this->StartIqCons = ip;

        this->StartObj = this->nlp->Objectives.size();
        this->StartEq = this->nlp->EqualityConstraints.size();
        this->StartIq = this->nlp->InequalityConstraints.size();

        this->num_obj_funs = 0;
        this->num_eq_funs = 0;
        this->num_iq_funs = 0;
    }

    int add_equality(ConstraintInterface eqfun, PhaseRegionFlags sreg, const Eigen::VectorXi &rxtuv,
                    const Eigen::VectorXi &rodepv, const Eigen::VectorXi &rstatpv,
                    ThreadingFlags Tmode);
    void add_partitioned_equality(const std::vector<ConstraintInterface> &eqfuns,
                                PhaseRegionFlags sreg, const Eigen::VectorXi &rxtuv,
                                const Eigen::VectorXi &rodepv, const Eigen::VectorXi &rstatpv,
                                const std::vector<ThreadingFlags> &Tmodes);

    int add_accumulation(ConstraintInterface eqfun, PhaseRegionFlags sreg,
                        const Eigen::VectorXi &rxtuv, const Eigen::VectorXi &rodepv,
                        const Eigen::VectorXi &rstatpv, ConstraintInterface accfun,
                        const Eigen::VectorXi &accpv, ThreadingFlags Tmode);

    int add_inequality(ConstraintInterface iqfun, PhaseRegionFlags sreg,
                      const Eigen::VectorXi &rxtuv, const Eigen::VectorXi &rodepv,
                      const Eigen::VectorXi &rstatpv, ThreadingFlags Tmode);

    void add_partitioned_inequality(const std::vector<ConstraintInterface> &iqfuns,
                                  PhaseRegionFlags sreg, const Eigen::VectorXi &rxtuv,
                                  const Eigen::VectorXi &rodepv, const Eigen::VectorXi &rstatpv,
                                  const std::vector<ThreadingFlags> &Tmodes);

    int add_objective(ObjectiveInterface objfun, PhaseRegionFlags sreg, const Eigen::VectorXi &rxtuv,
                     const Eigen::VectorXi &rodepv, const Eigen::VectorXi &rstatpv,
                     ThreadingFlags Tmode);

    int get_xtu_var_loc(int vloc, int State) const {
        int v = 0;
        if (this->BlockedControls) {
            if (vloc < XtVars()) {
                v = this->ODEFirstStateLocs[vloc] + State * this->XtVars();
            } else {
                int unum = State / (this->DefectCardinalStates - 1);
                if (unum > (this->num_defects - 1))
                    unum = this->num_defects - 1;
                v = this->ODEFirstStateLocs[vloc] + unum * this->UVars();
            }
        } else {
            v = this->ODEFirstStateLocs[vloc] + State * this->XtUVars();
        }
        return v;
    }

    int get_xtu_var_loc(int vloc, int State, int Defect) const {
        int v = 0;
        if (this->BlockedControls) {
            if (vloc < XtVars()) {
                v = this->ODEFirstStateLocs[vloc] + State * this->XtVars();
            } else {
                v = this->ODEFirstStateLocs[vloc] + Defect * this->UVars();
            }
        } else {
            v = this->ODEFirstStateLocs[vloc] + State * this->XtUVars();
        }
        return v;
    }

    std::array<Eigen::MatrixXi, 2> make_Vindex_Cindex(PhaseRegionFlags sreg, const VectorXi &rxtuv,
                                                      const VectorXi &rodepv,
                                                      const VectorXi &rstatpv, int orows,
                                                      int &NextCLoc) const;

    std::array<Eigen::MatrixXi, 2> make_Vindex_Cindex(PhaseRegionFlags sreg, const VectorXi &rxtuv,
                                                      const VectorXi &rodepv,
                                                      const VectorXi &rstatpv, int orows) const {
        int dummy = 0;
        return this->make_Vindex_Cindex(sreg, rxtuv, rodepv, rstatpv, orows, dummy);
    }

    Eigen::VectorXd make_solver_input(const std::vector<Eigen::VectorXd> &ActiveTraj,
                                    const Eigen::VectorXd &ActiveStaticParams) const;
    void collect_solver_output(const Eigen::VectorXd &Vars, std::vector<Eigen::VectorXd> &ActiveTraj,
                             Eigen::VectorXd &ActiveStaticParams) const;

    std::vector<Eigen::VectorXd> get_func_eq_multipliers(int Gindex,
                                                      const Eigen::VectorXd &EMultphase) const;

    std::vector<Eigen::VectorXd> get_func_iq_multipliers(int Gindex,
                                                      const Eigen::VectorXd &IMultphase) const;

    void print_stats(bool showfuns) const;

    static void Test();
};

} // namespace tycho::oc

