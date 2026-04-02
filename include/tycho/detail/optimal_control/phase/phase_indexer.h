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

namespace tycho::oc {

// Solvers types
using tycho::solvers::ConstraintInterface;
using tycho::solvers::NonLinearProgram;
using tycho::solvers::ObjectiveInterface;

// VF types
using vf::ThreadingFlags;

struct PhaseIndexer : ODESize<-1, -1, -1> {
    using VectorXi = Eigen::VectorXi;
    using MatrixXi = Eigen::MatrixXi;

    int static_p_vars_;
    int static_p_vars() const { return this->static_p_vars_; }

    std::shared_ptr<NonLinearProgram> nlp_;

    VectorXi ode_first_state_locs_;
    VectorXi ode_last_state_locs_;
    VectorXi ode_param_locs_;
    VectorXi static_param_locs_;

    int num_defects_;
    int num_states_;
    int num_controls_;

    bool blocked_controls_ = false;
    int blocked_control_start_;

    int defect_cardinal_states_;
    int num_nodal_states_;

    int num_phase_vars_;
    int num_phase_eq_cons_ = 0;
    int num_phase_iq_cons_ = 0;
    int next_phase_eq_con_ = 0;
    int next_phase_iq_con_ = 0;

    int start_obj_ = 0;
    int start_eq_ = 0;
    int start_iq_ = 0;

    int start_eq_cons_ = 0;
    int start_iq_cons_ = 0;

    int num_obj_funs_ = 0;
    int num_eq_funs_ = 0;
    int num_iq_funs_ = 0;

    PhaseIndexer() {}
    PhaseIndexer(int Xv, int Uv, int OPv, int SPv) {
        this->set_xvars(Xv);
        this->set_uvars(Uv);
        this->set_pvars(OPv);
        this->static_p_vars_ = SPv;
    }

    void set_dimensions(int DCS, int Dnum, bool BlockCon) {
        this->num_defects_ = Dnum;
        this->defect_cardinal_states_ = DCS;
        this->num_states_ = (this->defect_cardinal_states_ - 1) * this->num_defects_ + 1;
        this->num_nodal_states_ = this->num_defects_ + 1;

        this->blocked_controls_ = BlockCon;

        if (this->blocked_controls_) {
            this->num_phase_vars_ = this->num_states_ * this->xt_vars() +
                                    this->num_defects_ * this->u_vars() + this->p_vars() +
                                    this->static_p_vars();

            ode_first_state_locs_.setLinSpaced(this->xtu_vars(), 0, this->xtu_vars() - 1);
            ode_first_state_locs_.tail(this->u_vars()) += VectorXi::Constant(
                this->u_vars(), this->xt_vars() * (this->num_states_) - this->xt_vars());

            ode_last_state_locs_ =
                ode_first_state_locs_ +
                VectorXi::Constant(this->xtu_vars(), this->xt_vars() * (this->num_states_ - 1));
            ode_last_state_locs_.tail(this->u_vars()) =
                ode_first_state_locs_.tail(this->u_vars()) +
                VectorXi::Constant(this->u_vars(), this->u_vars() * (this->num_defects_ - 1));

        } else {
            this->num_phase_vars_ =
                this->num_states_ * this->xtu_vars() + this->p_vars() + this->static_p_vars();

            ode_first_state_locs_.setLinSpaced(this->xtu_vars(), 0, this->xtu_vars() - 1);
            ode_last_state_locs_ =
                ode_first_state_locs_ +
                VectorXi::Constant(this->xtu_vars(), this->xtu_vars() * (this->num_states_ - 1));
        }

        ode_param_locs_.setLinSpaced(this->p_vars(), 0, this->p_vars() - 1);
        ode_param_locs_ += Eigen::VectorXi::Constant(
            this->p_vars(), this->num_phase_vars_ - this->p_vars() - this->static_p_vars());

        static_param_locs_.setLinSpaced(this->static_p_vars(), 0, this->static_p_vars() - 1);
        static_param_locs_ += Eigen::VectorXi::Constant(
            this->static_p_vars(), this->num_phase_vars_ - this->static_p_vars());
    }
    void begin_indexing(std::shared_ptr<NonLinearProgram> np, int n, int ep, int ip) {
        this->nlp_ = np;

        this->num_phase_eq_cons_ = 0;
        this->num_phase_iq_cons_ = 0;

        this->ode_first_state_locs_ +=
            Eigen::VectorXi::Constant(this->ode_first_state_locs_.size(), n);
        this->ode_last_state_locs_ +=
            Eigen::VectorXi::Constant(this->ode_last_state_locs_.size(), n);
        this->ode_param_locs_ += Eigen::VectorXi::Constant(this->ode_param_locs_.size(), n);
        this->static_param_locs_ += Eigen::VectorXi::Constant(this->static_param_locs_.size(), n);
        this->next_phase_eq_con_ = ep;
        this->next_phase_iq_con_ = ip;

        this->start_eq_cons_ = ep;
        this->start_iq_cons_ = ip;

        this->start_obj_ = this->nlp_->objectives_.size();
        this->start_eq_ = this->nlp_->equality_constraints_.size();
        this->start_iq_ = this->nlp_->inequality_constraints_.size();

        this->num_obj_funs_ = 0;
        this->num_eq_funs_ = 0;
        this->num_iq_funs_ = 0;
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

    int add_objective(ObjectiveInterface objfun, PhaseRegionFlags sreg,
                      const Eigen::VectorXi &rxtuv, const Eigen::VectorXi &rodepv,
                      const Eigen::VectorXi &rstatpv, ThreadingFlags Tmode);

    int get_xtu_var_loc(int vloc, int State) const {
        int v = 0;
        if (this->blocked_controls_) {
            if (vloc < xt_vars()) {
                v = this->ode_first_state_locs_[vloc] + State * this->xt_vars();
            } else {
                int unum = State / (this->defect_cardinal_states_ - 1);
                if (unum > (this->num_defects_ - 1))
                    unum = this->num_defects_ - 1;
                v = this->ode_first_state_locs_[vloc] + unum * this->u_vars();
            }
        } else {
            v = this->ode_first_state_locs_[vloc] + State * this->xtu_vars();
        }
        return v;
    }

    int get_xtu_var_loc(int vloc, int State, int Defect) const {
        int v = 0;
        if (this->blocked_controls_) {
            if (vloc < xt_vars()) {
                v = this->ode_first_state_locs_[vloc] + State * this->xt_vars();
            } else {
                v = this->ode_first_state_locs_[vloc] + Defect * this->u_vars();
            }
        } else {
            v = this->ode_first_state_locs_[vloc] + State * this->xtu_vars();
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
    void collect_solver_output(const Eigen::VectorXd &Vars,
                               std::vector<Eigen::VectorXd> &ActiveTraj,
                               Eigen::VectorXd &ActiveStaticParams) const;

    std::vector<Eigen::VectorXd> get_func_eq_multipliers(int Gindex,
                                                         const Eigen::VectorXd &EMultphase) const;

    std::vector<Eigen::VectorXd> get_func_iq_multipliers(int Gindex,
                                                         const Eigen::VectorXd &IMultphase) const;

    void print_stats(bool showfuns) const;

    static void Test();
};

} // namespace tycho::oc
