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

/// @internal
/// @brief Maps a phase's discretized variables and constraints into NLP indices.
///
/// Owns the bookkeeping that translates a phase's mesh layout (nodal and defect
/// states, controls, ODE/static parameters) into absolute decision-variable and
/// constraint locations within the shared @c tycho::solvers::NonLinearProgram,
/// and registers each constraint/objective function with the NLP. This is
/// internal transcription plumbing, not user-facing.
/// @endinternal
struct PhaseIndexer : ODESize<-1, -1, -1> {
    using VectorXi = Eigen::VectorXi; ///< @brief Convenience alias for an integer vector.
    using MatrixXi = Eigen::MatrixXi; ///< @brief Convenience alias for an integer matrix.

    int static_p_vars_; ///< Number of static (non-ODE) parameter variables.
    /// @brief Number of static (non-ODE) parameter variables.
    /// @return The static-parameter count.
    int static_p_vars() const { return this->static_p_vars_; }

    std::shared_ptr<NonLinearProgram> nlp_; ///< The NLP being indexed into.

    VectorXi ode_first_state_locs_; ///< NLP locations of the first state's variables.
    VectorXi ode_last_state_locs_;  ///< NLP locations of the last state's variables.
    VectorXi ode_param_locs_;       ///< NLP locations of the ODE-parameter variables.
    VectorXi static_param_locs_;    ///< NLP locations of the static-parameter variables.

    int num_defects_;  ///< Number of defect (collocation) intervals.
    int num_states_;   ///< Total number of discretized states.
    int num_controls_; ///< Total number of control nodes.

    bool blocked_controls_ = false; ///< Whether controls are held block-constant.
    int blocked_control_start_;     ///< NLP start location of the blocked controls.

    int defect_cardinal_states_; ///< Number of cardinal states per defect interval.
    int num_nodal_states_;       ///< Number of nodal (mesh) states.

    int num_phase_vars_;        ///< Total number of decision variables in this phase.
    int num_phase_eq_cons_ = 0; ///< Number of equality-constraint rows in this phase.
    int num_phase_iq_cons_ = 0; ///< Number of inequality-constraint rows in this phase.
    int next_phase_eq_con_ = 0; ///< Next free equality-constraint row.
    int next_phase_iq_con_ = 0; ///< Next free inequality-constraint row.

    int start_obj_ = 0; ///< Index of this phase's first objective in the NLP.
    int start_eq_ = 0;  ///< Index of this phase's first equality function in the NLP.
    int start_iq_ = 0;  ///< Index of this phase's first inequality function in the NLP.

    int start_eq_cons_ = 0; ///< First equality-constraint row of this phase.
    int start_iq_cons_ = 0; ///< First inequality-constraint row of this phase.

    int num_obj_funs_ = 0; ///< Number of objective functions registered.
    int num_eq_funs_ = 0;  ///< Number of equality functions registered.
    int num_iq_funs_ = 0;  ///< Number of inequality functions registered.

    /// @brief Default constructor; dimensions must be set before indexing.
    PhaseIndexer() {}
    /// @brief Construct from the phase's variable dimensions.
    /// @param Xv   Number of state variables.
    /// @param Uv   Number of control variables.
    /// @param OPv  Number of ODE-parameter variables.
    /// @param SPv  Number of static-parameter variables.
    PhaseIndexer(int Xv, int Uv, int OPv, int SPv) {
        this->set_xvars(Xv);
        this->set_uvars(Uv);
        this->set_pvars(OPv);
        this->static_p_vars_ = SPv;
    }

    /// @brief Compute the phase variable layout from the mesh and control settings.
    /// @param DCS       Number of cardinal states per defect interval.
    /// @param Dnum      Number of defect intervals.
    /// @param BlockCon  Whether controls are held block-constant.
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
    /// @brief Bind to an NLP and offset all variable/constraint locations to its base.
    /// @param np  The NLP to index into.
    /// @param n   Decision-variable offset of this phase within the NLP.
    /// @param ep  Equality-constraint-row offset of this phase.
    /// @param ip  Inequality-constraint-row offset of this phase.
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

    /// @internal
    /// @brief Register an equality constraint over a phase region with the NLP.
    /// @param eqfun    The constraint function.
    /// @param sreg     Phase region the function acts on.
    /// @param rxtuv    Relative state/time/control indices the function reads.
    /// @param rodepv   Relative ODE-parameter indices the function reads.
    /// @param rstatpv  Relative static-parameter indices the function reads.
    /// @param Tmode    Threading mode for evaluation.
    /// @return The global index assigned to the registered function.
    /// @endinternal
    int add_equality(ConstraintInterface eqfun, PhaseRegionFlags sreg, const Eigen::VectorXi &rxtuv,
                     const Eigen::VectorXi &rodepv, const Eigen::VectorXi &rstatpv,
                     ThreadingFlags Tmode);
    /// @internal
    /// @brief Register a set of equality constraints partitioned across threads.
    /// @param eqfuns   The partitioned constraint functions.
    /// @param sreg     Phase region the functions act on.
    /// @param rxtuv    Relative state/time/control indices read.
    /// @param rodepv   Relative ODE-parameter indices read.
    /// @param rstatpv  Relative static-parameter indices read.
    /// @param Tmodes   Per-partition threading modes.
    /// @endinternal
    void add_partitioned_equality(const std::vector<ConstraintInterface> &eqfuns,
                                  PhaseRegionFlags sreg, const Eigen::VectorXi &rxtuv,
                                  const Eigen::VectorXi &rodepv, const Eigen::VectorXi &rstatpv,
                                  const std::vector<ThreadingFlags> &Tmodes);

    /// @internal
    /// @brief Register an accumulation (summed-over-path) constraint with the NLP.
    /// @param eqfun    The per-node constraint function whose outputs accumulate.
    /// @param sreg     Phase region the function acts on.
    /// @param rxtuv    Relative state/time/control indices read.
    /// @param rodepv   Relative ODE-parameter indices read.
    /// @param rstatpv  Relative static-parameter indices read.
    /// @param accfun   The function applied to the accumulated result.
    /// @param accpv    Parameter indices read by @p accfun.
    /// @param Tmode    Threading mode for evaluation.
    /// @return The global index assigned to the registered function.
    /// @endinternal
    int add_accumulation(ConstraintInterface eqfun, PhaseRegionFlags sreg,
                         const Eigen::VectorXi &rxtuv, const Eigen::VectorXi &rodepv,
                         const Eigen::VectorXi &rstatpv, ConstraintInterface accfun,
                         const Eigen::VectorXi &accpv, ThreadingFlags Tmode);

    /// @internal
    /// @brief Register an inequality constraint over a phase region with the NLP.
    /// @param iqfun    The constraint function.
    /// @param sreg     Phase region the function acts on.
    /// @param rxtuv    Relative state/time/control indices read.
    /// @param rodepv   Relative ODE-parameter indices read.
    /// @param rstatpv  Relative static-parameter indices read.
    /// @param Tmode    Threading mode for evaluation.
    /// @return The global index assigned to the registered function.
    /// @endinternal
    int add_inequality(ConstraintInterface iqfun, PhaseRegionFlags sreg,
                       const Eigen::VectorXi &rxtuv, const Eigen::VectorXi &rodepv,
                       const Eigen::VectorXi &rstatpv, ThreadingFlags Tmode);

    /// @internal
    /// @brief Register a set of inequality constraints partitioned across threads.
    /// @param iqfuns   The partitioned constraint functions.
    /// @param sreg     Phase region the functions act on.
    /// @param rxtuv    Relative state/time/control indices read.
    /// @param rodepv   Relative ODE-parameter indices read.
    /// @param rstatpv  Relative static-parameter indices read.
    /// @param Tmodes   Per-partition threading modes.
    /// @endinternal
    void add_partitioned_inequality(const std::vector<ConstraintInterface> &iqfuns,
                                    PhaseRegionFlags sreg, const Eigen::VectorXi &rxtuv,
                                    const Eigen::VectorXi &rodepv, const Eigen::VectorXi &rstatpv,
                                    const std::vector<ThreadingFlags> &Tmodes);

    /// @internal
    /// @brief Register an objective over a phase region with the NLP.
    /// @param objfun   The objective function.
    /// @param sreg     Phase region the function acts on.
    /// @param rxtuv    Relative state/time/control indices read.
    /// @param rodepv   Relative ODE-parameter indices read.
    /// @param rstatpv  Relative static-parameter indices read.
    /// @param Tmode    Threading mode for evaluation.
    /// @return The global index assigned to the registered function.
    /// @endinternal
    int add_objective(ObjectiveInterface objfun, PhaseRegionFlags sreg,
                      const Eigen::VectorXi &rxtuv, const Eigen::VectorXi &rodepv,
                      const Eigen::VectorXi &rstatpv, ThreadingFlags Tmode);

    /// @internal
    /// @brief NLP location of a variable at a given discretized state index.
    /// @param vloc   Relative variable location within the packed state vector.
    /// @param State  Discretized state index.
    /// @return Absolute decision-variable location in the NLP.
    /// @endinternal
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

    /// @internal
    /// @brief NLP location of a variable at a given state, resolving blocked controls.
    /// @param vloc    Relative variable location within the packed state vector.
    /// @param State   Discretized state index.
    /// @param Defect  Defect-interval index used for blocked-control variables.
    /// @return Absolute decision-variable location in the NLP.
    /// @endinternal
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

    /// @internal
    /// @brief Build the variable-index and constraint-index matrices for a region.
    /// @param sreg      Phase region a function acts on.
    /// @param rxtuv     Relative state/time/control indices read.
    /// @param rodepv    Relative ODE-parameter indices read.
    /// @param rstatpv   Relative static-parameter indices read.
    /// @param orows     Number of output (constraint) rows per evaluation.
    /// @param NextCLoc  In/out next free constraint-row location; advanced by this call.
    /// @return Array of {variable-index matrix, constraint-index matrix}.
    /// @endinternal
    std::array<Eigen::MatrixXi, 2> make_Vindex_Cindex(PhaseRegionFlags sreg, const VectorXi &rxtuv,
                                                      const VectorXi &rodepv,
                                                      const VectorXi &rstatpv, int orows,
                                                      int &NextCLoc) const;

    /// @internal
    /// @brief Convenience overload of @ref make_Vindex_Cindex that ignores the next-loc output.
    /// @param sreg     Phase region a function acts on.
    /// @param rxtuv    Relative state/time/control indices read.
    /// @param rodepv   Relative ODE-parameter indices read.
    /// @param rstatpv  Relative static-parameter indices read.
    /// @param orows    Number of output (constraint) rows per evaluation.
    /// @return Array of {variable-index matrix, constraint-index matrix}.
    /// @endinternal
    std::array<Eigen::MatrixXi, 2> make_Vindex_Cindex(PhaseRegionFlags sreg, const VectorXi &rxtuv,
                                                      const VectorXi &rodepv,
                                                      const VectorXi &rstatpv, int orows) const {
        int dummy = 0;
        return this->make_Vindex_Cindex(sreg, rxtuv, rodepv, rstatpv, orows, dummy);
    }

    /// @internal
    /// @brief Pack a per-phase trajectory and static params into the NLP decision vector.
    /// @param ActiveTraj          Per-state trajectory vectors of this phase.
    /// @param ActiveStaticParams  Static-parameter values of this phase.
    /// @return The packed decision-variable sub-vector for this phase.
    /// @endinternal
    Eigen::VectorXd make_solver_input(const std::vector<Eigen::VectorXd> &ActiveTraj,
                                      const Eigen::VectorXd &ActiveStaticParams) const;
    /// @internal
    /// @brief Unpack the NLP decision vector back into trajectory and static params.
    /// @param Vars                The solved decision-variable vector.
    /// @param ActiveTraj          Output per-state trajectory vectors.
    /// @param ActiveStaticParams  Output static-parameter values.
    /// @endinternal
    void collect_solver_output(const Eigen::VectorXd &Vars,
                               std::vector<Eigen::VectorXd> &ActiveTraj,
                               Eigen::VectorXd &ActiveStaticParams) const;

    /// @internal
    /// @brief Extract the per-node equality multipliers of one registered function.
    /// @param Gindex      Global index of the function.
    /// @param EMultphase  This phase's full equality-multiplier vector.
    /// @return Per-application multiplier vectors for the function.
    /// @endinternal
    std::vector<Eigen::VectorXd> get_func_eq_multipliers(int Gindex,
                                                         const Eigen::VectorXd &EMultphase) const;

    /// @internal
    /// @brief Extract the per-node inequality multipliers of one registered function.
    /// @param Gindex      Global index of the function.
    /// @param IMultphase  This phase's full inequality-multiplier vector.
    /// @return Per-application multiplier vectors for the function.
    /// @endinternal
    std::vector<Eigen::VectorXd> get_func_iq_multipliers(int Gindex,
                                                         const Eigen::VectorXd &IMultphase) const;

    /// @internal
    /// @brief Print this indexer's variable/constraint counts.
    /// @param showfuns  Whether to also print per-function details.
    /// @endinternal
    void print_stats(bool showfuns) const;

    /// @internal
    /// @brief Self-test entry point for the indexer.
    /// @endinternal
    static void Test();
};

} // namespace tycho::oc
