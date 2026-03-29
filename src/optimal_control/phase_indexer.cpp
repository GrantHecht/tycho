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

#include "tycho/detail/optimal_control/phase/phase_indexer.h"

using tycho::solvers::ConstraintFunction;
using tycho::solvers::ObjectiveFunction;

int tycho::oc::PhaseIndexer::add_equality(ConstraintInterface eqfun, PhaseRegionFlags sreg,
                                          const Eigen::VectorXi &rxtuv,
                                          const Eigen::VectorXi &rodepv,
                                          const Eigen::VectorXi &rstatpv, ThreadingFlags Tmode) {
    int index = this->nlp->equality_constraints_.size();
    int temp = this->next_phase_eq_con;
    auto vincin = this->make_Vindex_Cindex(sreg, rxtuv, rodepv, rstatpv, eqfun.output_rows(),
                                           this->next_phase_eq_con);
    this->nlp->equality_constraints_.emplace_back(ConstraintFunction(eqfun, vincin[0], vincin[1]));
    this->nlp->equality_constraints_.back().thread_mode_ = Tmode;
    this->num_phase_eq_cons += this->next_phase_eq_con - temp;
    this->num_eq_funs++;
    return index;
}

void tycho::oc::PhaseIndexer::add_partitioned_equality(
    const std::vector<ConstraintInterface> &eqfuns, PhaseRegionFlags sreg,
    const Eigen::VectorXi &rxtuv, const Eigen::VectorXi &rodepv, const Eigen::VectorXi &rstatpv,
    const std::vector<ThreadingFlags> &Tmodes) {
    int index = this->nlp->equality_constraints_.size();
    int temp = this->next_phase_eq_con;
    auto vincin = this->make_Vindex_Cindex(sreg, rxtuv, rodepv, rstatpv, eqfuns[0].output_rows(),
                                           this->next_phase_eq_con);
    for (int i = 0; i < eqfuns.size(); i++) {
        MatrixXi vindex = vincin[0].col(i);
        MatrixXi cindex = vincin[1].col(i);

        this->nlp->equality_constraints_.emplace_back(
            ConstraintFunction(eqfuns[i], vindex, cindex));
        this->nlp->equality_constraints_.back().thread_mode_ = Tmodes[i];
        this->num_eq_funs++;
    }

    this->num_phase_eq_cons += this->next_phase_eq_con - temp;
}

int tycho::oc::PhaseIndexer::add_accumulation(ConstraintInterface eqfun, PhaseRegionFlags sreg,
                                              const Eigen::VectorXi &rxtuv,
                                              const Eigen::VectorXi &rodepv,
                                              const Eigen::VectorXi &rstatpv,
                                              ConstraintInterface accfun,
                                              const Eigen::VectorXi &accpv, ThreadingFlags Tmode)

{
    int index = this->nlp->equality_constraints_.size();
    int temp = this->next_phase_eq_con;

    Eigen::VectorXi empty(0);
    empty.resize(0);

    auto vincin_acc = this->make_Vindex_Cindex(PhaseRegionFlags::Params, empty, empty, accpv,
                                               accfun.output_rows(), this->next_phase_eq_con);
    this->nlp->equality_constraints_.emplace_back(
        ConstraintFunction(accfun, vincin_acc[0], vincin_acc[1]));
    this->nlp->equality_constraints_.back().thread_mode_ = Tmode;
    this->nlp->equality_constraints_.back().index_data.unique_constraints = false;

    int dummy = 0;
    auto vincin =
        this->make_Vindex_Cindex(sreg, rxtuv, rodepv, rstatpv, eqfun.output_rows(), dummy);
    for (int i = 0; i < vincin[1].cols(); i++) {
        vincin[1].col(i) = vincin_acc[1].col(0);
    }

    this->nlp->equality_constraints_.emplace_back(ConstraintFunction(eqfun, vincin[0], vincin[1]));
    this->nlp->equality_constraints_.back().thread_mode_ = Tmode;
    this->nlp->equality_constraints_.back().index_data.unique_constraints = false;

    this->num_phase_eq_cons += this->next_phase_eq_con - temp;
    this->num_eq_funs += 2;
    return index + 1;
}

int tycho::oc::PhaseIndexer::add_inequality(ConstraintInterface iqfun, PhaseRegionFlags sreg,
                                            const Eigen::VectorXi &rxtuv,
                                            const Eigen::VectorXi &rodepv,
                                            const Eigen::VectorXi &rstatpv, ThreadingFlags Tmode) {
    int index = this->nlp->inequality_constraints_.size();
    int temp = this->next_phase_iq_con;
    auto vincin = this->make_Vindex_Cindex(sreg, rxtuv, rodepv, rstatpv, iqfun.output_rows(),
                                           this->next_phase_iq_con);
    this->nlp->inequality_constraints_.emplace_back(
        ConstraintFunction(iqfun, vincin[0], vincin[1]));
    this->nlp->inequality_constraints_.back().thread_mode_ = Tmode;
    this->num_phase_iq_cons += this->next_phase_iq_con - temp;
    this->num_iq_funs++;
    return index;
}

void tycho::oc::PhaseIndexer::add_partitioned_inequality(
    const std::vector<ConstraintInterface> &iqfuns, PhaseRegionFlags sreg,
    const Eigen::VectorXi &rxtuv, const Eigen::VectorXi &rodepv, const Eigen::VectorXi &rstatpv,
    const std::vector<ThreadingFlags> &Tmodes) {
    int index = this->nlp->inequality_constraints_.size();
    int temp = this->next_phase_iq_con;
    auto vincin = this->make_Vindex_Cindex(sreg, rxtuv, rodepv, rstatpv, iqfuns[0].output_rows(),
                                           this->next_phase_iq_con);
    for (int i = 0; i < iqfuns.size(); i++) {
        MatrixXi vindex = vincin[0].col(i);
        MatrixXi cindex = vincin[1].col(i);

        this->nlp->inequality_constraints_.emplace_back(
            ConstraintFunction(iqfuns[i], vindex, cindex));
        this->nlp->inequality_constraints_.back().thread_mode_ = Tmodes[i];
        this->num_iq_funs++;
    }

    this->num_phase_iq_cons += this->next_phase_iq_con - temp;
}

int tycho::oc::PhaseIndexer::add_objective(ObjectiveInterface objfun, PhaseRegionFlags sreg,
                                           const Eigen::VectorXi &rxtuv,
                                           const Eigen::VectorXi &rodepv,
                                           const Eigen::VectorXi &rstatpv, ThreadingFlags Tmode)

{
    int index = this->nlp->objectives_.size();
    auto vincin = this->make_Vindex_Cindex(sreg, rxtuv, rodepv, rstatpv, objfun.output_rows());
    this->nlp->objectives_.emplace_back(ObjectiveFunction(objfun, vincin[0]));
    this->nlp->objectives_.back().thread_mode_ = Tmode;
    this->num_obj_funs++;
    return index;
}

std::array<Eigen::MatrixXi, 2>
tycho::oc::PhaseIndexer::make_Vindex_Cindex(PhaseRegionFlags sreg, const VectorXi &rxtuv,
                                            const VectorXi &rodepv, const VectorXi &rstatpv,
                                            int orows, int &NextCLoc) const {
    int xsize = rxtuv.size();
    int opsize = rodepv.size();
    int spsize = rstatpv.size();
    int x1len = xsize + opsize + spsize;
    int x2len = 2 * xsize + opsize + spsize;
    int x3len = 3 * xsize + opsize + spsize;

    int xblen = DefectCardinalStates * xsize + opsize + spsize;

    int blockdeflen = DefectCardinalStates * this->xt_vars() + this->u_vars() + this->p_vars();

    int xpblen =
        DefectCardinalStates * xsize + (DefectCardinalStates - 1) * xsize + opsize + spsize;

    Eigen::MatrixXi v_index_;
    Eigen::MatrixXi c_index_;

    bool IsOnlyControl = true;
    for (int i = 0; i < rxtuv.size(); i++) {
        if (rxtuv[i] < this->xt_vars())
            IsOnlyControl = false;
    }

    auto VinLoopImpl = [&](int &loc, int col, const VectorXi &Locs, const VectorXi &Vars) {
        for (int i = 0; i < Vars.size(); i++) {
            v_index_(loc, col) = Locs[Vars[i]];
            loc++;
        }
    };
    auto PathLoopImpl = [&](int &loc, int col, int state) {
        for (int i = 0; i < rxtuv.size(); i++) {
            v_index_(loc, col) = this->get_xtu_var_loc(rxtuv[i], state);
            loc++;
        }
    };
    auto DefectPathLoopImpl = [&](int &loc, int col, int state, int Defect) {
        for (int i = 0; i < rxtuv.size(); i++) {
            v_index_(loc, col) = this->get_xtu_var_loc(rxtuv[i], state, Defect);
            loc++;
        }
    };

    auto CinSet = [&](int cols) {
        c_index_.resize(orows, cols);
        if (orows > 0) {
            for (int i = 0; i < cols; i++) {
                for (int j = 0; j < orows; j++) {
                    c_index_(j, i) = NextCLoc;
                    NextCLoc++;
                }
            }
        }
    };

    switch (sreg) {
    case PhaseRegionFlags::ODEParams:
    case PhaseRegionFlags::StaticParams:
    case PhaseRegionFlags::Params: {
        int loc = 0;
        CinSet(1);
        v_index_.resize(opsize + spsize, 1);
        VinLoopImpl(loc, 0, this->ODEParamLocs, rodepv);
        VinLoopImpl(loc, 0, this->StaticParamLocs, rstatpv);

    } break;
    case PhaseRegionFlags::Front: {
        int loc = 0;
        CinSet(1);
        v_index_.resize(x1len, 1);

        VinLoopImpl(loc, 0, this->ODEFirstStateLocs, rxtuv);
        VinLoopImpl(loc, 0, this->ODEParamLocs, rodepv);
        VinLoopImpl(loc, 0, this->StaticParamLocs, rstatpv);
    } break;
    case PhaseRegionFlags::Back: {
        int loc = 0;
        CinSet(1);
        v_index_.resize(x1len, 1);

        VinLoopImpl(loc, 0, this->ODELastStateLocs, rxtuv);
        VinLoopImpl(loc, 0, this->ODEParamLocs, rodepv);
        VinLoopImpl(loc, 0, this->StaticParamLocs, rstatpv);
    } break;
    case PhaseRegionFlags::FrontandBack: {
        int loc = 0;
        CinSet(1);
        v_index_.resize(x2len, 1);

        VinLoopImpl(loc, 0, this->ODEFirstStateLocs, rxtuv);
        VinLoopImpl(loc, 0, this->ODELastStateLocs, rxtuv);
        VinLoopImpl(loc, 0, this->ODEParamLocs, rodepv);
        VinLoopImpl(loc, 0, this->StaticParamLocs, rstatpv);
    } break;
    case PhaseRegionFlags::BackandFront: {
        int loc = 0;
        CinSet(1);
        v_index_.resize(x2len, 1);

        VinLoopImpl(loc, 0, this->ODELastStateLocs, rxtuv);
        VinLoopImpl(loc, 0, this->ODEFirstStateLocs, rxtuv);
        VinLoopImpl(loc, 0, this->ODEParamLocs, rodepv);
        VinLoopImpl(loc, 0, this->StaticParamLocs, rstatpv);
    } break;
    case PhaseRegionFlags::Path: {
        if (IsOnlyControl && this->BlockedControls) {
            CinSet(this->num_defects);
            v_index_.resize(x1len, this->num_defects);

            for (int i = 0; i < this->num_defects; i++) {
                int loc = 0;
                PathLoopImpl(loc, i, i * (this->DefectCardinalStates - 1));
                VinLoopImpl(loc, i, this->ODEParamLocs, rodepv);
                VinLoopImpl(loc, i, this->StaticParamLocs, rstatpv);
            }
        } else {
            CinSet(this->num_states);
            v_index_.resize(x1len, this->num_states);

            for (int i = 0; i < this->num_states; i++) {
                int loc = 0;
                PathLoopImpl(loc, i, i);
                VinLoopImpl(loc, i, this->ODEParamLocs, rodepv);
                VinLoopImpl(loc, i, this->StaticParamLocs, rstatpv);
            }
        }
    } break;
    case PhaseRegionFlags::InnerPath: {
        if (IsOnlyControl && this->BlockedControls) {
            CinSet(this->num_defects - 2);
            v_index_.resize(x1len, this->num_defects - 2);

            for (int i = 1; i < (this->num_defects - 1); i++) {
                int loc = 0;
                PathLoopImpl(loc, i - 1, i * (this->DefectCardinalStates - 1));
                VinLoopImpl(loc, i - 1, this->ODEParamLocs, rodepv);
                VinLoopImpl(loc, i - 1, this->StaticParamLocs, rstatpv);
            }
        } else {
            CinSet(this->num_states - 2);
            v_index_.resize(x1len, this->num_states - 2);

            for (int i = 1; i < (this->num_states - 1); i++) {
                int loc = 0;
                PathLoopImpl(loc, i - 1, i);
                VinLoopImpl(loc, i - 1, this->ODEParamLocs, rodepv);
                VinLoopImpl(loc, i - 1, this->StaticParamLocs, rstatpv);
            }
        }
    } break;
    case PhaseRegionFlags::NodalPath: {
        if (IsOnlyControl && this->BlockedControls) {
            CinSet(this->num_defects);
            v_index_.resize(x1len, this->num_defects);

            for (int i = 0; i < this->num_defects; i++) {
                int loc = 0;
                PathLoopImpl(loc, i, i * (this->DefectCardinalStates - 1));
                VinLoopImpl(loc, i, this->ODEParamLocs, rodepv);
                VinLoopImpl(loc, i, this->StaticParamLocs, rstatpv);
            }
        } else {
            CinSet(this->num_defects + 1);
            v_index_.resize(x1len, this->num_defects + 1);

            for (int i = 0; i < (this->num_defects + 1); i++) {
                int loc = 0;
                PathLoopImpl(loc, i, i * (this->DefectCardinalStates - 1));
                VinLoopImpl(loc, i, this->ODEParamLocs, rodepv);
                VinLoopImpl(loc, i, this->StaticParamLocs, rstatpv);
            }
        }
    } break;
    case PhaseRegionFlags::PairWisePath: {
        if (IsOnlyControl && this->BlockedControls) {
            CinSet(this->num_defects - 1);
            v_index_.resize(x2len, this->num_defects - 1);

            for (int i = 0; i < this->num_defects - 1; i++) {
                int loc = 0;
                PathLoopImpl(loc, i, i * (this->DefectCardinalStates - 1));
                PathLoopImpl(loc, i, (i + 1) * (this->DefectCardinalStates - 1));
                VinLoopImpl(loc, i, this->ODEParamLocs, rodepv);
                VinLoopImpl(loc, i, this->StaticParamLocs, rstatpv);
            }
        } else {
            CinSet(this->num_states - 1);
            v_index_.resize(x2len, this->num_states - 1);

            for (int i = 0; i < (this->num_states - 1); i++) {
                int loc = 0;
                PathLoopImpl(loc, i, i);
                PathLoopImpl(loc, i, (i + 1));
                VinLoopImpl(loc, i, this->ODEParamLocs, rodepv);
                VinLoopImpl(loc, i, this->StaticParamLocs, rstatpv);
            }
        }
    } break;
    case PhaseRegionFlags::FrontNodalBackPath: {
        if (IsOnlyControl && this->BlockedControls) {
            CinSet(this->num_defects - 2);
            v_index_.resize(x3len, this->num_defects - 2);

            for (int i = 1; i < (this->num_defects - 1); i++) {
                int loc = 0;
                VinLoopImpl(loc, i - 1, this->ODEFirstStateLocs, rxtuv);
                PathLoopImpl(loc, i - 1, i * (this->DefectCardinalStates - 1));
                VinLoopImpl(loc, i - 1, this->ODELastStateLocs, rxtuv);
                VinLoopImpl(loc, i, this->ODEParamLocs, rodepv);
                VinLoopImpl(loc, i, this->StaticParamLocs, rstatpv);
            }
        } else {
            CinSet(this->num_defects - 1);
            v_index_.resize(x3len, this->num_defects - 1);

            for (int i = 1; i < (this->num_defects); i++) {
                int loc = 0;
                VinLoopImpl(loc, i - 1, this->ODEFirstStateLocs, rxtuv);
                PathLoopImpl(loc, i - 1, i * (this->DefectCardinalStates - 1));
                VinLoopImpl(loc, i - 1, this->ODELastStateLocs, rxtuv);
                VinLoopImpl(loc, i - 1, this->ODEParamLocs, rodepv);
                VinLoopImpl(loc, i - 1, this->StaticParamLocs, rstatpv);
            }
        }
    } break;
    case PhaseRegionFlags::DefectPath: {
        CinSet(this->num_defects);
        v_index_.resize(xblen, this->num_defects);
        for (int i = 0; i < this->num_defects; i++) {
            int loc = 0;
            for (int j = 0; j < this->DefectCardinalStates; j++) {
                DefectPathLoopImpl(loc, i, i * (this->DefectCardinalStates - 1) + j, i);
            }
            VinLoopImpl(loc, i, this->ODEParamLocs, rodepv);
            VinLoopImpl(loc, i, this->StaticParamLocs, rstatpv);
        }
    } break;
    case PhaseRegionFlags::BlockDefectPath: {
        CinSet(this->num_defects);
        v_index_.resize(blockdeflen, this->num_defects);
        for (int i = 0; i < this->num_defects; i++) {
            int loc = 0;
            for (int j = 0; j < this->DefectCardinalStates; j++) {
                for (int k = 0; k < this->xt_vars(); k++) {
                    v_index_(loc, i) =
                        this->get_xtu_var_loc(k, i * (this->DefectCardinalStates - 1) + j, i);
                    loc++;
                }
            }
            for (int k = this->xt_vars(); k < this->xtu_vars(); k++) {
                v_index_(loc, i) =
                    this->get_xtu_var_loc(k, i * (this->DefectCardinalStates - 1), i);
                loc++;
            }
            VinLoopImpl(loc, i, this->ODEParamLocs, rodepv);
            // VinLoopImpl(loc, i, this->StaticParamLocs, rstatpv);
        }
    } break;
    case PhaseRegionFlags::DefectPairWisePath: {
        CinSet(this->num_defects - 1);
        v_index_.resize(xpblen, this->num_defects - 1);
        for (int i = 0; i < this->num_defects - 1; i++) {
            int loc = 0;
            for (int j = 0; j < 2 * this->DefectCardinalStates - 1; j++) {
                DefectPathLoopImpl(loc, i, i * (this->DefectCardinalStates - 1) + j, i);
            }
            VinLoopImpl(loc, i, this->ODEParamLocs, rodepv);
            VinLoopImpl(loc, i, this->StaticParamLocs, rstatpv);
        }
    } break;
    default: {
        throw std::invalid_argument("Indexing for requested phase region has not been implemented");
    }
    }

    return std::array<Eigen::MatrixXi, 2>{v_index_, c_index_};
}

Eigen::VectorXd
tycho::oc::PhaseIndexer::make_solver_input(const std::vector<Eigen::VectorXd> &ActiveTraj,
                                           const Eigen::VectorXd &ActiveStaticParams) const {
    Eigen::VectorXd Vars(this->num_phase_vars);

    if (this->BlockedControls) {
        for (int i = 0; i < this->num_states; i++) {
            Vars.segment(i * this->xt_vars(), this->xt_vars()) =
                ActiveTraj[i].head(this->xt_vars());
        }
        for (int i = 0; i < this->num_defects; i++) {
            Vars.segment(this->num_states * this->xt_vars() + i * this->u_vars(), this->u_vars()) =
                ActiveTraj[i * (this->DefectCardinalStates - 1)].segment(this->xt_vars(),
                                                                         this->u_vars());
        }

        if (this->p_vars() > 0) {
            Vars.segment(this->num_states * this->xt_vars() + this->num_defects * this->u_vars(),
                         this->p_vars()) = ActiveTraj[0].tail(this->p_vars());
        }
        if (this->StatPVars() > 0) {
            Vars.tail(this->StatPVars()) = ActiveStaticParams;
        }
    } else {
        for (int i = 0; i < this->num_states; i++) {
            Vars.segment(i * this->xtu_vars(), this->xtu_vars()) =
                ActiveTraj[i].head(this->xtu_vars());
        }
        if (this->p_vars() > 0) {
            Vars.segment(this->num_states * this->xtu_vars(), this->p_vars()) =
                ActiveTraj[0].tail(this->p_vars());
        }
        if (this->StatPVars() > 0) {
            Vars.segment(this->num_states * this->xtu_vars() + this->p_vars(), this->StatPVars()) =
                ActiveStaticParams;
        }
    }

    return Vars;
}

void tycho::oc::PhaseIndexer::collect_solver_output(const Eigen::VectorXd &Vars,
                                                    std::vector<Eigen::VectorXd> &ActiveTraj,
                                                    Eigen::VectorXd &ActiveStaticParams) const {
    Eigen::VectorXd opv(this->p_vars());

    if (this->BlockedControls) {
        if (this->p_vars() > 0) {
            opv = Vars.segment(this->num_states * this->xt_vars() +
                                   this->num_defects * this->u_vars(),
                               this->p_vars());
        }
        if (this->StatPVars() > 0) {
            ActiveStaticParams = Vars.tail(this->StatPVars());
        }

        for (int i = 0; i < this->num_states; i++) {
            ActiveTraj[i].head(this->xt_vars()) =
                Vars.segment(i * this->xt_vars(), this->xt_vars());
            int unum = i / (this->DefectCardinalStates - 1);
            if (unum > (this->num_defects - 1))
                unum = this->num_defects - 1;

            ActiveTraj[i].segment(this->xt_vars(), this->u_vars()) = Vars.segment(
                this->num_states * this->xt_vars() + unum * this->u_vars(), this->u_vars());

            if (this->p_vars() > 0) {
                ActiveTraj[i].tail(this->p_vars()) = opv;
            }
        }

    } else {
        if (this->p_vars() > 0) {
            opv = Vars.segment(this->num_states * this->xtu_vars(), this->p_vars());
        }
        if (this->StatPVars() > 0) {
            ActiveStaticParams = Vars.segment(this->num_states * this->xtu_vars() + this->p_vars(),
                                              this->StatPVars());
        }
        for (int i = 0; i < this->num_states; i++) {
            ActiveTraj[i].head(this->xtu_vars()) =
                Vars.segment(i * this->xtu_vars(), this->xtu_vars());
            if (this->p_vars() > 0) {
                ActiveTraj[i].tail(this->p_vars()) = opv;
            }
        }
    }
}

std::vector<Eigen::VectorXd>
tycho::oc::PhaseIndexer::get_func_eq_multipliers(int Gindex,
                                                 const Eigen::VectorXd &EMultphase) const {

    Eigen::MatrixXi CDX = this->nlp->equality_constraints_[Gindex].index_data.c_index_;
    std::vector<Eigen::VectorXd> mults;
    for (int i = 0; i < CDX.cols(); i++) {
        Eigen::VectorXd lmi(CDX.rows());
        for (int j = 0; j < CDX.rows(); j++) {
            int ind = CDX(j, i) - this->StartEqCons;
            lmi[j] = EMultphase[ind];
        }
        mults.push_back(lmi);
    }
    return mults;
}

std::vector<Eigen::VectorXd>
tycho::oc::PhaseIndexer::get_func_iq_multipliers(int Gindex,
                                                 const Eigen::VectorXd &IMultphase) const {
    Eigen::MatrixXi CDX = this->nlp->inequality_constraints_[Gindex].index_data.c_index_;
    std::vector<Eigen::VectorXd> mults;
    for (int i = 0; i < CDX.cols(); i++) {
        Eigen::VectorXd lmi(CDX.rows());
        for (int j = 0; j < CDX.rows(); j++) {
            int ind = CDX(j, i) - this->StartIqCons;
            lmi[j] = IMultphase[ind];
        }
        mults.push_back(lmi);
    }
    return mults;
}

void tycho::oc::PhaseIndexer::print_stats(bool showfuns) const {
    using std::cout;
    using std::endl;

    cout << "# ODEXtUVars:   " << this->xtu_vars() << endl;
    cout << "# ODEParams:    " << this->p_vars() << endl;
    cout << "# StaticParams: " << this->StatPVars() << endl << endl;

    cout << "# Defects:      " << this->num_defects << endl;
    cout << "# States:       " << this->num_states << endl;
    cout << "# Variables:    " << this->num_phase_vars << endl;
    cout << "# equal_cons_:    " << this->num_phase_eq_cons << endl;
    cout << "# InEqualCons:  " << this->num_phase_iq_cons << endl << endl;

    if (showfuns) {
        cout << "Objective Functions" << endl << endl;
        cout << "____________________________________________________________" << endl << endl;
        for (int i = 0; i < this->num_obj_funs; i++) {
            cout << "************************************************************" << endl << endl;

            this->nlp->objectives_[this->StartObj + i].print_data();
        }
        cout << "Equality Constraints" << endl << endl;
        cout << "____________________________________________________________" << endl << endl;
        for (int i = 0; i < this->num_eq_funs; i++) {
            cout << "************************************************************" << endl << endl;

            this->nlp->equality_constraints_[this->StartEq + i].print_data();
        }
        cout << "Inequality Constraints" << endl << endl;
        cout << "____________________________________________________________" << endl << endl;
        for (int i = 0; i < this->num_iq_funs; i++) {
            cout << "************************************************************" << endl << endl;

            this->nlp->inequality_constraints_[this->StartIq + i].print_data();
        }
    }
}

void tycho::oc::PhaseIndexer::Test() {
    using std::cin;
    using std::cout;
    using std::endl;

    int Xv = 6;
    int Uv = 3;
    int Pv = 1;
    int Spv = 2;

    int dnum = 5;

    int dcs = 3;

    PhaseIndexer s(Xv, Uv, Pv, Spv);

    s.set_dimensions(dcs, dnum, false);

    Eigen::VectorXi xv(2);
    xv << 6, 7;

    Eigen::VectorXi pv(0);
    // pv << 0;
    Eigen::VectorXi sv(0);
    // sv << 0,1;

    std::cout << " Front" << endl;
    std::cout << s.make_Vindex_Cindex(PhaseRegionFlags::Front, xv, pv, sv, 1)[0] << endl << endl;

    std::cout << " Back" << endl;
    std::cout << s.make_Vindex_Cindex(PhaseRegionFlags::Back, xv, pv, sv, 1)[0] << endl << endl;

    std::cout << " FrontandBack" << endl;
    std::cout << s.make_Vindex_Cindex(PhaseRegionFlags::FrontandBack, xv, pv, sv, 1)[0] << endl
              << endl;

    std::cout << " BackandFront" << endl;
    std::cout << s.make_Vindex_Cindex(PhaseRegionFlags::BackandFront, xv, pv, sv, 1)[0] << endl
              << endl;

    std::cout << " Path" << endl;
    std::cout << s.make_Vindex_Cindex(PhaseRegionFlags::Path, xv, pv, sv, 1)[0] << endl << endl;

    std::cout << " InnerPath" << endl;
    std::cout << s.make_Vindex_Cindex(PhaseRegionFlags::InnerPath, xv, pv, sv, 1)[0] << endl
              << endl;

    std::cout << " NodalPath" << endl;
    std::cout << s.make_Vindex_Cindex(PhaseRegionFlags::NodalPath, xv, pv, sv, 1)[0] << endl
              << endl;

    std::cout << " PairwisePath" << endl;
    std::cout << s.make_Vindex_Cindex(PhaseRegionFlags::PairWisePath, xv, pv, sv, 1)[0] << endl
              << endl;

    std::cout << " FrontNodalBackPath" << endl;
    std::cout << s.make_Vindex_Cindex(PhaseRegionFlags::FrontNodalBackPath, xv, pv, sv, 1)[0]
              << endl
              << endl;

    std::cout << " DefectPath" << endl;
    std::cout << s.make_Vindex_Cindex(PhaseRegionFlags::DefectPath, xv, pv, sv, 1)[0] << endl
              << endl;

    std::cout << " DefectPairWisePath" << endl;
    std::cout << s.make_Vindex_Cindex(PhaseRegionFlags::DefectPairWisePath, xv, pv, sv, 1)[0]
              << endl
              << endl;
}
