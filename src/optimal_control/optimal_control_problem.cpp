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

#include "tycho/detail/optimal_control/phase/optimal_control_problem.h"

#include "tycho/detail/vf/scaling/auto_scaling_utils.h"

using tycho::solvers::ConstraintFunction;
using tycho::solvers::ObjectiveFunction;
using tycho::vf::IOScaled;

Eigen::VectorXd tycho::oc::OptimalControlProblem::get_input_scale(
    LinkFlags lflag, Eigen::Vector<PhaseRegionFlags, -1> regs, std::vector<VectorXi> phases_to_link,
    std::vector<VectorXi> XtUVars, std::vector<VectorXi> OPVars, std::vector<VectorXi> SPVars,
    std::vector<VectorXi> LVars) {

    std::vector<VectorXd> scales;
    int size = 0;

    if (phases_to_link.size() > 0) {

        for (int i = 0; i < phases_to_link[0].size(); i++) {

            int pnum = phases_to_link[0][i];
            auto flag = regs[i];
            auto XtUV = XtUVars[i];
            auto OPV = OPVars[i];
            auto SPV = SPVars[i];

            VectorXd scale = this->phases[pnum]->get_input_scale(flag, XtUV, OPV, SPV);
            scales.push_back(scale);
            size += scale.size();
        }
    }

    if (LVars.size() > 0) {

        VectorXd lscales(LVars[0].size());
        for (int i = 0; i < LVars[0].size(); i++) {
            lscales[i] = this->LPUnits[LVars[0][i]];
            size++;
        }
        scales.push_back(lscales);
    }

    VectorXd input_scales(size);

    int start = 0;

    for (int i = 0; i < scales.size(); i++) {
        size = scales[i].size();
        input_scales.segment(start, size) = scales[i];
        start += size;
    }

    return input_scales;
}

std::vector<Eigen::VectorXd> tycho::oc::OptimalControlProblem::get_test_inputs(
    LinkFlags lflag, Eigen::Vector<PhaseRegionFlags, -1> regs, std::vector<VectorXi> phases_to_link,
    std::vector<VectorXi> XtUVars, std::vector<VectorXi> OPVars, std::vector<VectorXi> SPVars,
    std::vector<VectorXi> LVars) {

    int nappl = std::max(phases_to_link.size(), LVars.size());

    std::vector<Eigen::VectorXd> test_inputs;

    for (int j = 0; j < nappl; j++) {
        int size = 0;

        std::vector<VectorXd> inputs;

        for (int i = 0; i < phases_to_link[j].size(); i++) {

            int pnum = phases_to_link[j][i];
            auto flag = regs[i];
            auto XtUV = XtUVars[i];
            auto OPV = OPVars[i];
            auto SPV = SPVars[i];

            VectorXd input = this->phases[pnum]->get_test_inputs(flag, XtUV, OPV, SPV)[0];
            inputs.push_back(input);
            size += input.size();
        }

        VectorXd linput(LVars[j].size());
        for (int i = 0; i < LVars[j].size(); i++) {
            linput[i] = this->active_link_params_[LVars[j][i]];
            size++;
        }
        inputs.push_back(linput);

        VectorXd test_input(size);

        int start = 0;

        for (int i = 0; i < inputs.size(); i++) {
            size = inputs[i].size();
            test_input.segment(start, size) = inputs[i];
            start += size;
        }

        test_inputs.push_back(test_input);
    }

    return test_inputs;
}

void tycho::oc::OptimalControlProblem::transcribe_phases() {

    if (this->phases.size() > 0) {

        this->num_phase_vars.resize(this->phases.size());
        this->num_phase_eq_cons.resize(this->phases.size());
        this->num_phase_iq_cons.resize(this->phases.size());

        for (int i = 0; i < this->phases.size(); i++) {
            this->phases[i]->NumPartitions = this->NumPartitions;
            this->phases[i]->init_indexing();
            this->num_phase_vars[i] = this->phases[i]->indexer_.num_phase_vars;
        }

        this->phases[0]->transcribe_phase(0, 0, 0, this->nlp, 0);
        this->num_phase_eq_cons[0] = this->phases[0]->indexer_.num_phase_eq_cons;
        this->num_phase_iq_cons[0] = this->phases[0]->indexer_.num_phase_iq_cons;

        for (int i = 1; i < this->phases.size(); i++) {
            int Vstart = this->num_phase_vars.segment(0, i).sum();
            int Estart = this->num_phase_eq_cons.segment(0, i).sum();
            int Istart = this->num_phase_iq_cons.segment(0, i).sum();

            this->phases[i]->transcribe_phase(Vstart, Estart, Istart, this->nlp, i);
            this->num_phase_eq_cons[i] = this->phases[i]->indexer_.num_phase_eq_cons;
            this->num_phase_iq_cons[i] = this->phases[i]->indexer_.num_phase_iq_cons;
        }

    } else {

        this->num_phase_vars.resize(1);
        this->num_phase_eq_cons.resize(1);
        this->num_phase_iq_cons.resize(1);

        this->num_phase_vars[0] = 0;
        this->num_phase_eq_cons[0] = 0;
        this->num_phase_iq_cons[0] = 0;
    }
}

void tycho::oc::OptimalControlProblem::check_functions() {
    /*
    Loops through all user defined functions and checks that they do not
    reference non-existent variables. Should be run prior to any transcribing any
    problem functions. First checks each phase, then all link functions.
    */

    for (int i = 0; i < this->phases.size(); i++) {
        this->phases[i]->check_functions(i);
    }

    auto CheckFunc = [&](std::string type, auto &func) {
        for (int i = 0; i < func.phases_to_link_.size(); i++) {
            for (int j = 0; j < func.phases_to_link_[i].size(); j++) {
                int pnum = func.phases_to_link_[i][j];
                if (pnum >= this->phases.size() || pnum < 0) {
                    fmt::print(fmt::fg(fmt::color::red),
                               "Transcription Error!!!\n"
                               "{0:} references non-existant phase:{1:}\n"
                               " Function Storage Index:{2:}\n"
                               " Function Name:{3:}\n",
                               type, pnum, func.storage_index_, func.func_.name());
                    throw std::invalid_argument("");
                }

                if (func.xtu_vars_[j].size() > 0) {
                    if (func.xtu_vars_[j].maxCoeff() >= this->Phase(pnum)->xtu_p_vars() ||
                        func.xtu_vars_[j].minCoeff() < 0) {

                        fmt::print(
                            fmt::fg(fmt::color::red),
                            "Transcription Error!!!\n"
                            "{0:} function state variable indices out of bounds in phase:{1:}\n"
                            " Function Storage Index:{2:}\n"
                            " Function Name:{3:}\n",
                            type, pnum, func.storage_index_, func.func_.name());
                        throw std::invalid_argument("");
                    }
                }
                if (func.op_vars_[j].size() > 0) {
                    if (func.op_vars_[j].maxCoeff() >= this->Phase(pnum)->p_vars() ||
                        func.op_vars_[j].minCoeff() < 0) {

                        fmt::print(
                            fmt::fg(fmt::color::red),
                            "Transcription Error!!!\n"
                            "{0:} function ODE Param variable indices out of bounds in phase:{1:}\n"
                            " Function Storage Index:{2:}\n"
                            " Function Name:{3:}\n",
                            type, pnum, func.storage_index_, func.func_.name());
                        throw std::invalid_argument("");
                    }
                }
                if (func.sp_vars_[j].size() > 0) {
                    if (func.sp_vars_[j].maxCoeff() >= this->Phase(pnum)->num_stat_params_ ||
                        func.sp_vars_[j].minCoeff() < 0) {
                        fmt::print(fmt::fg(fmt::color::red),
                                   "Transcription Error!!!\n"
                                   "{0:} function Static Param variable indices out of bounds in "
                                   "phase:{1:}\n"
                                   " Function Storage Index:{2:}\n"
                                   " Function Name:{3:}\n",
                                   type, pnum, func.storage_index_, func.func_.name());
                        throw std::invalid_argument("");
                    }
                }
            }
        }

        for (int i = 0; i < func.link_params_.size(); i++) {
            if (func.link_params_[i].size() > 0) {

                if (func.link_params_[i].maxCoeff() >= this->num_link_params ||
                    func.link_params_[i].minCoeff() < 0) {

                    fmt::print(fmt::fg(fmt::color::red),
                               "Transcription Error!!!\n"
                               "{0:} function link parameter variable indices out of bounds\n"
                               " Function Storage Index:{1:}\n"
                               " Function Name:{2:}\n",
                               type, func.storage_index_, func.func_.name());
                    throw std::invalid_argument("");
                }
            }
        }
    };

    std::string eq = "Link Equality constraint";
    std::string iq = "Link Inequality constraint";
    std::string obj = "Link Objective";

    for (auto &[key, f] : this->LinkEqualities)
        CheckFunc(eq, f);
    for (auto &[key, f] : this->LinkInequalities)
        CheckFunc(iq, f);
    for (auto &[key, f] : this->LinkObjectives)
        CheckFunc(obj, f);
}

void tycho::oc::OptimalControlProblem::transcribe_links() {

    int NextEq = this->num_phase_eq_cons.sum();
    int NextIq = this->num_phase_iq_cons.sum();

    int LinkVarStart = this->num_phase_vars.sum();
    this->LinkParamLocs.resize(this->num_link_params);
    for (int i = 0; i < this->num_link_params; i++) {
        this->LinkParamLocs[i] = LinkVarStart + i;
    }

    this->StartObj = int(this->nlp->Objectives.size());
    this->StartEq = int(this->nlp->EqualityConstraints.size());
    this->StartIq = int(this->nlp->InequalityConstraints.size());
    this->num_eq_funs = 0;
    this->num_iq_funs = 0;
    this->num_obj_funs = 0;
    for (auto &[key, Eq] : this->LinkEqualities) {
        auto VC = this->make_link_Vindex_Cindex(
            Eq.link_flag_, Eq.phase_reg_flags_, Eq.phases_to_link_, Eq.xtu_vars_, Eq.op_vars_,
            Eq.sp_vars_, Eq.link_params_, Eq.func_.output_rows(), NextEq);

        auto Func = Eq.func_;

        if (this->auto_scaling_) {
            VectorXd input_scales =
                this->get_input_scale(Eq.link_flag_, Eq.phase_reg_flags_, Eq.phases_to_link_,
                                      Eq.xtu_vars_, Eq.op_vars_, Eq.sp_vars_, Eq.link_params_);
            VectorXd output_scales(Func.output_rows());
            output_scales = Eq.output_scales_;
            Func = IOScaled<decltype(Func)>(Eq.func_, input_scales, output_scales);
        }

        this->nlp->EqualityConstraints.emplace_back(ConstraintFunction(Func, VC[0], VC[1]));
        Eq.global_index_ = this->nlp->EqualityConstraints.size() - 1;
        this->num_eq_funs++;
    }
    for (auto &[key, Iq] : this->LinkInequalities) {
        auto VC = this->make_link_Vindex_Cindex(
            Iq.link_flag_, Iq.phase_reg_flags_, Iq.phases_to_link_, Iq.xtu_vars_, Iq.op_vars_,
            Iq.sp_vars_, Iq.link_params_, Iq.func_.output_rows(), NextIq);

        auto Func = Iq.func_;

        if (this->auto_scaling_) {
            VectorXd input_scales =
                this->get_input_scale(Iq.link_flag_, Iq.phase_reg_flags_, Iq.phases_to_link_,
                                      Iq.xtu_vars_, Iq.op_vars_, Iq.sp_vars_, Iq.link_params_);
            VectorXd output_scales(Func.output_rows());
            output_scales = Iq.output_scales_;
            Func = IOScaled<decltype(Func)>(Iq.func_, input_scales, output_scales);
        }

        this->nlp->InequalityConstraints.emplace_back(ConstraintFunction(Func, VC[0], VC[1]));
        Iq.global_index_ = this->nlp->InequalityConstraints.size() - 1;
        this->num_iq_funs++;
    }
    for (auto &[key, Ob] : this->LinkObjectives) {
        int dummy = 0;
        auto VC = this->make_link_Vindex_Cindex(
            Ob.link_flag_, Ob.phase_reg_flags_, Ob.phases_to_link_, Ob.xtu_vars_, Ob.op_vars_,
            Ob.sp_vars_, Ob.link_params_, Ob.func_.output_rows(), dummy);

        auto Func = Ob.func_;

        if (this->auto_scaling_) {
            VectorXd input_scales =
                this->get_input_scale(Ob.link_flag_, Ob.phase_reg_flags_, Ob.phases_to_link_,
                                      Ob.xtu_vars_, Ob.op_vars_, Ob.sp_vars_, Ob.link_params_);
            VectorXd output_scales(Func.output_rows());
            output_scales = Ob.output_scales_;
            Func = IOScaled<decltype(Func)>(Ob.func_, input_scales, output_scales);
        }

        this->nlp->Objectives.emplace_back(ObjectiveFunction(Func, VC[0]));
        Ob.global_index_ = this->nlp->Objectives.size() - 1;

        this->num_obj_funs++;
    }

    this->num_link_eq_cons = NextEq - this->num_phase_eq_cons.sum();
    this->num_link_iq_cons = NextIq - this->num_phase_iq_cons.sum();
}

void tycho::oc::OptimalControlProblem::calc_auto_scales() {
    for (int i = 0; i < this->phases.size(); i++) {
        this->phases[i]->calc_auto_scales();
    }

    auto calc_impl = [&](auto &funcmap) {
        for (auto &[key, func] : funcmap) {
            if (func.scale_mode_ == ScaleModes::AUTO) {
                VectorXd input_scales = this->get_input_scale(
                    func.link_flag_, func.phase_reg_flags_, func.phases_to_link_, func.xtu_vars_,
                    func.op_vars_, func.sp_vars_, func.link_params_);
                std::vector<VectorXd> test_inputs = this->get_test_inputs(
                    func.link_flag_, func.phase_reg_flags_, func.phases_to_link_, func.xtu_vars_,
                    func.op_vars_, func.sp_vars_, func.link_params_);
                VectorXd output_scales =
                    calc_jacobian_row_scales(func.func_, input_scales, test_inputs);
                func.output_scales_ = output_scales;
            } else {
            }
        }
    };

    calc_impl(this->LinkEqualities);
    calc_impl(this->LinkInequalities);
    calc_impl(this->LinkObjectives);
}

std::vector<double> tycho::oc::OptimalControlProblem::get_objective_scales() {
    std::vector<double> scales;
    for (auto &[key, obj] : this->LinkObjectives) {
        if (obj.scale_mode_ == ScaleModes::AUTO) {
            scales.push_back(obj.output_scales_[0]);
        }
    }
    for (auto phase : this->phases) {
        auto pscales = phase->get_objective_scales();

        for (auto s : pscales) {
            scales.push_back(s);
        }
    }
    return scales;
}

void tycho::oc::OptimalControlProblem::update_objective_scales(double scale) {
    for (auto &[key, obj] : this->LinkObjectives) {
        if (obj.scale_mode_ == ScaleModes::AUTO) {
            obj.output_scales_[0] = scale;
        }
    }
    for (auto phase : this->phases) {
        phase->update_objective_scales(scale);
    }
}

void tycho::oc::OptimalControlProblem::transcribe(bool showstats, bool showfuns) {

    this->nlp = std::make_shared<NonLinearProgram>(this->NumPartitions);

    check_functions();
    if (this->auto_scaling_) {
        this->calc_auto_scales();
        if (this->sync_objective_scales_) {
            // Ensure that all objectives have same scale factor to preserve meaning of un-scaled
            // problem Common scale is mean of all separate scale factors
            auto objscales = this->get_objective_scales();
            if (objscales.size() > 0) {
                double meanobjscale = std::accumulate(objscales.begin(), objscales.end(), 0.0) /
                                      double(objscales.size());
                this->update_objective_scales(meanobjscale);
            }
        }
    }

    this->transcribe_phases();
    this->transcribe_links();

    this->num_prob_vars = this->num_phase_vars.sum() + this->num_link_params;
    this->num_prob_eq_cons = this->num_phase_eq_cons.sum() + this->num_link_eq_cons;
    this->num_prob_iq_cons = this->num_phase_iq_cons.sum() + this->num_link_iq_cons;
    if (showstats)
        this->print_stats(showfuns);
    this->nlp->make_NLP(this->num_prob_vars, this->num_prob_eq_cons, this->num_prob_iq_cons);
    this->optimizer->setNLP(this->nlp);

    //////DO NOT GET RID OF THIS!!!!!!//
    this->do_transcription_ = false;
}

tycho::ConvergenceFlags tycho::oc::OptimalControlProblem::psipot_call_impl(JetJobModes mode) {

    this->check_transcriptions();
    if (this->do_transcription_)
        this->transcribe();
    VectorXd Input = this->make_solver_input();
    VectorXd Output;

    switch (mode) {
    case JetJobModes::Solve:
        Output = this->optimizer->solve(Input);
        break;
    case JetJobModes::Optimize:
        Output = this->optimizer->optimize(Input);
        break;
    case JetJobModes::SolveOptimize:
        Output = this->optimizer->solve_optimize(Input);
        break;
    case JetJobModes::SolveOptimizeSolve:
        Output = this->optimizer->solve_optimize_solve(Input);
        break;
    case JetJobModes::OptimizeSolve:
        Output = this->optimizer->optimize_solve(Input);
        break;
    default:
        throw std::invalid_argument("Unrecognized PSIOPT mode");
    }

    this->collect_solver_output(Output);

    this->collect_post_opt_info(this->optimizer->LastEqCons, this->optimizer->LastEqLmults,
                                this->optimizer->LastIqCons, this->optimizer->LastIqLmults);

    return this->optimizer->ConvergeFlag;
}

tycho::ConvergenceFlags tycho::oc::OptimalControlProblem::ocp_call_impl(JetJobModes mode) {
    if (this->print_mesh_info_ && this->adaptive_mesh_) {
        fmt::print(fmt::fg(fmt::color::white), "{0:=^{1}}\n", "", 65);
        fmt::print(fmt::fg(fmt::color::dim_gray), "Beginning");
        fmt::print(": ");
        fmt::print(fmt::fg(fmt::color::royal_blue), "Adaptive Mesh Refinement");
        fmt::print("\n");
    }

    utils::Timer Runtimer;
    Runtimer.start();

    PSIOPT::ConvergenceFlags flag = this->psipot_call_impl(mode);

    JetJobModes nextmode = mode;
    if (this->solve_only_first_) {
        switch (mode) {
        case JetJobModes::SolveOptimize:
            nextmode = JetJobModes::Optimize;
            break;
        case JetJobModes::SolveOptimizeSolve:
            nextmode = JetJobModes::OptimizeSolve;
            break;
        default:
            break;
        }
    }

    if (this->adaptive_mesh_) {

        if (flag >= this->mesh_abort_flag_) {
            if (this->print_mesh_info_) {
                fmt::print(fmt::fg(fmt::color::red),
                           "Mesh Iteration 0 Failed to Solve: Aborting\n");
            }
        } else {
            init_meshs();
            for (int i = 0; i < this->max_mesh_iters_; i++) {
                if (check_meshs(this->print_mesh_info_)) {
                    if (this->print_mesh_info_) {
                        this->print_meshs(i);
                        fmt::print(fmt::fg(fmt::color::lime_green), "All Meshes Converged\n");
                    }

                    break;
                } else if (i == this->max_mesh_iters_ - 1) {
                    if (this->print_mesh_info_) {
                        this->print_meshs(i);
                        fmt::print(fmt::fg(fmt::color::red), "All Meshes Not Converged\n");
                    }
                    break;
                } else {
                    update_meshs(this->print_mesh_info_);
                    if (this->print_mesh_info_)
                        this->print_meshs(i);
                }
                flag = this->psipot_call_impl(nextmode);
                if (flag >= this->mesh_abort_flag_) {
                    if (this->print_mesh_info_) {
                        fmt::print(fmt::fg(fmt::color::red),
                                   "Mesh Iteration {0:} Failed to Solve: Aborting\n", i + 1);
                    }
                    break;
                }
            }
        }
    }

    if (this->print_mesh_info_ && this->adaptive_mesh_) {

        Runtimer.stop();
        double tseconds = double(Runtimer.count<std::chrono::microseconds>()) / 1000000;
        fmt::print("Total Time:");
        if (tseconds > 0.5) {
            fmt::print(fmt::fg(fmt::color::cyan), "{0:>10.4f} s\n", tseconds);
        } else {
            fmt::print(fmt::fg(fmt::color::cyan), "{0:>10.2f} ms\n", tseconds * 1000);
        }

        fmt::print(fmt::fg(fmt::color::dim_gray), "Finished ");
        fmt::print(": ");
        fmt::print(fmt::fg(fmt::color::royal_blue), "Adaptive Mesh Refinement");
        fmt::print("\n");
        fmt::print(fmt::fg(fmt::color::white), "{0:=^{1}}\n", "", 65);
    }

    return flag;
}

void tycho::oc::OptimalControlProblem::print_stats(bool showfuns) {
    using std::cout;
    using std::endl;
    cout << "Problem Statistics" << endl << endl;

    cout << "# Variables:    " << num_prob_vars << endl;
    cout << "# EqualCons:    " << num_prob_eq_cons << endl;
    cout << "# InEqualCons:  " << num_prob_iq_cons << endl;
    cout << "# Phases:       " << this->phases.size() << endl << endl;

    for (int i = 0; i < this->phases.size(); i++) {
        cout << "____________________________________________________________" << endl << endl;
        cout << "Phase: " << i << " Statistics" << endl << endl;
        this->phases[i]->indexer_.print_stats(showfuns);
        cout << "____________________________________________________________" << endl << endl;
    }

    cout << "____________________________________________________________" << endl << endl;
    cout << "Link Statistics" << endl << endl;
    cout << "# Link Params:    " << num_link_params << endl;
    cout << "# Link EqualCons:    " << num_link_eq_cons << endl;
    cout << "# Link InEqualCons:  " << num_link_iq_cons << endl;

    cout << "____________________________________________________________" << endl << endl;

    if (showfuns) {
        cout << "Objective Functions" << endl << endl;
        cout << "____________________________________________________________" << endl << endl;
        for (int i = 0; i < this->num_obj_funs; i++) {
            cout << "************************************************************" << endl << endl;
            this->nlp->Objectives[this->StartObj + i].print_data();
        }
        cout << "Equality Constraints" << endl << endl;
        cout << "____________________________________________________________" << endl << endl;
        for (int i = 0; i < this->num_eq_funs; i++) {
            cout << "************************************************************" << endl << endl;
            this->nlp->EqualityConstraints[this->StartEq + i].print_data();
        }
        cout << "Inequality Constraints" << endl << endl;
        cout << "____________________________________________________________" << endl << endl;
        for (int i = 0; i < this->num_iq_funs; i++) {
            cout << "************************************************************" << endl << endl;
            this->nlp->InequalityConstraints[this->StartIq + i].print_data();
        }
    }
}

std::array<Eigen::MatrixXi, 2> tycho::oc::OptimalControlProblem::make_link_Vindex_Cindex(
    LinkFlags Reg, const Eigen::Matrix<PhaseRegionFlags, -1, 1> &PhaseRegs,
    const std::vector<Eigen::VectorXi> &PTL, const std::vector<Eigen::VectorXi> &xtv,
    const std::vector<Eigen::VectorXi> &opv, const std::vector<Eigen::VectorXi> &spv,
    const std::vector<Eigen::VectorXi> &lv, int orows, int &NextCLoc) const {
    using std::cout;
    using std::endl;
    MatrixXi Vindex;
    MatrixXi Cindex;

    switch (Reg) {
    case LinkFlags::PathToPath: {

        int cols = 0;
        std::vector<MatrixXi> vtemps(PTL.size());
        std::vector<MatrixXi> vtemps2(PhaseRegs.size());

        int irows = 0;
        for (int i = 0; i < PTL.size(); i++) {
            int sz = 0;
            irows = 0;

            for (int j = 0; j < PhaseRegs.size(); j++) {
                auto VinTemp = this->phases[PTL[i][j]]->indexer_.make_Vindex_Cindex(
                    PhaseRegs[j], xtv[j], opv[j], spv[j], 1)[0];
                vtemps2[j] = VinTemp;
                irows += VinTemp.rows();
                if (j == 0)
                    sz = VinTemp.cols();
                else {
                    if (sz != VinTemp.cols()) {
                        throw std::invalid_argument("Phases cannot be linked path to path");
                    }
                }
            }
            cols += sz;
            vtemps[i].resize(irows, sz);

            int start = 0;
            for (int j = 0; j < PhaseRegs.size(); j++) {
                vtemps[i].middleRows(start, vtemps2[j].rows()) = vtemps2[j];
                start += vtemps2[j].rows();
            }
        }

        Vindex.resize(irows, cols);

        int start = 0;
        for (int i = 0; i < PTL.size(); i++) {
            Vindex.middleCols(start, vtemps[i].cols()) = vtemps[i];
            start += vtemps[i].cols();
        }

        Cindex.resize(orows, cols);
        for (int i = 0; i < cols; i++) {
            for (int j = 0; j < orows; j++) {
                Cindex(j, i) = NextCLoc;
                NextCLoc++;
            }
        }

        break;
    }
    case LinkFlags::ReadRegions: {
    }
    default: {
        int cols = std::max(PTL.size(), lv.size());
        Cindex.resize(orows, cols);

        for (int i = 0; i < cols; i++) {
            for (int j = 0; j < orows; j++) {
                Cindex(j, i) = NextCLoc;
                NextCLoc++;
            }
        }
        int irows = lv[0].size();
        for (int i = 0; i < PhaseRegs.size(); i++) {
            irows += xtv[i].size() + opv[i].size() + spv[i].size();
        }
        Vindex.resize(irows, cols);
        for (int i = 0; i < cols; i++) {
            int start = 0;
            for (int j = 0; j < PhaseRegs.size(); j++) {

                auto VinTemp = this->phases[PTL[i][j]]->indexer_.make_Vindex_Cindex(
                    PhaseRegs[j], xtv[j], opv[j], spv[j], 1)[0];
                Vindex.col(i).segment(start, VinTemp.rows()) = VinTemp;
                start += VinTemp.rows();
            }
            for (int j = 0; j < lv[i].size(); j++) {
                Vindex(start, i) = this->LinkParamLocs[lv[i][j]];
                start++;
            }
        }
        break;
    }
    }

    return std::array<Eigen::MatrixXi, 2>{Vindex, Cindex};
}
