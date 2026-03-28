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
            linput[i] = this->ActiveLinkParams[LVars[j][i]];
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
            this->num_phase_vars[i] = this->phases[i]->indexer.num_phase_vars;
        }

        this->phases[0]->transcribe_phase(0, 0, 0, this->nlp, 0);
        this->num_phase_eq_cons[0] = this->phases[0]->indexer.num_phase_eq_cons;
        this->num_phase_iq_cons[0] = this->phases[0]->indexer.num_phase_iq_cons;

        for (int i = 1; i < this->phases.size(); i++) {
            int Vstart = this->num_phase_vars.segment(0, i).sum();
            int Estart = this->num_phase_eq_cons.segment(0, i).sum();
            int Istart = this->num_phase_iq_cons.segment(0, i).sum();

            this->phases[i]->transcribe_phase(Vstart, Estart, Istart, this->nlp, i);
            this->num_phase_eq_cons[i] = this->phases[i]->indexer.num_phase_eq_cons;
            this->num_phase_iq_cons[i] = this->phases[i]->indexer.num_phase_iq_cons;
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
        for (int i = 0; i < func.PhasesTolink.size(); i++) {
            for (int j = 0; j < func.PhasesTolink[i].size(); j++) {
                int pnum = func.PhasesTolink[i][j];
                if (pnum >= this->phases.size() || pnum < 0) {
                    fmt::print(fmt::fg(fmt::color::red),
                               "Transcription Error!!!\n"
                               "{0:} references non-existant phase:{1:}\n"
                               " Function Storage Index:{2:}\n"
                               " Function Name:{3:}\n",
                               type, pnum, func.StorageIndex, func.Func.name());
                    throw std::invalid_argument("");
                }

                if (func.XtUVars[j].size() > 0) {
                    if (func.XtUVars[j].maxCoeff() >= this->Phase(pnum)->XtUPVars() ||
                        func.XtUVars[j].minCoeff() < 0) {

                        fmt::print(
                            fmt::fg(fmt::color::red),
                            "Transcription Error!!!\n"
                            "{0:} function state variable indices out of bounds in phase:{1:}\n"
                            " Function Storage Index:{2:}\n"
                            " Function Name:{3:}\n",
                            type, pnum, func.StorageIndex, func.Func.name());
                        throw std::invalid_argument("");
                    }
                }
                if (func.OPVars[j].size() > 0) {
                    if (func.OPVars[j].maxCoeff() >= this->Phase(pnum)->PVars() ||
                        func.OPVars[j].minCoeff() < 0) {

                        fmt::print(
                            fmt::fg(fmt::color::red),
                            "Transcription Error!!!\n"
                            "{0:} function ODE Param variable indices out of bounds in phase:{1:}\n"
                            " Function Storage Index:{2:}\n"
                            " Function Name:{3:}\n",
                            type, pnum, func.StorageIndex, func.Func.name());
                        throw std::invalid_argument("");
                    }
                }
                if (func.SPVars[j].size() > 0) {
                    if (func.SPVars[j].maxCoeff() >= this->Phase(pnum)->num_stat_params ||
                        func.SPVars[j].minCoeff() < 0) {
                        fmt::print(fmt::fg(fmt::color::red),
                                   "Transcription Error!!!\n"
                                   "{0:} function Static Param variable indices out of bounds in "
                                   "phase:{1:}\n"
                                   " Function Storage Index:{2:}\n"
                                   " Function Name:{3:}\n",
                                   type, pnum, func.StorageIndex, func.Func.name());
                        throw std::invalid_argument("");
                    }
                }
            }
        }

        for (int i = 0; i < func.LinkParams.size(); i++) {
            if (func.LinkParams[i].size() > 0) {

                if (func.LinkParams[i].maxCoeff() >= this->num_link_params ||
                    func.LinkParams[i].minCoeff() < 0) {

                    fmt::print(fmt::fg(fmt::color::red),
                               "Transcription Error!!!\n"
                               "{0:} function link parameter variable indices out of bounds\n"
                               " Function Storage Index:{1:}\n"
                               " Function Name:{2:}\n",
                               type, func.StorageIndex, func.Func.name());
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
        auto VC = this->make_link_Vindex_Cindex(Eq.LinkFlag, Eq.PhaseRegFlags, Eq.PhasesTolink,
                                                Eq.XtUVars, Eq.OPVars, Eq.SPVars, Eq.LinkParams,
                                                Eq.Func.output_rows(), NextEq);

        auto Func = Eq.Func;

        if (this->AutoScaling) {
            VectorXd input_scales =
                this->get_input_scale(Eq.LinkFlag, Eq.PhaseRegFlags, Eq.PhasesTolink, Eq.XtUVars,
                                      Eq.OPVars, Eq.SPVars, Eq.LinkParams);
            VectorXd output_scales(Func.output_rows());
            output_scales = Eq.OutputScales;
            Func = IOScaled<decltype(Func)>(Eq.Func, input_scales, output_scales);
        }

        this->nlp->EqualityConstraints.emplace_back(ConstraintFunction(Func, VC[0], VC[1]));
        Eq.GlobalIndex = this->nlp->EqualityConstraints.size() - 1;
        this->num_eq_funs++;
    }
    for (auto &[key, Iq] : this->LinkInequalities) {
        auto VC = this->make_link_Vindex_Cindex(Iq.LinkFlag, Iq.PhaseRegFlags, Iq.PhasesTolink,
                                                Iq.XtUVars, Iq.OPVars, Iq.SPVars, Iq.LinkParams,
                                                Iq.Func.output_rows(), NextIq);

        auto Func = Iq.Func;

        if (this->AutoScaling) {
            VectorXd input_scales =
                this->get_input_scale(Iq.LinkFlag, Iq.PhaseRegFlags, Iq.PhasesTolink, Iq.XtUVars,
                                      Iq.OPVars, Iq.SPVars, Iq.LinkParams);
            VectorXd output_scales(Func.output_rows());
            output_scales = Iq.OutputScales;
            Func = IOScaled<decltype(Func)>(Iq.Func, input_scales, output_scales);
        }

        this->nlp->InequalityConstraints.emplace_back(ConstraintFunction(Func, VC[0], VC[1]));
        Iq.GlobalIndex = this->nlp->InequalityConstraints.size() - 1;
        this->num_iq_funs++;
    }
    for (auto &[key, Ob] : this->LinkObjectives) {
        int dummy = 0;
        auto VC = this->make_link_Vindex_Cindex(Ob.LinkFlag, Ob.PhaseRegFlags, Ob.PhasesTolink,
                                                Ob.XtUVars, Ob.OPVars, Ob.SPVars, Ob.LinkParams,
                                                Ob.Func.output_rows(), dummy);

        auto Func = Ob.Func;

        if (this->AutoScaling) {
            VectorXd input_scales =
                this->get_input_scale(Ob.LinkFlag, Ob.PhaseRegFlags, Ob.PhasesTolink, Ob.XtUVars,
                                      Ob.OPVars, Ob.SPVars, Ob.LinkParams);
            VectorXd output_scales(Func.output_rows());
            output_scales = Ob.OutputScales;
            Func = IOScaled<decltype(Func)>(Ob.Func, input_scales, output_scales);
        }

        this->nlp->Objectives.emplace_back(ObjectiveFunction(Func, VC[0]));
        Ob.GlobalIndex = this->nlp->Objectives.size() - 1;

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
            if (func.ScaleMode == ScaleModes::AUTO) {
                VectorXd input_scales =
                    this->get_input_scale(func.LinkFlag, func.PhaseRegFlags, func.PhasesTolink,
                                          func.XtUVars, func.OPVars, func.SPVars, func.LinkParams);
                std::vector<VectorXd> test_inputs =
                    this->get_test_inputs(func.LinkFlag, func.PhaseRegFlags, func.PhasesTolink,
                                          func.XtUVars, func.OPVars, func.SPVars, func.LinkParams);
                VectorXd output_scales =
                    calc_jacobian_row_scales(func.Func, input_scales, test_inputs);
                func.OutputScales = output_scales;
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
        if (obj.ScaleMode == ScaleModes::AUTO) {
            scales.push_back(obj.OutputScales[0]);
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
        if (obj.ScaleMode == ScaleModes::AUTO) {
            obj.OutputScales[0] = scale;
        }
    }
    for (auto phase : this->phases) {
        phase->update_objective_scales(scale);
    }
}

void tycho::oc::OptimalControlProblem::transcribe(bool showstats, bool showfuns) {

    this->nlp = std::make_shared<NonLinearProgram>(this->NumPartitions);

    check_functions();
    if (this->AutoScaling) {
        this->calc_auto_scales();
        if (this->SyncObjectiveScales) {
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
    this->do_transcription = false;
}

tycho::ConvergenceFlags tycho::oc::OptimalControlProblem::psipot_call_impl(JetJobModes mode) {

    this->check_transcriptions();
    if (this->do_transcription)
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
    if (this->PrintMeshInfo && this->AdaptiveMesh) {
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
    if (this->SolveOnlyFirst) {
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

    if (this->AdaptiveMesh) {

        if (flag >= this->MeshAbortFlag) {
            if (this->PrintMeshInfo) {
                fmt::print(fmt::fg(fmt::color::red),
                           "Mesh Iteration 0 Failed to Solve: Aborting\n");
            }
        } else {
            init_meshs();
            for (int i = 0; i < this->MaxMeshIters; i++) {
                if (check_meshs(this->PrintMeshInfo)) {
                    if (this->PrintMeshInfo) {
                        this->print_meshs(i);
                        fmt::print(fmt::fg(fmt::color::lime_green), "All Meshes Converged\n");
                    }

                    break;
                } else if (i == this->MaxMeshIters - 1) {
                    if (this->PrintMeshInfo) {
                        this->print_meshs(i);
                        fmt::print(fmt::fg(fmt::color::red), "All Meshes Not Converged\n");
                    }
                    break;
                } else {
                    update_meshs(this->PrintMeshInfo);
                    if (this->PrintMeshInfo)
                        this->print_meshs(i);
                }
                flag = this->psipot_call_impl(nextmode);
                if (flag >= this->MeshAbortFlag) {
                    if (this->PrintMeshInfo) {
                        fmt::print(fmt::fg(fmt::color::red),
                                   "Mesh Iteration {0:} Failed to Solve: Aborting\n", i + 1);
                    }
                    break;
                }
            }
        }
    }

    if (this->PrintMeshInfo && this->AdaptiveMesh) {

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
        this->phases[i]->indexer.print_stats(showfuns);
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
                auto VinTemp = this->phases[PTL[i][j]]->indexer.make_Vindex_Cindex(
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

                auto VinTemp = this->phases[PTL[i][j]]->indexer.make_Vindex_Cindex(
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
