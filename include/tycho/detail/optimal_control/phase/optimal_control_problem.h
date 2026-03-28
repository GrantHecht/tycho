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

#include "tycho/detail/optimal_control/core/interface_types.h"
#include "tycho/detail/optimal_control/core/link_function.h"
#include "tycho/detail/optimal_control/phase/ode_phase_base.h"
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
using tycho::solvers::OptimizationProblemBase;
using tycho::solvers::NonLinearProgram;
using tycho::solvers::PSIOPT;

// VF types
using vf::Arguments;
using vf::GenericFunction;
using vf::StackedOutputs;

struct OptimalControlProblem : OptimizationProblemBase {
    using VectorXi = Eigen::VectorXi;
    using MatrixXi = Eigen::MatrixXi;

    using VectorXd = Eigen::VectorXd;
    using MatrixXd = Eigen::MatrixXd;

    using VectorFunctionalX = GenericFunction<-1, -1>;
    using ScalarFunctionalX = GenericFunction<-1, 1>;

    using RegVec = Eigen::Matrix<PhaseRegionFlags, -1, 1>;

    using LinkConstraint = LinkFunction<VectorFunctionalX>;
    using LinkObjective = LinkFunction<ScalarFunctionalX>;
    using StateConstraint = StateFunction<VectorFunctionalX>;
    using StateObjective = StateFunction<ScalarFunctionalX>;
    using StateIntegral = StateFunction<ScalarFunctionalX>;
    using PhasePtr = std::shared_ptr<ODEPhaseBase>;

    using PhaseIndexPack =
        std::tuple<int, std::string, Eigen::VectorXi, Eigen::VectorXi, Eigen::VectorXi>;
    using PhaseIndexPackPtr =
        std::tuple<PhasePtr, std::string, Eigen::VectorXi, Eigen::VectorXi, Eigen::VectorXi>;

    using PhaseRefType = std::variant<int, PhasePtr, std::string>;

    using PhasePack =
        std::tuple<PhaseRefType, RegionType, VarIndexType, VarIndexType, VarIndexType>;

    std::vector<PhasePtr> phases;
    std::vector<std::string> phase_names;

    bool do_transcription = true;
    void reset_transcription() { this->do_transcription = true; };

    std::map<int, LinkConstraint> LinkEqualities;
    std::map<int, LinkConstraint> LinkInequalities;
    std::map<int, LinkObjective> LinkObjectives;

    VectorXd ActiveLinkParams;
    Eigen::VectorXd LPUnits;
    bool AutoScaling = false;
    bool SyncObjectiveScales = true;

    std::map<std::string, Eigen::VectorXi> LPidxs;

    void set_link_params(VectorXd parm, VectorXd units) {
        if (units.size() != parm.size()) {
            throw std::invalid_argument(
                "Size of link parameter vector and scaling units vector must match");
        }

        this->ActiveLinkParams = parm;
        this->num_link_params = parm.size();
        this->reset_transcription();
        this->LPUnits = units;
    }
    void set_link_params(VectorXd parm) {
        VectorXd units(parm.size());
        units.setOnes();
        return this->set_link_params(parm, units);
    }

    void set_link_param_vgroups(std::map<std::string, Eigen::VectorXi> lpidxs) {
        this->LPidxs = lpidxs;
    }
    void add_link_param_vgroups(std::map<std::string, Eigen::VectorXi> lpidxs) {
        for (auto &[key, value] : lpidxs) {
            this->LPidxs[key] = value;
        }
    }
    void add_link_param_vgroup(Eigen::VectorXi idx, std::string key) { this->LPidxs[key] = idx; }
    void add_link_param_vgroup(int idx, std::string key) {
        VectorXi tmp(1);
        tmp << idx;
        this->LPidxs[key] = tmp;
    }
    VectorXi get_lp_idx(std::string key) const {
        if (LPidxs.count(key) == 0) {
            throw std::invalid_argument(
                fmt::format("No LinkParam variable index group with name: {0:} exists.", key));
        }
        return this->LPidxs.at(key);
    }

    VectorXd return_link_params() { return this->ActiveLinkParams; }

    bool MultipliersLoaded = false;
    bool PostOptInfoValid = false;

    void invalidate_post_opt_info() { this->PostOptInfoValid = false; };

    VectorXd ActiveEqLmults;
    VectorXd ActiveIqLmults;
    VectorXd ActiveEqCons;
    VectorXd ActiveIqCons;

    VectorXi LinkParamLocs;

    VectorXi num_phase_vars;
    VectorXi num_phase_eq_cons;
    VectorXi num_phase_iq_cons;

    int num_link_params = 0;
    int num_link_eq_cons = 0;
    int num_link_iq_cons = 0;

    int StartObj = 0;
    int StartEq = 0;
    int StartIq = 0;

    int num_obj_funs = 0;
    int num_eq_funs = 0;
    int num_iq_funs = 0;

    int num_prob_vars = 0;
    int num_prob_eq_cons = 0;
    int num_prob_iq_cons = 0;

    ///////////////////////////////
    bool AdaptiveMesh = false;
    bool PrintMeshInfo = true;
    bool SolveOnlyFirst = true;

    int MaxMeshIters = 10;
    PSIOPT::ConvergenceFlags MeshAbortFlag = PSIOPT::ConvergenceFlags::DIVERGING;

    bool MeshConverged = false;

    void set_auto_scaling(bool autoscale, bool applytophases) {
        this->AutoScaling = autoscale;
        if (applytophases) {
            for (auto phase : this->phases) {
                phase->set_auto_scaling(autoscale);
            }
        }
        this->reset_transcription();
        this->invalidate_post_opt_info();
    }

    void set_adaptive_mesh(bool amesh, bool applytophases) {
        this->AdaptiveMesh = amesh;
        if (applytophases) {
            for (auto phase : this->phases) {
                phase->set_adaptive_mesh(amesh);
            }
        }
    }

    void set_mesh_tol(double t) {
        for (auto phase : this->phases) {
            phase->set_mesh_tol(t);
        }
    }
    void set_mesh_red_factor(double t) {
        for (auto phase : this->phases) {
            phase->set_mesh_red_factor(t);
        }
    }
    void set_mesh_inc_factor(double t) {
        for (auto phase : this->phases) {
            phase->set_mesh_inc_factor(t);
        }
    }
    void set_mesh_err_factor(double t) {
        for (auto phase : this->phases) {
            phase->set_mesh_err_factor(t);
        }
    }
    void set_max_mesh_iters(int it) { this->MaxMeshIters = it; }
    void set_min_segments(int it) {
        for (auto phase : this->phases) {
            phase->set_min_segments(it);
        }
    }
    void set_max_segments(int it) {
        for (auto phase : this->phases) {
            phase->set_max_segments(it);
        }
    }
    void set_mesh_error_criteria(MeshErrorAggregation m) {
        for (auto phase : this->phases) {
            phase->set_mesh_error_criteria(m);
        }
    }
    void set_mesh_error_criteria(const std::string &m) {
        for (auto phase : this->phases) {
            phase->set_mesh_error_criteria(m);
        }
    }
    void set_mesh_error_estimator(MeshErrorEstimators m) {
        for (auto phase : this->phases) {
            phase->set_mesh_error_estimator(m);
        }
    }
    void set_mesh_error_estimator(const std::string &m) {
        for (auto phase : this->phases) {
            phase->set_mesh_error_estimator(m);
        }
    }
    void set_mesh_error_distributor(MeshErrorAggregation m) {
        for (auto phase : this->phases) {
            phase->set_mesh_error_distributor(m);
        }
    }
    void set_mesh_error_distributor(const std::string &m) {
        for (auto phase : this->phases) {
            phase->set_mesh_error_distributor(m);
        }
    }

    ///////////////////////////////
    OptimalControlProblem() {}
    OptimalControlProblem(std::vector<PhasePtr> ps) { this->add_phases(ps); }

    int add_phase(PhasePtr p) {
        this->reset_transcription();
        this->phases.push_back(p);
        int index = int(this->phases.size()) - 1;
        this->phase_names.push_back(std::to_string(index));
        check_dupilcate_phases();
        return index;
    }

    std::vector<int> add_phases(std::vector<PhasePtr> ps) {
        std::vector<int> idxs;
        for (auto p : ps) {
            idxs.push_back(this->add_phase(p));
        }
        return idxs;
    }

    int add_phase(PhasePtr p, const std::string &name) {
        this->reset_transcription();
        this->phases.push_back(p);
        int index = int(this->phases.size()) - 1;
        this->phase_names.push_back(name);
        check_dupilcate_phases();
        return index;
    }

    int get_phase_num(const std::string &name) {
        auto nameit = std::find(phase_names.begin(), phase_names.end(), name);
        if (nameit == phase_names.end()) {
            fmt::print(fmt::fg(fmt::color::red),
                       "Transcription Error!!!\n"
                       "No phase with name '{0}' exists in OptimalControlProblem.\n",
                       name);
            throw std::invalid_argument("");
        }
        return int(nameit - phase_names.begin());
    }

    int get_phase_num(PhasePtr p) {
        auto ptrit = std::find_if(phases.begin(), phases.end(),
                                  [&](PhasePtr pt) { return pt.get() == p.get(); });
        if (ptrit == phases.end()) {
            fmt::print(fmt::fg(fmt::color::red),
                       "Transcription Error!!!\n"
                       "The requested phase does not exist in OptimalControlProblem\n");
            throw std::invalid_argument("");
        }
        return int(ptrit - phases.begin());
    }

    int get_phase_num(PhaseRefType phase_t) {
        int phasenum;

        if (std::holds_alternative<int>(phase_t)) {
            phasenum = std::get<int>(phase_t);
        } else if (std::holds_alternative<PhasePtr>(phase_t)) {
            phasenum = this->get_phase_num(std::get<PhasePtr>(phase_t));
        } else if (std::holds_alternative<std::string>(phase_t)) {
            phasenum = this->get_phase_num(std::get<std::string>(phase_t));
        }
        return phasenum;
    }

    std::vector<VectorXi> ptl_from_phase_names(std::vector<std::vector<std::string>> ptlnamevec) {
        std::vector<VectorXi> ptl;
        for (auto &appl : ptlnamevec) {
            VectorXi ptlv(appl.size());
            for (int i = 0; i < appl.size(); i++) {
                ptlv[i] = this->get_phase_num(appl[i]);
            }
            ptl.push_back(ptlv);
        }
        return ptl;
    }

    std::vector<VectorXi> ptl_from_phases(std::vector<std::vector<PhasePtr>> ptlnamevec) {
        std::vector<VectorXi> ptl;
        for (auto &appl : ptlnamevec) {
            VectorXi ptlv(appl.size());
            for (int i = 0; i < appl.size(); i++) {
                ptlv[i] = this->get_phase_num(appl[i]);
            }
            ptl.push_back(ptlv);
        }
        return ptl;
    }

    void remove_phase(int ith) {
        this->reset_transcription();
        if (ith < 0)
            ith = (this->phases.size() + ith);
        this->phases.erase(this->phases.begin() + ith);
        this->phase_names.erase(this->phase_names.begin() + ith);
    }
    PhasePtr Phase(int ith) {
        if (ith < 0)
            ith = (this->phases.size() + ith);
        return this->phases[ith];
    }

    VectorXi get_lp_vars(VarIndexType LPvars_t) const {

        VectorXi LPvars;

        if (std::holds_alternative<int>(LPvars_t)) {
            LPvars.resize(1);
            LPvars[0] = std::get<int>(LPvars_t);
        } else if (std::holds_alternative<VectorXi>(LPvars_t)) {
            LPvars = std::get<VectorXi>(LPvars_t);
        } else if (std::holds_alternative<std::string>(LPvars_t)) {
            LPvars = this->get_lp_idx(std::get<std::string>(LPvars_t));
        } else if (std::holds_alternative<std::vector<std::string>>(LPvars_t)) {
            std::vector<VectorXi> varvec;
            int size = 0;
            auto tmpvars = std::get<std::vector<std::string>>(LPvars_t);
            for (auto tmpv : tmpvars) {
                varvec.push_back(this->get_lp_idx(tmpv));
                size += varvec.back().size();
            }
            LPvars.resize(size);
            int next = 0;
            for (auto varv : varvec) {
                for (int i = 0; i < varv.size(); i++) {
                    LPvars[next] = varv[i];
                    next++;
                }
            }
        }

        return LPvars;
    }

    template <class FuncHolder, class FuncType>
    FuncHolder make_func_impl(FuncType fun, std::vector<PhasePack> packs, VarIndexType lv,
                            ScaleType scale_t) {

        int npacks = packs.size();
        std::vector<Eigen::VectorXi> PTL;
        VectorXi phasenums(npacks);
        Eigen::Matrix<PhaseRegionFlags, -1, 1> RegFlags(npacks);
        std::vector<Eigen::VectorXi> xtvs(npacks);
        std::vector<Eigen::VectorXi> opvs(npacks);
        std::vector<Eigen::VectorXi> spvs(npacks);

        for (int i = 0; i < npacks; i++) {

            auto phase_t = std::get<0>(packs[i]); // Name of phase, either phaseptr,int, or string
            int phasenum = get_phase_num(phase_t);

            if (phasenum < 0 || phasenum >= this->phases.size()) {
                throw std::invalid_argument(
                    fmt::format("Function references non - existent phase : {0:}\n", phasenum));
            }

            phasenums[i] = phasenum;

            RegFlags[i] = this->phases[phasenum]->get_region(std::get<1>(packs[i]));
            xtvs[i] = this->phases[phasenum]->get_xt_up_vars(RegFlags[i], std::get<2>(packs[i]));
            opvs[i] = this->phases[phasenum]->get_op_vars(RegFlags[i], std::get<3>(packs[i]));
            spvs[i] = this->phases[phasenum]->get_sp_vars(RegFlags[i], std::get<4>(packs[i]));
        }

        std::vector<Eigen::VectorXi> lvs;
        lvs.push_back(get_lp_vars(lv));
        PTL.push_back(phasenums);

        auto func = FuncHolder(fun, RegFlags, PTL, xtvs, opvs, spvs, lvs, scale_t);
        return func;
    }

    template <class FuncHolder, class FuncType>
    FuncHolder make_func_impl(FuncType fun, PhaseRefType p0, RegionType reg0, VarIndexType XtUV0,
                            VarIndexType OPV0, VarIndexType SPV0, PhaseRefType p1, RegionType reg1,
                            VarIndexType XtUV1, VarIndexType OPV1, VarIndexType SPV1,
                            VarIndexType lv, ScaleType scale_t) {

        auto pack0 = PhasePack{p0, reg0, XtUV0, OPV0, SPV0};
        auto pack1 = PhasePack{p1, reg1, XtUV1, OPV1, SPV1};

        auto packs = std::vector{pack0, pack1};
        return make_func_impl<FuncHolder, FuncType>(fun, packs, lv, scale_t);
    }

    template <class FuncHolder, class FuncType>
    FuncHolder make_func_impl(FuncType fun, PhaseRefType p0, RegionType reg0_t, VarIndexType v0,
                            PhaseRefType p1, RegionType reg1_t, VarIndexType v1, VarIndexType lv,
                            ScaleType scale_t) {

        VarIndexType xtv0, opv0, spv0, xtv1, opv1, spv1;

        xtv0 = VectorXi();
        opv0 = VectorXi();
        spv0 = VectorXi();

        xtv1 = VectorXi();
        opv1 = VectorXi();
        spv1 = VectorXi();

        PhaseRegionFlags reg0, reg1;

        if (std::holds_alternative<PhaseRegionFlags>(reg0_t)) {
            reg0 = std::get<PhaseRegionFlags>(reg0_t);
        } else if (std::holds_alternative<std::string>(reg0_t)) {
            reg0 = strto_PhaseRegionFlag(std::get<std::string>(reg0_t));
        }

        if (std::holds_alternative<PhaseRegionFlags>(reg1_t)) {
            reg1 = std::get<PhaseRegionFlags>(reg1_t);
        } else if (std::holds_alternative<std::string>(reg1_t)) {
            reg1 = strto_PhaseRegionFlag(std::get<std::string>(reg1_t));
        }

        if (reg0 == PhaseRegionFlags::ODEParams)
            opv0 = v0;
        else if (reg0 == PhaseRegionFlags::StaticParams)
            spv0 = v0;
        else
            xtv0 = v0;

        if (reg1 == PhaseRegionFlags::ODEParams)
            opv1 = v1;
        else if (reg1 == PhaseRegionFlags::StaticParams)
            spv1 = v1;
        else
            xtv1 = v1;

        auto pack0 = PhasePack{p0, reg0, xtv0, opv0, spv0};
        auto pack1 = PhasePack{p1, reg1, xtv1, opv1, spv1};
        auto packs = std::vector{pack0, pack1};

        return make_func_impl<FuncHolder, FuncType>(fun, packs, lv, scale_t);
    }

    template <class FuncHolder, class FuncType>
    FuncHolder make_func_impl(FuncType fun, PhaseRefType p0, RegionType reg0_t, VarIndexType v0,
                            PhaseRefType p1, RegionType reg1_t, VarIndexType v1,
                            ScaleType scale_t) {

        VectorXi empty;

        return make_func_impl<FuncHolder, FuncType>(fun, p0, reg0_t, v0, p1, reg1_t, v1, empty,
                                                  scale_t);
    }

    /////////////////////////////////////////////////

    template <class FuncMap>
    void remove_func_impl(FuncMap &map, int index, const std::string &funcstr) {
        this->reset_transcription();
        this->invalidate_post_opt_info();
        if (index == -1 && map.size() > 0) {
            index = map.rbegin()->first;
        }

        if (map.count(index) == 0) {
            throw std::invalid_argument(fmt::format(
                "No {0:} with index {1:} exists in Optimal Control Problem.", funcstr, index));
        }
        map.erase(index);
    }

    template <class FuncType, class FuncMap>
    int add_func_impl(FuncType func, FuncMap &map, const std::string &funcstr) {
        this->reset_transcription();
        this->invalidate_post_opt_info();
        int index = map.size() == 0 ? 0 : map.rbegin()->first + 1;
        map[index] = func;
        map[index].StorageIndex = index;

        check_function_size(map.at(index), funcstr);
        return index;
    }

    ////////////////////////////////////////////////
    int add_link_equal_con(LinkConstraint lc) {
        return add_func_impl(lc, this->LinkEqualities, "Link Equality Constraint");
    }
    ////////////////////////////////////////////

    /////////////// THE NEW EQUALCON INTERFACE//////////////////////////////

    int add_link_equal_con(VectorFunctionalX lc, std::vector<PhasePack> packs, VarIndexType lv,
                        ScaleType scale_t) {
        auto Func = this->make_func_impl<LinkConstraint, VectorFunctionalX>(lc, packs, lv, scale_t);
        return add_func_impl(Func, this->LinkEqualities, "Link Equality Constraint");
    }

    int add_link_equal_con(VectorFunctionalX lc, PhaseRefType p0, RegionType reg0, VarIndexType XtUV0,
                        VarIndexType OPV0, VarIndexType SPV0, PhaseRefType p1, RegionType reg1,
                        VarIndexType XtUV1, VarIndexType OPV1, VarIndexType SPV1, VarIndexType lv,
                        ScaleType scale_t) {

        auto Func = this->make_func_impl<LinkConstraint, VectorFunctionalX>(
            lc, p0, reg0, XtUV0, OPV0, SPV0, p1, reg1, XtUV1, OPV1, SPV1, lv, scale_t);
        return add_func_impl(Func, this->LinkEqualities, "Link Equality Constraint");
    }

    int add_link_equal_con(VectorFunctionalX lc, PhaseRefType p0, RegionType reg0, VarIndexType v0,
                        PhaseRefType p1, RegionType reg1, VarIndexType v1, VarIndexType lv,
                        ScaleType scale_t) {

        auto Func = this->make_func_impl<LinkConstraint, VectorFunctionalX>(lc, p0, reg0, v0, p1,
                                                                          reg1, v1, lv, scale_t);
        return add_func_impl(Func, this->LinkEqualities, "Link Equality Constraint");
    }
    int add_link_equal_con(VectorFunctionalX lc, PhaseRefType p0, RegionType reg0, VarIndexType v0,
                        PhaseRefType p1, RegionType reg1, VarIndexType v1, ScaleType scale_t) {

        auto Func = this->make_func_impl<LinkConstraint, VectorFunctionalX>(lc, p0, reg0, v0, p1,
                                                                          reg1, v1, scale_t);
        return add_func_impl(Func, this->LinkEqualities, "Link Equality Constraint");
    }

    std::vector<int> add_forward_link_equal_con(PhaseRefType iphase_t, PhaseRefType fphase_t,
                                            VarIndexType vars, ScaleType scale_t) {

        int iphase = get_phase_num(iphase_t);
        int fphase = get_phase_num(fphase_t);

        if (iphase < 0)
            iphase = (this->phases.size() + iphase);
        if (fphase < 0)
            fphase = (this->phases.size() + fphase);

        if (iphase < 0 || iphase >= this->phases.size()) {
            throw std::invalid_argument(fmt::format(
                "Link Equality constraint references non-existent phase:{0:}\n", iphase));
        }

        int vsize = this->phases[iphase]->get_xt_up_vars(PhaseRegionFlags::Front, vars).size();

        auto args = Arguments<-1>(2 * vsize);
        auto func = args.head<-1>(vsize) - args.tail<-1>(vsize);

        std::vector<int> idxs;
        for (int i = iphase; i < fphase; i++) {

            int idx = this->add_link_equal_con(func, i, "Last", vars, i + 1, "First", vars, scale_t);

            idxs.push_back(idx);
        }

        return idxs;
    }

    int add_direct_link_equal_con(PhaseRefType p0, RegionType reg0_t, VarIndexType v0, PhaseRefType p1,
                              RegionType reg1_t, VarIndexType v1, ScaleType scale_t) {

        int phase = get_phase_num(p0);

        if (phase < 0)
            phase = (this->phases.size() + phase);

        if (phase < 0 || phase >= this->phases.size()) {
            throw std::invalid_argument(fmt::format(
                "Link Equality constraint references non-existent phase:{0:}\n", phase));
        }

        PhaseRegionFlags reg0 = get_PhaseRegion(reg0_t);

        int vsize = this->phases[phase]->get_xt_up_vars(reg0, v0).size();

        auto args = Arguments<-1>(2 * vsize);
        auto func = args.head<-1>(vsize) - args.tail<-1>(vsize);

        return this->add_link_equal_con(func, p0, reg0_t, v0, p1, reg1_t, v1, scale_t);
    }

    std::vector<int> add_param_link_equal_con(PhaseRefType iphase_t, PhaseRefType fphase_t,
                                          RegionType reg0_t, VarIndexType vars, ScaleType scale_t) {

        PhaseRegionFlags reg0 = get_PhaseRegion(reg0_t);

        if (reg0 != PhaseRegionFlags::ODEParams && reg0 != PhaseRegionFlags::StaticParams) {
            throw std::invalid_argument("Phase Region must be ODEParams or StaticParams");
        }

        int iphase = get_phase_num(iphase_t);
        int fphase = get_phase_num(fphase_t);

        if (iphase < 0)
            iphase = (this->phases.size() + iphase);
        if (fphase < 0)
            fphase = (this->phases.size() + fphase);

        if (iphase < 0 || iphase >= this->phases.size()) {
            throw std::invalid_argument(fmt::format(
                "Link Equality constraint references non-existent phase:{0:}\n", iphase));
        }

        std::vector<int> idxs;
        for (int i = iphase; i < fphase; i++) {

            int idx = this->add_direct_link_equal_con(i, reg0, vars, i + 1, reg0, vars, scale_t);

            idxs.push_back(idx);
        }

        return idxs;
    }

    int add_link_param_equal_con(VectorFunctionalX lc, std::vector<VectorXi> lpvs, ScaleType scale_t) {
        std::vector<Eigen::VectorXi> empty;
        return this->add_link_equal_con(
            LinkConstraint(lc, LinkFlags::LinkParams, empty, empty, empty, empty, lpvs, scale_t));
    }
    int add_link_param_equal_con(VectorFunctionalX lc, VectorXi lpv, ScaleType scale_t) {
        std::vector<Eigen::VectorXi> lpvs;
        lpvs.push_back(lpv);
        return this->add_link_param_equal_con(lc, lpvs, scale_t);
    }

    //////////////////////////////////////////////////////////////

    int add_link_equal_con(VectorFunctionalX lc, RegVec regs, std::vector<VectorXi> ptl,
                        std::vector<VectorXi> xtuvs, std::vector<VectorXi> opvs,
                        std::vector<VectorXi> spvs, std::vector<VectorXi> lpvs) {
        auto LC = LinkConstraint(lc, regs, ptl, xtuvs, opvs, spvs, lpvs);
        return this->add_link_equal_con(LC);
    }

    int add_link_equal_con(VectorFunctionalX lc, std::vector<std::string> regs,
                        std::vector<VectorXi> ptl, std::vector<VectorXi> xtuvs,
                        std::vector<VectorXi> opvs, std::vector<VectorXi> spvs,
                        std::vector<VectorXi> lpvs) {
        auto LC = LinkConstraint(lc, regs, ptl, xtuvs, opvs, spvs, lpvs);
        return this->add_link_equal_con(LC);
    }

    int add_link_equal_con(VectorFunctionalX lc, LinkFlags regs, std::vector<VectorXi> ptl,
                        std::vector<VectorXi> xtuvs, std::vector<VectorXi> opvs,
                        std::vector<VectorXi> spvs, std::vector<VectorXi> lpvs) {
        auto LC = LinkConstraint(lc, regs, ptl, xtuvs, opvs, spvs, lpvs);
        return this->add_link_equal_con(LC);
    }

    int add_link_equal_con(VectorFunctionalX lc, std::string regs, std::vector<VectorXi> ptl,
                        std::vector<VectorXi> xtuvs, std::vector<VectorXi> opvs,
                        std::vector<VectorXi> spvs, std::vector<VectorXi> lpvs) {
        auto LC = LinkConstraint(lc, regs, ptl, xtuvs, opvs, spvs, lpvs);
        return this->add_link_equal_con(LC);
    }

    int add_link_equal_con(VectorFunctionalX lc, RegVec regs, std::vector<std::vector<PhasePtr>> ptl,
                        std::vector<VectorXi> xtuvs, std::vector<VectorXi> opvs,
                        std::vector<VectorXi> spvs, std::vector<VectorXi> lpvs) {
        auto LC = LinkConstraint(lc, regs, ptl_from_phases(ptl), xtuvs, opvs, spvs, lpvs);
        return this->add_link_equal_con(LC);
    }

    int add_link_equal_con(VectorFunctionalX lc, std::vector<std::string> regs,
                        std::vector<std::vector<PhasePtr>> ptl, std::vector<VectorXi> xtuvs,
                        std::vector<VectorXi> opvs, std::vector<VectorXi> spvs,
                        std::vector<VectorXi> lpvs) {
        auto LC = LinkConstraint(lc, regs, ptl_from_phases(ptl), xtuvs, opvs, spvs, lpvs);
        return this->add_link_equal_con(LC);
    }

    int add_link_equal_con(VectorFunctionalX lc, LinkFlags regs,
                        std::vector<std::vector<PhasePtr>> ptl, std::vector<VectorXi> xtuvs,
                        std::vector<VectorXi> opvs, std::vector<VectorXi> spvs,
                        std::vector<VectorXi> lpvs) {
        auto LC = LinkConstraint(lc, regs, ptl_from_phases(ptl), xtuvs, opvs, spvs, lpvs);
        return this->add_link_equal_con(LC);
    }

    int add_link_equal_con(VectorFunctionalX lc, std::string regs,
                        std::vector<std::vector<PhasePtr>> ptl, std::vector<VectorXi> xtuvs,
                        std::vector<VectorXi> opvs, std::vector<VectorXi> spvs,
                        std::vector<VectorXi> lpvs) {
        auto LC = LinkConstraint(lc, regs, ptl_from_phases(ptl), xtuvs, opvs, spvs, lpvs);
        return this->add_link_equal_con(LC);
    }

    ////////////////////////////////////////////////

    int add_link_equal_con(VectorFunctionalX lc, RegVec regs, std::vector<VectorXi> ptl,
                        std::vector<VectorXi> xtuvs, std::vector<VectorXi> lpvs) {
        std::vector<Eigen::VectorXi> empty;
        empty.resize(regs.size());
        auto LC = LinkConstraint(lc, regs, ptl, xtuvs, empty, empty, lpvs);
        return this->add_link_equal_con(LC);
    }

    int add_link_equal_con(VectorFunctionalX lc, RegVec regs, std::vector<std::vector<PhasePtr>> ptl,
                        std::vector<VectorXi> xtuvs, std::vector<VectorXi> lpvs) {
        return this->add_link_equal_con(lc, regs, ptl_from_phases(ptl), xtuvs, lpvs);
    }

    int add_link_equal_con(VectorFunctionalX lc, LinkFlags regs, std::vector<VectorXi> ptl,
                        VectorXi xtuvs) {
        auto LC = LinkConstraint(lc, regs, ptl, xtuvs);
        return this->add_link_equal_con(LC);
    }
    int add_link_equal_con(VectorFunctionalX lc, std::string regs, std::vector<VectorXi> ptl,
                        VectorXi xtuvs) {
        auto LC = LinkConstraint(lc, regs, ptl, xtuvs);
        return this->add_link_equal_con(LC);
    }

    int add_link_equal_con(VectorFunctionalX lc, LinkFlags regs,
                        std::vector<std::vector<PhasePtr>> ptl, VectorXi xtuvs) {
        return this->add_link_equal_con(lc, regs, ptl_from_phases(ptl), xtuvs);
    }
    int add_link_equal_con(VectorFunctionalX lc, std::string regs,
                        std::vector<std::vector<PhasePtr>> ptl, VectorXi xtuvs) {
        return this->add_link_equal_con(lc, regs, ptl_from_phases(ptl), xtuvs);
    }

    /////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////

    int add_link_param_equal_con(VectorFunctionalX lc, std::vector<VectorXi> lpvs) {
        std::vector<Eigen::VectorXi> empty;
        return this->add_link_equal_con(
            LinkConstraint(lc, LinkFlags::LinkParams, empty, empty, empty, empty, lpvs));
    }
    int add_link_param_equal_con(VectorFunctionalX lc, VectorXi lpvs) {
        std::vector<Eigen::VectorXi> lpvss;
        lpvss.push_back(lpvs);
        return this->add_link_param_equal_con(lc, lpvss);
    }

    int add_forward_link_equal_con(int iphase, int fphase, VectorXi vars) {
        if (iphase < 0)
            iphase = (this->phases.size() + iphase);
        if (fphase < 0)
            fphase = (this->phases.size() + fphase);

        std::vector<Eigen::VectorXi> PTL;
        for (int i = iphase; i < fphase; i++) {
            VectorXi pl(2);
            pl[0] = i;
            pl[1] = i + 1;
            PTL.push_back(pl);
        }
        std::vector<Eigen::VectorXi> xtv(2);
        xtv[0] = vars;
        xtv[1] = vars;

        auto args = Arguments<-1>(2 * vars.size());
        auto func = args.head<-1>(vars.size()) - args.tail<-1>(vars.size());

        return this->add_link_equal_con(LinkConstraint(func, LinkFlags::BackToFront, PTL, xtv));
    }

    int add_forward_link_equal_con(PhasePtr iphase, PhasePtr fphase, VectorXi vars) {
        return this->add_forward_link_equal_con(get_phase_num(iphase), get_phase_num(fphase), vars);
    }
    /////////////////////////////////////////////////////////////////////////

    int add_direct_link_equal_con(LinkFlags LinkFlag, int iphase, VectorXi v1, int fphase,
                              VectorXi v2) {
        if (iphase < 0)
            iphase = (this->phases.size() + iphase);
        if (fphase < 0)
            fphase = (this->phases.size() + fphase);

        std::vector<Eigen::VectorXi> PTL;
        VectorXi pl(2);
        pl[0] = iphase;
        pl[1] = fphase;
        PTL.push_back(pl);

        std::vector<Eigen::VectorXi> xtv(2);
        xtv[0] = v1;
        xtv[1] = v2;

        auto args = Arguments<-1>(2 * v1.size());
        auto func = args.head<-1>(v1.size()) - args.tail<-1>(v2.size());
        return this->add_link_equal_con(LinkConstraint(func, LinkFlag, PTL, xtv));
    }

    int add_direct_link_equal_con(VectorFunctionalX lc, int iphase, PhaseRegionFlags f1, VectorXi v1,
                              int fphase, PhaseRegionFlags f2, VectorXi v2) {
        if (iphase < 0)
            iphase = (this->phases.size() + iphase);
        if (fphase < 0)
            fphase = (this->phases.size() + fphase);

        std::vector<Eigen::VectorXi> PTL;
        VectorXi pl(2);
        pl[0] = iphase;
        pl[1] = fphase;
        PTL.push_back(pl);

        Eigen::Matrix<PhaseRegionFlags, -1, 1> RegFlags(2);
        RegFlags[0] = f1;
        RegFlags[1] = f2;

        std::vector<Eigen::VectorXi> xtv(2);
        std::vector<Eigen::VectorXi> opv(2);
        std::vector<Eigen::VectorXi> spv(2);

        if (f1 == PhaseRegionFlags::ODEParams)
            opv[0] = v1;
        else if (f1 == PhaseRegionFlags::StaticParams)
            spv[0] = v1;
        else
            xtv[0] = v1;

        if (f2 == PhaseRegionFlags::ODEParams)
            opv[1] = v2;
        else if (f2 == PhaseRegionFlags::StaticParams)
            spv[1] = v2;
        else
            xtv[1] = v2;

        std::vector<Eigen::VectorXi> lv(1);

        return this->add_link_equal_con(LinkConstraint(lc, RegFlags, PTL, xtv, opv, spv, lv));
    }

    int add_direct_link_equal_con(VectorFunctionalX lc, int iphase, std::string f1, VectorXi v1,
                              int fphase, std::string f2, VectorXi v2) {
        return this->add_direct_link_equal_con(lc, iphase, strto_PhaseRegionFlag(f1), v1, fphase,
                                           strto_PhaseRegionFlag(f2), v2);
    }

    int add_direct_link_equal_con(int iphase, PhaseRegionFlags f1, VectorXi v1, int fphase,
                              PhaseRegionFlags f2, VectorXi v2) {
        auto args = Arguments<-1>(2 * v1.size());
        auto func = args.head<-1>(v1.size()) - args.tail<-1>(v2.size());
        return this->add_direct_link_equal_con(func, iphase, f1, v1, fphase, f2, v2);
    }

    int add_direct_link_equal_con(int iphase, std::string f1, VectorXi v1, int fphase, std::string f2,
                              VectorXi v2) {
        return this->add_direct_link_equal_con(iphase, strto_PhaseRegionFlag(f1), v1, fphase,
                                           strto_PhaseRegionFlag(f2), v2);
    }

    int add_direct_link_equal_con(PhasePtr iphase, PhaseRegionFlags f1, VectorXi v1, PhasePtr fphase,
                              PhaseRegionFlags f2, VectorXi v2) {
        return this->add_direct_link_equal_con(get_phase_num(iphase), f1, v1, get_phase_num(fphase), f2,
                                           v2);
    }

    int add_direct_link_equal_con(PhasePtr iphase, std::string f1, VectorXi v1, PhasePtr fphase,
                              std::string f2, VectorXi v2) {
        return this->add_direct_link_equal_con(get_phase_num(iphase), f1, v1, get_phase_num(fphase), f2,
                                           v2);
    }

    int add_direct_link_equal_con(VectorFunctionalX lc, PhasePtr iphase, PhaseRegionFlags f1,
                              VectorXi v1, PhasePtr fphase, PhaseRegionFlags f2, VectorXi v2) {
        return this->add_direct_link_equal_con(lc, get_phase_num(iphase), f1, v1, get_phase_num(fphase), f2,
                                           v2);
    }
    int add_direct_link_equal_con(VectorFunctionalX lc, PhasePtr iphase, std::string f1, VectorXi v1,
                              PhasePtr fphase, std::string f2, VectorXi v2) {
        return this->add_direct_link_equal_con(lc, get_phase_num(iphase), f1, v1, get_phase_num(fphase), f2,
                                           v2);
    }

    ////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////

    int add_link_inequal_con(LinkConstraint lc) {
        return add_func_impl(lc, this->LinkInequalities, "Link Inequality Constraint");
    }

    int add_link_inequal_con(VectorFunctionalX lc, std::vector<PhasePack> packs, VarIndexType lv,
                          ScaleType scale_t) {
        auto Func = this->make_func_impl<LinkConstraint, VectorFunctionalX>(lc, packs, lv, scale_t);
        return add_func_impl(Func, this->LinkInequalities, "Link Inequality Constraint");
    }

    int add_link_inequal_con(VectorFunctionalX lc, PhaseRefType p0, RegionType reg0,
                          VarIndexType XtUV0, VarIndexType OPV0, VarIndexType SPV0, PhaseRefType p1,
                          RegionType reg1, VarIndexType XtUV1, VarIndexType OPV1, VarIndexType SPV1,
                          VarIndexType lv, ScaleType scale_t) {

        auto Func = this->make_func_impl<LinkConstraint, VectorFunctionalX>(
            lc, p0, reg0, XtUV0, OPV0, SPV0, p1, reg1, XtUV1, OPV1, SPV1, lv, scale_t);
        return add_func_impl(Func, this->LinkInequalities, "Link Inequality Constraint");
    }

    int add_link_inequal_con(VectorFunctionalX lc, PhaseRefType p0, RegionType reg0, VarIndexType v0,
                          PhaseRefType p1, RegionType reg1, VarIndexType v1, VarIndexType lv,
                          ScaleType scale_t) {

        auto Func = this->make_func_impl<LinkConstraint, VectorFunctionalX>(lc, p0, reg0, v0, p1,
                                                                          reg1, v1, lv, scale_t);
        return add_func_impl(Func, this->LinkInequalities, "Link Inequality Constraint");
    }
    int add_link_inequal_con(VectorFunctionalX lc, PhaseRefType p0, RegionType reg0, VarIndexType v0,
                          PhaseRefType p1, RegionType reg1, VarIndexType v1, ScaleType scale_t) {

        auto Func = this->make_func_impl<LinkConstraint, VectorFunctionalX>(lc, p0, reg0, v0, p1,
                                                                          reg1, v1, scale_t);
        return add_func_impl(Func, this->LinkInequalities, "Link Inequality Constraint");
    }
    int add_link_param_inequal_con(VectorFunctionalX lc, std::vector<VectorXi> lpvs,
                               ScaleType scale_t) {
        std::vector<Eigen::VectorXi> empty;
        return this->add_link_inequal_con(
            LinkConstraint(lc, LinkFlags::LinkParams, empty, empty, empty, empty, lpvs, scale_t));
    }
    int add_link_param_inequal_con(VectorFunctionalX lc, VectorXi lpv, ScaleType scale_t) {
        std::vector<Eigen::VectorXi> lpvs;
        lpvs.push_back(lpv);
        return this->add_link_param_inequal_con(lc, lpvs, scale_t);
    }

    //////////////////////////

    int add_link_inequal_con(VectorFunctionalX lc, RegVec regs, std::vector<VectorXi> ptl,
                          std::vector<VectorXi> xtuvs, std::vector<VectorXi> opvs,
                          std::vector<VectorXi> spvs, std::vector<VectorXi> lpvs) {
        auto LC = LinkConstraint(lc, regs, ptl, xtuvs, opvs, spvs, lpvs);
        return this->add_link_inequal_con(LC);
    }

    int add_link_inequal_con(VectorFunctionalX lc, std::vector<std::string> regs,
                          std::vector<VectorXi> ptl, std::vector<VectorXi> xtuvs,
                          std::vector<VectorXi> opvs, std::vector<VectorXi> spvs,
                          std::vector<VectorXi> lpvs) {
        auto LC = LinkConstraint(lc, regs, ptl, xtuvs, opvs, spvs, lpvs);
        return this->add_link_inequal_con(LC);
    }

    int add_link_inequal_con(VectorFunctionalX lc, LinkFlags regs, std::vector<VectorXi> ptl,
                          std::vector<VectorXi> xtuvs, std::vector<VectorXi> opvs,
                          std::vector<VectorXi> spvs, std::vector<VectorXi> lpvs) {
        auto LC = LinkConstraint(lc, regs, ptl, xtuvs, opvs, spvs, lpvs);
        return this->add_link_inequal_con(LC);
    }

    int add_link_inequal_con(VectorFunctionalX lc, std::string regs, std::vector<VectorXi> ptl,
                          std::vector<VectorXi> xtuvs, std::vector<VectorXi> opvs,
                          std::vector<VectorXi> spvs, std::vector<VectorXi> lpvs) {
        auto LC = LinkConstraint(lc, regs, ptl, xtuvs, opvs, spvs, lpvs);
        return this->add_link_inequal_con(LC);
    }

    int add_link_inequal_con(VectorFunctionalX lc, RegVec regs, std::vector<std::vector<PhasePtr>> ptl,
                          std::vector<VectorXi> xtuvs, std::vector<VectorXi> opvs,
                          std::vector<VectorXi> spvs, std::vector<VectorXi> lpvs) {
        auto LC = LinkConstraint(lc, regs, ptl_from_phases(ptl), xtuvs, opvs, spvs, lpvs);
        return this->add_link_inequal_con(LC);
    }

    int add_link_inequal_con(VectorFunctionalX lc, std::vector<std::string> regs,
                          std::vector<std::vector<PhasePtr>> ptl, std::vector<VectorXi> xtuvs,
                          std::vector<VectorXi> opvs, std::vector<VectorXi> spvs,
                          std::vector<VectorXi> lpvs) {
        auto LC = LinkConstraint(lc, regs, ptl_from_phases(ptl), xtuvs, opvs, spvs, lpvs);
        return this->add_link_inequal_con(LC);
    }

    int add_link_inequal_con(VectorFunctionalX lc, LinkFlags regs,
                          std::vector<std::vector<PhasePtr>> ptl, std::vector<VectorXi> xtuvs,
                          std::vector<VectorXi> opvs, std::vector<VectorXi> spvs,
                          std::vector<VectorXi> lpvs) {
        auto LC = LinkConstraint(lc, regs, ptl_from_phases(ptl), xtuvs, opvs, spvs, lpvs);
        return this->add_link_inequal_con(LC);
    }

    int add_link_inequal_con(VectorFunctionalX lc, std::string regs,
                          std::vector<std::vector<PhasePtr>> ptl, std::vector<VectorXi> xtuvs,
                          std::vector<VectorXi> opvs, std::vector<VectorXi> spvs,
                          std::vector<VectorXi> lpvs) {
        auto LC = LinkConstraint(lc, regs, ptl_from_phases(ptl), xtuvs, opvs, spvs, lpvs);
        return this->add_link_inequal_con(LC);
    }

    ////////////////////////////////////////////////

    int add_link_inequal_con(VectorFunctionalX lc, LinkFlags regs, std::vector<VectorXi> ptl,
                          VectorXi xtuvs) {
        auto LC = LinkConstraint(lc, regs, ptl, xtuvs);
        return this->add_link_inequal_con(LC);
    }

    int add_link_inequal_con(VectorFunctionalX lc, LinkFlags regs,
                          std::vector<std::vector<PhasePtr>> ptl, VectorXi xtuvs) {
        return this->add_link_inequal_con(lc, regs, ptl_from_phases(ptl), xtuvs);
    }

    int add_link_inequal_con(VectorFunctionalX lc, std::string regs, std::vector<VectorXi> ptl,
                          VectorXi xtuvs) {
        auto LC = LinkConstraint(lc, regs, ptl, xtuvs);
        return this->add_link_inequal_con(LC);
    }

    int add_link_inequal_con(VectorFunctionalX lc, std::string regs,
                          std::vector<std::vector<PhasePtr>> ptl, VectorXi xtuvs) {
        return this->add_link_inequal_con(lc, regs, ptl_from_phases(ptl), xtuvs);
    }

    int add_link_inequal_con(VectorFunctionalX lc, RegVec regs, std::vector<VectorXi> ptl,
                          std::vector<VectorXi> xtuvs, std::vector<VectorXi> lpvs) {
        std::vector<Eigen::VectorXi> empty;
        empty.resize(regs.size());
        auto LC = LinkConstraint(lc, regs, ptl, xtuvs, empty, empty, lpvs);
        return this->add_link_inequal_con(LC);
    }

    int add_link_inequal_con(VectorFunctionalX lc, RegVec regs, std::vector<std::vector<PhasePtr>> ptl,
                          std::vector<VectorXi> xtuvs, std::vector<VectorXi> lpvs) {
        return this->add_link_inequal_con(lc, regs, ptl_from_phases(ptl), xtuvs, lpvs);
    }

    int add_link_param_inequal_con(VectorFunctionalX lc, std::vector<VectorXi> lpvs) {
        std::vector<Eigen::VectorXi> empty;
        return this->add_link_inequal_con(
            LinkConstraint(lc, LinkFlags::LinkParams, empty, empty, empty, empty, lpvs));
    }
    int add_link_param_inequal_con(VectorFunctionalX lc, VectorXi lpvs) {
        std::vector<Eigen::VectorXi> lpvss;
        lpvss.push_back(lpvs);
        return this->add_link_param_inequal_con(lc, lpvss);
    }
    ////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////

    int add_link_objective(LinkObjective lc) {
        return add_func_impl(lc, this->LinkObjectives, "Link Objective");
    }

    int add_link_objective(ScalarFunctionalX lc, std::vector<PhasePack> packs, VarIndexType lv,
                         ScaleType scale_t) {
        auto Func = this->make_func_impl<LinkObjective, ScalarFunctionalX>(lc, packs, lv, scale_t);
        return add_func_impl(Func, this->LinkObjectives, "Link Objective");
    }

    int add_link_objective(ScalarFunctionalX lc, PhaseRefType p0, RegionType reg0, VarIndexType XtUV0,
                         VarIndexType OPV0, VarIndexType SPV0, PhaseRefType p1, RegionType reg1,
                         VarIndexType XtUV1, VarIndexType OPV1, VarIndexType SPV1, VarIndexType lv,
                         ScaleType scale_t) {

        auto Func = this->make_func_impl<LinkObjective, ScalarFunctionalX>(
            lc, p0, reg0, XtUV0, OPV0, SPV0, p1, reg1, XtUV1, OPV1, SPV1, lv, scale_t);
        return add_func_impl(Func, this->LinkObjectives, "Link Objective");
    }

    int add_link_objective(ScalarFunctionalX lc, PhaseRefType p0, RegionType reg0, VarIndexType v0,
                         PhaseRefType p1, RegionType reg1, VarIndexType v1, VarIndexType lv,
                         ScaleType scale_t) {

        auto Func = this->make_func_impl<LinkObjective, ScalarFunctionalX>(lc, p0, reg0, v0, p1, reg1,
                                                                         v1, lv, scale_t);
        return add_func_impl(Func, this->LinkObjectives, "Link Objective");
    }
    int add_link_objective(ScalarFunctionalX lc, PhaseRefType p0, RegionType reg0, VarIndexType v0,
                         PhaseRefType p1, RegionType reg1, VarIndexType v1, ScaleType scale_t) {

        auto Func = this->make_func_impl<LinkObjective, ScalarFunctionalX>(lc, p0, reg0, v0, p1, reg1,
                                                                         v1, scale_t);
        return add_func_impl(Func, this->LinkObjectives, "Link Objective");
    }
    int add_link_param_objective(ScalarFunctionalX lc, std::vector<VectorXi> lpvs, ScaleType scale_t) {
        std::vector<Eigen::VectorXi> empty;
        return this->add_link_objective(
            LinkObjective(lc, LinkFlags::LinkParams, empty, empty, empty, empty, lpvs, scale_t));
    }
    int add_link_param_objective(ScalarFunctionalX lc, VectorXi lpv, ScaleType scale_t) {
        std::vector<Eigen::VectorXi> lpvs;
        lpvs.push_back(lpv);
        return this->add_link_param_objective(lc, lpvs, scale_t);
    }

    //////////////////////////

    int add_link_objective(ScalarFunctionalX lc, RegVec regs, std::vector<VectorXi> ptl,
                         std::vector<VectorXi> xtuvs, std::vector<VectorXi> opvs,
                         std::vector<VectorXi> spvs, std::vector<VectorXi> lpvs) {
        auto LC = LinkObjective(lc, regs, ptl, xtuvs, opvs, spvs, lpvs);
        return this->add_link_objective(LC);
    }

    int add_link_objective(ScalarFunctionalX lc, std::vector<std::string> regs,
                         std::vector<VectorXi> ptl, std::vector<VectorXi> xtuvs,
                         std::vector<VectorXi> opvs, std::vector<VectorXi> spvs,
                         std::vector<VectorXi> lpvs) {
        auto LC = LinkObjective(lc, regs, ptl, xtuvs, opvs, spvs, lpvs);
        return this->add_link_objective(LC);
    }

    int add_link_objective(ScalarFunctionalX lc, LinkFlags regs, std::vector<VectorXi> ptl,
                         std::vector<VectorXi> xtuvs, std::vector<VectorXi> opvs,
                         std::vector<VectorXi> spvs, std::vector<VectorXi> lpvs) {
        auto LC = LinkObjective(lc, regs, ptl, xtuvs, opvs, spvs, lpvs);
        return this->add_link_objective(LC);
    }

    int add_link_objective(ScalarFunctionalX lc, std::string regs, std::vector<VectorXi> ptl,
                         std::vector<VectorXi> xtuvs, std::vector<VectorXi> opvs,
                         std::vector<VectorXi> spvs, std::vector<VectorXi> lpvs) {
        auto LC = LinkObjective(lc, regs, ptl, xtuvs, opvs, spvs, lpvs);
        return this->add_link_objective(LC);
    }

    int add_link_objective(ScalarFunctionalX lc, RegVec regs, std::vector<std::vector<PhasePtr>> ptl,
                         std::vector<VectorXi> xtuvs, std::vector<VectorXi> opvs,
                         std::vector<VectorXi> spvs, std::vector<VectorXi> lpvs) {
        auto LC = LinkObjective(lc, regs, ptl_from_phases(ptl), xtuvs, opvs, spvs, lpvs);
        return this->add_link_objective(LC);
    }

    int add_link_objective(ScalarFunctionalX lc, std::vector<std::string> regs,
                         std::vector<std::vector<PhasePtr>> ptl, std::vector<VectorXi> xtuvs,
                         std::vector<VectorXi> opvs, std::vector<VectorXi> spvs,
                         std::vector<VectorXi> lpvs) {
        auto LC = LinkObjective(lc, regs, ptl_from_phases(ptl), xtuvs, opvs, spvs, lpvs);
        return this->add_link_objective(LC);
    }

    int add_link_objective(ScalarFunctionalX lc, LinkFlags regs,
                         std::vector<std::vector<PhasePtr>> ptl, std::vector<VectorXi> xtuvs,
                         std::vector<VectorXi> opvs, std::vector<VectorXi> spvs,
                         std::vector<VectorXi> lpvs) {
        auto LC = LinkObjective(lc, regs, ptl_from_phases(ptl), xtuvs, opvs, spvs, lpvs);
        return this->add_link_objective(LC);
    }

    int add_link_objective(ScalarFunctionalX lc, std::string regs,
                         std::vector<std::vector<PhasePtr>> ptl, std::vector<VectorXi> xtuvs,
                         std::vector<VectorXi> opvs, std::vector<VectorXi> spvs,
                         std::vector<VectorXi> lpvs) {
        auto LC = LinkObjective(lc, regs, ptl_from_phases(ptl), xtuvs, opvs, spvs, lpvs);
        return this->add_link_objective(LC);
    }

    ////////////////////////////////////////////////

    int add_link_objective(ScalarFunctionalX lc, LinkFlags regs, std::vector<VectorXi> ptl,
                         VectorXi xtuvs) {
        auto LC = LinkObjective(lc, regs, ptl, xtuvs);
        return this->add_link_objective(LC);
    }

    int add_link_objective(ScalarFunctionalX lc, LinkFlags regs,
                         std::vector<std::vector<PhasePtr>> ptl, VectorXi xtuvs) {
        return this->add_link_objective(lc, regs, ptl_from_phases(ptl), xtuvs);
    }

    int add_link_objective(ScalarFunctionalX lc, std::string regs, std::vector<VectorXi> ptl,
                         VectorXi xtuvs) {
        auto LC = LinkObjective(lc, regs, ptl, xtuvs);
        return this->add_link_objective(LC);
    }

    int add_link_objective(ScalarFunctionalX lc, std::string regs,
                         std::vector<std::vector<PhasePtr>> ptl, VectorXi xtuvs) {
        return this->add_link_objective(lc, regs, ptl_from_phases(ptl), xtuvs);
    }

    int add_link_objective(ScalarFunctionalX lc, RegVec regs, std::vector<VectorXi> ptl,
                         std::vector<VectorXi> xtuvs, std::vector<VectorXi> lpvs) {
        std::vector<Eigen::VectorXi> empty;
        empty.resize(regs.size());
        auto LC = LinkObjective(lc, regs, ptl, xtuvs, empty, empty, lpvs);
        return this->add_link_objective(LC);
    }

    int add_link_objective(ScalarFunctionalX lc, RegVec regs, std::vector<std::vector<PhasePtr>> ptl,
                         std::vector<VectorXi> xtuvs, std::vector<VectorXi> lpvs) {
        return this->add_link_objective(lc, regs, ptl_from_phases(ptl), xtuvs, lpvs);
    }

    int add_link_param_objective(ScalarFunctionalX lc, std::vector<VectorXi> lpvs) {
        std::vector<Eigen::VectorXi> empty;
        return this->add_link_objective(
            LinkObjective(lc, LinkFlags::LinkParams, empty, empty, empty, empty, lpvs));
    }
    int add_link_param_objective(ScalarFunctionalX lc, VectorXi lpvs) {
        std::vector<Eigen::VectorXi> lpvss;
        lpvss.push_back(lpvs);
        return this->add_link_param_objective(lc, lpvss);
    }

    ///////////////////////////////////////////////////

    void remove_link_equal_con(int index) {
        this->remove_func_impl(this->LinkEqualities, index, "Equality Constraint");
    }
    void remove_link_inequal_con(int index) {
        this->remove_func_impl(this->LinkInequalities, index, "Inequality Constraint");
    }
    void remove_link_objective(int index) {
        this->remove_func_impl(this->LinkObjectives, index, "Link Objective");
    }
    ///////////////////////////////////////////////////

    std::vector<Eigen::VectorXd> return_link_equal_con_vals(int index) const {
        if (!this->PostOptInfoValid) {
            throw std::invalid_argument(" Post optimization info unavailable.");
        }
        if (this->LinkEqualities.count(index) == 0) {
            throw std::invalid_argument(fmt::format(
                "No Equality Constraint with index {0:} exists in Optimal Control Problem.",
                index));
        }

        int Gindex = this->LinkEqualities.at(index).GlobalIndex;
        auto Cindex = this->nlp->EqualityConstraints[Gindex].index_data.Cindex;
        int offset = this->num_phase_eq_cons.sum();

        std::vector<Eigen::VectorXd> Allvals;
        for (int i = 0; i < Cindex.cols(); i++) {
            VectorXd vals(Cindex.rows());
            for (int j = 0; j < Cindex.rows(); j++) {
                int idx = Cindex(j, i) - offset;
                vals[j] = this->ActiveEqCons[idx];
            }
            Allvals.push_back(vals);
        }
        return Allvals;
    }

    std::vector<Eigen::VectorXd> return_link_equal_con_lmults(int index) const {
        if (!this->PostOptInfoValid) {
            throw std::invalid_argument(" Post optimization info unavailable.");
        }
        if (this->LinkEqualities.count(index) == 0) {
            throw std::invalid_argument(fmt::format(
                "No Equality Constraint with index {0:} exists in Optimal Control Problem.",
                index));
        }

        int Gindex = this->LinkEqualities.at(index).GlobalIndex;
        auto Cindex = this->nlp->EqualityConstraints[Gindex].index_data.Cindex;
        int offset = this->num_phase_eq_cons.sum();

        std::vector<Eigen::VectorXd> Allvals;
        for (int i = 0; i < Cindex.cols(); i++) {
            VectorXd vals(Cindex.rows());
            for (int j = 0; j < Cindex.rows(); j++) {
                int idx = Cindex(j, i) - offset;
                vals[j] = this->ActiveEqLmults[idx];
            }
            Allvals.push_back(vals);
        }
        return Allvals;
    }

    std::vector<Eigen::VectorXd> return_link_inequal_con_vals(int index) const {
        if (!this->PostOptInfoValid) {
            throw std::invalid_argument(" Post optimization info unavailable.");
        }
        if (this->LinkInequalities.count(index) == 0) {
            throw std::invalid_argument(fmt::format(
                "No Inequality Constraint with index {0:} exists in Optimal Control Problem.",
                index));
        }
        int Gindex = this->LinkInequalities.at(index).GlobalIndex;
        auto Cindex = this->nlp->InequalityConstraints[Gindex].index_data.Cindex;
        int offset = this->num_phase_iq_cons.sum();

        std::vector<Eigen::VectorXd> Allvals;
        for (int i = 0; i < Cindex.cols(); i++) {
            VectorXd vals(Cindex.rows());
            for (int j = 0; j < Cindex.rows(); j++) {
                int idx = Cindex(j, i) - offset;
                vals[j] = this->ActiveIqCons[idx];
            }
            Allvals.push_back(vals);
        }
        return Allvals;
    }

    std::vector<Eigen::VectorXd> return_link_inequal_con_lmults(int index) const {
        if (!this->PostOptInfoValid) {
            throw std::invalid_argument(" Post optimization info unavailable.");
        }
        if (this->LinkInequalities.count(index) == 0) {
            throw std::invalid_argument(fmt::format(
                "No Inequality Constraint with index {0:} exists in Optimal Control Problem.",
                index));
        }

        int Gindex = this->LinkInequalities.at(index).GlobalIndex;
        auto Cindex = this->nlp->InequalityConstraints[Gindex].index_data.Cindex;
        int offset = this->num_phase_iq_cons.sum();

        std::vector<Eigen::VectorXd> Allvals;
        for (int i = 0; i < Cindex.cols(); i++) {
            VectorXd vals(Cindex.rows());
            for (int j = 0; j < Cindex.rows(); j++) {
                int idx = Cindex(j, i) - offset;
                vals[j] = this->ActiveIqLmults[idx];
            }
            Allvals.push_back(vals);
        }
        return Allvals;
    }

    Eigen::VectorXd return_link_equal_con_scales(int index) {
        return this->LinkEqualities.at(index).OutputScales;
    }
    Eigen::VectorXd return_link_inequal_con_scales(int index) {
        return this->LinkInequalities.at(index).OutputScales;
    }
    Eigen::VectorXd return_link_objective_scales(int index) {
        return this->LinkObjectives.at(index).OutputScales;
    }

    ///////////////////////////////////////////////////

    Eigen::VectorXd get_input_scale(LinkFlags lflag, Eigen::Vector<PhaseRegionFlags, -1> regs,
                                    std::vector<VectorXi> phases_to_link,
                                    std::vector<VectorXi> XtUVars, std::vector<VectorXi> OPVars,
                                    std::vector<VectorXi> SPVars, std::vector<VectorXi> LVars);

    std::vector<Eigen::VectorXd>
    get_test_inputs(LinkFlags lflag, Eigen::Vector<PhaseRegionFlags, -1> regs,
                    std::vector<VectorXi> phases_to_link, std::vector<VectorXi> XtUVars,
                    std::vector<VectorXi> OPVars, std::vector<VectorXi> SPVars,
                    std::vector<VectorXi> LVars);

    void check_transcriptions() {
        for (int i = 0; i < this->phases.size(); i++) {
            if (this->phases[i]->do_transcription) {
                this->do_transcription = true;
            }
        }
    }

    void transcribe_phases();

    void check_dupilcate_phases() {
        for (int i = 0; i < this->phases.size(); i++) {
            for (int j = 0; j < this->phases.size(); j++) {
                if (j != i) {

                    if (this->phase_names[i] == this->phase_names[j]) {
                        fmt::print(
                            fmt::fg(fmt::color::red),
                            "Transcription Error!!!\n"
                            "OptimalControlProblem contains Two phases with identical names\n");
                        throw std::invalid_argument("");
                    }

                    if (this->phases[i].get() == this->phases[j].get()) {

                        fmt::print(
                            fmt::fg(fmt::color::red),
                            "Transcription Error!!!\n"
                            "Same phase detected more than once in optimal control problem. \n");
                        throw std::invalid_argument("");
                    }
                }
            }
        }
    }

    void check_functions();

    template <class T> void check_function_size(const T &func, std::string ftype) {
        int irows = func.Func.input_rows();
        switch (func.LinkFlag) {
        case LinkFlags::BackToFront:
        case LinkFlags::FrontToBack:
        case LinkFlags::FrontToFront:
        case LinkFlags::BackToBack:
        case LinkFlags::ParamsToParams:
        case LinkFlags::LinkParams:
        case LinkFlags::PathToPath:
        case LinkFlags::ReadRegions: {

            if (func.LinkParams.size() != func.PhasesTolink.size() &&
                func.PhasesTolink.size() > 0) {
                fmt::print(fmt::fg(fmt::color::red),
                           "Transcription Error!!!\n"
                           "LinkParam Vector Must be same size as PTL Vector "
                           "even if each element of LinkParam Vector is empty (See Docs).");
                throw std::invalid_argument("");
            }

            if (func.LinkParams.size() > 0 && func.PhasesTolink.size() == 0) {
                for (int i = 0; i < func.LinkParams.size(); i++) {
                    int isize = func.LinkParams[i].size();
                    if (irows != isize) {
                        fmt::print(fmt::fg(fmt::color::red),
                                   "Transcription Error!!!\n"
                                   "Input size of {0:} (IRows = {1:}) does not match that implied "
                                   "by indexing "
                                   "parameters (IRows = {2:}).\n",
                                   ftype, irows, isize);
                        throw std::invalid_argument("");
                    }
                }
            } else {
                for (int i = 0; i < func.PhasesTolink.size(); i++) {
                    int isize = func.LinkParams[i].size();

                    if (func.PhasesTolink[i].size() != func.XtUVars.size()) {
                        fmt::print(fmt::fg(fmt::color::red),
                                   "Transcription Error!!!\n"
                                   "Size of PTL vector element must equal size of Phase State "
                                   "Variables Vector");
                        throw std::invalid_argument("");
                    }
                    if (func.PhasesTolink[i].size() != func.OPVars.size()) {
                        fmt::print(fmt::fg(fmt::color::red),
                                   "Transcription Error!!!\n"
                                   "Size of PTL vector element must equal size of Phase ODEParam "
                                   "Variables Vector");
                        throw std::invalid_argument("");
                    }
                    if (func.PhasesTolink[i].size() != func.SPVars.size()) {
                        fmt::print(fmt::fg(fmt::color::red),
                                   "Transcription Error!!!\n"
                                   "Size of PTL vector element must equal size of Phase "
                                   "StaticParam Variables Vector");
                        throw std::invalid_argument("");
                    }
                    if (func.PhasesTolink[i].size() != func.PhaseRegFlags.size()) {
                        fmt::print(fmt::fg(fmt::color::red),
                                   "Transcription Error!!!\n"
                                   "Size of PTL vector element must equal size of Phase Region "
                                   "Flag Vector");
                        throw std::invalid_argument("");
                    }
                    for (int j = 0; j < func.PhasesTolink[i].size(); j++) {
                        auto flag = func.PhaseRegFlags[j];
                        int xmult = 1;
                        switch (flag) {
                        case PhaseRegionFlags::Front:
                        case PhaseRegionFlags::Back:
                        case PhaseRegionFlags::Path:
                        case PhaseRegionFlags::ODEParams:
                        case PhaseRegionFlags::StaticParams:
                        case PhaseRegionFlags::Params:
                            xmult = 1;
                            break;
                        case PhaseRegionFlags::FrontandBack:
                        case PhaseRegionFlags::BackandFront:
                            xmult = 2;
                            break;
                        default: {

                            fmt::print(fmt::fg(fmt::color::red),
                                       "Transcription Error!!!\n"
                                       "Invalid Phase Region requested in link function\n"
                                       "Only the following regions are supported\n"
                                       "    Front, Back, Path,ODEParams,StaticParams, Params, "
                                       "FrontandBack, BackAndFront\n");

                            throw std::invalid_argument("");
                            break;
                        }
                        }
                        isize += func.XtUVars[j].size() * xmult + func.OPVars[j].size() +
                                 func.SPVars[j].size();
                    }
                    if (irows != isize) {
                        fmt::print(fmt::fg(fmt::color::red),
                                   "Transcription Error!!!\n"
                                   "Input size of {0:} (IRows = {1:}) does not match that implied "
                                   "by indexing "
                                   "parameters (IRows = {2:}).\n",
                                   ftype, irows, isize);
                        throw std::invalid_argument("");
                    }
                }
            }

            break;
        }
        default: {
            break;
        }
        }
    }

    void transcribe_links();

    void calc_auto_scales();

    std::vector<double> get_objective_scales();
    void update_objective_scales(double scale);

    void transcribe(bool showstats, bool showfuns);

    void transcribe() { this->transcribe(false, false); }

    void jet_initialize() {
        this->setNumPartitions(1, 1);
        this->optimizer->PrintLevel = 10;
        this->PrintMeshInfo = false;
        this->transcribe();
    }
    void jet_release() {
        this->optimizer->release();
        this->initPartitions();
        this->optimizer->PrintLevel = 0;
        this->PrintMeshInfo = true;
        this->nlp = std::shared_ptr<NonLinearProgram>();
        for (auto &phase : this->phases)
            phase->jet_release();
        this->reset_transcription();
        this->invalidate_post_opt_info();
    }

    ////////////////////////////////////////////////////////////
  protected:
    void init_meshs() {
        this->MeshConverged = false;
        for (auto &phase : this->phases) {
            if (phase->AdaptiveMesh) {
                phase->init_mesh_refinement();
            }
        }
    }

    bool check_meshs(bool printinfo) {
        this->MeshConverged = true;
        for (auto &phase : this->phases) {
            if (phase->AdaptiveMesh) {
                if (!phase->check_mesh())
                    MeshConverged = false;
            }
        }

        return this->MeshConverged;
    }
    void update_meshs(bool printinfo) {
        for (auto &phase : this->phases) {
            if (phase->AdaptiveMesh) {
                if (!phase->MeshConverged) {
                    phase->update_mesh();
                }
            }
        }
    }

    void print_meshs(int iter) {
        MeshIterateInfo::print_header(iter);
        for (int i = 0; i < this->phases.size(); i++) {
            if (this->phases[i]->AdaptiveMesh) {
                this->phases[i]->MeshIters.back().print(i);
            }
        }
    }

    std::array<Eigen::MatrixXi, 2> make_link_Vindex_Cindex(
        LinkFlags Reg, const Eigen::Matrix<PhaseRegionFlags, -1, 1> &PhaseRegs,
        const std::vector<Eigen::VectorXi> &PTL, const std::vector<Eigen::VectorXi> &xtv,
        const std::vector<Eigen::VectorXi> &opv, const std::vector<Eigen::VectorXi> &spv,
        const std::vector<Eigen::VectorXi> &lv, int orows, int &NextCLoc) const;

    PSIOPT::ConvergenceFlags psipot_call_impl(JetJobModes mode);

    PSIOPT::ConvergenceFlags ocp_call_impl(JetJobModes mode);

    VectorXd make_solver_input() const {
        VectorXd Vars(this->num_prob_vars);

        for (int i = 0; i < this->phases.size(); i++) {
            int Start = 0;
            if (i > 0)
                Start = this->num_phase_vars.segment(0, i).sum();
            Vars.segment(Start, this->num_phase_vars[i]) = this->phases[i]->make_solver_input();
        }
        if (this->AutoScaling && this->LPUnits.size() > 0) {
            Vars.tail(this->num_link_params) = this->ActiveLinkParams.cwiseQuotient(this->LPUnits);
        } else {
            Vars.tail(this->num_link_params) = this->ActiveLinkParams;
        }

        return Vars;
    }

    void collect_solver_output(const VectorXd &Vars) {
        for (int i = 0; i < this->phases.size(); i++) {
            int Start = 0;
            if (i > 0)
                Start = this->num_phase_vars.segment(0, i).sum();
            this->phases[i]->collect_solver_output(Vars.segment(Start, this->num_phase_vars[i]));
        }
        if (this->AutoScaling && this->LPUnits.size() > 0) {
            this->ActiveLinkParams = Vars.tail(this->num_link_params).cwiseProduct(this->LPUnits);
        } else {
            this->ActiveLinkParams = Vars.tail(this->num_link_params);
        }
    }
    void collect_solver_multipliers(const VectorXd &EM, const VectorXd &IM) {
        this->MultipliersLoaded = true;
        for (int i = 0; i < this->phases.size(); i++) {
            int EStart = 0;
            if (i > 0)
                EStart = this->num_phase_eq_cons.segment(0, i).sum();
            int IStart = 0;
            if (i > 0)
                IStart = this->num_phase_iq_cons.segment(0, i).sum();
            this->phases[i]->collect_solver_multipliers(EM.segment(EStart, this->num_phase_eq_cons[i]),
                                                      IM.segment(IStart, this->num_phase_iq_cons[i]));
        }
        this->ActiveEqLmults = EM.tail(this->num_link_eq_cons);
        this->ActiveIqLmults = IM.tail(this->num_link_iq_cons);
    }

    void collect_post_opt_info(const VectorXd &EC, const VectorXd &EM, const VectorXd &IC,
                            const VectorXd &IM) {
        this->MultipliersLoaded = true;
        this->PostOptInfoValid = true;

        for (int i = 0; i < this->phases.size(); i++) {
            int EStart = 0;
            if (i > 0)
                EStart = this->num_phase_eq_cons.segment(0, i).sum();
            int IStart = 0;
            if (i > 0)
                IStart = this->num_phase_iq_cons.segment(0, i).sum();
            this->phases[i]->collect_post_opt_info(EC.segment(EStart, this->num_phase_eq_cons[i]),
                                                EM.segment(EStart, this->num_phase_eq_cons[i]),
                                                IC.segment(IStart, this->num_phase_iq_cons[i]),
                                                IM.segment(IStart, this->num_phase_iq_cons[i]));
        }
        this->ActiveEqLmults = EM.tail(this->num_link_eq_cons);
        this->ActiveIqLmults = IM.tail(this->num_link_iq_cons);
        this->ActiveEqCons = EC.tail(this->num_link_eq_cons);
        this->ActiveIqCons = IC.tail(this->num_link_iq_cons);
    }

  public:
    PSIOPT::ConvergenceFlags solve() { return ocp_call_impl(JetJobModes::Solve); }
    PSIOPT::ConvergenceFlags optimize() { return ocp_call_impl(JetJobModes::Optimize); }
    PSIOPT::ConvergenceFlags solve_optimize() { return ocp_call_impl(JetJobModes::SolveOptimize); }
    PSIOPT::ConvergenceFlags solve_optimize_solve() {
        return ocp_call_impl(JetJobModes::SolveOptimizeSolve);
    }
    PSIOPT::ConvergenceFlags optimize_solve() { return ocp_call_impl(JetJobModes::OptimizeSolve); }

    void print_stats(bool showfuns);
};

} // namespace tycho::oc

