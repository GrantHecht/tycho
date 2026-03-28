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

#include "tycho/detail/optimal_control/phase/ode_phase_base.h"

#include "tycho/detail/optimal_control/transcription/lgl_control_splines.h"
#include "tycho/detail/optimal_control/transcription/lgl_integrals.h"
#include "tycho/detail/optimal_control/transcription/mesh_spacing_constraints.h"
#include "tycho/detail/vf/common/value_lock.h"
#include "tycho/detail/vf/scaling/auto_scaling_utils.h"

int tycho::oc::ODEPhaseBase::add_boundary_value(RegionType reg, VarIndexType args,
                                          const std::variant<double, VectorXd> &value_t,
                                          ScaleType scale_t) {
    Eigen::VectorXd value;
    if (std::holds_alternative<double>(value_t)) {
        value.resize(1);
        value[0] = std::get<double>(value_t);
    } else if (std::holds_alternative<VectorXd>(value_t)) {
        value = std::get<VectorXd>(value_t);
    }
    auto Func = Arguments<-1>(value.size()) - value;
    return this->add_equal_con(reg, Func, args, scale_t);
}

int tycho::oc::ODEPhaseBase::add_delta_var_equal_con(VarIndexType var, double value, double scale,
                                             ScaleType scale_t) {
    auto args = Arguments<2>();
    auto x0 = args.coeff<0>();
    auto x1 = args.coeff<1>();
    auto func = ((x1 - x0) - value) * scale;
    return this->add_equal_con(PhaseRegionFlags::FrontandBack, func, var, scale_t);
}

int tycho::oc::ODEPhaseBase::add_value_lock(RegionType reg, VarIndexType args, ScaleType scale_t) {
    int argsize = this->get_xt_up_vars(get_region(reg), args).size();
    return this->add_equal_con(reg, LockArgs<-1>(argsize), args, scale_t);
}

int tycho::oc::ODEPhaseBase::add_periodicity_con(VarIndexType args, ScaleType scale_t) {
    int argsize = this->get_xt_up_vars(PhaseRegionFlags::FrontandBack, args).size();
    auto X = Arguments<-1>(argsize * 2);
    auto Func = X.head(argsize) - X.tail(argsize);
    return this->add_equal_con(PhaseRegionFlags::FrontandBack, Func, args, scale_t);
}
/////////////////////////////////////////////////////////////////////////////////////////////

int tycho::oc::ODEPhaseBase::add_lu_var_bound(RegionType reg, VarIndexType var, double lowerbound,
                                       double upperbound, double lbscale, double ubscale,
                                       ScaleType scale_t) {
    if (lowerbound > upperbound) {
        fmt::print(fmt::fg(fmt::color::red),
                   "Transcription Error!!!\n"
                   "Lower-bound({0:.3e}) greater than Upper-bound({1:.3e}) \n",
                   lowerbound, upperbound);
        throw std::invalid_argument("");
    }
    check_lbscale(lbscale);
    check_ubscale(ubscale);

    auto x = Arguments<1>();
    auto lowbound = (lowerbound - x) * lbscale;
    auto ubound = (x - upperbound) * ubscale;
    auto lubound = StackedOutputs{lowbound, ubound};

    return this->add_inequal_con(reg, lubound, var, scale_t);
}
int tycho::oc::ODEPhaseBase::add_lu_var_bound(PhaseRegionFlags reg, int var, double lowerbound,
                                       double upperbound, double lbscale, double ubscale) {

    if (lowerbound > upperbound) {
        fmt::print(fmt::fg(fmt::color::red),
                   "Transcription Error!!!\n"
                   "Lower-bound({0:.3e}) greater than Upper-bound({1:.3e}) \n",
                   lowerbound, upperbound);
        throw std::invalid_argument("");
    }
    check_lbscale(lbscale);
    check_ubscale(ubscale);

    auto x = Arguments<1>();
    auto lowbound = (lowerbound - x) * lbscale;
    auto ubound = (x - upperbound) * ubscale;

    auto lubound = StackedOutputs{lowbound, ubound};
    VectorXi v(1);
    v[0] = var;
    return this->add_inequal_con(StateConstraint(lubound, reg, v));
}

int tycho::oc::ODEPhaseBase::add_lower_var_bound(RegionType reg, VarIndexType var, double lowerbound,
                                          double lbscale, ScaleType scale_t) {
    check_lbscale(lbscale);
    auto x = Arguments<1>();
    auto lbound = (lowerbound - x) * lbscale;

    return this->add_inequal_con(reg, lbound, var, scale_t);
}

int tycho::oc::ODEPhaseBase::add_upper_var_bound(RegionType reg, VarIndexType var, double upperbound,
                                          double ubscale, ScaleType scale_t) {
    check_ubscale(ubscale);
    auto x = Arguments<1>();
    auto ubound = (x - upperbound) * ubscale;
    return this->add_inequal_con(reg, ubound, var, scale_t);
}

int tycho::oc::ODEPhaseBase::add_lu_func_bound(RegionType reg, ScalarFunctionalX func,
                                        VarIndexType XtUPvars, VarIndexType OPvars,
                                        VarIndexType SPvars, double lowerbound, double upperbound,
                                        double lbscale, double ubscale, ScaleType scale_t) {

    if (lowerbound > upperbound) {
        fmt::print(fmt::fg(fmt::color::red),
                   "Transcription Error!!!\n"
                   "Lower-bound({0:.3e}) greater than Upper-bound({1:.3e}) \n",
                   lowerbound, upperbound);
        throw std::invalid_argument("");
    }
    check_lbscale(lbscale);
    check_ubscale(ubscale);
    auto x = Arguments<1>();
    auto lubound = StackedOutputs{(lowerbound - x) * lbscale, (x - upperbound) * ubscale};
    auto lufun = lubound.eval(func);
    return this->add_inequal_con(reg, lufun, XtUPvars, OPvars, SPvars, scale_t);
}

int tycho::oc::ODEPhaseBase::add_lower_func_bound(RegionType reg, ScalarFunctionalX func,
                                           VarIndexType XtUPvars, VarIndexType OPvars,
                                           VarIndexType SPvars, double lowerbound, double lbscale,
                                           ScaleType scale_t) {
    check_lbscale(lbscale);

    Vector1<double> rhs;
    rhs[0] = lowerbound;
    auto lbfun = ((-1.0 * func) + rhs) * lbscale;

    return this->add_inequal_con(reg, lbfun, XtUPvars, OPvars, SPvars, scale_t);
}

int tycho::oc::ODEPhaseBase::add_upper_func_bound(RegionType reg, ScalarFunctionalX func,
                                           VarIndexType XtUPvars, VarIndexType OPvars,
                                           VarIndexType SPvars, double upperbound, double ubscale,
                                           ScaleType scale_t) {

    check_ubscale(ubscale);

    Vector1<double> rhs;
    rhs[0] = -upperbound;
    auto ubfun = ((func) + rhs) * ubscale;
    return this->add_inequal_con(reg, ubfun, XtUPvars, OPvars, SPvars, scale_t);
}

int tycho::oc::ODEPhaseBase::add_lu_norm_bound(RegionType reg, VarIndexType XtUPvars, double lowerbound,
                                        double upperbound, double lbscale, double ubscale,
                                        ScaleType scale_t) {
    if (lowerbound > upperbound) {
        fmt::print(fmt::fg(fmt::color::red),
                   "Transcription Error!!!\n"
                   "Lower-bound({0:.3e}) greater than Upper-bound({1:.3e}) \n",
                   lowerbound, upperbound);
        throw std::invalid_argument("");
    }
    check_lbscale(lbscale);
    check_ubscale(ubscale);

    int size = this->get_xt_up_vars(get_region(reg), XtUPvars).size();

    auto impl = [&](auto sz) {
        auto x = Arguments<1>();
        auto lubound = StackedOutputs{(lowerbound - x) * lbscale, (x - upperbound) * ubscale};
        auto normfun = lubound.eval(Arguments<sz.value>(size).norm());
        return this->add_inequal_con(reg, normfun, XtUPvars, scale_t);
    };

    switch (size) {
    case 2:
        return impl(int_const<2>());
    case 3:
        return impl(int_const<3>());
    case 4:
        return impl(int_const<4>());
    default:
        return impl(int_const<-1>());
    }
    return 0;
}

int tycho::oc::ODEPhaseBase::add_lu_squared_norm_bound(RegionType reg, VarIndexType XtUPvars,
                                               double lowerbound, double upperbound, double lbscale,
                                               double ubscale, ScaleType scale_t) {
    if (lowerbound > upperbound) {
        fmt::print(fmt::fg(fmt::color::red),
                   "Transcription Error!!!\n"
                   "Lower-bound({0:.3e}) greater than Upper-bound({1:.3e}) \n",
                   lowerbound, upperbound);
        throw std::invalid_argument("");
    }
    check_lbscale(lbscale);
    check_ubscale(ubscale);

    int size = this->get_xt_up_vars(get_region(reg), XtUPvars).size();

    auto impl = [&](auto sz) {
        auto x = Arguments<1>();
        auto lubound = StackedOutputs{(lowerbound - x) * lbscale, (x - upperbound) * ubscale};
        auto normfun = lubound.eval(Arguments<sz.value>(size).squared_norm());
        return this->add_inequal_con(reg, normfun, XtUPvars, scale_t);
    };

    switch (size) {
    case 2:
        return impl(int_const<2>());
    case 3:
        return impl(int_const<3>());
    case 4:
        return impl(int_const<4>());
    default:
        return impl(int_const<-1>());
    }
    return 0;
}

int tycho::oc::ODEPhaseBase::add_lower_norm_bound(RegionType reg, VarIndexType XtUPvars, double lowerbound,
                                           double lbscale, ScaleType scale_t) {
    check_lbscale(lbscale);

    int size = this->get_xt_up_vars(get_region(reg), XtUPvars).size();
    auto impl = [&](auto sz) {
        auto x = Arguments<1>();
        auto normfun = (lowerbound - Arguments<sz.value>(size).norm()) * lbscale;
        return this->add_inequal_con(reg, normfun, XtUPvars, scale_t);
    };
    switch (size) {
    case 2:
        return impl(int_const<2>());
    case 3:
        return impl(int_const<3>());
    case 4:
        return impl(int_const<4>());
    default:
        return impl(int_const<-1>());
    }
}

int tycho::oc::ODEPhaseBase::add_lower_squared_norm_bound(RegionType reg, VarIndexType XtUPvars,
                                                  double lowerbound, double lbscale,
                                                  ScaleType scale_t) {
    check_lbscale(lbscale);

    int size = this->get_xt_up_vars(get_region(reg), XtUPvars).size();
    auto impl = [&](auto sz) {
        auto x = Arguments<1>();
        auto normfun = (lowerbound - Arguments<sz.value>(size).squared_norm()) * lbscale;
        return this->add_inequal_con(reg, normfun, XtUPvars, scale_t);
    };
    switch (size) {
    case 2:
        return impl(int_const<2>());
    case 3:
        return impl(int_const<3>());
    case 4:
        return impl(int_const<4>());
    default:
        return impl(int_const<-1>());
    }
}

int tycho::oc::ODEPhaseBase::add_upper_norm_bound(RegionType reg, VarIndexType XtUPvars, double upperbound,
                                           double ubscale, ScaleType scale_t) {
    check_ubscale(ubscale);

    int size = this->get_xt_up_vars(get_region(reg), XtUPvars).size();
    auto impl = [&](auto sz) {
        auto x = Arguments<1>();
        auto normfun = (Arguments<sz.value>(size).norm() - upperbound) * ubscale;
        return this->add_inequal_con(reg, normfun, XtUPvars, scale_t);
    };
    switch (size) {
    case 2:
        return impl(int_const<2>());
    case 3:
        return impl(int_const<3>());
    case 4:
        return impl(int_const<4>());
    default:
        return impl(int_const<-1>());
    }
    return 0;
}

int tycho::oc::ODEPhaseBase::add_upper_squared_norm_bound(RegionType reg, VarIndexType XtUPvars,
                                                  double upperbound, double ubscale,
                                                  ScaleType scale_t) {
    check_ubscale(ubscale);

    int size = this->get_xt_up_vars(get_region(reg), XtUPvars).size();
    auto impl = [&](auto sz) {
        auto x = Arguments<1>();
        auto normfun = (Arguments<sz.value>(size).squared_norm() - upperbound) * ubscale;
        return this->add_inequal_con(reg, normfun, XtUPvars, scale_t);
    };
    switch (size) {
    case 2:
        return impl(int_const<2>());
    case 3:
        return impl(int_const<3>());
    case 4:
        return impl(int_const<4>());
    default:
        return impl(int_const<-1>());
    }
    return 0;
}

int tycho::oc::ODEPhaseBase::add_lower_delta_var_bound(RegionType reg, VarIndexType var, double lowerbound,
                                               double lbscale, ScaleType scale_t) {
    check_lbscale(lbscale);

    auto args = Arguments<2>();
    auto x0 = args.coeff<0>();
    auto x1 = args.coeff<1>();
    auto func = (lowerbound - (x1 - x0)) * lbscale;

    return this->add_inequal_con(reg, func, var, scale_t);
}

int tycho::oc::ODEPhaseBase::add_upper_delta_var_bound(RegionType reg, VarIndexType var, double upperbound,
                                               double ubscale, ScaleType scale_t) {
    check_ubscale(ubscale);
    auto args = Arguments<2>();
    auto x0 = args.coeff<0>();
    auto x1 = args.coeff<1>();
    auto func = ((x1 - x0) - upperbound) * ubscale;
    return this->add_inequal_con(reg, func, var, scale_t);
}

//////////////////////////////

int tycho::oc::ODEPhaseBase::add_value_objective(RegionType reg, VarIndexType var, double scale,
                                           ScaleType scale_t) {
    auto obj = Arguments<1>() * scale;
    return this->add_state_objective(reg, obj, var, scale_t);
}

int tycho::oc::ODEPhaseBase::add_delta_var_objective(VarIndexType var, double scale, ScaleType scale_t) {
    auto args = Arguments<2>();
    auto x0 = args.coeff<0>();
    auto x1 = args.coeff<1>();
    auto func = ((x1 - x0)) * scale;

    return this->add_state_objective(PhaseRegionFlags::FrontandBack, func, var, scale_t);
}

std::vector<Eigen::VectorXd> tycho::oc::ODEPhaseBase::return_costate_traj() const {
    if (!this->PostOptInfoValid) {
        throw std::invalid_argument(
            "No costates to return,a solve or optimize call must be made before "
            "returning the costate trajectory ");
    }

    auto TrajTemp =
        this->indexer.get_func_eq_multipliers(this->DynamicsFuncIndex, this->ActiveEqLmults);

    std::vector<Eigen::VectorXd> tmp;
    int k = 0;
    int stride = (this->num_tran_card_states - 1);
    auto getSpace = [&](int i) {
        if (this->num_tran_card_states == 4) {
            return LGLCoeffs<4>::InteriorSpacings[i];
        } else if (this->num_tran_card_states == 3) {
            return LGLCoeffs<3>::InteriorSpacings[i];
        } else if (this->num_tran_card_states == 2) {
            return LGLCoeffs<2>::InteriorSpacings[i];
        } else {
            std::invalid_argument("Costate estimation Not Implemented for specified Transcription "
                                  "Mode");
            return 0.0;
        }
    };
    for (auto &T : TrajTemp) {
        VectorXd x0 = this->ActiveTraj[k * (stride)];
        VectorXd xf = this->ActiveTraj[k * (stride) + stride];
        double t0 = x0[this->TVar()];
        double h = xf[this->TVar()] - x0[this->TVar()];
        for (int i = 0; i < stride; i++) {
            VectorXd cs(this->XVars() + 1);
            cs.head(this->XVars()) = T.segment(i * this->XVars(), this->XVars());
            cs[this->TVar()] = t0 + h * getSpace(i);
            tmp.push_back(cs);
        }
        k++;
    }

    std::vector<Eigen::VectorXd> Costates(this->ActiveTraj.size());

    for (int i = 0; i < (this->ActiveTraj.size()); i++) {

        int idx0 = i - 1;
        int idx1 = i;

        if (i == 0) {
            idx0++;
            idx1++;
        } else if (i == (this->ActiveTraj.size() - 1)) {
            idx0--;
            idx1--;
        }
        auto t0 = tmp[idx0][this->TVar()];
        auto t1 = tmp[idx1][this->TVar()];
        auto tm = this->ActiveTraj[i][this->TVar()];
        Costates[i] = tmp[idx0] + ((tm - t0) / (t1 - t0)) * (tmp[idx1] - tmp[idx0]);
    }

    return Costates;
}

std::vector<Eigen::VectorXd> tycho::oc::ODEPhaseBase::return_traj_error() const {

    if (!this->PostOptInfoValid) {
        throw std::invalid_argument(
            "No trajectory errors to return,a solve or optimize call must be made before "
            "returning the trajectory error ");
    }

    auto ErrTrajTemp =
        this->indexer.get_func_eq_multipliers(this->DynamicsFuncIndex, this->ActiveEqCons);

    std::vector<Eigen::VectorXd> ErrTraj;
    int k = 0;
    int stride = (this->num_tran_card_states - 1);
    auto getSpace = [&](int i) {
        if (this->num_tran_card_states == 4) {
            return LGLCoeffs<4>::InteriorSpacings[i];
        } else if (this->num_tran_card_states == 3) {
            return LGLCoeffs<3>::InteriorSpacings[i];
        } else if (this->num_tran_card_states == 2) {
            return LGLCoeffs<2>::InteriorSpacings[i];
        } else {
            std::invalid_argument("Error estimation Not Implemented for specified Transcription "
                                  "Mode");
            return 0.0;
        }
    };
    for (auto &T : ErrTrajTemp) {
        VectorXd x0 = this->ActiveTraj[k * (stride)];
        VectorXd xf = this->ActiveTraj[k * (stride) + stride];
        double t0 = x0[this->TVar()];
        double h = xf[this->TVar()] - x0[this->TVar()];
        for (int i = 0; i < stride; i++) {
            VectorXd cs(this->XVars() + 1);
            cs.head(this->XVars()) = T.segment(i * this->XVars(), this->XVars());
            cs[this->TVar()] = t0 + h * getSpace(i);
            ErrTraj.push_back(cs);
        }
        k++;
    }
    return ErrTraj;
}

void tycho::oc::ODEPhaseBase::set_traj(const std::vector<Eigen::VectorXd> &mesh, Eigen::VectorXd DBS,
                                  Eigen::VectorXi DPB, bool LerpTraj) {

    if (mesh.size() == 0) {
        throw std::invalid_argument("Input trajectory is empty");
    }
    if (DPB.sum() < 2) {
        throw std::invalid_argument("Phase must have least 2 segments/defects.");
    }
    int msize = mesh[0].size();
    if (msize != this->Table.XtUVars) {
        std::cout << "User Input Error in function setInitTraj for ODE:" << this->Table.ode.name()
                  << std::endl;
        std::cout << " Dimension of Input States(" << msize
                  << ") does not match expected dimensions of the ODE(" << this->Table.XtUVars
                  << ")" << std::endl;
        throw std::invalid_argument("");
    }
    if ((DBS.size() - 1) != DPB.size()) {
        std::cout << "User Input Error in function setInitTraj for ODE:" << this->Table.ode.name()
                  << std::endl;
        std::cout << "  Size of Defect Bin Spacing(" << DBS.size()
                  << ") not consistent with size of Defects Per Bin(" << DPB.size() << ")"
                  << std::endl;
        throw std::invalid_argument("");
    }
    for (auto &X : mesh) {
        if (X.hasNaN()) {
            throw std::invalid_argument("NaN detected in State Vector in ODEPhaseBase::set_traj");
        }
    }

    this->Table.load_uneven_data(DPB.sum(), mesh);

    if (LerpTraj) {

        double t0 = mesh[0][this->TVar()];
        double tf = mesh.back()[this->TVar()];
        Eigen::VectorXd intime_nd(mesh.size());
        for (int i = 0; i < mesh.size(); i++) {
            intime_nd[i] = (mesh[i][this->TVar()] - t0) / (tf - t0);
        }

        int nstates = (this->num_tran_card_states - 1) * DPB.sum() + 1;
        Eigen::VectorXd outtime_nd(nstates);
        outtime_nd.setZero();

        Eigen::VectorXd Tspacing = this->Table.Tspacing;
        Eigen::VectorXd cvect(Tspacing.size());

        this->ActiveTraj.resize(nstates);

        for (int i = 0, start = 0; i < DBS.size() - 1; i++) {
            double bint0 = DBS[i];
            double bintf = DBS[i + 1];
            double segdt = (bintf - bint0) / DPB[i];
            for (int j = 0; j < DPB[i]; j++) {
                cvect.setConstant(outtime_nd[start]);
                outtime_nd.segment(start, Tspacing.size()) = cvect + Tspacing * segdt;
                start += (this->num_tran_card_states - 1);
            }
        }

        for (int i = 0, elem = 0; i < nstates; i++) {
            double ti = outtime_nd[i];
            auto it = std::upper_bound(intime_nd.cbegin() + elem, intime_nd.cend(), ti);
            elem = int(it - intime_nd.cbegin()) - 1;
            elem = std::clamp(elem, 0, int(intime_nd.size() - 2));
            double helem = intime_nd[elem + 1] - intime_nd[elem];
            double tndlocal = (ti - intime_nd[elem]) / (helem);
            this->ActiveTraj[i] = mesh[elem] * (1.0 - tndlocal) + mesh[elem + 1] * tndlocal;
        }

    } else {
        this->ActiveTraj = this->Table.NDdistribute(DBS, DPB);
    }

    this->DefBinSpacing = DBS;
    this->DefsPerBin = DPB;
    this->num_defects = DPB.sum();
    this->TrajectoryLoaded = true;
    this->reset_transcription();
    this->invalidate_post_opt_info();
}

void tycho::oc::ODEPhaseBase::set_traj(const std::vector<Eigen::VectorXd> &mesh) {
    /// Sets phase to have same number of defects and spacing as input trajectory

    int numdefects = (mesh.size() - 1) / (this->num_tran_card_states - 1);
    int numrem = (mesh.size() - 1) % (this->num_tran_card_states - 1);

    if (numrem != 0) {
        throw std::invalid_argument(
            "Number of states in mesh inconsistent with transcription type.");
    }
    if (mesh.size() == 0) {
        throw std::invalid_argument("Input trajectory is empty");
    }
    int msize = mesh[0].size();
    if (msize != this->Table.XtUVars) {
        std::cout << "User Input Error in function setInitTraj for ODE:" << this->Table.ode.name()
                  << std::endl;
        std::cout << " Dimension of Input States(" << msize
                  << ") does not match expected dimensions of the ODE(" << this->Table.XtUVars
                  << ")" << std::endl;
        throw std::invalid_argument("");
    }
    for (auto &X : mesh) {
        if (X.hasNaN()) {
            throw std::invalid_argument("NaN detected in State Vector in ODEPhaseBase::set_traj");
        }
    }

    Eigen::VectorXd DBS;

    DBS.resize(numdefects + 1);
    DBS[0] = 0;

    double t0 = mesh.front()[this->TVar()];
    double tf = mesh.back()[this->TVar()];

    for (int i = 0; i < numdefects; i++) {
        int elem = (this->num_tran_card_states - 1) * (i + 1);
        double ti = mesh[elem][this->TVar()];
        DBS[i + 1] = (ti - t0) / (tf - t0);
    }

    Eigen::VectorXi DPB;

    DPB.resize(numdefects);
    DPB.setOnes();

    this->set_traj(mesh, DBS, DPB, false);
}

void tycho::oc::ODEPhaseBase::refine_traj_manual(VectorXd DBS, VectorXi DPB) {
    if ((DBS.size() - 1) != DPB.size()) {
        std::cout << "User Input Error in function setInitTraj for ODE:" << this->Table.ode.name()
                  << std::endl;
        std::cout << "  Size of Defect Bin Spacing(" << DBS.size()
                  << ") not consistent with size of Defects Per Bin(" << DPB.size() << ")"
                  << std::endl;
        throw std::invalid_argument("");
    }

    this->Table.load_exact_data(this->ActiveTraj);
    this->ActiveTraj = this->Table.NDdistribute(DBS, DPB);
    this->DefBinSpacing = DBS;
    this->DefsPerBin = DPB;
    this->num_defects = DPB.sum();
    this->reset_transcription();
    this->invalidate_post_opt_info();
}

std::vector<Eigen::VectorXd> tycho::oc::ODEPhaseBase::refine_traj_equal(int n) {
    this->check_mesh();
    this->MeshIters.back().up_numsegs = n;
    Eigen::VectorXi dpb = VectorXi::Ones(n);
    Eigen::VectorXd bins = this->MeshIters.back().calc_bins(n);
    this->refine_traj_manual(bins, dpb);

    return this->ActiveTraj;
}

void tycho::oc::ODEPhaseBase::refine_traj_auto() {
    this->check_mesh();
    this->update_mesh();
    if (this->PrintMeshInfo) {
        this->MeshIters.back().print(0);
    }
}

void tycho::oc::ODEPhaseBase::sub_variables(PhaseRegionFlags reg, VectorXi indices, VectorXd vals) {
    switch (reg) {
    case PhaseRegionFlags::Front: {
        for (int i = 0; i < indices.size(); i++) {
            this->ActiveTraj[0][indices[i]] = vals[i];
        }
        break;
    }
    case PhaseRegionFlags::Back: {
        for (int i = 0; i < indices.size(); i++) {
            this->ActiveTraj.back()[indices[i]] = vals[i];
        }
        break;
    }
    case PhaseRegionFlags::Path: {
        for (int i = 0; i < indices.size(); i++) {
            for (int j = 0; j < this->ActiveTraj.size(); j++) {
                this->ActiveTraj[j][indices[i]] = vals[i];
            }
        }
        break;
    }
    case PhaseRegionFlags::StaticParams: {
        for (int i = 0; i < indices.size(); i++) {
            this->ActiveStaticParams[indices[i]] = vals[i];
        }
        break;
    }
    default: {
        throw std::invalid_argument("Variable Substitution Not Allowed for specified Phase Region");
    }
    }
}

void tycho::oc::ODEPhaseBase::transcribe_integrals() {
    auto MakeInt = [&](auto cs, auto xv, auto pv, const ScalarFunctionalX &integrand, int xvv,
                       int pvv) {
        auto integral =
            LGLIntegral<ScalarFunctionalX, cs.value, xv.value, pv.value>{integrand, xvv, pvv};
        integral.EnableVectorization = this->EnableVectorization;

        return ObjectiveInterface(integral);
    };
    auto SwitchX = [&](auto cs, int xv, auto pv, const ScalarFunctionalX &integrand, int xvv,
                       int pvv) {
        switch (xv) {
        case 1:
            return MakeInt(cs, int_const<1>(), pv, integrand, xvv, pvv);
        case 2:
            return MakeInt(cs, int_const<2>(), pv, integrand, xvv, pvv);
        case 3:
            return MakeInt(cs, int_const<3>(), pv, integrand, xvv, pvv);
        default:
            return MakeInt(cs, int_const<-1>(), pv, integrand, xvv, pvv);
        }
    };
    auto SwitchP = [&](auto cs, int xv, int pv, const ScalarFunctionalX &integrand, int xvv,
                       int pvv) {
        switch (pv) {
        case 0:
            return SwitchX(cs, xv, int_const<0>(), integrand, xvv, pvv);
        default:
            return MakeInt(cs, int_const<-1>(), int_const<-1>(), integrand, xvv, pvv);
        }
    };
    auto SwitchC = [&](int cs, int xv, int pv, const ScalarFunctionalX &integrand, int xvv,
                       int pvv) {
        switch (cs) {
        case 2:
            return SwitchP(int_const<2>(), xv, pv, integrand, xvv, pvv);
        case 3:
            return SwitchP(int_const<3>(), xv, pv, integrand, xvv, pvv);
        case 4:
            return SwitchP(int_const<4>(), xv, pv, integrand, xvv, pvv);
        default: {
            throw std::invalid_argument("Integral Type not implemented");
            return SwitchP(int_const<2>(), xv, pv, integrand, xvv, pvv);
        }
        }
    };

    for (auto &[key, ob] : this->user_integrands) {
        int xp = ob.XtUVars.size();
        int sop = ob.OPVars.size() + ob.SPVars.size();
        VectorXi xtrap(xp + 1);
        xtrap.head(xp) = ob.XtUVars;
        xtrap[xp] = this->TVar();

        ObjectiveInterface obj;
        PhaseRegionFlags PhaseReg = PhaseRegionFlags::PairWisePath;

        auto Func = ob.Func;
        if (this->AutoScaling) {
            VectorXd input_scales =
                this->get_input_scale(ob.RegionFlag, ob.XtUVars, ob.OPVars, ob.SPVars);
            VectorXd output_scales(Func.output_rows());
            output_scales.setOnes();
            output_scales = ob.OutputScales;

            Func = IOScaled<decltype(Func)>(ob.Func, input_scales, output_scales);
        }

        if (this->IntegralMode == IntegralModes::BaseIntegral &&
            ((this->TranscriptionMode == TranscriptionModes::LGL5) ||
             (this->TranscriptionMode == TranscriptionModes::LGL7))) {
            if (this->TranscriptionMode == TranscriptionModes::LGL5) {
                obj = SwitchC(3, xp, sop, Func, xp, sop);
            } else if (this->TranscriptionMode == TranscriptionModes::LGL7) {
                obj = SwitchC(4, xp, sop, Func, xp, sop);
            }
            PhaseReg = PhaseRegionFlags::DefectPath;
        } else {
            obj = SwitchC(2, xp, sop, Func, xp, sop);
            if (this->TranscriptionMode == TranscriptionModes::LGL3 ||
                this->TranscriptionMode == TranscriptionModes::Trapezoidal ||
                this->TranscriptionMode == TranscriptionModes::CentralShooting) {
                PhaseReg = PhaseRegionFlags::DefectPath;

            } else {
                PhaseReg = PhaseRegionFlags::PairWisePath;
            }
        }

        ThreadingFlags ThreadMode =
            ob.Func.thread_safe() ? ThreadingFlags::ByApplication : ThreadingFlags::MainThread;

        int Gindex =
            this->indexer.add_objective(obj, PhaseReg, xtrap, ob.OPVars, ob.SPVars, ThreadMode);

        int PLindex = Gindex - this->indexer.StartObj;
        ob.GlobalIndex = Gindex;
        ob.PhaseLocalIndex = PLindex;
    }

    for (auto &[key, ob] : this->user_param_integrands) {
        int xp = ob.XtUVars.size();
        int sop = ob.OPVars.size() + ob.SPVars.size();
        VectorXi xtrap(xp + 1);
        xtrap.head(xp) = ob.XtUVars;
        xtrap[xp] = this->TVar();

        ObjectiveInterface obj;
        PhaseRegionFlags PhaseReg = PhaseRegionFlags::PairWisePath;

        auto Func = ob.Func;
        double AccScale = 1.0;

        if (this->AutoScaling) {
            VectorXd input_scales =
                this->get_input_scale(ob.RegionFlag, ob.XtUVars, ob.OPVars, ob.SPVars);
            VectorXd output_scales(Func.output_rows());
            output_scales.setOnes();
            output_scales = ob.OutputScales;

            Func = IOScaled<decltype(Func)>(ob.Func, input_scales, output_scales);

            double pscale = this->SPUnits[ob.EXTVars[0]];

            AccScale = pscale * output_scales[0] / (this->XtUPUnits[this->TVar()]);
        }

        if (this->IntegralMode == IntegralModes::BaseIntegral &&
            ((this->TranscriptionMode == TranscriptionModes::LGL5) ||
             (this->TranscriptionMode == TranscriptionModes::LGL7))) {
            if (this->TranscriptionMode == TranscriptionModes::LGL5) {
                obj = SwitchC(3, xp, sop, Func, xp, sop);
            } else if (this->TranscriptionMode == TranscriptionModes::LGL7) {
                obj = SwitchC(4, xp, sop, Func, xp, sop);
            }
            PhaseReg = PhaseRegionFlags::DefectPath;
        } else {
            obj = SwitchC(2, xp, sop, Func, xp, sop);
            PhaseReg = PhaseRegionFlags::PairWisePath;
        }

        ThreadingFlags ThreadMode =
            ob.Func.thread_safe() ? ThreadingFlags::ByApplication : ThreadingFlags::MainThread;

        auto AccFunc = Arguments<1>() * -AccScale;

        int Gindex = this->indexer.add_accumulation(obj, PhaseReg, xtrap, ob.OPVars, ob.SPVars,
                                                   AccFunc, ob.EXTVars, ThreadMode);

        int PLindex = Gindex - this->indexer.StartObj;
        ob.GlobalIndex = Gindex;
        ob.PhaseLocalIndex = PLindex;
    }
}
void tycho::oc::ODEPhaseBase::transcribe_basic_funcs() {
    for (auto &[key, eq] : this->user_equalities) {
        ThreadingFlags ThreadMode =
            eq.Func.thread_safe() ? ThreadingFlags::ByApplication : ThreadingFlags::MainThread;
        if (eq.RegionFlag == PhaseRegionFlags::Path ||
            eq.RegionFlag == PhaseRegionFlags::PairWisePath)
            eq.Func.enable_vectorization(this->EnableVectorization);

        auto Func = eq.Func;
        if (this->AutoScaling) {
            VectorXd input_scales =
                this->get_input_scale(eq.RegionFlag, eq.XtUVars, eq.OPVars, eq.SPVars);
            VectorXd output_scales(Func.output_rows());
            output_scales.setOnes();
            output_scales = eq.OutputScales;
            Func = IOScaled<decltype(Func)>(eq.Func, input_scales, output_scales);
        }

        int Gindex = this->indexer.add_equality(Func, eq.RegionFlag, eq.XtUVars, eq.OPVars,
                                               eq.SPVars, ThreadMode);

        int PLindex = Gindex - this->indexer.StartEq;
        eq.GlobalIndex = Gindex;
        eq.PhaseLocalIndex = PLindex;
    }
    for (auto &[key, iq] : this->user_inequalities) {
        ThreadingFlags ThreadMode =
            iq.Func.thread_safe() ? ThreadingFlags::ByApplication : ThreadingFlags::MainThread;
        if (iq.RegionFlag == PhaseRegionFlags::Path ||
            iq.RegionFlag == PhaseRegionFlags::PairWisePath) {

            iq.Func.enable_vectorization(this->EnableVectorization);
        }

        auto Func = iq.Func;
        if (this->AutoScaling) {
            VectorXd input_scales =
                this->get_input_scale(iq.RegionFlag, iq.XtUVars, iq.OPVars, iq.SPVars);
            VectorXd output_scales(Func.output_rows());
            output_scales.setOnes();
            output_scales = iq.OutputScales;

            Func = IOScaled<decltype(Func)>(iq.Func, input_scales, output_scales);
        }

        int Gindex = this->indexer.add_inequality(Func, iq.RegionFlag, iq.XtUVars, iq.OPVars,
                                                 iq.SPVars, ThreadMode);
        int PLindex = Gindex - this->indexer.StartIq;
        iq.GlobalIndex = Gindex;
        iq.PhaseLocalIndex = PLindex;
    }
    for (auto &[key, ob] : this->user_state_objectives) {
        ThreadingFlags ThreadMode =
            ob.Func.thread_safe() ? ThreadingFlags::ByApplication : ThreadingFlags::MainThread;
        if (ob.RegionFlag == PhaseRegionFlags::Path ||
            ob.RegionFlag == PhaseRegionFlags::PairWisePath)
            ob.Func.enable_vectorization(this->EnableVectorization);

        auto Func = ob.Func;
        if (this->AutoScaling) {
            VectorXd input_scales =
                this->get_input_scale(ob.RegionFlag, ob.XtUVars, ob.OPVars, ob.SPVars);
            VectorXd output_scales(Func.output_rows());
            output_scales.setOnes();
            output_scales = ob.OutputScales;

            Func = IOScaled<decltype(Func)>(ob.Func, input_scales, output_scales);
        }

        int Gindex = this->indexer.add_objective(Func, ob.RegionFlag, ob.XtUVars, ob.OPVars,
                                                ob.SPVars, ThreadMode);
        int PLindex = Gindex - this->indexer.StartObj;
        ob.GlobalIndex = Gindex;
        ob.PhaseLocalIndex = PLindex;
    }
}
void tycho::oc::ODEPhaseBase::transcribe_axis_funcs() {
    VectorXd cspace(this->num_defects + 1);
    int start = 0;
    for (int i = 0; i < this->DefsPerBin.size(); i++) {
        cspace.segment(start, this->DefsPerBin[i] + 1)
            .setLinSpaced(this->DefBinSpacing[i], this->DefBinSpacing[i + 1]);
        start += this->DefsPerBin[i];
    }

    std::vector<ConstraintInterface> AxisFuncs;
    std::vector<ThreadingFlags> Tmodes;
    Eigen::VectorXi bins(this->NumPartitions + 1);
    bins.setLinSpaced(0, this->indexer.num_nodal_states);

    for (int i = 0; i < this->indexer.num_nodal_states - 2; i++) {
        AxisFuncs.emplace_back(SingleMeshSpacing(cspace[i + 1]));
        ThreadingFlags thrt = ThreadingFlags::Thread0;
        for (int j = 0; j < this->NumPartitions; j++) {
            if (i >= bins(j) && i < bins(j + 1)) {
                thrt = static_cast<ThreadingFlags>(j);
            }
        }
        Tmodes.push_back(thrt);
    }

    VectorXi tloc(1);
    tloc[0] = this->TVar();

    VectorXi empty(0);
    empty.resize(0);

    if (this->TranscriptionMode == TranscriptionModes::LGL7) {
        LGLMeshSpacing<4> axcon7;
        this->indexer.add_equality(axcon7, PhaseRegionFlags::DefectPath, tloc, empty, empty,
                                  ThreadingFlags::ByApplication);
    } else if (this->TranscriptionMode == TranscriptionModes::LGL5) {
        LGLMeshSpacing<3> axcon5;
        this->indexer.add_equality(axcon5, PhaseRegionFlags::DefectPath, tloc, empty, empty,
                                  ThreadingFlags::ByApplication);
    }

    this->indexer.add_partitioned_equality(AxisFuncs, PhaseRegionFlags::FrontNodalBackPath, tloc,
                                         empty, empty, Tmodes);
}

void tycho::oc::ODEPhaseBase::transcribe_control_funcs() {
    VectorXi StateT(this->XtUVars());
    for (int i = 0; i < this->XtUVars(); i++)
        StateT[i] = i;
    VectorXi TUvarT = StateT.segment(this->TVar(), 1 + this->UVars());
    VectorXi empty(0);
    empty.resize(0);

    std::vector<VectorXi> TUis(this->UVars());
    for (int i = 0; i < this->UVars(); i++) {
        TUis[i].resize(2);
        TUis[i][0] = this->TVar();
        TUis[i][1] = this->TVar() + 1 + i;
    }
    this->ControlFuncsIndex = -1;

    if (this->TranscriptionMode == TranscriptionModes::LGL7) {
        if (this->UVars() > 0) {
            if (this->ControlMode == ControlModes::HighestOrderSpline) {
                LGLControlSpline<4, -1, 2> lgl7spln2(this->UVars());

                this->ControlFuncsIndex =
                    this->indexer.add_equality(lgl7spln2, PhaseRegionFlags::DefectPairWisePath,
                                              TUvarT, empty, empty, ThreadingFlags::ByApplication);

            } else if (this->ControlMode == ControlModes::FirstOrderSpline) {
                LGLControlSpline<4, -1, 1> lgl7spln1(this->UVars());

                this->ControlFuncsIndex =
                    this->indexer.add_equality(lgl7spln1, PhaseRegionFlags::DefectPairWisePath,
                                              TUvarT, empty, empty, ThreadingFlags::ByApplication);
            }
        }
    } else if (this->TranscriptionMode == TranscriptionModes::LGL5) {
        if (this->UVars() > 0) {
            if (this->ControlMode == ControlModes::HighestOrderSpline ||
                this->ControlMode == ControlModes::FirstOrderSpline) {
                LGLControlSpline<3, -1, 1> lgl5spln1(this->UVars());

                this->ControlFuncsIndex =
                    this->indexer.add_equality(lgl5spln1, PhaseRegionFlags::DefectPairWisePath,
                                              TUvarT, empty, empty, ThreadingFlags::ByApplication);
            }
        }
    }
}

void tycho::oc::ODEPhaseBase::check_functions(int pnum) {

    /*
   Loops through all user defined functions and checks that they do not
   reference non-existent variables. Should be run prior to any transcribing any
   problem functions.
   */

    auto CheckFun = [&](const std::string &type, auto &func) {
        if (func.XtUVars.size() > 0) {
            if (func.XtUVars.maxCoeff() >= this->XtUPVars() || func.XtUVars.minCoeff() < 0) {

                fmt::print(fmt::fg(fmt::color::red),
                           "Transcription Error!!!\n"
                           "{0:} function state variable indices out of bounds in phase:{1:}\n"
                           " Function Storage Index:{2:}\n"
                           " Function Name:{3:}\n",
                           type, pnum, func.StorageIndex, func.Func.name());
                throw std::invalid_argument("");
            }
        }
        if (func.OPVars.size() > 0) {
            if (func.OPVars.maxCoeff() >= this->PVars() || func.OPVars.minCoeff() < 0) {

                fmt::print(fmt::fg(fmt::color::red),
                           "Transcription Error!!!\n"
                           "{0:} function ODE Param variable indices out of bounds in phase:{1:}\n"
                           " Function Storage Index:{2:}\n"
                           " Function Name:{3:}\n",
                           type, pnum, func.StorageIndex, func.Func.name());
                throw std::invalid_argument("");
            }
        }
        if (func.SPVars.size() > 0) {
            if (func.SPVars.maxCoeff() >= this->num_stat_params || func.SPVars.minCoeff() < 0) {
                fmt::print(
                    fmt::fg(fmt::color::red),
                    "Transcription Error!!!\n"
                    "{0:} function Static Param variable indices out of bounds in phase:{1:}\n"
                    " Function Storage Index:{2:}\n"
                    " Function Name:{3:}\n",
                    type, pnum, func.StorageIndex, func.Func.name());
                throw std::invalid_argument("");
            }
        }
        if (func.EXTVars.size() > 0) {
            if (func.EXTVars.maxCoeff() >= this->num_stat_params || func.EXTVars.minCoeff() < 0) {

                fmt::print(fmt::fg(fmt::color::red),
                           "Transcription Error!!!\n"
                           "{0:} function Integral Static Param variable indices out of bounds in "
                           "phase:{1:}\n"
                           " Function Storage Index:{2:}\n"
                           " Function Name:{3:}\n",
                           type, pnum, func.StorageIndex, func.Func.name());
                throw std::invalid_argument("");
            }
        }
    };

    std::string eq = "Equality constraint";
    std::string iq = "Inequality constraint";
    std::string sobj = "State objective";
    std::string iobj = "Integral objective";
    std::string ipcon = "Integral parameter";

    for (auto &[key, f] : this->user_equalities)
        CheckFun(eq, f);
    for (auto &[key, f] : this->user_inequalities)
        CheckFun(iq, f);
    for (auto &[key, f] : this->user_integrands)
        CheckFun(iobj, f);
    for (auto &[key, f] : this->user_param_integrands)
        CheckFun(ipcon, f);
    for (auto &[key, f] : this->user_state_objectives)
        CheckFun(sobj, f);
}

Eigen::VectorXd tycho::oc::ODEPhaseBase::get_input_scale(PhaseRegionFlags flag, VectorXi XtUV,
                                                     VectorXi OPV, VectorXi SPV) const {

    int nloops;
    switch (flag) {
    case PhaseRegionFlags::Front:
    case PhaseRegionFlags::Back:
    case PhaseRegionFlags::Path:
    case PhaseRegionFlags::Params:
    case PhaseRegionFlags::ODEParams:
    case PhaseRegionFlags::StaticParams:
    case PhaseRegionFlags::InnerPath: {
        nloops = 1;
        break;
    }
    case PhaseRegionFlags::FrontandBack:
    case PhaseRegionFlags::BackandFront:
    case PhaseRegionFlags::PairWisePath: {
        nloops = 2;
        break;
    }
    default: {
        throw std::invalid_argument("Cannot scale this phase region");
        break;
    }
    }

    int isize = XtUV.size() * nloops + OPV.size() + SPV.size();
    VectorXd scales(isize);

    int next = 0;
    for (int n = 0; n < nloops; n++) {
        for (int i = 0; i < XtUV.size(); i++) {
            scales[next] = this->XtUPUnits[XtUV[i]];
            next++;
        }
    }
    for (int i = 0; i < OPV.size(); i++) {
        scales[next] = this->XtUPUnits[OPV[i] + this->XtUVars()];
        next++;
    }
    for (int i = 0; i < SPV.size(); i++) {
        scales[next] = this->SPUnits[SPV[i]];
        next++;
    }

    return scales;
}

std::vector<Eigen::VectorXd> tycho::oc::ODEPhaseBase::get_test_inputs(PhaseRegionFlags flag,
                                                                  VectorXi XtUV, VectorXi OPV,
                                                                  VectorXi SPV) const {

    std::vector<std::vector<int>> test_states;

    int nloops = 0;
    switch (flag) {
    case PhaseRegionFlags::Front: {
        test_states.push_back({0});
        break;
    }
    case PhaseRegionFlags::Back: {
        test_states.push_back({int(this->ActiveTraj.size() - 1)});
        break;
    }
    case PhaseRegionFlags::Params: {
        test_states.push_back({0});
        break;
    }
    case PhaseRegionFlags::ODEParams: {
        test_states.push_back({0});
        break;
    }
    case PhaseRegionFlags::StaticParams: {
        test_states.push_back({0});
        break;
    }
    case PhaseRegionFlags::Path: {
        for (int i = 0; i < this->ActiveTraj.size(); i++)
            test_states.push_back({i});
        break;
    }
    case PhaseRegionFlags::InnerPath: {
        for (int i = 1; i < this->ActiveTraj.size() - 1; i++)
            test_states.push_back({i});
        break;
    }
    case PhaseRegionFlags::FrontandBack: {
        test_states.push_back({0, int(this->ActiveTraj.size() - 1)});
        break;
    }
    case PhaseRegionFlags::BackandFront: {
        test_states.push_back({int(this->ActiveTraj.size() - 1), 0});
        break;
    }
    case PhaseRegionFlags::PairWisePath: {
        for (int i = 0; i < this->ActiveTraj.size() - 1; i++)
            test_states.push_back({i, i + 1});
        break;
    }
    default: {
        throw std::invalid_argument("Cannot scale this phase region");
        break;
    }
    }

    int isize = XtUV.size() * test_states[0].size() + OPV.size() + SPV.size();

    std::vector<Eigen::VectorXd> inputs;
    VectorXd input_scales = this->get_input_scale(flag, XtUV, OPV, SPV);

    for (int ncalls = 0; ncalls < test_states.size(); ncalls++) {

        VectorXd input(isize);

        int next = 0;

        for (int i = 0; i < test_states[ncalls].size(); i++) {
            int state = test_states[ncalls][i];
            for (int j = 0; j < XtUV.size(); j++) {
                input[next] = this->ActiveTraj[state][XtUV[j]];
                next++;
            }
        }
        for (int j = 0; j < OPV.size(); j++) {
            input[next] = this->ActiveTraj[0][OPV[j] + this->XtUVars()];
            next++;
        }
        for (int j = 0; j < SPV.size(); j++) {
            input[next] = this->ActiveStaticParams[SPV[j]];
            next++;
        }
        input = input.cwiseQuotient(input_scales);

        inputs.push_back(input);
    }

    return inputs;
}

void tycho::oc::ODEPhaseBase::calc_auto_scales() {
    auto calc_impl = [&](auto &funcmap) {
        for (auto &[key, func] : funcmap) {
            if (func.ScaleMode == ScaleModes::AUTO) {
                VectorXd input_scales =
                    this->get_input_scale(func.RegionFlag, func.XtUVars, func.OPVars, func.SPVars);
                std::vector<VectorXd> test_inputs =
                    this->get_test_inputs(func.RegionFlag, func.XtUVars, func.OPVars, func.SPVars);
                VectorXd output_scales =
                    calc_jacobian_row_scales(func.Func, input_scales, test_inputs);
                func.OutputScales = output_scales;
            } else {
            }
        }
    };

    calc_impl(this->user_equalities);
    calc_impl(this->user_inequalities);
    calc_impl(this->user_state_objectives);
    calc_impl(this->user_integrands);
    calc_impl(this->user_param_integrands);
}

std::vector<double> tycho::oc::ODEPhaseBase::get_objective_scales() {
    // If we have mixed integral and state objectives, we assume that they have the same units
    // since the output scales are computed for the integrands we need to divide by tstar
    // before averaging to make the units consistent

    std::vector<double> scales;
    for (auto &[key, obj] : this->user_state_objectives) {
        if (obj.ScaleMode == ScaleModes::AUTO) {
            // OutputScales units 1/obj
            scales.push_back(obj.OutputScales[0]);
        }
    }
    for (auto &[key, obj] : this->user_integrands) {
        if (obj.ScaleMode == ScaleModes::AUTO) {
            // OutputScales units tstar/obj
            // Divide by tstar, since this function is the integrand not the total integral
            scales.push_back(obj.OutputScales[0] / this->XtUPUnits[this->TVar()]);
        }
    }

    return scales;
}

void tycho::oc::ODEPhaseBase::update_objective_scales(double scale) {
    for (auto &[key, obj] : this->user_state_objectives) {
        if (obj.ScaleMode == ScaleModes::AUTO) {
            obj.OutputScales[0] = scale;
        }
    }
    for (auto &[key, obj] : this->user_integrands) {
        if (obj.ScaleMode == ScaleModes::AUTO) {
            // Multiply by tstar, since this function is the integrand not the total integral
            obj.OutputScales[0] = scale * this->XtUPUnits[this->TVar()];
        }
    }
}

void tycho::oc::ODEPhaseBase::transcribe_phase(int vo, int eqo, int iqo,
                                           std::shared_ptr<NonLinearProgram> np, int pnum)

{
    this->indexer.begin_indexing(np, vo, eqo, iqo);

    this->transcribe_dynamics();
    this->transcribe_axis_funcs();
    this->transcribe_control_funcs();
    this->transcribe_integrals();
    this->transcribe_basic_funcs();

    //////DO NOT GET RID OF THIS!!!!!!//
    this->do_transcription = false;
    ////////////////////////////////////
}

void tycho::oc::ODEPhaseBase::transcribe(bool showstats, bool showfuns) {
    this->nlp = std::make_shared<NonLinearProgram>(this->NumPartitions);

    this->init_indexing();

    this->check_functions(0);
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

    this->transcribe_phase(0, 0, 0, this->nlp, 0);
    if (showstats)
        this->indexer.print_stats(showfuns);

    if (this->indexer.num_phase_eq_cons > this->indexer.num_phase_vars) {
        fmt::print(fmt::fg(fmt::color::yellow),
                   "Transcription Warning!!!\n"
                   "Number of Equality Constraints({0:}) in phase exceeds number of free "
                   "variables({1:}).\n"
                   "You likely have a redundant constraint.\n",
                   this->indexer.num_phase_eq_cons, this->indexer.num_phase_vars);
    }

    this->nlp->make_NLP(this->indexer.num_phase_vars, this->indexer.num_phase_eq_cons,
                        this->indexer.num_phase_iq_cons);

    this->optimizer->setNLP(this->nlp);
}

void tycho::oc::ODEPhaseBase::test_partitions(int i, int j, int n) {
    this->reset_transcription();

    auto nlp1 = std::make_shared<NonLinearProgram>(i);
    this->init_indexing();
    this->transcribe_phase(0, 0, 0, nlp1, 0);
    if (false)
        this->indexer.print_stats(false);
    nlp1->make_NLP(this->indexer.num_phase_vars, this->indexer.num_phase_eq_cons,
                   this->indexer.num_phase_iq_cons);

    auto nlp2 = std::make_shared<NonLinearProgram>(j);
    this->init_indexing();
    this->transcribe_phase(0, 0, 0, nlp2, 0);
    if (false)
        this->indexer.print_stats(false);
    nlp2->make_NLP(this->indexer.num_phase_vars, this->indexer.num_phase_eq_cons,
                   this->indexer.num_phase_iq_cons);

    Eigen::VectorXd v = this->make_solver_input();
    NonLinearProgram::NLPTest(v, n, nlp1, nlp2);

    this->reset_transcription();
}

bool tycho::oc::ODEPhaseBase::check_mesh() {

    Eigen::VectorXd tsnd;
    Eigen::MatrixXd mesh_errors;
    Eigen::MatrixXd mesh_dist;

    this->Table.load_exact_data(this->ActiveTraj);

    if (this->MeshErrorEstimator == MeshErrorEstimators::INTEGRATOR ||
        this->TranscriptionMode == TranscriptionModes::CentralShooting) {
        this->get_meshinfo_integrator(tsnd, mesh_errors, mesh_dist);
    } else if (this->MeshErrorEstimator == MeshErrorEstimators::DEBOOR) {
        this->get_meshinfo_deboor(tsnd, mesh_errors, mesh_dist);
    } else {
        throw std::invalid_argument("Unknown mesh error estimator");
    }

    Eigen::VectorXd error = mesh_errors.colwise().lpNorm<Eigen::Infinity>();
    Eigen::VectorXd dist = mesh_dist.colwise().lpNorm<Eigen::Infinity>();

    this->MeshIters.emplace_back(this->num_defects, this->MeshTol, tsnd, error, dist);

    if (this->MeshErrorCriteria == MeshErrorAggregation::ENDTOEND ||
        this->MeshErrorDistributor == MeshErrorAggregation::ENDTOEND) {
        Eigen::VectorXd error_vec = this->calc_global_error();
        this->MeshIters.back().global_error = error_vec.lpNorm<Eigen::Infinity>();
    }

    double error_crit;

    switch (this->MeshErrorCriteria) {
    case MeshErrorAggregation::MAX:
        error_crit = this->MeshIters.back().max_error;
        break;
    case MeshErrorAggregation::AVG:
        error_crit = this->MeshIters.back().avg_error;
        break;
    case MeshErrorAggregation::GEOMETRIC:
        error_crit = this->MeshIters.back().gmean_error;
        break;
    case MeshErrorAggregation::ENDTOEND:
        error_crit = this->MeshIters.back().global_error;
        break;
    default:
        throw std::invalid_argument("Unknown MeshErrorAggregation");
    }

    this->MeshConverged = (error_crit < this->MeshTol);
    this->MeshIters.back().converged = this->MeshConverged;

    return this->MeshConverged;
}

void tycho::oc::ODEPhaseBase::update_mesh() {

    double ntemp = 0;
    for (int i = 0; i < this->MeshIters.back().error.size() - 1; i++) {

        double nsegs =
            std::pow((this->MeshIters.back().error[i] * this->MeshErrFactor) / this->MeshTol,
                     1 / (this->Order + 1));
        ntemp += std::max(this->MeshRedFactor, nsegs);
    }
    int n = int(std::ceil(ntemp)) + this->NumExtraSegs;

    n = std::clamp(n, int(this->num_defects * this->MeshRedFactor),
                   int(this->num_defects * this->MeshIncFactor));
    n = std::clamp(n, this->MinSegments, this->MaxSegments);

    Eigen::VectorXd bins = this->MeshIters.back().calc_bins(n);

    if (this->DetectControlSwitches && this->UVars() > 0) {
        Eigen::VectorXd switchvec = this->calcSwitches();
        std::vector<double> stmp;

        for (int i = 0; i < bins.size() - 1; i++) {
            for (int j = 0; j < switchvec.size(); j++) {
                if (switchvec[j] > bins[i] && switchvec[j] < bins[i + 1]) {
                    stmp.push_back(2 * bins[i] / 3.0 + bins[i + 1] / 3.0);
                    stmp.push_back(bins[i] / 3.0 + 2 * bins[i + 1] / 3.0);

                    break;
                }
            }
        }
        switchvec.resize(stmp.size());
        for (int i = 0; i < stmp.size(); i++) {
            switchvec[i] = stmp[i];
        }

        Eigen::VectorXd binstmp(bins.size() + switchvec.size());
        binstmp.head(bins.size()) = bins;
        binstmp.tail(switchvec.size()) = switchvec;
        std::sort(binstmp.begin(), binstmp.end());

        bins = binstmp;
    }

    Eigen::VectorXi dpb = VectorXi::Ones(bins.size() - 1);
    this->MeshIters.back().up_numsegs = bins.size() - 1;

    this->refine_traj_manual(bins, dpb);
}

Eigen::VectorXd tycho::oc::ODEPhaseBase::calcSwitches() {

    Eigen::MatrixXd uvals(this->UVars(), this->ActiveTraj.size());
    Eigen::VectorXd tsnd(this->ActiveTraj.size());

    double T0 = this->ActiveTraj[0][this->TVar()];
    double TF = this->ActiveTraj.back()[this->TVar()];

    for (int i = 0; i < this->ActiveTraj.size(); i++) {
        uvals.col(i) = this->ActiveTraj[i].segment(this->XtVars(), this->UVars());
        tsnd[i] = (this->ActiveTraj[i][this->TVar()] - T0) / (TF - T0);
    }

    Eigen::VectorXd umin = uvals.rowwise().minCoeff();
    Eigen::VectorXd umax = uvals.rowwise().maxCoeff();
    Eigen::VectorXd ones(this->UVars());
    ones.setOnes();

    Eigen::MatrixXd und(this->UVars(), this->ActiveTraj.size());

    for (int i = 0; i < this->ActiveTraj.size(); i++) {
        und.col(i) = (uvals.col(i) - umin).cwiseQuotient(ones + umax - umin);
    }

    Eigen::VectorXd udiff;
    Eigen::VectorXd unddiff;
    std::vector<double> switches;

    for (int i = 0; i < this->ActiveTraj.size() - 1; i++) {
        udiff = (uvals.col(i + 1) - uvals.col(i)).cwiseAbs();
        unddiff = (uvals.col(i + 1) - uvals.col(i)).cwiseAbs();
        if (udiff.maxCoeff() > this->AbsSwitchTol && unddiff.maxCoeff() > this->RelSwitchTol) {
            double t = tsnd[i + 1] / 2.0 + tsnd[i] / 2.0;
            switches.push_back(t);
        }
    }

    return stdvector_to_eigenvector(switches);
}

Tycho::PSIOPT::ConvergenceFlags tycho::oc::ODEPhaseBase::psipot_call_impl(JetJobModes mode) {
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

Tycho::PSIOPT::ConvergenceFlags tycho::oc::ODEPhaseBase::phase_call_impl(JetJobModes mode) {

    if (this->PrintMeshInfo && this->AdaptiveMesh) {
        fmt::print(fmt::fg(fmt::color::white), "{0:=^{1}}\n", "", 65);
        fmt::print(fmt::fg(fmt::color::dim_gray), "Beginning");
        fmt::print(": ");
        fmt::print(fmt::fg(fmt::color::royal_blue), "Adaptive Mesh Refinement");
        fmt::print("\n");
    }

    Utils::Timer Runtimer;

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
            init_mesh_refinement();
            for (int i = 0; i < this->MaxMeshIters; i++) {
                if (check_mesh() && !((i == 0) && this->ForceOneMeshIter)) {
                    if (this->PrintMeshInfo) {
                        MeshIterateInfo::print_header(i);
                        this->MeshIters.back().print(0);
                        fmt::print(fmt::fg(fmt::color::lime_green), "Mesh Converged\n");
                    }
                    break;
                } else if (i == this->MaxMeshIters - 1) {
                    if (this->PrintMeshInfo) {
                        MeshIterateInfo::print_header(i);
                        this->MeshIters.back().print(0);
                        fmt::print(fmt::fg(fmt::color::red), "Mesh Not Converged\n");
                    }
                    break;
                } else {
                    update_mesh();
                    if (this->PrintMeshInfo) {
                        MeshIterateInfo::print_header(i);
                        this->MeshIters.back().print(0);
                    }
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
