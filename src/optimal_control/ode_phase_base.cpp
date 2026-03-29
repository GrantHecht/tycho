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

#include "tycho/detail/optimal_control/phase/ode_phase_base.h"

#include "tycho/detail/optimal_control/transcription/lgl_control_splines.h"
#include "tycho/detail/optimal_control/transcription/lgl_integrals.h"
#include "tycho/detail/optimal_control/transcription/mesh_spacing_constraints.h"
#include "tycho/detail/vf/common/value_lock.h"
#include "tycho/detail/vf/scaling/auto_scaling_utils.h"

using tycho::vf::Arguments;
using tycho::vf::IOScaled;
using tycho::vf::LockArgs;
using tycho::vf::StackedOutputs;

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
    auto func = Arguments<-1>(value.size()) - value;
    return this->add_equal_con(reg, func, args, scale_t);
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
    auto func = X.head(argsize) - X.tail(argsize);
    return this->add_equal_con(PhaseRegionFlags::FrontandBack, func, args, scale_t);
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

int tycho::oc::ODEPhaseBase::add_lower_var_bound(RegionType reg, VarIndexType var,
                                                 double lowerbound, double lbscale,
                                                 ScaleType scale_t) {
    check_lbscale(lbscale);
    auto x = Arguments<1>();
    auto lbound = (lowerbound - x) * lbscale;

    return this->add_inequal_con(reg, lbound, var, scale_t);
}

int tycho::oc::ODEPhaseBase::add_upper_var_bound(RegionType reg, VarIndexType var,
                                                 double upperbound, double ubscale,
                                                 ScaleType scale_t) {
    check_ubscale(ubscale);
    auto x = Arguments<1>();
    auto ubound = (x - upperbound) * ubscale;
    return this->add_inequal_con(reg, ubound, var, scale_t);
}

int tycho::oc::ODEPhaseBase::add_lu_func_bound(RegionType reg, ScalarFunctionalX func,
                                               VarIndexType XtUPvars, VarIndexType OPvars,
                                               VarIndexType SPvars, double lowerbound,
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
    auto lubound = StackedOutputs{(lowerbound - x) * lbscale, (x - upperbound) * ubscale};
    auto lufun = lubound.eval(func);
    return this->add_inequal_con(reg, lufun, XtUPvars, OPvars, SPvars, scale_t);
}

int tycho::oc::ODEPhaseBase::add_lower_func_bound(RegionType reg, ScalarFunctionalX func,
                                                  VarIndexType XtUPvars, VarIndexType OPvars,
                                                  VarIndexType SPvars, double lowerbound,
                                                  double lbscale, ScaleType scale_t) {
    check_lbscale(lbscale);

    Vector1<double> rhs;
    rhs[0] = lowerbound;
    auto lbfun = ((-1.0 * func) + rhs) * lbscale;

    return this->add_inequal_con(reg, lbfun, XtUPvars, OPvars, SPvars, scale_t);
}

int tycho::oc::ODEPhaseBase::add_upper_func_bound(RegionType reg, ScalarFunctionalX func,
                                                  VarIndexType XtUPvars, VarIndexType OPvars,
                                                  VarIndexType SPvars, double upperbound,
                                                  double ubscale, ScaleType scale_t) {

    check_ubscale(ubscale);

    Vector1<double> rhs;
    rhs[0] = -upperbound;
    auto ubfun = ((func) + rhs) * ubscale;
    return this->add_inequal_con(reg, ubfun, XtUPvars, OPvars, SPvars, scale_t);
}

int tycho::oc::ODEPhaseBase::add_lu_norm_bound(RegionType reg, VarIndexType XtUPvars,
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
                                                       double lowerbound, double upperbound,
                                                       double lbscale, double ubscale,
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

int tycho::oc::ODEPhaseBase::add_lower_norm_bound(RegionType reg, VarIndexType XtUPvars,
                                                  double lowerbound, double lbscale,
                                                  ScaleType scale_t) {
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

int tycho::oc::ODEPhaseBase::add_upper_norm_bound(RegionType reg, VarIndexType XtUPvars,
                                                  double upperbound, double ubscale,
                                                  ScaleType scale_t) {
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

int tycho::oc::ODEPhaseBase::add_lower_delta_var_bound(RegionType reg, VarIndexType var,
                                                       double lowerbound, double lbscale,
                                                       ScaleType scale_t) {
    check_lbscale(lbscale);

    auto args = Arguments<2>();
    auto x0 = args.coeff<0>();
    auto x1 = args.coeff<1>();
    auto func = (lowerbound - (x1 - x0)) * lbscale;

    return this->add_inequal_con(reg, func, var, scale_t);
}

int tycho::oc::ODEPhaseBase::add_upper_delta_var_bound(RegionType reg, VarIndexType var,
                                                       double upperbound, double ubscale,
                                                       ScaleType scale_t) {
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

int tycho::oc::ODEPhaseBase::add_delta_var_objective(VarIndexType var, double scale,
                                                     ScaleType scale_t) {
    auto args = Arguments<2>();
    auto x0 = args.coeff<0>();
    auto x1 = args.coeff<1>();
    auto func = ((x1 - x0)) * scale;

    return this->add_state_objective(PhaseRegionFlags::FrontandBack, func, var, scale_t);
}

std::vector<Eigen::VectorXd> tycho::oc::ODEPhaseBase::return_costate_traj() const {
    if (!this->post_opt_info_valid_) {
        throw std::invalid_argument(
            "No costates to return,a solve or optimize call must be made before "
            "returning the costate trajectory ");
    }

    auto traj_temp =
        this->indexer_.get_func_eq_multipliers(this->dynamics_func_index_, this->active_eq_lmults_);

    std::vector<Eigen::VectorXd> tmp;
    int k = 0;
    int stride = (this->num_tran_card_states_ - 1);
    auto get_space = [&](int i) {
        if (this->num_tran_card_states_ == 4) {
            return LGLCoeffs<4>::InteriorSpacings[i];
        } else if (this->num_tran_card_states_ == 3) {
            return LGLCoeffs<3>::InteriorSpacings[i];
        } else if (this->num_tran_card_states_ == 2) {
            return LGLCoeffs<2>::InteriorSpacings[i];
        } else {
            std::invalid_argument("Costate estimation Not Implemented for specified Transcription "
                                  "Mode");
            return 0.0;
        }
    };
    for (auto &T : traj_temp) {
        VectorXd x0 = this->active_traj_[k * (stride)];
        VectorXd xf = this->active_traj_[k * (stride) + stride];
        double t0 = x0[this->t_var()];
        double h = xf[this->t_var()] - x0[this->t_var()];
        for (int i = 0; i < stride; i++) {
            VectorXd cs(this->x_vars() + 1);
            cs.head(this->x_vars()) = T.segment(i * this->x_vars(), this->x_vars());
            cs[this->t_var()] = t0 + h * get_space(i);
            tmp.push_back(cs);
        }
        k++;
    }

    std::vector<Eigen::VectorXd> costates(this->active_traj_.size());

    for (int i = 0; i < (this->active_traj_.size()); i++) {

        int idx0 = i - 1;
        int idx1 = i;

        if (i == 0) {
            idx0++;
            idx1++;
        } else if (i == (this->active_traj_.size() - 1)) {
            idx0--;
            idx1--;
        }
        auto t0 = tmp[idx0][this->t_var()];
        auto t1 = tmp[idx1][this->t_var()];
        auto tm = this->active_traj_[i][this->t_var()];
        costates[i] = tmp[idx0] + ((tm - t0) / (t1 - t0)) * (tmp[idx1] - tmp[idx0]);
    }

    return costates;
}

std::vector<Eigen::VectorXd> tycho::oc::ODEPhaseBase::return_traj_error() const {

    if (!this->post_opt_info_valid_) {
        throw std::invalid_argument(
            "No trajectory errors to return,a solve or optimize call must be made before "
            "returning the trajectory error ");
    }

    auto err_traj_temp =
        this->indexer_.get_func_eq_multipliers(this->dynamics_func_index_, this->active_eq_cons_);

    std::vector<Eigen::VectorXd> err_traj;
    int k = 0;
    int stride = (this->num_tran_card_states_ - 1);
    auto get_space = [&](int i) {
        if (this->num_tran_card_states_ == 4) {
            return LGLCoeffs<4>::InteriorSpacings[i];
        } else if (this->num_tran_card_states_ == 3) {
            return LGLCoeffs<3>::InteriorSpacings[i];
        } else if (this->num_tran_card_states_ == 2) {
            return LGLCoeffs<2>::InteriorSpacings[i];
        } else {
            std::invalid_argument("Error estimation Not Implemented for specified Transcription "
                                  "Mode");
            return 0.0;
        }
    };
    for (auto &T : err_traj_temp) {
        VectorXd x0 = this->active_traj_[k * (stride)];
        VectorXd xf = this->active_traj_[k * (stride) + stride];
        double t0 = x0[this->t_var()];
        double h = xf[this->t_var()] - x0[this->t_var()];
        for (int i = 0; i < stride; i++) {
            VectorXd cs(this->x_vars() + 1);
            cs.head(this->x_vars()) = T.segment(i * this->x_vars(), this->x_vars());
            cs[this->t_var()] = t0 + h * get_space(i);
            err_traj.push_back(cs);
        }
        k++;
    }
    return err_traj;
}

void tycho::oc::ODEPhaseBase::set_traj(const std::vector<Eigen::VectorXd> &mesh,
                                       Eigen::VectorXd DBS, Eigen::VectorXi DPB, bool LerpTraj) {

    if (mesh.size() == 0) {
        throw std::invalid_argument("Input trajectory is empty");
    }
    if (DPB.sum() < 2) {
        throw std::invalid_argument("Phase must have least 2 segments/defects.");
    }
    int msize = mesh[0].size();
    if (msize != this->table_.xtu_vars_) {
        std::cout << "User Input Error in function setInitTraj for ODE:" << this->table_.ode_.name()
                  << std::endl;
        std::cout << " Dimension of Input States(" << msize
                  << ") does not match expected dimensions of the ODE(" << this->table_.xtu_vars_
                  << ")" << std::endl;
        throw std::invalid_argument("");
    }
    if ((DBS.size() - 1) != DPB.size()) {
        std::cout << "User Input Error in function setInitTraj for ODE:" << this->table_.ode_.name()
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

    this->table_.load_uneven_data(DPB.sum(), mesh);

    if (LerpTraj) {

        double t0 = mesh[0][this->t_var()];
        double tf = mesh.back()[this->t_var()];
        Eigen::VectorXd intime_nd(mesh.size());
        for (int i = 0; i < mesh.size(); i++) {
            intime_nd[i] = (mesh[i][this->t_var()] - t0) / (tf - t0);
        }

        int nstates = (this->num_tran_card_states_ - 1) * DPB.sum() + 1;
        Eigen::VectorXd outtime_nd(nstates);
        outtime_nd.setZero();

        Eigen::VectorXd tspacing = this->table_.t_spacing_;
        Eigen::VectorXd cvect(tspacing.size());

        this->active_traj_.resize(nstates);

        for (int i = 0, start = 0; i < DBS.size() - 1; i++) {
            double bint0 = DBS[i];
            double bintf = DBS[i + 1];
            double segdt = (bintf - bint0) / DPB[i];
            for (int j = 0; j < DPB[i]; j++) {
                cvect.setConstant(outtime_nd[start]);
                outtime_nd.segment(start, tspacing.size()) = cvect + tspacing * segdt;
                start += (this->num_tran_card_states_ - 1);
            }
        }

        for (int i = 0, elem = 0; i < nstates; i++) {
            double ti = outtime_nd[i];
            auto it = std::upper_bound(intime_nd.cbegin() + elem, intime_nd.cend(), ti);
            elem = int(it - intime_nd.cbegin()) - 1;
            elem = std::clamp(elem, 0, int(intime_nd.size() - 2));
            double helem = intime_nd[elem + 1] - intime_nd[elem];
            double tndlocal = (ti - intime_nd[elem]) / (helem);
            this->active_traj_[i] = mesh[elem] * (1.0 - tndlocal) + mesh[elem + 1] * tndlocal;
        }

    } else {
        this->active_traj_ = this->table_.nd_distribute(DBS, DPB);
    }

    this->def_bin_spacing_ = DBS;
    this->defs_per_bin_ = DPB;
    this->num_defects_ = DPB.sum();
    this->trajectory_loaded_ = true;
    this->reset_transcription();
    this->invalidate_post_opt_info();
}

void tycho::oc::ODEPhaseBase::set_traj(const std::vector<Eigen::VectorXd> &mesh) {
    /// Sets phase to have same number of defects and spacing as input trajectory

    int numdefects = (mesh.size() - 1) / (this->num_tran_card_states_ - 1);
    int numrem = (mesh.size() - 1) % (this->num_tran_card_states_ - 1);

    if (numrem != 0) {
        throw std::invalid_argument(
            "Number of states in mesh inconsistent with transcription type.");
    }
    if (mesh.size() == 0) {
        throw std::invalid_argument("Input trajectory is empty");
    }
    int msize = mesh[0].size();
    if (msize != this->table_.xtu_vars_) {
        std::cout << "User Input Error in function setInitTraj for ODE:" << this->table_.ode_.name()
                  << std::endl;
        std::cout << " Dimension of Input States(" << msize
                  << ") does not match expected dimensions of the ODE(" << this->table_.xtu_vars_
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

    double t0 = mesh.front()[this->t_var()];
    double tf = mesh.back()[this->t_var()];

    for (int i = 0; i < numdefects; i++) {
        int elem = (this->num_tran_card_states_ - 1) * (i + 1);
        double ti = mesh[elem][this->t_var()];
        DBS[i + 1] = (ti - t0) / (tf - t0);
    }

    Eigen::VectorXi DPB;

    DPB.resize(numdefects);
    DPB.setOnes();

    this->set_traj(mesh, DBS, DPB, false);
}

void tycho::oc::ODEPhaseBase::refine_traj_manual(VectorXd DBS, VectorXi DPB) {
    if ((DBS.size() - 1) != DPB.size()) {
        std::cout << "User Input Error in function setInitTraj for ODE:" << this->table_.ode_.name()
                  << std::endl;
        std::cout << "  Size of Defect Bin Spacing(" << DBS.size()
                  << ") not consistent with size of Defects Per Bin(" << DPB.size() << ")"
                  << std::endl;
        throw std::invalid_argument("");
    }

    this->table_.load_exact_data(this->active_traj_);
    this->active_traj_ = this->table_.nd_distribute(DBS, DPB);
    this->def_bin_spacing_ = DBS;
    this->defs_per_bin_ = DPB;
    this->num_defects_ = DPB.sum();
    this->reset_transcription();
    this->invalidate_post_opt_info();
}

std::vector<Eigen::VectorXd> tycho::oc::ODEPhaseBase::refine_traj_equal(int n) {
    this->check_mesh();
    this->mesh_iters_.back().up_numsegs = n;
    Eigen::VectorXi dpb = VectorXi::Ones(n);
    Eigen::VectorXd bins = this->mesh_iters_.back().calc_bins(n);
    this->refine_traj_manual(bins, dpb);

    return this->active_traj_;
}

void tycho::oc::ODEPhaseBase::refine_traj_auto() {
    this->check_mesh();
    this->update_mesh();
    if (this->print_mesh_info_) {
        this->mesh_iters_.back().print(0);
    }
}

void tycho::oc::ODEPhaseBase::sub_variables(PhaseRegionFlags reg, VectorXi indices, VectorXd vals) {
    switch (reg) {
    case PhaseRegionFlags::Front: {
        for (int i = 0; i < indices.size(); i++) {
            this->active_traj_[0][indices[i]] = vals[i];
        }
        break;
    }
    case PhaseRegionFlags::Back: {
        for (int i = 0; i < indices.size(); i++) {
            this->active_traj_.back()[indices[i]] = vals[i];
        }
        break;
    }
    case PhaseRegionFlags::Path: {
        for (int i = 0; i < indices.size(); i++) {
            for (int j = 0; j < this->active_traj_.size(); j++) {
                this->active_traj_[j][indices[i]] = vals[i];
            }
        }
        break;
    }
    case PhaseRegionFlags::StaticParams: {
        for (int i = 0; i < indices.size(); i++) {
            this->active_static_params_[indices[i]] = vals[i];
        }
        break;
    }
    default: {
        throw std::invalid_argument("Variable Substitution Not Allowed for specified Phase Region");
    }
    }
}

void tycho::oc::ODEPhaseBase::transcribe_integrals() {
    auto make_int = [&](auto cs, auto xv, auto pv, const ScalarFunctionalX &integrand, int xvv,
                        int pvv) {
        auto integral =
            LGLIntegral<ScalarFunctionalX, cs.value, xv.value, pv.value>{integrand, xvv, pvv};
        integral.enable_vectorization_ = this->enable_vectorization_;

        return ObjectiveInterface(integral);
    };
    auto switch_x = [&](auto cs, int xv, auto pv, const ScalarFunctionalX &integrand, int xvv,
                        int pvv) {
        switch (xv) {
        case 1:
            return make_int(cs, int_const<1>(), pv, integrand, xvv, pvv);
        case 2:
            return make_int(cs, int_const<2>(), pv, integrand, xvv, pvv);
        case 3:
            return make_int(cs, int_const<3>(), pv, integrand, xvv, pvv);
        default:
            return make_int(cs, int_const<-1>(), pv, integrand, xvv, pvv);
        }
    };
    auto switch_p = [&](auto cs, int xv, int pv, const ScalarFunctionalX &integrand, int xvv,
                        int pvv) {
        switch (pv) {
        case 0:
            return switch_x(cs, xv, int_const<0>(), integrand, xvv, pvv);
        default:
            return make_int(cs, int_const<-1>(), int_const<-1>(), integrand, xvv, pvv);
        }
    };
    auto SwitchC = [&](int cs, int xv, int pv, const ScalarFunctionalX &integrand, int xvv,
                       int pvv) {
        switch (cs) {
        case 2:
            return switch_p(int_const<2>(), xv, pv, integrand, xvv, pvv);
        case 3:
            return switch_p(int_const<3>(), xv, pv, integrand, xvv, pvv);
        case 4:
            return switch_p(int_const<4>(), xv, pv, integrand, xvv, pvv);
        default: {
            throw std::invalid_argument("Integral Type not implemented");
            return switch_p(int_const<2>(), xv, pv, integrand, xvv, pvv);
        }
        }
    };

    for (auto &[key, ob] : this->user_integrands) {
        int xp = ob.xtu_vars_.size();
        int sop = ob.op_vars_.size() + ob.sp_vars_.size();
        VectorXi xtrap(xp + 1);
        xtrap.head(xp) = ob.xtu_vars_;
        xtrap[xp] = this->t_var();

        ObjectiveInterface obj;
        PhaseRegionFlags PhaseReg = PhaseRegionFlags::PairWisePath;

        auto func = ob.func_;
        if (this->auto_scaling_) {
            VectorXd input_scales =
                this->get_input_scale(ob.region_flag_, ob.xtu_vars_, ob.op_vars_, ob.sp_vars_);
            VectorXd output_scales(func.output_rows());
            output_scales.setOnes();
            output_scales = ob.output_scales_;

            func = IOScaled<decltype(func)>(ob.func_, input_scales, output_scales);
        }

        if (this->integral_mode_ == IntegralModes::BaseIntegral &&
            ((this->transcription_mode_ == TranscriptionModes::LGL5) ||
             (this->transcription_mode_ == TranscriptionModes::LGL7))) {
            if (this->transcription_mode_ == TranscriptionModes::LGL5) {
                obj = SwitchC(3, xp, sop, func, xp, sop);
            } else if (this->transcription_mode_ == TranscriptionModes::LGL7) {
                obj = SwitchC(4, xp, sop, func, xp, sop);
            }
            PhaseReg = PhaseRegionFlags::DefectPath;
        } else {
            obj = SwitchC(2, xp, sop, func, xp, sop);
            if (this->transcription_mode_ == TranscriptionModes::LGL3 ||
                this->transcription_mode_ == TranscriptionModes::Trapezoidal ||
                this->transcription_mode_ == TranscriptionModes::CentralShooting) {
                PhaseReg = PhaseRegionFlags::DefectPath;

            } else {
                PhaseReg = PhaseRegionFlags::PairWisePath;
            }
        }

        ThreadingFlags thread_mode =
            ob.func_.thread_safe() ? ThreadingFlags::ByApplication : ThreadingFlags::MainThread;

        int Gindex = this->indexer_.add_objective(obj, PhaseReg, xtrap, ob.op_vars_, ob.sp_vars_,
                                                  thread_mode);

        int PLindex = Gindex - this->indexer_.StartObj;
        ob.global_index_ = Gindex;
        ob.phase_local_index_ = PLindex;
    }

    for (auto &[key, ob] : this->user_param_integrands) {
        int xp = ob.xtu_vars_.size();
        int sop = ob.op_vars_.size() + ob.sp_vars_.size();
        VectorXi xtrap(xp + 1);
        xtrap.head(xp) = ob.xtu_vars_;
        xtrap[xp] = this->t_var();

        ObjectiveInterface obj;
        PhaseRegionFlags PhaseReg = PhaseRegionFlags::PairWisePath;

        auto func = ob.func_;
        double AccScale = 1.0;

        if (this->auto_scaling_) {
            VectorXd input_scales =
                this->get_input_scale(ob.region_flag_, ob.xtu_vars_, ob.op_vars_, ob.sp_vars_);
            VectorXd output_scales(func.output_rows());
            output_scales.setOnes();
            output_scales = ob.output_scales_;

            func = IOScaled<decltype(func)>(ob.func_, input_scales, output_scales);

            double pscale = this->sp_units_[ob.ext_vars_[0]];

            AccScale = pscale * output_scales[0] / (this->xtup_units_[this->t_var()]);
        }

        if (this->integral_mode_ == IntegralModes::BaseIntegral &&
            ((this->transcription_mode_ == TranscriptionModes::LGL5) ||
             (this->transcription_mode_ == TranscriptionModes::LGL7))) {
            if (this->transcription_mode_ == TranscriptionModes::LGL5) {
                obj = SwitchC(3, xp, sop, func, xp, sop);
            } else if (this->transcription_mode_ == TranscriptionModes::LGL7) {
                obj = SwitchC(4, xp, sop, func, xp, sop);
            }
            PhaseReg = PhaseRegionFlags::DefectPath;
        } else {
            obj = SwitchC(2, xp, sop, func, xp, sop);
            PhaseReg = PhaseRegionFlags::PairWisePath;
        }

        ThreadingFlags thread_mode =
            ob.func_.thread_safe() ? ThreadingFlags::ByApplication : ThreadingFlags::MainThread;

        auto AccFunc = Arguments<1>() * -AccScale;

        int Gindex = this->indexer_.add_accumulation(obj, PhaseReg, xtrap, ob.op_vars_, ob.sp_vars_,
                                                     AccFunc, ob.ext_vars_, thread_mode);

        int PLindex = Gindex - this->indexer_.StartObj;
        ob.global_index_ = Gindex;
        ob.phase_local_index_ = PLindex;
    }
}
void tycho::oc::ODEPhaseBase::transcribe_basic_funcs() {
    for (auto &[key, eq] : this->user_equalities) {
        ThreadingFlags thread_mode =
            eq.func_.thread_safe() ? ThreadingFlags::ByApplication : ThreadingFlags::MainThread;
        if (eq.region_flag_ == PhaseRegionFlags::Path ||
            eq.region_flag_ == PhaseRegionFlags::PairWisePath)
            eq.func_.enable_vectorization(this->enable_vectorization_);

        auto func = eq.func_;
        if (this->auto_scaling_) {
            VectorXd input_scales =
                this->get_input_scale(eq.region_flag_, eq.xtu_vars_, eq.op_vars_, eq.sp_vars_);
            VectorXd output_scales(func.output_rows());
            output_scales.setOnes();
            output_scales = eq.output_scales_;
            func = IOScaled<decltype(func)>(eq.func_, input_scales, output_scales);
        }

        int Gindex = this->indexer_.add_equality(func, eq.region_flag_, eq.xtu_vars_, eq.op_vars_,
                                                 eq.sp_vars_, thread_mode);

        int PLindex = Gindex - this->indexer_.StartEq;
        eq.global_index_ = Gindex;
        eq.phase_local_index_ = PLindex;
    }
    for (auto &[key, iq] : this->user_inequalities) {
        ThreadingFlags thread_mode =
            iq.func_.thread_safe() ? ThreadingFlags::ByApplication : ThreadingFlags::MainThread;
        if (iq.region_flag_ == PhaseRegionFlags::Path ||
            iq.region_flag_ == PhaseRegionFlags::PairWisePath) {

            iq.func_.enable_vectorization(this->enable_vectorization_);
        }

        auto func = iq.func_;
        if (this->auto_scaling_) {
            VectorXd input_scales =
                this->get_input_scale(iq.region_flag_, iq.xtu_vars_, iq.op_vars_, iq.sp_vars_);
            VectorXd output_scales(func.output_rows());
            output_scales.setOnes();
            output_scales = iq.output_scales_;

            func = IOScaled<decltype(func)>(iq.func_, input_scales, output_scales);
        }

        int Gindex = this->indexer_.add_inequality(func, iq.region_flag_, iq.xtu_vars_, iq.op_vars_,
                                                   iq.sp_vars_, thread_mode);
        int PLindex = Gindex - this->indexer_.StartIq;
        iq.global_index_ = Gindex;
        iq.phase_local_index_ = PLindex;
    }
    for (auto &[key, ob] : this->user_state_objectives) {
        ThreadingFlags thread_mode =
            ob.func_.thread_safe() ? ThreadingFlags::ByApplication : ThreadingFlags::MainThread;
        if (ob.region_flag_ == PhaseRegionFlags::Path ||
            ob.region_flag_ == PhaseRegionFlags::PairWisePath)
            ob.func_.enable_vectorization(this->enable_vectorization_);

        auto func = ob.func_;
        if (this->auto_scaling_) {
            VectorXd input_scales =
                this->get_input_scale(ob.region_flag_, ob.xtu_vars_, ob.op_vars_, ob.sp_vars_);
            VectorXd output_scales(func.output_rows());
            output_scales.setOnes();
            output_scales = ob.output_scales_;

            func = IOScaled<decltype(func)>(ob.func_, input_scales, output_scales);
        }

        int Gindex = this->indexer_.add_objective(func, ob.region_flag_, ob.xtu_vars_, ob.op_vars_,
                                                  ob.sp_vars_, thread_mode);
        int PLindex = Gindex - this->indexer_.StartObj;
        ob.global_index_ = Gindex;
        ob.phase_local_index_ = PLindex;
    }
}
void tycho::oc::ODEPhaseBase::transcribe_axis_funcs() {
    VectorXd cspace(this->num_defects_ + 1);
    int start = 0;
    for (int i = 0; i < this->defs_per_bin_.size(); i++) {
        cspace.segment(start, this->defs_per_bin_[i] + 1)
            .setLinSpaced(this->def_bin_spacing_[i], this->def_bin_spacing_[i + 1]);
        start += this->defs_per_bin_[i];
    }

    std::vector<ConstraintInterface> AxisFuncs;
    std::vector<ThreadingFlags> Tmodes;
    Eigen::VectorXi bins(this->NumPartitions + 1);
    bins.setLinSpaced(0, this->indexer_.num_nodal_states);

    for (int i = 0; i < this->indexer_.num_nodal_states - 2; i++) {
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
    tloc[0] = this->t_var();

    VectorXi empty(0);
    empty.resize(0);

    if (this->transcription_mode_ == TranscriptionModes::LGL7) {
        LGLMeshSpacing<4> axcon7;
        this->indexer_.add_equality(axcon7, PhaseRegionFlags::DefectPath, tloc, empty, empty,
                                    ThreadingFlags::ByApplication);
    } else if (this->transcription_mode_ == TranscriptionModes::LGL5) {
        LGLMeshSpacing<3> axcon5;
        this->indexer_.add_equality(axcon5, PhaseRegionFlags::DefectPath, tloc, empty, empty,
                                    ThreadingFlags::ByApplication);
    }

    this->indexer_.add_partitioned_equality(AxisFuncs, PhaseRegionFlags::FrontNodalBackPath, tloc,
                                            empty, empty, Tmodes);
}

void tycho::oc::ODEPhaseBase::transcribe_control_funcs() {
    VectorXi StateT(this->xtu_vars());
    for (int i = 0; i < this->xtu_vars(); i++)
        StateT[i] = i;
    VectorXi TUvarT = StateT.segment(this->t_var(), 1 + this->u_vars());
    VectorXi empty(0);
    empty.resize(0);

    std::vector<VectorXi> TUis(this->u_vars());
    for (int i = 0; i < this->u_vars(); i++) {
        TUis[i].resize(2);
        TUis[i][0] = this->t_var();
        TUis[i][1] = this->t_var() + 1 + i;
    }
    this->control_funcs_index_ = -1;

    if (this->transcription_mode_ == TranscriptionModes::LGL7) {
        if (this->u_vars() > 0) {
            if (this->control_mode_ == ControlModes::HighestOrderSpline) {
                LGLControlSpline<4, -1, 2> lgl7spln2(this->u_vars());

                this->control_funcs_index_ = this->indexer_.add_equality(
                    lgl7spln2, PhaseRegionFlags::DefectPairWisePath, TUvarT, empty, empty,
                    ThreadingFlags::ByApplication);

            } else if (this->control_mode_ == ControlModes::FirstOrderSpline) {
                LGLControlSpline<4, -1, 1> lgl7spln1(this->u_vars());

                this->control_funcs_index_ = this->indexer_.add_equality(
                    lgl7spln1, PhaseRegionFlags::DefectPairWisePath, TUvarT, empty, empty,
                    ThreadingFlags::ByApplication);
            }
        }
    } else if (this->transcription_mode_ == TranscriptionModes::LGL5) {
        if (this->u_vars() > 0) {
            if (this->control_mode_ == ControlModes::HighestOrderSpline ||
                this->control_mode_ == ControlModes::FirstOrderSpline) {
                LGLControlSpline<3, -1, 1> lgl5spln1(this->u_vars());

                this->control_funcs_index_ = this->indexer_.add_equality(
                    lgl5spln1, PhaseRegionFlags::DefectPairWisePath, TUvarT, empty, empty,
                    ThreadingFlags::ByApplication);
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
        if (func.xtu_vars_.size() > 0) {
            if (func.xtu_vars_.maxCoeff() >= this->xtu_p_vars() || func.xtu_vars_.minCoeff() < 0) {

                fmt::print(fmt::fg(fmt::color::red),
                           "Transcription Error!!!\n"
                           "{0:} function state variable indices out of bounds in phase:{1:}\n"
                           " Function Storage Index:{2:}\n"
                           " Function Name:{3:}\n",
                           type, pnum, func.storage_index_, func.func_.name());
                throw std::invalid_argument("");
            }
        }
        if (func.op_vars_.size() > 0) {
            if (func.op_vars_.maxCoeff() >= this->p_vars() || func.op_vars_.minCoeff() < 0) {

                fmt::print(fmt::fg(fmt::color::red),
                           "Transcription Error!!!\n"
                           "{0:} function ODE Param variable indices out of bounds in phase:{1:}\n"
                           " Function Storage Index:{2:}\n"
                           " Function Name:{3:}\n",
                           type, pnum, func.storage_index_, func.func_.name());
                throw std::invalid_argument("");
            }
        }
        if (func.sp_vars_.size() > 0) {
            if (func.sp_vars_.maxCoeff() >= this->num_stat_params_ ||
                func.sp_vars_.minCoeff() < 0) {
                fmt::print(
                    fmt::fg(fmt::color::red),
                    "Transcription Error!!!\n"
                    "{0:} function Static Param variable indices out of bounds in phase:{1:}\n"
                    " Function Storage Index:{2:}\n"
                    " Function Name:{3:}\n",
                    type, pnum, func.storage_index_, func.func_.name());
                throw std::invalid_argument("");
            }
        }
        if (func.ext_vars_.size() > 0) {
            if (func.ext_vars_.maxCoeff() >= this->num_stat_params_ ||
                func.ext_vars_.minCoeff() < 0) {

                fmt::print(fmt::fg(fmt::color::red),
                           "Transcription Error!!!\n"
                           "{0:} function Integral Static Param variable indices out of bounds in "
                           "phase:{1:}\n"
                           " Function Storage Index:{2:}\n"
                           " Function Name:{3:}\n",
                           type, pnum, func.storage_index_, func.func_.name());
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
            scales[next] = this->xtup_units_[XtUV[i]];
            next++;
        }
    }
    for (int i = 0; i < OPV.size(); i++) {
        scales[next] = this->xtup_units_[OPV[i] + this->xtu_vars()];
        next++;
    }
    for (int i = 0; i < SPV.size(); i++) {
        scales[next] = this->sp_units_[SPV[i]];
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
        test_states.push_back({int(this->active_traj_.size() - 1)});
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
        for (int i = 0; i < this->active_traj_.size(); i++)
            test_states.push_back({i});
        break;
    }
    case PhaseRegionFlags::InnerPath: {
        for (int i = 1; i < this->active_traj_.size() - 1; i++)
            test_states.push_back({i});
        break;
    }
    case PhaseRegionFlags::FrontandBack: {
        test_states.push_back({0, int(this->active_traj_.size() - 1)});
        break;
    }
    case PhaseRegionFlags::BackandFront: {
        test_states.push_back({int(this->active_traj_.size() - 1), 0});
        break;
    }
    case PhaseRegionFlags::PairWisePath: {
        for (int i = 0; i < this->active_traj_.size() - 1; i++)
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
                input[next] = this->active_traj_[state][XtUV[j]];
                next++;
            }
        }
        for (int j = 0; j < OPV.size(); j++) {
            input[next] = this->active_traj_[0][OPV[j] + this->xtu_vars()];
            next++;
        }
        for (int j = 0; j < SPV.size(); j++) {
            input[next] = this->active_static_params_[SPV[j]];
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
            if (func.scale_mode_ == ScaleModes::AUTO) {
                VectorXd input_scales = this->get_input_scale(func.region_flag_, func.xtu_vars_,
                                                              func.op_vars_, func.sp_vars_);
                std::vector<VectorXd> test_inputs = this->get_test_inputs(
                    func.region_flag_, func.xtu_vars_, func.op_vars_, func.sp_vars_);
                VectorXd output_scales =
                    calc_jacobian_row_scales(func.func_, input_scales, test_inputs);
                func.output_scales_ = output_scales;
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
        if (obj.scale_mode_ == ScaleModes::AUTO) {
            // OutputScales units 1/obj
            scales.push_back(obj.output_scales_[0]);
        }
    }
    for (auto &[key, obj] : this->user_integrands) {
        if (obj.scale_mode_ == ScaleModes::AUTO) {
            // OutputScales units tstar/obj
            // Divide by tstar, since this function is the integrand not the total integral
            scales.push_back(obj.output_scales_[0] / this->xtup_units_[this->t_var()]);
        }
    }

    return scales;
}

void tycho::oc::ODEPhaseBase::update_objective_scales(double scale) {
    for (auto &[key, obj] : this->user_state_objectives) {
        if (obj.scale_mode_ == ScaleModes::AUTO) {
            obj.output_scales_[0] = scale;
        }
    }
    for (auto &[key, obj] : this->user_integrands) {
        if (obj.scale_mode_ == ScaleModes::AUTO) {
            // Multiply by tstar, since this function is the integrand not the total integral
            obj.output_scales_[0] = scale * this->xtup_units_[this->t_var()];
        }
    }
}

void tycho::oc::ODEPhaseBase::transcribe_phase(int vo, int eqo, int iqo,
                                               std::shared_ptr<NonLinearProgram> np, int pnum)

{
    this->indexer_.begin_indexing(np, vo, eqo, iqo);

    this->transcribe_dynamics();
    this->transcribe_axis_funcs();
    this->transcribe_control_funcs();
    this->transcribe_integrals();
    this->transcribe_basic_funcs();

    //////DO NOT GET RID OF THIS!!!!!!//
    this->do_transcription_ = false;
    ////////////////////////////////////
}

void tycho::oc::ODEPhaseBase::transcribe(bool showstats, bool showfuns) {
    this->nlp = std::make_shared<NonLinearProgram>(this->NumPartitions);

    this->init_indexing();

    this->check_functions(0);
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

    this->transcribe_phase(0, 0, 0, this->nlp, 0);
    if (showstats)
        this->indexer_.print_stats(showfuns);

    if (this->indexer_.num_phase_eq_cons > this->indexer_.num_phase_vars) {
        fmt::print(fmt::fg(fmt::color::yellow),
                   "Transcription Warning!!!\n"
                   "Number of Equality Constraints({0:}) in phase exceeds number of free "
                   "variables({1:}).\n"
                   "You likely have a redundant constraint.\n",
                   this->indexer_.num_phase_eq_cons, this->indexer_.num_phase_vars);
    }

    this->nlp->make_NLP(this->indexer_.num_phase_vars, this->indexer_.num_phase_eq_cons,
                        this->indexer_.num_phase_iq_cons);

    this->optimizer->setNLP(this->nlp);
}

void tycho::oc::ODEPhaseBase::test_partitions(int i, int j, int n) {
    this->reset_transcription();

    auto nlp1 = std::make_shared<NonLinearProgram>(i);
    this->init_indexing();
    this->transcribe_phase(0, 0, 0, nlp1, 0);
    if (false)
        this->indexer_.print_stats(false);
    nlp1->make_NLP(this->indexer_.num_phase_vars, this->indexer_.num_phase_eq_cons,
                   this->indexer_.num_phase_iq_cons);

    auto nlp2 = std::make_shared<NonLinearProgram>(j);
    this->init_indexing();
    this->transcribe_phase(0, 0, 0, nlp2, 0);
    if (false)
        this->indexer_.print_stats(false);
    nlp2->make_NLP(this->indexer_.num_phase_vars, this->indexer_.num_phase_eq_cons,
                   this->indexer_.num_phase_iq_cons);

    Eigen::VectorXd v = this->make_solver_input();
    NonLinearProgram::NLPTest(v, n, nlp1, nlp2);

    this->reset_transcription();
}

bool tycho::oc::ODEPhaseBase::check_mesh() {

    Eigen::VectorXd tsnd;
    Eigen::MatrixXd mesh_errors;
    Eigen::MatrixXd mesh_dist;

    this->table_.load_exact_data(this->active_traj_);

    if (this->mesh_error_estimator_ == MeshErrorEstimators::INTEGRATOR ||
        this->transcription_mode_ == TranscriptionModes::CentralShooting) {
        this->get_meshinfo_integrator(tsnd, mesh_errors, mesh_dist);
    } else if (this->mesh_error_estimator_ == MeshErrorEstimators::DEBOOR) {
        this->get_meshinfo_deboor(tsnd, mesh_errors, mesh_dist);
    } else {
        throw std::invalid_argument("Unknown mesh error estimator");
    }

    Eigen::VectorXd error = mesh_errors.colwise().lpNorm<Eigen::Infinity>();
    Eigen::VectorXd dist = mesh_dist.colwise().lpNorm<Eigen::Infinity>();

    this->mesh_iters_.emplace_back(this->num_defects_, this->mesh_tol_, tsnd, error, dist);

    if (this->mesh_error_criteria_ == MeshErrorAggregation::ENDTOEND ||
        this->mesh_error_distributor_ == MeshErrorAggregation::ENDTOEND) {
        Eigen::VectorXd error_vec = this->calc_global_error();
        this->mesh_iters_.back().global_error = error_vec.lpNorm<Eigen::Infinity>();
    }

    double error_crit;

    switch (this->mesh_error_criteria_) {
    case MeshErrorAggregation::MAX:
        error_crit = this->mesh_iters_.back().max_error;
        break;
    case MeshErrorAggregation::AVG:
        error_crit = this->mesh_iters_.back().avg_error;
        break;
    case MeshErrorAggregation::GEOMETRIC:
        error_crit = this->mesh_iters_.back().gmean_error;
        break;
    case MeshErrorAggregation::ENDTOEND:
        error_crit = this->mesh_iters_.back().global_error;
        break;
    default:
        throw std::invalid_argument("Unknown MeshErrorAggregation");
    }

    this->mesh_converged_ = (error_crit < this->mesh_tol_);
    this->mesh_iters_.back().converged = this->mesh_converged_;

    return this->mesh_converged_;
}

void tycho::oc::ODEPhaseBase::update_mesh() {

    double ntemp = 0;
    for (int i = 0; i < this->mesh_iters_.back().error.size() - 1; i++) {

        double nsegs =
            std::pow((this->mesh_iters_.back().error[i] * this->mesh_err_factor_) / this->mesh_tol_,
                     1 / (this->order_ + 1));
        ntemp += std::max(this->mesh_red_factor_, nsegs);
    }
    int n = int(std::ceil(ntemp)) + this->num_extra_segs_;

    n = std::clamp(n, int(this->num_defects_ * this->mesh_red_factor_),
                   int(this->num_defects_ * this->mesh_inc_factor_));
    n = std::clamp(n, this->min_segments_, this->max_segments_);

    Eigen::VectorXd bins = this->mesh_iters_.back().calc_bins(n);

    if (this->detect_control_switches_ && this->u_vars() > 0) {
        Eigen::VectorXd switchvec = this->calc_switches();
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
    this->mesh_iters_.back().up_numsegs = bins.size() - 1;

    this->refine_traj_manual(bins, dpb);
}

Eigen::VectorXd tycho::oc::ODEPhaseBase::calc_switches() {

    Eigen::MatrixXd uvals(this->u_vars(), this->active_traj_.size());
    Eigen::VectorXd tsnd(this->active_traj_.size());

    double t0 = this->active_traj_[0][this->t_var()];
    double tf = this->active_traj_.back()[this->t_var()];

    for (int i = 0; i < this->active_traj_.size(); i++) {
        uvals.col(i) = this->active_traj_[i].segment(this->xt_vars(), this->u_vars());
        tsnd[i] = (this->active_traj_[i][this->t_var()] - t0) / (tf - t0);
    }

    Eigen::VectorXd umin = uvals.rowwise().minCoeff();
    Eigen::VectorXd umax = uvals.rowwise().maxCoeff();
    Eigen::VectorXd ones(this->u_vars());
    ones.setOnes();

    Eigen::MatrixXd und(this->u_vars(), this->active_traj_.size());

    for (int i = 0; i < this->active_traj_.size(); i++) {
        und.col(i) = (uvals.col(i) - umin).cwiseQuotient(ones + umax - umin);
    }

    Eigen::VectorXd udiff;
    Eigen::VectorXd unddiff;
    std::vector<double> switches;

    for (int i = 0; i < this->active_traj_.size() - 1; i++) {
        udiff = (uvals.col(i + 1) - uvals.col(i)).cwiseAbs();
        unddiff = (uvals.col(i + 1) - uvals.col(i)).cwiseAbs();
        if (udiff.maxCoeff() > this->abs_switch_tol_ &&
            unddiff.maxCoeff() > this->rel_switch_tol_) {
            double t = tsnd[i + 1] / 2.0 + tsnd[i] / 2.0;
            switches.push_back(t);
        }
    }

    return utils::stdvector_to_eigenvector(switches);
}

tycho::ConvergenceFlags tycho::oc::ODEPhaseBase::psipot_call_impl(JetJobModes mode) {
    if (this->do_transcription_)
        this->transcribe();
    VectorXd input = this->make_solver_input();
    VectorXd output;

    switch (mode) {
    case JetJobModes::Solve:
        output = this->optimizer->solve(input);
        break;
    case JetJobModes::Optimize:
        output = this->optimizer->optimize(input);
        break;
    case JetJobModes::SolveOptimize:
        output = this->optimizer->solve_optimize(input);
        break;
    case JetJobModes::SolveOptimizeSolve:
        output = this->optimizer->solve_optimize_solve(input);
        break;
    case JetJobModes::OptimizeSolve:
        output = this->optimizer->optimize_solve(input);
        break;
    default:
        throw std::invalid_argument("Unrecognized PSIOPT mode");
    }

    this->collect_solver_output(output);

    this->collect_post_opt_info(this->optimizer->LastEqCons, this->optimizer->LastEqLmults,
                                this->optimizer->LastIqCons, this->optimizer->LastIqLmults);

    return this->optimizer->ConvergeFlag;
}

tycho::ConvergenceFlags tycho::oc::ODEPhaseBase::phase_call_impl(JetJobModes mode) {

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
            init_mesh_refinement();
            for (int i = 0; i < this->max_mesh_iters_; i++) {
                if (check_mesh() && !((i == 0) && this->force_one_mesh_iter_)) {
                    if (this->print_mesh_info_) {
                        MeshIterateInfo::print_header(i);
                        this->mesh_iters_.back().print(0);
                        fmt::print(fmt::fg(fmt::color::lime_green), "Mesh Converged\n");
                    }
                    break;
                } else if (i == this->max_mesh_iters_ - 1) {
                    if (this->print_mesh_info_) {
                        MeshIterateInfo::print_header(i);
                        this->mesh_iters_.back().print(0);
                        fmt::print(fmt::fg(fmt::color::red), "Mesh Not Converged\n");
                    }
                    break;
                } else {
                    update_mesh();
                    if (this->print_mesh_info_) {
                        MeshIterateInfo::print_header(i);
                        this->mesh_iters_.back().print(0);
                    }
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
