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
#include <sstream>

#include "tycho/detail/integrators/rk_steppers.h"
#include "tycho/detail/optimal_control/transcription/lgl_interp_functions.h"
#include "tycho/detail/optimal_control/transcription/lgl_interp_table.h"
#include "tycho/detail/vf/type_erasure/generic_conditional.h"
#include "tycho/detail/vf/type_erasure/generic_function.h"
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

#include "tycho/detail/utils/memory_management.h"

// Forward-declare GenericODE in its home namespace (tycho::oc) so that
// the integrators namespace can reference it before ode.h is included.
namespace tycho::oc {
template <class BaseType, int _XV, int _UV, int _PV> struct GenericODE;
template <class DODE, class Integrator> struct CentralShootingDefect;
} // namespace tycho::oc

namespace tycho::integrators {

// Import cross-namespace types from vf, utils, and oc.
using oc::GenericODE;
using oc::InterpFunction;
using oc::LGLInterpTable;
using utils::BumpAllocator;
using utils::SZ_SUM;
using utils::TempSpec;
using vf::Arguments;
using vf::Constant;
using vf::GenericConditional;
using vf::GenericFunction;
using vf::NestedFunction;
using vf::ParsedInput;
using vf::StackedOutputs;
using vf::VectorFunction;

// CentralShootingDefect lives in tycho::oc; use the qualified friend below.

template <class DODE>
struct Integrator : VectorFunction<Integrator<DODE>, SZ_SUM<DODE::IRC, 1>::value, DODE::IRC> {

    using Base = VectorFunction<Integrator<DODE>, SZ_SUM<DODE::IRC, 1>::value, DODE::IRC>;
    DENSE_FUNCTION_BASE_TYPES(Base);

    template <class Scalar> using ODEState = typename DODE::template Input<Scalar>;
    template <class Scalar> using ODEDeriv = typename DODE::template Output<Scalar>;

    using EventPack = std::tuple<GenericFunction<-1, 1>, int, int>;
    using EventLocsType = std::vector<std::vector<ODEState<double>>>;

    /// <summary>
    /// The type for the differentiable stepper function.
    /// Psuedo ODE is a compostion of the ode and control function(if any)
    /// </summary>
    /// <typeparam name="PseudoODE"></typeparam>
    template <class PseudoODE, RKOptions RKOp> using StepperType = RKStepper<PseudoODE, RKOp>;

    /// <summary>
    /// Wraps stepper types with RKoptions types
    /// </summary>
    using StepperWrapperType = GenericFunction<SZ_SUM<DODE::IRC, 1>::value, DODE::IRC>;
    using ControllerType = GenericFunction<-1, -1>;
    using StopFuncType = GenericConditional<-1>;

    using IntegRet = ODEState<double>;
    using DenseRet = std::vector<ODEState<double>>;
    using STMRet = std::tuple<IntegRet, Jacobian<double>>;

    using IntegEventRet = std::tuple<IntegRet, EventLocsType>;
    using DenseEventRet = std::tuple<DenseRet, EventLocsType>;
    using STMEventRet = std::tuple<IntegRet, Jacobian<double>, EventLocsType>;

    using ControlIndexType =
        std::variant<int, Eigen::VectorXi, std::string, std::vector<std::string>>;

    friend oc::CentralShootingDefect<DODE, Integrator>;

  protected:
    DODE ode;
    bool usecontroller = false;
    ControllerType controller;
    StepperWrapperType stepper;
    RKOptions rk_method_ = RKOptions::DOPRI54;

  public:
    Integrator() { this->enable_vectorization_ = true; }

    Integrator(const DODE &dode, std::string meth, double defstep) : Integrator() {
        // Use in_place_type to sidestep MSVC variant overload-resolution
        // ambiguity between the int and Eigen::VectorXi alternatives.
        ControlIndexType empty_ci{std::in_place_type<Eigen::VectorXi>};
        this->set_method(meth, dode, defstep, false, ControllerType{}, empty_ci);
        this->set_abs_tol(1.0e-12); // Must Be called after set_method!!!
        this->set_rel_tol(0);       // Must Be called after set_method!!!
    }
    Integrator(const DODE &dode, double defstep) : Integrator(dode, "DOPRI87", defstep) {}
    Integrator(const DODE &dode, std::string meth, double defstep, const ControllerType &ucon,
               const ControlIndexType &varlocs_t)
        : Integrator() {

        this->set_method(meth, dode, defstep, true, ucon, varlocs_t);
        this->set_abs_tol(1.0e-12); // Must Be called after set_method!!!
        this->set_rel_tol(0);       // Must Be called after set_method!!!
    }
    // VectorXi overloads: explicitly wrap into ControlIndexType to avoid MSVC
    // variant implicit-conversion ambiguity (int vs VectorXi alternatives).
    Integrator(const DODE &dode, std::string meth, double defstep, const ControllerType &ucon,
               const Eigen::VectorXi &varlocs)
        : Integrator(dode, meth, defstep, ucon,
                     ControlIndexType{std::in_place_type<Eigen::VectorXi>, varlocs}) {}
    Integrator(const DODE &dode, double defstep, const ControllerType &ucon,
               const ControlIndexType &varlocs_t)
        : Integrator(dode, "DOPRI87", defstep, ucon, varlocs_t) {}
    Integrator(const DODE &dode, double defstep, const ControllerType &ucon,
               const Eigen::VectorXi &varlocs)
        : Integrator(dode, "DOPRI87", defstep, ucon,
                     ControlIndexType{std::in_place_type<Eigen::VectorXi>, varlocs}) {}
    Integrator(const DODE &dode, double defstep, const Eigen::VectorXd &v) : Integrator() {

        Eigen::VectorXi tloc(1);
        tloc[0] = dode.TVar();
        GenericFunction<-1, -1> ucon = Constant<-1, -1>(1, v);
        this->set_method("DOPRI87", dode, defstep, true, ucon, tloc);
        this->set_abs_tol(1.0e-12); // Must Be called after set_method!!!
        this->set_rel_tol(0);       // Must Be called after set_method!!!
    }
    Integrator(const DODE &dode, std::string meth, double defstep,
               std::shared_ptr<LGLInterpTable> tab, const Eigen::VectorXi &ulocs)
        : Integrator() {

        Eigen::VectorXi varlocs(1);
        varlocs[0] = dode.TVar();
        ControllerType ucon = InterpFunction<-1>(tab, ulocs);
        this->set_method(meth, dode, defstep, true, ucon, varlocs);
        this->set_abs_tol(1.0e-12); // Must Be called after set_method!!!
        this->set_rel_tol(0);       // Must Be called after set_method!!!
    }
    Integrator(const DODE &dode, double defstep, std::shared_ptr<LGLInterpTable> tab,
               const Eigen::VectorXi &ulocs)
        : Integrator(dode, "DOPRI87", defstep, tab, ulocs) {}

    Integrator(const DODE &dode, std::string meth, double defstep,
               std::shared_ptr<LGLInterpTable> tab)
        : Integrator() {

        // Bug waiting to happen when LGL interp table is re-factored
        if (dode.input_rows() != tab->XtUVars || dode.XVars() != tab->XVars) {
            throw std::invalid_argument("Table data does not match expected dimension of ODE."
                                        " Please provide the indices variables in the table you "
                                        "want to interpret as controls.\n");
        }
        Eigen::VectorXi ulocs;
        ulocs.setLinSpaced(dode.UVars(), dode.TVar() + 1, dode.TVar() + dode.UVars());

        Eigen::VectorXi varlocs(1);
        varlocs[0] = dode.TVar();
        ControllerType ucon = InterpFunction<-1>(tab, ulocs);
        this->set_method(meth, dode, defstep, true, ucon, varlocs);
        this->set_abs_tol(1.0e-12); // Must Be called after set_method!!!
        this->set_rel_tol(0);       // Must Be called after set_method!!!
    }
    Integrator(const DODE &dode, double defstep, std::shared_ptr<LGLInterpTable> tab)
        : Integrator(dode, "DOPRI87", defstep, tab) {}

    void set_method(std::string str, const DODE &dode, double defstep, bool usecontrol,
                    const GenericFunction<-1, -1> &ucon, const ControlIndexType &varlocs_t) {

        this->set_step_sizes(defstep, defstep / 10000, defstep * 10000);

        if (str == "DOPRI54" || str == "DP54") {
            this->rk_method_ = RKOptions::DOPRI54;
            this->error_order_ = 4;
            // Using DOPRI5 rather than DOPRI54 here is not a mistake
            this->init_stepper_and_controller<RKOptions::DOPRI5>(dode, usecontrol, ucon, varlocs_t);
        } else if (str == "DOPRI87" || str == "DP87") {
            this->rk_method_ = RKOptions::DOPRI87;
            this->error_order_ = 7;
            this->init_stepper_and_controller<RKOptions::DOPRI87>(dode, usecontrol, ucon,
                                                                  varlocs_t);
        } else {
            throw std::invalid_argument("Invalid integration method '{0:}'.");
        }
    }

    template <RKOptions RKOp>
    void init_stepper_and_controller(const DODE &odet, bool usecontrol,
                                     const GenericFunction<-1, -1> &ucon,
                                     const ControlIndexType &varlocs_t) {

        this->ode = odet;
        Eigen::VectorXi varlocs = this->get_var_locs(varlocs_t);
        this->set_io_rows(this->ode.input_rows() + 1, this->ode.input_rows());

        this->usecontroller = usecontrol;

        auto Stepper = StepperType<DODE, RKOp>(ode);
        constexpr int IRC = decltype(Stepper)::IRC;
        constexpr int DUV = (DODE::UV == 1) ? -1 : DODE::UV;
        if constexpr (DODE::UV == 0) {
            this->stepper = StepperWrapperType(Stepper);
            this->controller = Arguments<-1>(ode.UVars());
        } else {
            if (!this->usecontroller) {
                this->stepper = StepperWrapperType(Stepper);
                this->controller = Arguments<-1>(ode.UVars());

            } else {

                if (ucon.output_rows() != ode.UVars()) {
                    throw std::invalid_argument(
                        "Controller output size does not match number of ode control variables");
                }

                if (ucon.input_rows() != varlocs.size()) {
                    throw std::invalid_argument("Controller input size is inconsistent with "
                                                "specified number of input state variables");
                }

                Arguments<DODE::IRC> odeargs(ode.input_rows());
                ParsedInput<GenericFunction<-1, -1>, DODE::IRC, DUV> controllerfunc(
                    ucon, varlocs, ode.input_rows());
                this->controller = controllerfunc;

                if constexpr (DODE::PV == 0) {

                    auto ode_args = StackedOutputs{odeargs.template head<DODE::XtV>(ode.XtVars()),
                                                   controllerfunc};
                    auto ode_expr = NestedFunction<DODE, decltype(ode_args)>(ode, ode_args);
                    auto gen_ode =
                        GenericODE<GenericFunction<-1, -1>, DODE::XV, DODE::UV, DODE::PV>(
                            ode_expr, ode.XVars(), ode.UVars(), ode.PVars());
                    auto stepper_u = ode_args.eval(StepperType<decltype(gen_ode), RKOp>(gen_ode));
                    this->stepper = StepperWrapperType(stepper_u);
                } else {

                    auto ode_args =
                        StackedOutputs{odeargs.template head<DODE::XtV>(ode.XtVars()),
                                       controllerfunc, odeargs.template tail<-1>(ode.PVars())};
                    auto ode_expr = NestedFunction<DODE, decltype(ode_args)>(ode, ode_args);
                    auto gen_ode =
                        GenericODE<GenericFunction<-1, -1>, DODE::XV, DODE::UV, DODE::PV>(
                            ode_expr, ode.XVars(), ode.UVars(), ode.PVars());
                    auto stepper_up = ode_args.eval(StepperType<decltype(gen_ode), RKOp>(gen_ode));
                    this->stepper = StepperWrapperType(stepper_up);
                }
            }
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////

    GenericFunction<-1, -1> get_stepper() { return this->stepper; }

    Eigen::VectorXi get_var_locs(const ControlIndexType &varlocs_t) {

        Eigen::VectorXi varlocs;

        /////////////////////////////////////////////////
        if (std::holds_alternative<int>(varlocs_t)) {
            varlocs.resize(1);
            varlocs[0] = std::get<int>(varlocs_t);
        } else if (std::holds_alternative<Eigen::VectorXi>(varlocs_t)) {
            varlocs = std::get<Eigen::VectorXi>(varlocs_t);
        } else if (std::holds_alternative<std::string>(varlocs_t)) {
            varlocs = this->ode.idx(std::get<std::string>(varlocs_t));
        } else if (std::holds_alternative<std::vector<std::string>>(varlocs_t)) {

            std::vector<Eigen::VectorXi> varvec;
            int size = 0;
            auto tmpvars = std::get<std::vector<std::string>>(varlocs_t);

            for (auto tmpv : tmpvars) {
                varvec.push_back(this->ode.idx(tmpv));
                size += varvec.back().size();
            }
            varlocs.resize(size);
            int next = 0;
            for (auto varv : varvec) {
                for (int i = 0; i < varv.size(); i++) {
                    varlocs[next] = varv[i];
                    next++;
                }
            }
        }
        return varlocs;
    }

    double error_order_ = 7.0;
    double min_step_size_ = 0.1;
    double def_step_size_ = 0.1;
    double max_step_size_ = 0.1;
    double max_step_change_ = 3.0;
    bool adaptive_ = true;
    bool fast_adaptive_stm_ = true;
    double event_tol_ = 1.0e-6;
    int max_event_iters_ = 10;
    bool vectorize_batch_calls_ = true;

    double step_frac_ = .9;
    double err_pow_fac_ = 1;

    ODEDeriv<double> abs_tols_;
    ODEDeriv<double> rel_tols_;

    void set_abs_tol(double tol) { this->abs_tols_.setConstant(this->ode.XVars(), abs(tol)); }
    void set_rel_tol(double tol) { this->rel_tols_.setConstant(this->ode.XVars(), abs(tol)); }

    void set_abs_tols(ODEDeriv<double> tol) {
        if (tol.size() != this->ode.XVars()) {
            throw std::invalid_argument("Incorrectly sized tolerance vector.");
        }
        this->abs_tols_ = tol;
    }
    void set_rel_tols(ODEDeriv<double> tol) {
        if (tol.size() != this->ode.XVars()) {
            throw std::invalid_argument("Incorrectly sized tolerance vector.");
        }
        this->rel_tols_ = tol;
    }

    ODEDeriv<double> get_abs_tols() const { return this->abs_tols_; }
    ODEDeriv<double> get_rel_tols() const { return this->rel_tols_; }

    void set_step_sizes(double defstep, double minstep, double maxstep) {
        if (defstep < minstep) {
            throw ::std::invalid_argument(
                "Default integrator stepsize must be greater than minimum stepsize.");
        }
        if (defstep > maxstep) {
            throw ::std::invalid_argument(
                "Default integrator stepsize must be less maximum stepsize.");
        }
        if (minstep > maxstep) {
            throw ::std::invalid_argument(
                "Minimum integrator stepsize must be greater than minimum stepsize.");
        }
        if (defstep < 0 || minstep < 0 || maxstep < 0) {
            throw ::std::invalid_argument("Stepsizes must be positive numbers (this doesnt mean "
                                          "you cant integrate backwards).");
        }

        this->def_step_size_ = defstep;
        this->min_step_size_ = minstep;
        this->max_step_size_ = maxstep;
    }

    /////////////////////////////////////////////////////////////////////////////////////

  protected:
    double calc_hnext(double h, double err, double accerr) const {
        return this->step_frac_ * h *
               pow((accerr / err), 1.0 / (this->error_order_ + err_pow_fac_));
    }

    template <class State> void update_control(State &xtup) const {
        if constexpr (DODE::UV != 0) {
            if (this->usecontroller) {
                this->controller.compute(
                    xtup, xtup.template segment<DODE::UV>(this->ode.TVar() + 1, this->ode.UVars()));
            }
        }
    }

    template <RKOptions RKOp, class Scalar>
    inline void stepper_compute_impl(const ODEState<Scalar> &x, Scalar tf, ODEState<Scalar> &xf,
                                     ODEState<Scalar> &xf_est, bool dofsal,
                                     ODEDeriv<Scalar> &xdot_prev, bool domidpoint,
                                     ODEState<Scalar> &xf_mid) const {

        using RKData = RKCoeffs<RKOp>;
        constexpr int Stages = RKData::Stages;
        constexpr int Stgsm1 = RKData::Stages - 1;
        constexpr bool is_diag_ = RKData::is_diag_;

        auto Impl = [&](auto &k_vals, auto &xtup) {
            xtup = x;
            Scalar t0 = xtup[this->ode.TVar()];
            Scalar h = tf - t0;

            if (dofsal || domidpoint) {
                k_vals[0] = xdot_prev * h;
            } else {
                this->update_control(xtup);
                this->ode.compute(xtup, k_vals[0]);
                k_vals[0] *= h;
            }

            if constexpr (true) {
                for (int i = 0; i < Stgsm1; i++) {
                    Scalar ti = t0 + RKData::Times[i] * h;
                    xtup = x;
                    xtup[this->ode.TVar()] = ti;
                    const int ip1 = i + 1;
                    const int js = is_diag_ ? i : 0;
                    for (int j = js; j < ip1; j++) {
                        xtup.template segment<DODE::XV>(0, this->ode.XVars()) +=
                            Scalar(RKData::ACoeffs[i][j]) * k_vals[j];
                    }

                    this->update_control(xtup);
                    this->ode.compute(xtup, k_vals[ip1]);

                    k_vals[ip1] *= h;
                }
            } else {

                const int tvar = this->ode.TVar();

                tycho::utils::constexpr_for_loop(
                    std::integral_constant<int, 0>(), std::integral_constant<int, Stgsm1>(),
                    [&](auto i) {
                        Scalar ti = t0 + RKData::Times[i.value] * h;
                        xtup = x;
                        xtup[tvar] = ti;
                        constexpr int ip1 = i.value + 1;
                        const int js = is_diag_ ? i.value : 0;

                        tycho::utils::constexpr_for_loop(
                            std::integral_constant<int, 0>(), std::integral_constant<int, ip1>(),
                            [&](auto j) {
                                if constexpr (RKData::ACoeffs[i.value][j.value] != 0.0) {
                                    xtup.template segment<DODE::XV>(0, tvar) +=
                                        Scalar(RKData::ACoeffs[i.value][j.value]) * k_vals[j.value];
                                }
                            });

                        this->update_control(xtup);
                        this->ode.compute(xtup, k_vals[ip1]);

                        k_vals[ip1] *= h;
                    });
            }

            xtup = x;
            xtup[this->ode.TVar()] = tf;
            for (int i = 0; i < Stages; i++) {
                xtup.template segment<DODE::XV>(0, this->ode.XVars()) +=
                    Scalar(RKData::BCoeffs[i]) * k_vals[i];
            }

            this->update_control(xtup);
            xf = xtup; // Next State

            if (dofsal || domidpoint) {
                if constexpr (RKData::FSAL) {
                    xdot_prev = k_vals.back() * (1.0 / h);
                }
            }

            xtup = x;
            xtup[this->ode.TVar()] = tf;

            for (int i = 0; i < Stages; i++) {
                xtup.template segment<DODE::XV>(0, this->ode.XVars()) +=
                    Scalar(RKData::CCoeffs[i]) * k_vals[i];
            }

            xf_est = xtup; // Estimate

            if (domidpoint) {

                xtup = x;
                xtup[this->ode.TVar()] = t0 + h / 2.0;

                for (int i = 0; i < Stages; i++) {
                    xtup.template segment<DODE::XV>(0, this->ode.XVars()) +=
                        Scalar(RKData::MidCoeffs[i] / 2.0) * k_vals[i];
                }

                if constexpr (!RKData::FSAL) {
                    k_vals.back().setZero();
                    this->ode.compute(xf, k_vals.back());
                    xtup.template segment<DODE::XV>(0, this->ode.XVars()) +=
                        Scalar(RKData::MidCoeffs.back() / 2.0) * k_vals.back() * h;
                    xdot_prev = k_vals.back();
                }

                this->update_control(xtup);

                xf_mid = xtup;
            }
        };

        BumpAllocator::allocate_run(
            Impl, ArrayOfTempSpecs<ODEDeriv<Scalar>, Stages>(this->ode.output_rows(), 1),
            TempSpec<ODEState<Scalar>>(this->ode.input_rows(), 1));
    }

    template <class Scalar>
    inline void stepper_compute(const ODEState<Scalar> &x, Scalar tf, ODEState<Scalar> &xf,
                                ODEState<Scalar> &xf_est, ODEDeriv<Scalar> &xdot_prev,
                                bool domidpoint, ODEState<Scalar> &xf_mid) const {

        switch (this->rk_method_) {
        case RKOptions::DOPRI54: {
            this->stepper_compute_impl<RKOptions::DOPRI54, Scalar>(x, tf, xf, xf_est, true,
                                                                   xdot_prev, domidpoint, xf_mid);
        } break;
        case RKOptions::DOPRI87: {
            this->stepper_compute_impl<RKOptions::DOPRI87, Scalar>(x, tf, xf, xf_est, false,
                                                                   xdot_prev, domidpoint, xf_mid);
        } break;
        default: {
        }
        }
    }

    Output<double> integrate_impl(const ODEState<double> &x, double tf,
                                  const std::vector<EventPack> &events,
                                  std::vector<std::vector<Eigen::Vector2d>> &eventtimes,
                                  bool storestates, bool storederivs, bool storemidpoints,
                                  std::vector<ODEState<double>> &states,
                                  std::vector<ODEDeriv<double>> &derivs) const {

        if (x.size() != this->ode.input_rows()) {
            throw std::invalid_argument("Incorrectly sized input state.");
        }

        double t0 = x[this->ode.TVar()];
        double H = tf - t0;
        int numsteps = int(abs(H / this->def_step_size_)) + 1;
        double h = .9 * (H / double(numsteps));

        ODEState<double> xi = x;
        this->update_control(xi);

        ODEState<double> xnext = xi;
        ODEState<double> xnext_est = xi;
        ODEState<double> xnext_mid = xi;

        ODEDeriv<double> xdoti(this->ode.output_rows());
        xdoti.setZero();
        this->ode.compute(xi, xdoti);
        ODEDeriv<double> xdotnext = xdoti;

        std::vector<Vector1<double>> prev_event_vals(events.size());
        std::vector<Vector1<double>> next_event_vals(events.size());

        for (int j = 0; j < events.size(); j++) {
            prev_event_vals[j].setZero();
            next_event_vals[j].setZero();

            if (std::get<0>(events[j]).input_rows() != this->ode.input_rows()) {
                throw std::invalid_argument(
                    "Input size of event function must equal input size of ode.");
            }

            std::get<0>(events[j]).compute(xi, prev_event_vals[j]);
        }

        eventtimes.resize(events.size());

        if (storestates) {
            states.resize(0);
            derivs.resize(0);
            if (storemidpoints) {
                states.reserve(numsteps * 2 + 2);
                if (storederivs)
                    derivs.reserve(numsteps * 2 + 2);
            } else {
                states.reserve(numsteps + 2);
                if (storederivs)
                    derivs.reserve(numsteps + 2);
            }
            states.push_back(xi);
            if (storederivs)
                derivs.push_back(xdoti);
        }

        ODEDeriv<double> abs_error;
        ODEDeriv<double> abs_error_max;
        ODEDeriv<double> err_vec;

        bool hit_minimum = false;
        int minimum_count = 0;
        int i = 0;
        bool continueloop = true;

        while (continueloop) {

            double tnext = xi[this->ode.TVar()] + h;

            if (H > 0.0) {
                if ((tnext - tf) >= 0.0) {
                    h = tf - xi[this->ode.TVar()];
                    tnext = tf;
                    continueloop = false;
                }
            } else {
                if ((tnext - tf) <= 0.0) {
                    h = tf - xi[this->ode.TVar()];
                    tnext = tf;
                    continueloop = false;
                }
            }

            xnext.setZero();
            xnext_est.setZero();
            xnext_mid.setZero();
            xdotnext = xdoti;

            this->stepper_compute(xi, tnext, xnext, xnext_est, xdotnext,
                                  storemidpoints || storederivs, xnext_mid);

            if (this->adaptive_) {
                abs_error =
                    (xnext.head(this->ode.XVars()) - xnext_est.head(this->ode.XVars())).cwiseAbs();

                err_vec = this->abs_tols_ +
                          xnext.head(this->ode.XVars()).cwiseAbs().cwiseProduct(this->rel_tols_);

                abs_error_max = abs_error.cwiseQuotient(err_vec);
                int worst = 0;
                abs_error_max.maxCoeff(&worst);

                double err = abs_error[worst];
                double acc = err_vec[worst];
                double hnext = calc_hnext(h, err, acc);

                if (hnext / h > this->max_step_change_)
                    h *= this->max_step_change_;
                else if (hnext / h < 1. / this->max_step_change_)
                    h /= this->max_step_change_;
                else
                    h = hnext;

                if (abs(h) > this->max_step_size_)
                    h = this->max_step_size_ * h / abs(h);

                if (abs(h) < this->min_step_size_) {
                    h = this->min_step_size_ * h / abs(h);
                    hit_minimum = true;
                    minimum_count++;
                } else {
                    hit_minimum = false;
                }
                if ((err - acc) > 0 && !hit_minimum) {
                    continueloop = true;
                    continue;
                }
            }

            bool eventbreak = false;
            for (int j = 0; j < events.size(); j++) {
                next_event_vals[j].setZero();
                std::get<0>(events[j]).compute(xnext, next_event_vals[j]);

                double vprev = prev_event_vals[j][0];
                double vnext = next_event_vals[j][0];

                int dir = std::get<1>(events[j]);

                double vprod = vprev * vnext;

                if (vprod < 0.0) {
                    if ((dir > 0 && vnext > 0) || (dir < 0 && vnext < 0) || dir == 0) {
                        Eigen::Vector2d times;
                        times[0] = xi[this->ode.TVar()];
                        times[1] = xnext[this->ode.TVar()];
                        eventtimes[j].push_back(times);
                        int stop = std::get<2>(events[j]);

                        if (stop != 0) {
                            if (eventtimes[j].size() == stop) {
                                eventbreak = true;
                            }
                        }
                    }
                }
            }

            xi = xnext;
            xdoti = xdotnext;
            prev_event_vals = next_event_vals;

            if (storestates) {
                if (storemidpoints) {
                    states.push_back(xnext_mid);
                    if (storederivs) {
                        xdotnext.setZero();
                        this->ode.compute(xnext_mid, xdotnext);
                        derivs.push_back(xdotnext);
                    }
                }
                states.push_back(xi);
                if (storederivs)
                    derivs.push_back(xdoti);
            }

            if (eventbreak)
                break;
            i++;
        }
        return xi;
    }

    std::vector<Output<double>>
    integrate_impl_vectorized(const std::vector<ODEState<double>> &xs, const Eigen::VectorXd &tfs,
                              const std::vector<EventPack> &events,
                              std::vector<std::vector<std::vector<Eigen::Vector2d>>> &eventtimes_s,
                              bool storestates, bool storederivs, bool storemidpoints,
                              std::vector<std::vector<ODEState<double>>> &states_s,
                              std::vector<std::vector<ODEDeriv<double>>> &derivs_s) const {

        if (xs.size() != tfs.size()) {
            throw std::invalid_argument("Number of initial states and final times must match.");
        }
        if (xs.size() == 0) {
            throw std::invalid_argument("Must supply at least one initial state.");
        }

        int ntrajs = xs.size();

        Eigen::VectorXd hs(ntrajs);
        Eigen::VectorXd h_spans(ntrajs);
        std::vector<ODEState<double>> xis = xs;
        std::vector<ODEDeriv<double>> xdotis(ntrajs);
        Eigen::VectorXd tnexts(ntrajs);

        std::vector<std::vector<Vector1<double>>> prev_event_vals_s(ntrajs);
        std::vector<std::vector<Vector1<double>>> next_event_vals_s(ntrajs);

        for (int i = 0; i < ntrajs; i++) {
            if (xis[i].size() != this->ode.input_rows()) {
                throw std::invalid_argument("Incorrectly sized input state.");
            }
            this->update_control(xis[i]);

            double t0 = xis[i][this->ode.TVar()];
            h_spans[i] = tfs[i] - t0;
            int numsteps = int(abs(h_spans[i] / this->def_step_size_)) + 1;
            hs[i] = .9 * (h_spans[i] / double(numsteps));

            xdotis[i].resize(this->ode.output_rows());
            xdotis[i].setZero();
            this->ode.compute(xis[i], xdotis[i]);

            prev_event_vals_s[i].resize(events.size());
            next_event_vals_s[i].resize(events.size());

            for (int j = 0; j < events.size(); j++) {
                prev_event_vals_s[i][j].setZero();
                next_event_vals_s[i][j].setZero();

                if (std::get<0>(events[j]).input_rows() != this->ode.input_rows()) {
                    throw std::invalid_argument(
                        "Input size of event function must equal input size of ode.");
                }

                std::get<0>(events[j]).compute(xis[i], prev_event_vals_s[i][j]);
            }
            if (events.size() > 0) {
                eventtimes_s[i].resize(events.size());
            }

            if (storestates) {
                states_s[i].resize(0);
                derivs_s[i].resize(0);
                if (storemidpoints) {
                    states_s[i].reserve(numsteps * 2 + 2);
                    if (storederivs)
                        derivs_s[i].reserve(numsteps * 2 + 2);
                } else {
                    states_s[i].reserve(numsteps + 2);
                    if (storederivs)
                        derivs_s[i].reserve(numsteps + 2);
                }
                states_s[i].push_back(xis[i]);
                if (storederivs)
                    derivs_s[i].push_back(xdotis[i]);
            }
        }

        using SuperScalar = tycho::DefaultSuperScalar;

        ODEState<double> xnext(this->ode.input_rows());
        ODEState<double> xnext_est(this->ode.input_rows());
        ODEState<double> xnext_mid(this->ode.input_rows());
        ODEDeriv<double> xdotnext(this->ode.output_rows());

        ODEState<double> xi(this->ode.input_rows());

        ODEState<SuperScalar> xi_ss(this->ode.input_rows());

        ODEState<SuperScalar> xnext_ss(this->ode.input_rows());
        ODEState<SuperScalar> xnext_est_ss(this->ode.input_rows());
        ODEState<SuperScalar> xnext_mid_ss(this->ode.input_rows());
        ODEDeriv<SuperScalar> xdotnext_ss(this->ode.output_rows());
        SuperScalar tnext_ss;

        ODEDeriv<double> abs_error(ode.XVars());
        ODEDeriv<double> abs_error_max(ode.XVars());
        ODEDeriv<double> err_vec(ode.XVars());
        ODEDeriv<SuperScalar> abs_error_ss(ode.XVars());

        std::vector<bool> continueloops(ntrajs, true);
        std::vector<bool> hit_minimums(ntrajs, false);

        int numrunning = ntrajs;
        int lastrunning = ntrajs - 1;

        if (ntrajs < SuperScalar::SizeAtCompileTime) {

            for (int V = 0; V < SuperScalar::SizeAtCompileTime; V++) {
                for (int k = 0; k < this->ode.input_rows(); k++) {
                    xi_ss[k][V] = xis[0][k];
                }
                for (int k = 0; k < this->ode.output_rows(); k++) {
                    xdotnext_ss[k][V] = xdotis[0][k];
                }
            }
        }

        while (numrunning > 0) {

            std::array<int, SuperScalar::SizeAtCompileTime> idxs;

            int V = 0;

            for (int i = 0; i < ntrajs; i++) {
                if (continueloops[i]) {
                    double tnext = xis[i][this->ode.TVar()] + hs[i];

                    if (h_spans[i] > 0.0) {
                        if ((tnext - tfs[i]) >= 0.0) {
                            hs[i] = tfs[i] - xis[i][this->ode.TVar()];
                            tnext = tfs[i];
                            continueloops[i] = false;
                        }
                    } else {
                        if ((tnext - tfs[i]) <= 0.0) {
                            hs[i] = tfs[i] - xis[i][this->ode.TVar()];
                            tnext = tfs[i];
                            continueloops[i] = false;
                        }
                    }

                    for (int k = 0; k < this->ode.input_rows(); k++) {
                        xi_ss[k][V] = xis[i][k];
                    }
                    for (int k = 0; k < this->ode.output_rows(); k++) {
                        xdotnext_ss[k][V] = xdotis[i][k];
                    }

                    idxs[V] = i;
                    tnext_ss[V] = tnext;
                    V++;

                    int Vmax = (i == lastrunning) && V != SuperScalar::SizeAtCompileTime
                                   ? V
                                   : SuperScalar::SizeAtCompileTime;

                    if (V == Vmax) {
                        V = 0;
                        xnext_ss.setZero();
                        xnext_est_ss.setZero();
                        xnext_mid_ss.setZero();

                        this->stepper_compute(xi_ss, tnext_ss, xnext_ss, xnext_est_ss, xdotnext_ss,
                                              storemidpoints || storederivs, xnext_mid_ss);

                        abs_error_ss = (xnext_ss.head(this->ode.XVars()) -
                                        xnext_est_ss.head(this->ode.XVars()))
                                           .cwiseAbs();

                        for (int V = 0; V < Vmax; V++) {

                            int itmp = idxs[V];

                            for (int k = 0; k < this->ode.input_rows(); k++) {
                                xnext[k] = xnext_ss[k][V];
                                xnext_mid[k] = xnext_mid_ss[k][V];
                            }
                            for (int k = 0; k < this->ode.output_rows(); k++) {
                                xdotnext[k] = xdotnext_ss[k][V];
                                abs_error[k] = abs_error_ss[k][V];
                            }

                            if (this->adaptive_) {

                                double h = hs[itmp];

                                err_vec = this->abs_tols_ + xnext.head(this->ode.XVars())
                                                                .cwiseAbs()
                                                                .cwiseProduct(this->rel_tols_);

                                abs_error_max = abs_error.cwiseQuotient(err_vec);
                                int worst = 0;
                                abs_error_max.maxCoeff(&worst);

                                double err = abs_error[worst];
                                double acc = err_vec[worst];
                                double hnext = calc_hnext(h, err, acc);

                                if (hnext / h > this->max_step_change_)
                                    h *= this->max_step_change_;
                                else if (hnext / h < 1. / this->max_step_change_)
                                    h /= this->max_step_change_;
                                else
                                    h = hnext;

                                if (abs(h) > this->max_step_size_)
                                    h = this->max_step_size_ * h / abs(h);

                                if (abs(h) < this->min_step_size_) {
                                    h = this->min_step_size_ * h / abs(h);
                                    hit_minimums[itmp] = true;
                                } else {
                                    hit_minimums[itmp] = false;
                                }

                                hs[itmp] = h;

                                if ((err - acc) > 0 && !hit_minimums[itmp]) {
                                    continueloops[itmp] = true;
                                    continue;
                                }
                            }

                            bool eventbreak = false;
                            for (int j = 0; j < events.size(); j++) {
                                next_event_vals_s[itmp][j].setZero();
                                std::get<0>(events[j]).compute(xnext, next_event_vals_s[itmp][j]);

                                double vprev = prev_event_vals_s[itmp][j][0];
                                double vnext = next_event_vals_s[itmp][j][0];

                                int dir = std::get<1>(events[j]);

                                double vprod = vprev * vnext;

                                if (vprod < 0.0) {
                                    if ((dir > 0 && vnext > 0) || (dir < 0 && vnext < 0) ||
                                        dir == 0) {
                                        Eigen::Vector2d times;
                                        times[0] = xis[itmp][this->ode.TVar()];
                                        times[1] = xnext[this->ode.TVar()];
                                        eventtimes_s[itmp][j].push_back(times);
                                        int stop = std::get<2>(events[j]);

                                        if (stop != 0) {
                                            if (eventtimes_s[itmp][j].size() == stop) {
                                                eventbreak = true;
                                            }
                                        }
                                    }
                                }
                            }

                            xis[itmp] = xnext;
                            xdotis[itmp] = xdotnext;
                            prev_event_vals_s[itmp] = next_event_vals_s[itmp];

                            if (storestates) {
                                if (storemidpoints) {
                                    states_s[itmp].push_back(xnext_mid);
                                    if (storederivs) {
                                        xdotnext.setZero();
                                        this->ode.compute(xnext_mid, xdotnext);
                                        derivs_s[itmp].push_back(xdotnext);
                                    }
                                }
                                states_s[itmp].push_back(xis[itmp]);
                                if (storederivs)
                                    derivs_s[itmp].push_back(xdotis[itmp]);
                            }

                            if (eventbreak)
                                continueloops[itmp] = false;
                        }
                    }
                }
            }

            numrunning = 0;

            for (int i = 0; i < ntrajs; i++) {
                if (continueloops[i]) {
                    lastrunning = i;
                    numrunning++;
                }
            }
        }
        return xis;
    }

    std::vector<std::vector<ODEState<double>>>
    find_events(std::shared_ptr<LGLInterpTable> tab, const std::vector<EventPack> &events,
                const std::vector<std::vector<Eigen::Vector2d>> &eventtimes) const {

        Eigen::VectorXi vars;
        vars.setLinSpaced(this->ode.input_rows(), 0, this->ode.input_rows() - 1);

        InterpFunction<-1> tabfunc(tab, vars);

        Vector1<double> x;
        Vector1<double> fx;
        Vector1<double> fl;

        Vector1<double> jx;

        std::vector<std::vector<ODEState<double>>> eventstates(events.size());

        for (int i = 0; i < events.size(); i++) {
            if (eventtimes[i].size() > 0) {

                auto func = std::get<0>(events[i]).eval(tabfunc);

                auto newton = [&](auto x0) {
                    x[0] = x0;
                    for (int k = 0; k < max_event_iters_; k++) {
                        fx.setZero();
                        jx.setZero();
                        func.compute_jacobian(x, fx, jx);
                        if (abs(fx[0]) < abs(event_tol_)) {
                            break;
                        }
                        x[0] = x[0] - fx[0] / jx[0];
                    }
                    return x[0];
                };

                auto bisect = [&](auto tlow, auto thigh, int iters) {
                    double tm = (tlow + thigh) / 2.0;
                    x[0] = tlow;
                    fl = func.compute(x);
                    int sgnfl = (fl[0] >= 0) - (fl[0] <= 0);

                    x[0] = tm;
                    fx = func.compute(x);
                    int sgnfx = (fx[0] >= 0) - (fx[0] <= 0);

                    for (int i = 0; i < 5; i++) {
                        if (sgnfx == sgnfl) {
                            tlow = tm;
                            sgnfl = sgnfx;
                        } else {
                            thigh = tm;
                        }

                        tm = (tlow + thigh) / 2.0;
                        if ((thigh - tlow) / 2.0 < abs(event_tol_))
                            break;

                        x[0] = tm;
                        fx = func.compute(x);
                        sgnfx = (fx[0] >= 0) - (fx[0] <= 0);
                    }

                    return std::array<double, 3>{tm, tlow, thigh};
                };

                for (auto &eventtime : eventtimes[i]) {

                    double tlow = eventtime[0];
                    double thigh = eventtime[1];

                    if (thigh < tlow) {
                        std::swap(tlow, thigh);
                    }

                    auto res = bisect(tlow, thigh, 2);
                    double tig = res[0];
                    double tlow2 = res[1];
                    double thigh2 = res[2];

                    double tevent = newton(tig);

                    if (tevent > tlow && tevent < thigh) {
                        ODEState<double> ei(this->ode.input_rows());
                        ei.setZero();
                        tab->InterpolateRef(tevent, ei);
                        eventstates[i].push_back(ei);
                    } else {

                        res = bisect(tlow2, thigh2, max_event_iters_);
                        tig = res[0];
                        tevent = newton(tig);

                        if (tevent > tlow && tevent < thigh) {
                            ODEState<double> ei(this->ode.input_rows());
                            ei.setZero();
                            tab->InterpolateRef(tevent, ei);
                            eventstates[i].push_back(ei);
                        } // else give up
                    }
                }
            }
        }

        return eventstates;
    }

    std::shared_ptr<LGLInterpTable> make_table(const std::vector<ODEState<double>> &xs,
                                               bool fifthorder) const {
        std::vector<Eigen::VectorXd> xs_in;
        for (auto &X : xs) {
            xs_in.push_back(X);
        }

        GenericFunction<-1, -1> odetemp;
        if constexpr (DODE::IsGenericODE) {
            odetemp = this->ode.func;
        } else {
            odetemp = this->ode;
        }
        TranscriptionModes m = fifthorder ? TranscriptionModes::LGL5 : TranscriptionModes::LGL3;
        std::shared_ptr<LGLInterpTable> tab = std::make_shared<LGLInterpTable>(
            odetemp, this->ode.XVars(), this->ode.UVars() + this->ode.PVars(), m);

        tab->load_exact_data(xs_in);

        return tab;
    }

    std::shared_ptr<LGLInterpTable> make_table(const std::vector<ODEState<double>> &xs,
                                               const std::vector<ODEDeriv<double>> &d_xs,
                                               bool fifthorder) const {

        GenericFunction<-1, -1> odetemp;
        if constexpr (DODE::IsGenericODE) {
            odetemp = this->ode.func;
        } else {
            odetemp = this->ode;
        }
        TranscriptionModes m = fifthorder ? TranscriptionModes::LGL5 : TranscriptionModes::LGL3;
        std::shared_ptr<LGLInterpTable> tab = std::make_shared<LGLInterpTable>(
            odetemp, this->ode.XVars(), this->ode.UVars() + this->ode.PVars(), m);

        tab->load_exact_data(xs, d_xs);

        return tab;
    }

    std::vector<ODEState<double>> midpoints_removed(const std::vector<ODEState<double>> &xs) const {
        std::vector<ODEState<double>> x_new;
        x_new.reserve((xs.size() - 1) / 2.0);
        for (int i = 0; i < xs.size(); i += 2) {
            x_new.push_back(xs[i]);
        }
        return x_new;
    }

    std::vector<Jacobian<double>>
    calculate_jacobians(const std::vector<std::vector<ODEState<double>>> &xs_s) const {

        constexpr int vsize = tycho::DefaultSuperScalar::SizeAtCompileTime;

        Input<tycho::DefaultSuperScalar> stepper_input_ss(this->input_rows());
        ODEState<tycho::DefaultSuperScalar> stepper_output_ss(this->ode.input_rows());
        Jacobian<tycho::DefaultSuperScalar> stepper_jacobian_ss(this->output_rows(),
                                                                this->input_rows());
        Hessian<tycho::DefaultSuperScalar> jxall_ss(this->input_rows(), this->input_rows());
        Jacobian<tycho::DefaultSuperScalar> jactmp_ss(this->output_rows(), this->input_rows());

        int ntrajs = xs_s.size();

        std::vector<int> idxs(ntrajs);

        std::iota(idxs.begin(), idxs.end(), 0);

        std::sort(idxs.begin(), idxs.end(),
                  [&](auto a, auto b) { return xs_s[a].size() < xs_s[b].size(); });

        int npacks = xs_s.size() / vsize;
        int nrem = xs_s.size() % vsize;
        if (nrem > 0)
            npacks++;

        std::vector<Hessian<double>> jxalls(ntrajs);
        std::vector<Jacobian<double>> jxs(ntrajs);

        if (ntrajs < tycho::DefaultSuperScalar::SizeAtCompileTime) {
            // Pad out inputs to prevent junk data being given to stepper
            for (int v = 0; v < tycho::DefaultSuperScalar::SizeAtCompileTime; v++) {
                for (int j = 0; j < this->ode.input_rows(); j++) {
                    stepper_input_ss[j][v] = xs_s[0][0][j];
                }
            }
        }

        for (int n = 0; n < npacks; n++) {

            int vmax = std::min(vsize, ntrajs - n * vsize);

            for (int v = 0; v < vmax; v++) {
                int idx = idxs[n * vsize + v];
                for (int j = 0; j < this->ode.input_rows(); j++) {
                    stepper_input_ss[j][v] = xs_s[idx][0][j];
                }
            }

            int ncalls = xs_s[idxs[n * vsize]].size() - 1;
            jxall_ss.setIdentity();
            jactmp_ss.setZero();

            for (int i = 0; i < ncalls; i++) {

                for (int v = 0; v < vmax; v++) {
                    int idx = idxs[n * vsize + v];
                    for (int j = 0; j < this->ode.input_rows(); j++) {
                        stepper_input_ss[j][v] = xs_s[idx][i][j];
                    }
                    stepper_input_ss[this->ode.input_rows()][v] =
                        xs_s[idx][i + 1][this->ode.TVar()];
                }

                stepper_output_ss.setZero();
                stepper_jacobian_ss.setZero();
                this->stepper.compute_jacobian(stepper_input_ss, stepper_output_ss,
                                               stepper_jacobian_ss);
                jactmp_ss.noalias() = stepper_jacobian_ss * jxall_ss;
                jxall_ss.template topRows<Base::ORC>(this->output_rows()) = jactmp_ss;
                stepper_input_ss.head(this->ode.input_rows()) = stepper_output_ss;
            }

            for (int v = 0; v < vmax; v++) {

                int idx = idxs[n * vsize + v];

                jxalls[idx].resize(this->input_rows(), this->input_rows());

                for (int k = 0; k < this->input_rows(); k++) {
                    for (int l = 0; l < this->input_rows(); l++) {
                        jxalls[idx](l, k) = jxall_ss(l, k)[v];
                    }
                }

                if (ncalls == xs_s[idx].size() - 1) {
                    jxs[idx] = jxalls[idx].template topRows<Base::ORC>(this->output_rows());
                } else {
                    std::vector<ODEState<double>> xs_tmp(xs_s[idx].begin() + ncalls,
                                                         xs_s[idx].end());

                    Jacobian<double> jtmp = this->calculate_jacobian(xs_tmp);

                    jxs[idx] = jtmp * jxalls[idx];
                }
            }
        }

        return jxs;
    }

    Jacobian<double> calculate_jacobian(const std::vector<ODEState<double>> &xs) const {

        Jacobian<double> jx(this->output_rows(), this->input_rows());
        jx.setZero();
        Hessian<double> jxall(this->input_rows(), this->input_rows());
        jxall.setIdentity();

        Input<double> stepper_input(this->input_rows());
        ODEState<double> stepper_output(this->ode.input_rows());
        Jacobian<double> stepper_jacobian(this->output_rows(), this->input_rows());
        Jacobian<double> jactmp(this->output_rows(), this->input_rows());

        int n = xs.size();
        int numsteps = xs.size() - 1;

        constexpr int vsize = tycho::DefaultSuperScalar::SizeAtCompileTime;

        Input<tycho::DefaultSuperScalar> stepper_input_ss(this->input_rows());
        ODEState<tycho::DefaultSuperScalar> stepper_output_ss(this->ode.input_rows());
        Jacobian<tycho::DefaultSuperScalar> stepper_jacobian_ss(this->output_rows(),
                                                                this->input_rows());

        auto scalar_impl = [&](int i) {
            stepper_input.head(this->ode.input_rows()) = xs[i];
            stepper_input[this->ode.input_rows()] = xs[i + 1][this->ode.TVar()];

            stepper_output.setZero();
            stepper_jacobian.setZero();

            this->stepper.compute_jacobian(stepper_input, stepper_output, stepper_jacobian);
            jactmp.noalias() = stepper_jacobian * jxall;
            jxall.template topRows<Base::ORC>(this->output_rows()) = jactmp;
        };

        auto vector_impl = [&](int i) {
            stepper_output_ss.setZero();
            stepper_jacobian_ss.setZero();

            for (int j = 0; j < vsize; j++) {
                for (int k = 0; k < this->ode.input_rows(); k++) {
                    stepper_input_ss.head(this->ode.input_rows())[k][j] = xs[i + j][k];
                }
                stepper_input_ss[this->ode.input_rows()][j] = xs[i + j + 1][this->ode.TVar()];
            }
            this->stepper.compute_jacobian(stepper_input_ss, stepper_output_ss,
                                           stepper_jacobian_ss);

            for (int j = 0; j < vsize; j++) {
                for (int k = 0; k < this->input_rows(); k++) {
                    for (int l = 0; l < this->output_rows(); l++) {
                        stepper_jacobian(l, k) = stepper_jacobian_ss(l, k)[j];
                    }
                }
                jactmp.noalias() = stepper_jacobian * jxall;
                jxall.template topRows<Base::ORC>(this->output_rows()) = jactmp;
            }
        };

        int packs = (this->enable_vectorization_) ? numsteps / vsize : 0;

        for (int i = 0; i < packs; i++) {
            vector_impl(i * vsize);
        }
        for (int i = packs * vsize; i < numsteps; i++) {
            scalar_impl(i);
        }

        jx = jxall.template topRows<Base::ORC>(this->output_rows());

        return jx;
    }

    std::tuple<Jacobian<double>, Hessian<double>>
    calculate_jacobian_hessian(const std::vector<ODEState<double>> &xs,
                               const ODEState<double> &lf) const {
        ODEState<double> xf(this->ode.input_rows());
        xf.setZero();

        Jacobian<double> jx(this->output_rows(), this->input_rows());
        jx.setZero();

        Jacobian<double> jxall(this->output_rows(), this->input_rows());
        jxall.setZero();
        jxall.leftCols(this->output_rows()).setIdentity();

        Hessian<double> hxall(this->input_rows(), this->input_rows());
        hxall.setZero();

        Input<double> stepper_input(this->input_rows());
        Input<double> stepper_grad(this->input_rows());

        ODEState<double> stepper_output(this->ode.input_rows());
        Jacobian<double> stepper_jacobian(this->output_rows(), this->input_rows());
        Hessian<double> stepper_hessian(this->input_rows(), this->input_rows());

        ODEState<double> stepper_adjvars = lf;

        Hessian<double> jtwist(this->input_rows(), this->input_rows());
        jtwist.setZero();
        jtwist(this->input_rows() - 1, this->input_rows() - 1) = 1.0;

        int numsteps = xs.size() - 1;

        for (int i = 0; i < numsteps; i++) {
            stepper_input.head(this->ode.input_rows()) = xs[numsteps - i - 1];
            stepper_input[this->ode.input_rows()] = xs[numsteps - i][this->ode.TVar()];

            stepper_output.setZero();
            stepper_jacobian.setZero();
            stepper_grad.setZero();
            stepper_hessian.setZero();

            this->stepper.compute_jacobian_adjointgradient_adjointhessian(
                stepper_input, stepper_output, stepper_jacobian, stepper_grad, stepper_hessian,
                stepper_adjvars);

            jtwist.topRows(this->output_rows()) = stepper_jacobian;
            jxall = jxall * jtwist;
            if (i == 0) {
                jxall.rightCols(1) = stepper_jacobian.rightCols(1);
            }

            hxall = jtwist.transpose() * hxall * jtwist;
            hxall += stepper_hessian;
            stepper_adjvars = stepper_grad.head(this->output_rows());
        }

        jx = jxall.template topRows<Base::ORC>(this->output_rows());

        return std::tuple<Jacobian<double>, Hessian<double>>{jx, hxall};
    }

    std::tuple<std::vector<Jacobian<double>>, std::vector<Hessian<double>>>
    calculate_jacobians_hessians(const std::vector<std::vector<ODEState<double>>> &xs_s,
                                 const std::vector<ODEState<double>> &lf_s) const {

        constexpr int vsize = tycho::DefaultSuperScalar::SizeAtCompileTime;

        Input<tycho::DefaultSuperScalar> stepper_input_ss(this->input_rows());
        ODEState<tycho::DefaultSuperScalar> stepper_output_ss(this->ode.input_rows());
        Input<tycho::DefaultSuperScalar> stepper_grad_ss(this->input_rows());
        Jacobian<tycho::DefaultSuperScalar> stepper_jacobian_ss(this->output_rows(),
                                                                this->input_rows());
        Hessian<tycho::DefaultSuperScalar> stepper_hessian_ss(this->input_rows(),
                                                              this->input_rows());
        ODEState<tycho::DefaultSuperScalar> stepper_adjvars_ss(this->ode.input_rows());

        Jacobian<tycho::DefaultSuperScalar> jxall_ss(this->output_rows(), this->input_rows());
        Hessian<tycho::DefaultSuperScalar> jtwist_ss(this->input_rows(), this->input_rows());
        Hessian<tycho::DefaultSuperScalar> hxall_ss(this->input_rows(), this->input_rows());

        Jacobian<double> jxall(this->output_rows(), this->input_rows());
        Hessian<double> jtwist(this->input_rows(), this->input_rows());
        Hessian<double> hxall(this->input_rows(), this->input_rows());

        jxall.setZero();
        jtwist.setZero();
        hxall.setZero();

        int ntrajs = xs_s.size();

        std::vector<int> idxs(ntrajs);

        std::iota(idxs.begin(), idxs.end(), 0);

        std::sort(idxs.begin(), idxs.end(),
                  [&](auto a, auto b) { return xs_s[a].size() < xs_s[b].size(); });

        int npacks = xs_s.size() / vsize;
        int nrem = xs_s.size() % vsize;
        if (nrem > 0)
            npacks++;

        std::vector<Jacobian<double>> jxs(ntrajs);
        std::vector<Hessian<double>> hxs(ntrajs);

        std::vector<std::tuple<Jacobian<double>, Hessian<double>>> hjxs(ntrajs);

        if (ntrajs < tycho::DefaultSuperScalar::SizeAtCompileTime) {
            // Pad out inputs to prevent junk data being given to stepper
            for (int v = 0; v < tycho::DefaultSuperScalar::SizeAtCompileTime; v++) {
                for (int j = 0; j < this->ode.input_rows(); j++) {
                    stepper_input_ss[j][v] = xs_s[0][0][j];
                }
            }
        }

        for (int n = 0; n < npacks; n++) {

            int vmax = std::min(vsize, ntrajs - n * vsize);
            stepper_adjvars_ss.setZero();

            for (int v = 0; v < vmax; v++) {
                int idx = idxs[n * vsize + v];
                for (int j = 0; j < this->ode.output_rows(); j++) {
                    stepper_adjvars_ss[j][v] = lf_s[idx][j];
                }
            }

            int ncalls = xs_s[idxs[n * vsize]].size() - 1;

            jxall_ss.setZero();
            jxall_ss.leftCols(this->output_rows()).setIdentity();
            hxall_ss.setZero();

            jtwist_ss.setZero();
            jtwist_ss(this->input_rows() - 1, this->input_rows() - 1) =
                tycho::DefaultSuperScalar(1.0);

            for (int i = 0; i < ncalls; i++) {

                for (int v = 0; v < vmax; v++) {
                    int idx = idxs[n * vsize + v];
                    for (int j = 0; j < this->ode.input_rows(); j++) {
                        stepper_input_ss[j][v] = (*(xs_s[idx].end() - i - 2))[j];
                    }
                    stepper_input_ss[this->ode.input_rows()][v] =
                        (*(xs_s[idx].end() - i - 1))[this->ode.TVar()];
                }

                stepper_output_ss.setZero();
                stepper_jacobian_ss.setZero();
                stepper_jacobian_ss.setZero();
                stepper_grad_ss.setZero();
                stepper_hessian_ss.setZero();

                this->stepper.compute_jacobian_adjointgradient_adjointhessian(
                    stepper_input_ss, stepper_output_ss, stepper_jacobian_ss, stepper_grad_ss,
                    stepper_hessian_ss, stepper_adjvars_ss);

                jtwist_ss.topRows(this->output_rows()) = stepper_jacobian_ss;
                jxall_ss = jxall_ss * jtwist_ss;
                if (i == 0) {
                    jxall_ss.rightCols(1) = stepper_jacobian_ss.rightCols(1);
                }

                hxall_ss = jtwist_ss.transpose() * hxall_ss * jtwist_ss;
                hxall_ss += stepper_hessian_ss;
                stepper_adjvars_ss = stepper_grad_ss.head(this->output_rows());
            }

            for (int v = 0; v < vmax; v++) {
                int idx = idxs[n * vsize + v];
                jxs[idx].resize(this->output_rows(), this->input_rows());
                hxs[idx].resize(this->input_rows(), this->input_rows());

                for (int k = 0; k < this->input_rows(); k++) {
                    for (int l = 0; l < this->output_rows(); l++) {
                        jxs[idx](l, k) = jxall_ss(l, k)[v];
                    }
                }

                for (int k = 0; k < this->input_rows(); k++) {
                    for (int l = 0; l < this->input_rows(); l++) {
                        hxs[idx](l, k) = hxall_ss(l, k)[v];
                    }
                }

                if (ncalls == xs_s[idx].size() - 1) {

                } else {
                    std::vector<ODEState<double>> xs_tmp(xs_s[idx].begin(),
                                                         xs_s[idx].end() - ncalls);

                    ODEState<double> lf(this->ode.input_rows());

                    for (int l = 0; l < this->output_rows(); l++) {
                        lf[l] = stepper_adjvars_ss[l][v];
                    }

                    auto [jtmp, htmp] = calculate_jacobian_hessian(xs_tmp, lf);

                    jtwist.topRows(this->output_rows()) = jtmp;
                    jtwist(this->input_rows() - 1, this->input_rows() - 1) = (1.0);

                    jxs[idx] = jxs[idx] * jtwist;
                    hxs[idx] = jtwist.transpose() * hxs[idx] * jtwist;
                    hxs[idx] += htmp;
                }
            }
        }

        return std::tuple{jxs, hxs};
    }

  public:
    ////////////////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////////////
    IntegRet integrate(const ODEState<double> &x0, double tf) const {

        ODEState<double> xf;
        std::vector<EventPack> events;
        std::vector<std::vector<Eigen::Vector2d>> eventtimes;

        bool storestates = false;
        bool storederivs = false;
        bool storemidpoints = false;
        std::vector<ODEState<double>> xs;
        std::vector<ODEDeriv<double>> d_xs;

        xf = this->integrate_impl(x0, tf, events, eventtimes, storestates, storederivs,
                                  storemidpoints, xs, d_xs);
        return xf;
    }

    std::vector<ODEState<double>> integrate(const std::vector<ODEState<double>> &x0s,
                                            const Eigen::VectorXd &tfs) const {

        if (!vectorize_batch_calls_) {

            std::vector<ODEState<double>> xfs(x0s.size());
            std::vector<EventPack> events;
            std::vector<std::vector<Eigen::Vector2d>> eventtimes;

            bool storestates = false;
            bool storederivs = false;
            bool storemidpoints = false;
            std::vector<ODEState<double>> xs;
            std::vector<ODEDeriv<double>> d_xs;

            for (int i = 0; i < x0s.size(); i++) {

                xfs[i] = this->integrate_impl(x0s[i], tfs[i], events, eventtimes, storestates,
                                              storederivs, storemidpoints, xs, d_xs);
            }

            return xfs;

        } else {

            std::vector<EventPack> events;
            std::vector<std::vector<std::vector<Eigen::Vector2d>>> eventtimes;

            bool storestates = false;
            bool storederivs = false;
            bool storemidpoints = false;
            std::vector<std::vector<ODEState<double>>> xs;
            std::vector<std::vector<ODEDeriv<double>>> d_xs;

            return integrate_impl_vectorized(x0s, tfs, events, eventtimes, storestates, storederivs,
                                             storemidpoints, xs, d_xs);
        }
    }

    std::vector<STMRet> integrate_stm(const std::vector<ODEState<double>> &x0s,
                                      const Eigen::VectorXd &tfs) const {

        if (!vectorize_batch_calls_) {
            std::vector<STMRet> rets(x0s.size());
            for (int i = 0; i < x0s.size(); i++) {
                rets[i] = this->integrate_stm(x0s[i], tfs[i]);
            }
            return rets;
        } else {

            std::vector<EventPack> events;
            std::vector<std::vector<std::vector<Eigen::Vector2d>>> eventtimes;

            bool storestates = true;
            bool storederivs = false;
            bool storemidpoints = false;
            std::vector<std::vector<ODEState<double>>> xs(x0s.size());
            std::vector<std::vector<ODEDeriv<double>>> d_xs(x0s.size());

            auto x_finals = integrate_impl_vectorized(x0s, tfs, events, eventtimes, storestates,
                                                      storederivs, storemidpoints, xs, d_xs);

            auto jacs = this->calculate_jacobians(xs);
            std::vector<STMRet> rets(x0s.size());

            for (int i = 0; i < x0s.size(); i++) {

                rets[i] = std::tuple{x_finals[i], jacs[i]};
            }
            return rets;
        }
    }

    std::vector<std::tuple<ODEState<double>, Jacobian<double>, Hessian<double>>>
    integrate_stm2(const std::vector<ODEState<double>> &x0s, const Eigen::VectorXd &tfs,
                   const std::vector<ODEState<double>> &lfs) const {

        if (!vectorize_batch_calls_) {
            std::vector<std::tuple<ODEState<double>, Jacobian<double>, Hessian<double>>> rets(
                x0s.size());
            for (int i = 0; i < x0s.size(); i++) {
                auto xs = this->integrate_dense(x0s[i], tfs[i]);
                auto [J, H] = this->calculate_jacobian_hessian(xs, lfs[i]);
                rets[i] = std::tuple{xs.back(), J, H};
            }
            return rets;
        } else {

            std::vector<EventPack> events;
            std::vector<std::vector<std::vector<Eigen::Vector2d>>> eventtimes;

            bool storestates = true;
            bool storederivs = false;
            bool storemidpoints = false;
            std::vector<std::vector<ODEState<double>>> xs(x0s.size());
            std::vector<std::vector<ODEDeriv<double>>> d_xs(x0s.size());

            auto x_finals = integrate_impl_vectorized(x0s, tfs, events, eventtimes, storestates,
                                                      storederivs, storemidpoints, xs, d_xs);

            auto [js, hs_out] = this->calculate_jacobians_hessians(xs, lfs);
            std::vector<std::tuple<ODEState<double>, Jacobian<double>, Hessian<double>>> rets(
                x0s.size());

            for (int i = 0; i < x0s.size(); i++) {
                rets[i] = std::tuple{x_finals[i], js[i], hs_out[i]};
            }
            return rets;
        }
    }

    IntegEventRet integrate(const ODEState<double> &x0, double tf,
                            const std::vector<EventPack> &events) const {

        ODEState<double> xf;
        std::vector<std::vector<Eigen::Vector2d>> eventtimes;

        bool storestates = true;
        bool storederivs = true;
        bool storemidpoints = true;
        std::vector<ODEState<double>> xs;
        std::vector<ODEDeriv<double>> d_xs;

        xf = this->integrate_impl(x0, tf, events, eventtimes, storestates, storederivs,
                                  storemidpoints, xs, d_xs);

        std::vector<std::vector<ODEState<double>>> eventlocs(events.size());
        for (auto etimes : eventtimes) {
            if (etimes.size() > 0) {
                auto tab = this->make_table(xs, d_xs, false);
                eventlocs = this->find_events(tab, events, eventtimes);
                break;
            }
        }

        return std::tuple{xf, eventlocs};
    }

    template <class... Args>
    auto integrate_parallel_impl(const std::vector<ODEState<double>> &x0s,
                                 const Eigen::VectorXd &tfs, int n_parts, Args &&...args)
        -> std::vector<decltype(Integrator::integrate(x0s[0], tfs[0], args...))> {

        if (x0s.size() != tfs.size()) {
            throw std::invalid_argument(
                "List of initial states and final times must be the same size");
        }

        using SingleRetType = decltype(Integrator::integrate(x0s[0], tfs[0], args...));
        using RetType = std::vector<SingleRetType>;

        int n = x0s.size();
        RetType results(n);

        auto job = [&](int start, int stop) {
            for (int i = start; i < stop; i++) {
                results[i] = this->integrate(x0s[i], tfs[i], args...);
            }
        };

        tycho::utils::parallel_blocks(n, job, n_parts);
        return results;
    }

    std::vector<IntegRet> integrate_parallel(const std::vector<ODEState<double>> &x0s,
                                             const Eigen::VectorXd &tfs, int n_parts) {
        return this->integrate_parallel_impl(x0s, tfs, n_parts);
    }
    std::vector<IntegEventRet> integrate_parallel(const std::vector<ODEState<double>> &x0s,
                                                  const Eigen::VectorXd &tfs,
                                                  const std::vector<EventPack> &events,
                                                  int n_parts) {
        return this->integrate_parallel_impl(x0s, tfs, n_parts, events);
    }
    /////////////////////////////////////////////////////////////////////////////////////

    DenseRet integrate_dense(const ODEState<double> &x0, double tf) const {

        ODEState<double> xf;
        std::vector<EventPack> events;
        std::vector<std::vector<Eigen::Vector2d>> eventtimes;

        bool storestates = true;
        bool storederivs = false;
        bool storemidpoints = false;
        std::vector<ODEState<double>> xs;
        std::vector<ODEDeriv<double>> d_xs;

        xf = this->integrate_impl(x0, tf, events, eventtimes, storestates, storederivs,
                                  storemidpoints, xs, d_xs);
        return xs;
    }

    DenseEventRet integrate_dense(const ODEState<double> &x0, double tf,
                                  const std::vector<EventPack> &events, bool alloutput) const {

        ODEState<double> xf;
        std::vector<std::vector<Eigen::Vector2d>> eventtimes;

        bool storestates = true;
        bool storederivs = true;
        bool storemidpoints = true;
        std::vector<ODEState<double>> xs;
        std::vector<ODEDeriv<double>> d_xs;

        xf = this->integrate_impl(x0, tf, events, eventtimes, storestates, storederivs,
                                  storemidpoints, xs, d_xs);

        std::vector<std::vector<ODEState<double>>> eventlocs(events.size());
        for (auto etimes : eventtimes) {
            if (etimes.size() > 0) {
                auto tab = this->make_table(xs, d_xs, false);
                eventlocs = this->find_events(tab, events, eventtimes);
                break;
            }
        }
        if (alloutput)
            return std::tuple{xs, eventlocs};
        else
            return std::tuple{midpoints_removed(xs), eventlocs};
    }

    DenseEventRet integrate_dense(const ODEState<double> &x0, double tf, int n,
                                  const std::vector<EventPack> &events) const {

        ODEState<double> xf;
        std::vector<std::vector<Eigen::Vector2d>> eventtimes;

        bool storestates = true;
        bool storederivs = true;
        bool storemidpoints = true;
        std::vector<ODEState<double>> xs;
        std::vector<ODEDeriv<double>> d_xs;

        xf = this->integrate_impl(x0, tf, events, eventtimes, storestates, storederivs,
                                  storemidpoints, xs, d_xs);

        auto tab = this->make_table(xs, d_xs, true);
        std::vector<std::vector<ODEState<double>>> eventlocs =
            this->find_events(tab, events, eventtimes);

        Eigen::VectorXd ts;
        ts.setLinSpaced(n, xs[0][this->ode.TVar()], xs.back()[this->ode.TVar()]);

        std::vector<ODEState<double>> x_interp(n);

        for (int i = 0; i < n; i++) {
            x_interp[i].resize(this->ode.input_rows());
            tab->InterpolateRef(ts[i], x_interp[i]);
        }

        return std::tuple{x_interp, eventlocs};
    }

    DenseRet integrate_dense(const ODEState<double> &x0, double tf, int n) const {

        ODEState<double> xf;
        std::vector<std::vector<Eigen::Vector2d>> eventtimes;

        bool storestates = true;
        bool storederivs = true;
        bool storemidpoints = true;
        std::vector<ODEState<double>> xs;
        std::vector<ODEDeriv<double>> d_xs;
        std::vector<EventPack> events;

        xf = this->integrate_impl(x0, tf, events, eventtimes, storestates, storederivs,
                                  storemidpoints, xs, d_xs);

        auto tab = this->make_table(xs, d_xs, true);

        Eigen::VectorXd ts;
        ts.setLinSpaced(n, xs[0][this->ode.TVar()], xs.back()[this->ode.TVar()]);

        std::vector<ODEState<double>> x_interp(n);

        for (int i = 0; i < n; i++) {
            x_interp[i].resize(this->ode.input_rows());
            tab->InterpolateRef(ts[i], x_interp[i]);
        }

        return x_interp;
    }

    DenseRet integrate_dense(const ODEState<double> &x0, double tf, int num_states,
                             std::function<bool(ConstEigenRef<Eigen::VectorXd>)> exitfun) const {
        VectorX<double> ts = VectorX<double>::LinSpaced(num_states, x0[this->ode.TVar()], tf);

        std::vector<ODEState<double>> xout;
        xout.reserve(num_states);
        xout.push_back(x0);
        for (int i = 1; i < num_states; i++) {
            xout.push_back(this->integrate(xout[i - 1], ts[i]));
            if (exitfun(xout.back()))
                break;
        }
        return xout;
    }

    template <class... Args>
    auto integrate_dense_parallel_impl(const std::vector<ODEState<double>> &x0s,
                                       const Eigen::VectorXd &tfs, int n_parts, Args &&...args)
        -> std::vector<decltype(Integrator::integrate_dense(x0s[0], tfs[0], args...))> {

        if (x0s.size() != tfs.size()) {
            throw std::invalid_argument(
                "List of initial states and final times must be the same size");
        }

        using SingleRetType = decltype(Integrator::integrate_dense(x0s[0], tfs[0], args...));
        using RetType = std::vector<SingleRetType>;

        int n = x0s.size();
        RetType results(n);

        auto job = [&](int start, int stop) {
            for (int i = start; i < stop; i++) {
                results[i] = this->integrate_dense(x0s[i], tfs[i], args...);
            }
        };

        tycho::utils::parallel_blocks(n, job, n_parts);
        return results;
    }

    std::vector<DenseRet> integrate_dense_parallel(const std::vector<ODEState<double>> &x0s,
                                                   const Eigen::VectorXd &tfs, int n_parts) {
        return this->integrate_dense_parallel_impl(x0s, tfs, n_parts);
    }
    std::vector<DenseEventRet> integrate_dense_parallel(const std::vector<ODEState<double>> &x0s,
                                                        const Eigen::VectorXd &tfs,
                                                        const std::vector<EventPack> &events,
                                                        int n_parts) {
        return this->integrate_dense_parallel_impl(x0s, tfs, n_parts, events, false);
    }
    /////////////////////////////////////////////////////////////////////////////////////

    template <class... Args>
    auto integrate_dense_parallel_impl_n(const std::vector<ODEState<double>> &x0s,
                                         const Eigen::VectorXd &tfs, const std::vector<int> &ns,
                                         int n_parts, Args &&...args)
        -> std::vector<decltype(Integrator::integrate_dense(x0s[0], tfs[0], ns[0], args...))> {

        if (x0s.size() != tfs.size()) {
            throw std::invalid_argument(
                "List of initial states and final times must be the same size");
        }
        if (x0s.size() != ns.size()) {
            throw std::invalid_argument(
                "List of initial states and state numbers must be the same size");
        }

        using SingleRetType = decltype(Integrator::integrate_dense(x0s[0], tfs[0], ns[0], args...));
        using RetType = std::vector<SingleRetType>;

        int n = x0s.size();
        RetType results(n);

        auto job = [&](int start, int stop) {
            for (int i = start; i < stop; i++) {
                results[i] = this->integrate_dense(x0s[i], tfs[i], ns[i], args...);
            }
        };

        tycho::utils::parallel_blocks(n, job, n_parts);
        return results;
    }

    std::vector<DenseEventRet> integrate_dense_parallel(const std::vector<ODEState<double>> &x0s,
                                                        const Eigen::VectorXd &tfs,
                                                        const std::vector<int> &ns,
                                                        const std::vector<EventPack> &events,
                                                        int n_parts) {
        return this->integrate_dense_parallel_impl_n(x0s, tfs, ns, n_parts, events);
    }
    std::vector<DenseRet> integrate_dense_parallel(const std::vector<ODEState<double>> &x0s,
                                                   const Eigen::VectorXd &tfs,
                                                   const std::vector<int> &ns, int n_parts) {
        return this->integrate_dense_parallel_impl_n(x0s, tfs, ns, n_parts);
    }
    /////////////////////////////////////////////////////////////////////////////////////

    STMRet integrate_stm(const ODEState<double> &x0, double tf) const {
        auto xs = this->integrate_dense(x0, tf);
        Jacobian<double> jx = this->calculate_jacobian(xs);
        return std::tuple{xs.back(), jx};
    }
    STMEventRet integrate_stm(const ODEState<double> &x0, double tf,
                              const std::vector<EventPack> &events) const {
        auto [xs, eventlocs] = this->integrate_dense(x0, tf, events, false);
        Jacobian<double> jx = this->calculate_jacobian(xs);
        return std::tuple{xs.back(), jx, eventlocs};
    }

    template <class... Args>
    auto integrate_stm_parallel_impl(const std::vector<ODEState<double>> &x0s,
                                     const Eigen::VectorXd &tfs, int n_parts, Args &&...args)
        -> std::vector<decltype(Integrator::integrate_stm(x0s[0], tfs[0], args...))> {

        if (x0s.size() != tfs.size()) {
            throw std::invalid_argument(
                "List of initial states and final times must be the same size");
        }

        using SingleRetType = decltype(Integrator::integrate_stm(x0s[0], tfs[0], args...));
        using RetType = std::vector<SingleRetType>;

        int n = x0s.size();
        RetType results(n);

        auto job = [&](int start, int stop) {
            for (int i = start; i < stop; i++) {
                results[i] = this->integrate_stm(x0s[i], tfs[i], args...);
            }
        };

        tycho::utils::parallel_blocks(n, job, n_parts);
        return results;
    }

    std::vector<STMRet> integrate_stm_parallel(const std::vector<ODEState<double>> &x0s,
                                               const Eigen::VectorXd &tfs, int n_parts) {
        return this->integrate_stm_parallel_impl(x0s, tfs, n_parts);
    }
    std::vector<STMEventRet> integrate_stm_parallel(const std::vector<ODEState<double>> &x0s,
                                                    const Eigen::VectorXd &tfs,
                                                    const std::vector<EventPack> &events,
                                                    int n_parts) {
        return this->integrate_stm_parallel_impl(x0s, tfs, n_parts, events);
    }

    STMRet integrate_stm_parallel(const ODEState<double> &x0, double tf, int n_parts) {
        VectorX<double> ts = VectorX<double>::LinSpaced(n_parts + 1, x0[this->ode.TVar()], tf);
        std::vector<ODEState<double>> xs(n_parts + 1);
        xs[0] = x0;

        std::vector<std::future<STMRet>> results(n_parts);

        Eigen::MatrixXd jxall(this->input_rows(), this->input_rows());
        jxall.setIdentity();

        auto stm_op = [&](int i) {
            auto xi = xs[i];
            auto tf1 = ts[i + 1];
            return this->integrate_stm(xi, tf1);
        };

        if (n_parts > 1 && tycho::utils::use_thread_pool()) {
            int submitted = 0;
            try {
                for (int i = 0; i < n_parts; i++) {
                    results[i] =
                        tycho::utils::thread_pool().submit_task([&stm_op, i] { return stm_op(i); });
                    submitted = i + 1;
                    if (i < (n_parts - 1))
                        xs[i + 1] = this->integrate(xs[i], ts[i + 1]);
                }
            } catch (...) {
                // If integrate() throws mid-loop, drain submitted futures to
                // prevent use-after-free of stack-captured references (&stm_op).
                auto ex = std::current_exception();
                for (int j = 0; j < submitted; j++) {
                    try {
                        results[j].get();
                    } catch (...) {
                    }
                }
                std::rethrow_exception(ex);
            }
            // If .get() throws, drain remaining futures before rethrowing to
            // prevent use-after-free of stack-captured references (&stm_op, etc.).
            std::exception_ptr ex;
            for (int i = 0; i < n_parts; i++) {
                try {
                    auto [xf, jx] = results[i].get();
                    jxall.topRows(this->output_rows()) = (jx * jxall).eval();
                    if (i == (n_parts - 1))
                        xs[i + 1] = xf;
                } catch (...) {
                    if (!ex)
                        ex = std::current_exception();
                    int suppressed = 0;
                    for (int j = i + 1; j < n_parts; j++) {
                        try {
                            results[j].get();
                        } catch (const std::exception &e) {
                            if (suppressed == 0)
                                std::fprintf(stderr,
                                             "[Tycho] integrate_stm_parallel: additional segment "
                                             "also failed: %s\n",
                                             e.what());
                            ++suppressed;
                        } catch (...) {
                            ++suppressed;
                        }
                    }
                    if (suppressed > 1)
                        std::fprintf(stderr,
                                     "[Tycho] integrate_stm_parallel: %d additional exceptions "
                                     "suppressed\n",
                                     suppressed - 1);
                    break;
                }
            }
            if (ex)
                std::rethrow_exception(ex);
        } else {
            for (int i = 0; i < n_parts; i++) {
                auto [xf, jx] = stm_op(i);
                jxall.topRows(this->output_rows()) = (jx * jxall).eval();
                xs[i + 1] = (i < n_parts - 1) ? this->integrate(xs[i], ts[i + 1]) : xf;
            }
        }

        STMRet tup_final;
        std::get<0>(tup_final) = xs.back();
        std::get<1>(tup_final) = jxall.topRows(this->output_rows());
        return tup_final;
    }

    /////////////////////////////////////////////////////////////////////////////////////

    template <class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();

        ODEState<Scalar> x0 = x.head(this->ode.input_rows());
        Scalar tf = x[this->ode.input_rows()];
        fx = this->integrate(x0, tf);
    }
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

        ODEState<Scalar> x0 = x.head(this->ode.input_rows());
        Scalar tf = x[this->ode.input_rows()];
        auto xs = this->integrate_dense(x0, tf);
        fx = xs.back();
        jx = this->calculate_jacobian(xs);
    }
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
        ConstMatrixBaseRef<JacType> jx_, ConstVectorBaseRef<AdjGradType> adjgrad_,
        ConstMatrixBaseRef<AdjHessType> adjhess_, ConstVectorBaseRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
        VectorBaseRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
        MatrixBaseRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

        ODEState<Scalar> x0 = x.head(this->ode.input_rows());
        ODEState<Scalar> lf = adjvars;
        Scalar tf = x[this->ode.input_rows()];

        auto xs = this->integrate_dense(x0, tf);
        fx = xs.back();

        std::tuple<Jacobian<double>, Hessian<double>> res =
            this->calculate_jacobian_hessian(xs, lf);

        jx = std::get<0>(res);
        adjhess = std::get<1>(res);
        adjgrad = jx.transpose() * adjvars;
    }

    /////////////////////////////////////////////////////////////////////////////////////
};

} // namespace tycho::integrators
