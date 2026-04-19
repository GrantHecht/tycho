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

#include "tycho/detail/integrators/adaptive_driver.h"
#include "tycho/detail/integrators/error_norm.h"
#include "tycho/detail/integrators/event_handler.h"
#include "tycho/detail/integrators/initial_dt.h"
#include "tycho/detail/integrators/parallel_driver.h"
#include "tycho/detail/integrators/rk_steppers.h"
#include "tycho/detail/integrators/step_controller.h"
#include "tycho/detail/integrators/stepper.h"
#include "tycho/detail/integrators/stm_driver.h"
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
// the integrators namespace can reference it before ode_.h is included.
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
using vf::CDiagRef;
using vf::CEigRef;
using vf::CMatRef;
using vf::Constant;
using vf::CVecRef;
using vf::DiagRef;
using vf::EigRef;
using vf::GenericConditional;
using vf::GenericFunction;
using vf::MatRef;
using vf::NestedFunction;
using vf::ParsedInput;
using vf::StackedOutputs;
using vf::VecRef;
using vf::VectorFunction;

// CentralShootingDefect lives in tycho::oc; use the qualified friend below.

/// Dispatch `fn` with a compile-time IVPAlg tag matching the runtime `alg`.
/// Used by Integrator to turn its runtime rk_method_ selector into a
/// compile-time Alg parameter for AdaptiveDriver<Alg,...> / ParallelDriver<Alg,...>.
/// Throws std::logic_error for internal tags that should never reach this
/// dispatch (RK4Classic, DOPRI5, *Trans). Public-selectable methods map
/// to their own std::integral_constant<IVPAlg, ...> tag.
template <class Fn> inline auto with_public_ivp_alg(IVPAlg alg, Fn &&fn) {
    switch (alg) {
    case IVPAlg::DOPRI54:
        return fn(std::integral_constant<IVPAlg, IVPAlg::DOPRI54>{});
    case IVPAlg::DOPRI87:
        return fn(std::integral_constant<IVPAlg, IVPAlg::DOPRI87>{});
    case IVPAlg::Tsit5:
        return fn(std::integral_constant<IVPAlg, IVPAlg::Tsit5>{});
    case IVPAlg::BS3:
        return fn(std::integral_constant<IVPAlg, IVPAlg::BS3>{});
    case IVPAlg::BS5:
        return fn(std::integral_constant<IVPAlg, IVPAlg::BS5>{});
    case IVPAlg::Vern7:
        return fn(std::integral_constant<IVPAlg, IVPAlg::Vern7>{});
    case IVPAlg::Vern8:
        return fn(std::integral_constant<IVPAlg, IVPAlg::Vern8>{});
    case IVPAlg::Vern9:
        return fn(std::integral_constant<IVPAlg, IVPAlg::Vern9>{});
    default:
        throw std::logic_error(
            "with_public_ivp_alg: unsupported IVPAlg for adaptive stepping (internal "
            "transcription tag passed where a public method was expected).");
    }
}

template <class DODE>
struct Integrator : VectorFunction<Integrator<DODE>, SZ_SUM<DODE::IRC, 1>::value, DODE::IRC> {

    using Base = VectorFunction<Integrator<DODE>, SZ_SUM<DODE::IRC, 1>::value, DODE::IRC>;
    VF_TYPE_ALIASES(Base);

    template <class Scalar> using ODEState = typename DODE::template Input<Scalar>;
    template <class Scalar> using ODEDeriv = typename DODE::template Output<Scalar>;

    using EventPack = std::tuple<GenericFunction<-1, 1>, int, int>;
    using EventLocsType = std::vector<std::vector<ODEState<double>>>;

    /// Differentiable stepper type: RK stepper over (ODE ∘ control) composed function.
    template <class PseudoODE, IVPAlg RKOp> using StepperType = RKStepper<PseudoODE, RKOp>;

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
    DODE ode_;
    bool use_controller_ = false;
    ControllerType controller_;
    StepperWrapperType stepper_;
    IVPAlg rk_method_ = IVPAlg::DOPRI87;

  public:
    Integrator() { this->enable_vectorization_ = true; }

    Integrator(const DODE &dode, IVPAlg meth, double defstep) : Integrator() {
        // Use in_place_type to sidestep MSVC variant overload-resolution
        // ambiguity between the int and Eigen::VectorXi alternatives.
        ControlIndexType empty_ci{std::in_place_type<Eigen::VectorXi>};
        this->set_method(meth, dode, defstep, false, ControllerType{}, empty_ci);
        this->set_abs_tol(1.0e-12); // Must Be called after set_method!!!
        this->set_rel_tol(0);       // Must Be called after set_method!!!
    }
    Integrator(const DODE &dode, double defstep) : Integrator(dode, IVPAlg::DOPRI87, defstep) {}
    Integrator(const DODE &dode, IVPAlg meth, double defstep, const ControllerType &ucon,
               const ControlIndexType &varlocs_t)
        : Integrator() {

        this->set_method(meth, dode, defstep, true, ucon, varlocs_t);
        this->set_abs_tol(1.0e-12); // Must Be called after set_method!!!
        this->set_rel_tol(0);       // Must Be called after set_method!!!
    }
    // VectorXi overloads: explicitly wrap into ControlIndexType to avoid MSVC
    // variant implicit-conversion ambiguity (int vs VectorXi alternatives).
    Integrator(const DODE &dode, IVPAlg meth, double defstep, const ControllerType &ucon,
               const Eigen::VectorXi &varlocs)
        : Integrator(dode, meth, defstep, ucon,
                     ControlIndexType{std::in_place_type<Eigen::VectorXi>, varlocs}) {}
    Integrator(const DODE &dode, double defstep, const ControllerType &ucon,
               const ControlIndexType &varlocs_t)
        : Integrator(dode, IVPAlg::DOPRI87, defstep, ucon, varlocs_t) {}
    Integrator(const DODE &dode, double defstep, const ControllerType &ucon,
               const Eigen::VectorXi &varlocs)
        : Integrator(dode, IVPAlg::DOPRI87, defstep, ucon,
                     ControlIndexType{std::in_place_type<Eigen::VectorXi>, varlocs}) {}
    Integrator(const DODE &dode, IVPAlg meth, double defstep, const Eigen::VectorXd &v)
        : Integrator() {

        Eigen::VectorXi tloc(1);
        tloc[0] = dode.t_var();
        GenericFunction<-1, -1> ucon = Constant<-1, -1>(1, v);
        this->set_method(meth, dode, defstep, true, ucon, tloc);
        this->set_abs_tol(1.0e-12); // Must Be called after set_method!!!
        this->set_rel_tol(0);       // Must Be called after set_method!!!
    }
    Integrator(const DODE &dode, double defstep, const Eigen::VectorXd &v)
        : Integrator(dode, IVPAlg::DOPRI87, defstep, v) {}
    Integrator(const DODE &dode, IVPAlg meth, double defstep, std::shared_ptr<LGLInterpTable> tab,
               const Eigen::VectorXi &ulocs)
        : Integrator() {

        Eigen::VectorXi varlocs(1);
        varlocs[0] = dode.t_var();
        ControllerType ucon = InterpFunction<-1>(tab, ulocs);
        this->set_method(meth, dode, defstep, true, ucon, varlocs);
        this->set_abs_tol(1.0e-12); // Must Be called after set_method!!!
        this->set_rel_tol(0);       // Must Be called after set_method!!!
    }
    Integrator(const DODE &dode, double defstep, std::shared_ptr<LGLInterpTable> tab,
               const Eigen::VectorXi &ulocs)
        : Integrator(dode, IVPAlg::DOPRI87, defstep, tab, ulocs) {}

    Integrator(const DODE &dode, IVPAlg meth, double defstep, std::shared_ptr<LGLInterpTable> tab)
        : Integrator() {

        // Bug waiting to happen when LGL interp table is re-factored
        if (dode.input_rows() != tab->xtu_vars_ || dode.x_vars() != tab->x_vars_) {
            throw std::invalid_argument("Table data does not match expected dimension of ODE."
                                        " Please provide the indices variables in the table you "
                                        "want to interpret as controls.\n");
        }
        Eigen::VectorXi ulocs;
        ulocs.setLinSpaced(dode.u_vars(), dode.t_var() + 1, dode.t_var() + dode.u_vars());

        Eigen::VectorXi varlocs(1);
        varlocs[0] = dode.t_var();
        ControllerType ucon = InterpFunction<-1>(tab, ulocs);
        this->set_method(meth, dode, defstep, true, ucon, varlocs);
        this->set_abs_tol(1.0e-12); // Must Be called after set_method!!!
        this->set_rel_tol(0);       // Must Be called after set_method!!!
    }
    Integrator(const DODE &dode, double defstep, std::shared_ptr<LGLInterpTable> tab)
        : Integrator(dode, IVPAlg::DOPRI87, defstep, tab) {}

    void set_method(IVPAlg alg, const DODE &dode, double defstep, bool usecontrol,
                    const GenericFunction<-1, -1> &ucon, const ControlIndexType &varlocs_t) {

        if (defstep <= 0.0) {
            throw ::std::invalid_argument(
                "Initial step size must be strictly positive (backward integration is driven "
                "by tf < t0, not by sign of h; zero would divide-by-zero in the fixed-step "
                "path).");
        }
        this->def_step_size_ = defstep;
        // HW auto-initdt stays enabled — user can opt out via
        // set_initial_step_size() or set_auto_initial_dt(false).
        this->use_hairer_wanner_initdt_ = true;

        switch (alg) {
        case IVPAlg::DOPRI54:
            this->rk_method_ = IVPAlg::DOPRI54;
            this->error_order_ = 4;
            // Using DOPRI5 rather than DOPRI54 here is not a mistake
            this->init_stepper_and_controller<IVPAlg::DOPRI5>(dode, usecontrol, ucon, varlocs_t);
            break;
        case IVPAlg::DOPRI87:
            this->rk_method_ = IVPAlg::DOPRI87;
            this->error_order_ = 7;
            this->init_stepper_and_controller<IVPAlg::DOPRI87>(dode, usecontrol, ucon, varlocs_t);
            break;
        case IVPAlg::Tsit5:
            this->rk_method_ = IVPAlg::Tsit5;
            this->error_order_ = 4;
            this->init_stepper_and_controller<IVPAlg::Tsit5Trans>(dode, usecontrol, ucon,
                                                                  varlocs_t);
            break;
        case IVPAlg::BS3:
            this->rk_method_ = IVPAlg::BS3;
            this->error_order_ = 2;
            this->init_stepper_and_controller<IVPAlg::BS3Trans>(dode, usecontrol, ucon, varlocs_t);
            break;
        case IVPAlg::BS5:
            this->rk_method_ = IVPAlg::BS5;
            this->error_order_ = 4;
            this->init_stepper_and_controller<IVPAlg::BS5Trans>(dode, usecontrol, ucon, varlocs_t);
            break;
        case IVPAlg::Vern7:
            this->rk_method_ = IVPAlg::Vern7;
            this->error_order_ = 6;
            this->init_stepper_and_controller<IVPAlg::Vern7Trans>(dode, usecontrol, ucon,
                                                                  varlocs_t);
            break;
        case IVPAlg::Vern8:
            this->rk_method_ = IVPAlg::Vern8;
            this->error_order_ = 7;
            this->init_stepper_and_controller<IVPAlg::Vern8Trans>(dode, usecontrol, ucon,
                                                                  varlocs_t);
            break;
        case IVPAlg::Vern9:
            this->rk_method_ = IVPAlg::Vern9;
            this->error_order_ = 8;
            this->init_stepper_and_controller<IVPAlg::Vern9Trans>(dode, usecontrol, ucon,
                                                                  varlocs_t);
            break;
        case IVPAlg::RK4Classic:
        case IVPAlg::DOPRI5:
        case IVPAlg::Tsit5Trans:
        case IVPAlg::BS3Trans:
        case IVPAlg::BS5Trans:
        case IVPAlg::Vern7Trans:
        case IVPAlg::Vern8Trans:
        case IVPAlg::Vern9Trans:
            throw std::invalid_argument(
                "Internal template-dispatch tags (RK4Classic, DOPRI5, Tsit5Trans, BS3Trans, "
                "BS5Trans, Vern7Trans, Vern8Trans, Vern9Trans) are not runtime-selectable. Use "
                "IVPAlg::DOPRI54, IVPAlg::DOPRI87, IVPAlg::Tsit5, IVPAlg::BS3, IVPAlg::BS5, "
                "IVPAlg::Vern7, IVPAlg::Vern8, or IVPAlg::Vern9.");
        default:
            throw std::logic_error("Integrator::set_method: unhandled IVPAlg enum value");
        }

        this->controller_variant_ = default_controller_for(alg);
        std::visit(
            [](auto &c) {
                c.validate();
                c.reset();
            },
            this->controller_variant_);
    }

    template <IVPAlg RKOp>
    void init_stepper_and_controller(const DODE &odet, bool usecontrol,
                                     const GenericFunction<-1, -1> &ucon,
                                     const ControlIndexType &varlocs_t) {

        this->ode_ = odet;
        Eigen::VectorXi varlocs = this->get_var_locs(varlocs_t);
        this->set_io_rows(this->ode_.input_rows() + 1, this->ode_.input_rows());

        this->use_controller_ = usecontrol;

        auto Stepper = StepperType<DODE, RKOp>(ode_);
        constexpr int IRC = decltype(Stepper)::IRC;
        constexpr int DUV = (DODE::UV == 1) ? -1 : DODE::UV;
        if constexpr (DODE::UV == 0) {
            this->stepper_ = StepperWrapperType(Stepper);
            this->controller_ = Arguments<-1>(ode_.u_vars());
        } else {
            if (!this->use_controller_) {
                this->stepper_ = StepperWrapperType(Stepper);
                this->controller_ = Arguments<-1>(ode_.u_vars());

            } else {

                if (ucon.output_rows() != ode_.u_vars()) {
                    throw std::invalid_argument(
                        "Controller output size does not match number of ode control variables");
                }

                if (ucon.input_rows() != varlocs.size()) {
                    throw std::invalid_argument("Controller input size is inconsistent with "
                                                "specified number of input state variables");
                }

                Arguments<DODE::IRC> odeargs(ode_.input_rows());
                ParsedInput<GenericFunction<-1, -1>, DODE::IRC, DUV> controllerfunc(
                    ucon, varlocs, ode_.input_rows());
                this->controller_ = controllerfunc;

                if constexpr (DODE::PV == 0) {

                    auto ode_args = StackedOutputs{odeargs.template head<DODE::XtV>(ode_.xt_vars()),
                                                   controllerfunc};
                    auto ode_expr = NestedFunction<DODE, decltype(ode_args)>(ode_, ode_args);
                    auto gen_ode =
                        GenericODE<GenericFunction<-1, -1>, DODE::XV, DODE::UV, DODE::PV>(
                            ode_expr, ode_.x_vars(), ode_.u_vars(), ode_.p_vars());
                    auto stepper_u = ode_args.eval(StepperType<decltype(gen_ode), RKOp>(gen_ode));
                    this->stepper_ = StepperWrapperType(stepper_u);
                } else {

                    auto ode_args =
                        StackedOutputs{odeargs.template head<DODE::XtV>(ode_.xt_vars()),
                                       controllerfunc, odeargs.template tail<-1>(ode_.p_vars())};
                    auto ode_expr = NestedFunction<DODE, decltype(ode_args)>(ode_, ode_args);
                    auto gen_ode =
                        GenericODE<GenericFunction<-1, -1>, DODE::XV, DODE::UV, DODE::PV>(
                            ode_expr, ode_.x_vars(), ode_.u_vars(), ode_.p_vars());
                    auto stepper_up = ode_args.eval(StepperType<decltype(gen_ode), RKOp>(gen_ode));
                    this->stepper_ = StepperWrapperType(stepper_up);
                }
            }
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////

    GenericFunction<-1, -1> get_stepper() { return this->stepper_; }

    Eigen::VectorXi get_var_locs(const ControlIndexType &varlocs_t) {

        Eigen::VectorXi varlocs;

        /////////////////////////////////////////////////
        if (std::holds_alternative<int>(varlocs_t)) {
            varlocs.resize(1);
            varlocs[0] = std::get<int>(varlocs_t);
        } else if (std::holds_alternative<Eigen::VectorXi>(varlocs_t)) {
            varlocs = std::get<Eigen::VectorXi>(varlocs_t);
        } else if (std::holds_alternative<std::string>(varlocs_t)) {
            varlocs = this->ode_.idx(std::get<std::string>(varlocs_t));
        } else if (std::holds_alternative<std::vector<std::string>>(varlocs_t)) {

            std::vector<Eigen::VectorXi> varvec;
            int size = 0;
            auto tmpvars = std::get<std::vector<std::string>>(varlocs_t);

            for (auto tmpv : tmpvars) {
                varvec.push_back(this->ode_.idx(tmpv));
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
    double def_step_size_ = 0.1;
    double max_step_change_ = 10.0;
    // Julia-style maxiters cap — safety net against runaway rejection loops or
    // pathological dynamics. The adaptive controller handles step-size control;
    // there is no hard min/max on dt, only a hard cap on iterations.
    int max_steps_ = 1'000'000;
    bool adaptive_ = true;
    bool fast_adaptive_stm_ = true;
    double event_tol_ = 1.0e-6;
    int max_event_iters_ = 10;
    bool vectorize_batch_calls_ = true;

    double step_frac_ = .9;
    double err_pow_fac_ = 1;

    // SP3 controller variant — selected per algorithm in set_method, overrideable
    // via set_controller(). Read-only during integrate calls: public wrappers
    // and parallel workers clone from this prototype into a local
    // `ControllerVariant` per call and operate on the local. Only
    // `set_method` / `set_controller` mutate this member, and neither is
    // concurrent-safe with an in-flight integrate.
    // Use the free `tycho::integrators::ControllerVariant` from step_controller.h
    // directly so extracted drivers (AdaptiveDriver, ParallelDriver) can share
    // the same variant type without going through the Integrator scope.
    ControllerVariant controller_variant_;
    ErrorNormType error_norm_type_ = ErrorNormType::RMS;

    /// Build a per-call controller for a worker (or main thread): clone the
    /// prototype and reset internal state so it starts from first-step
    /// semantics. Every integrate / integrate_*_core / parallel path uses
    /// this pattern; centralizing avoids the "forgot the reset()" class of
    /// bug that surfaced during the SP3 per-lane work (see commit ce4709b).
    ControllerVariant make_worker_controller() const {
        ControllerVariant c = this->controller_variant_;
        std::visit([](auto &cc) { cc.reset(); }, c);
        return c;
    }

    // Step statistics for `get_naccept()` / `get_nreject()` inspection.
    // Written only by single-threaded public wrappers via a post-call
    // writeback from local counters; the impl bodies (`integrate_impl`,
    // `integrate_impl_vectorized`) and all parallel workers keep their
    // counters entirely local and do not touch these members. `mutable`
    // is retained so the writeback can happen from const-qualified
    // wrappers (required because the VectorFunction::compute contract
    // forces `compute_impl` and friends to be const).
    mutable int naccept_ = 0;
    mutable int nreject_ = 0;

    // Count of event refinements that fell off the bisect+Newton fast path
    // and ALSO failed the wider-bracket retry inside find_events. Such
    // crossings are silently absent from the returned eventstates vector
    // (size mismatch with eventtimes) — exposing the count via
    // get_failed_event_count() lets callers detect that loss without
    // changing the existing return shape. Reset by find_events at the
    // start of each call (see :1554-area).
    mutable int n_failed_event_refinements_ = 0;

    // SP3 Hairer-Wanner initial-dt toggle. Default-on. An explicit
    // set_initial_step_size() call flips it off so a user-supplied initial
    // step is respected (principle of least surprise). Constructors set
    // def_step_size_ directly and leave HW enabled.
    bool use_hairer_wanner_initdt_ = true;

    ODEDeriv<double> abs_tols_;
    ODEDeriv<double> rel_tols_;

    void set_abs_tol(double tol) { this->abs_tols_.setConstant(this->ode_.x_vars(), abs(tol)); }
    void set_rel_tol(double tol) { this->rel_tols_.setConstant(this->ode_.x_vars(), abs(tol)); }

    void set_abs_tols(ODEDeriv<double> tol) {
        if (tol.size() != this->ode_.x_vars()) {
            throw std::invalid_argument("Incorrectly sized tolerance vector.");
        }
        for (Eigen::Index i = 0; i < tol.size(); ++i) {
            if (!(tol[i] >= 0.0)) {
                throw std::invalid_argument("abs_tols[" + std::to_string(i) +
                                            "] must be >= 0; got " + std::to_string(tol[i]));
            }
        }
        this->abs_tols_ = tol;
    }
    void set_rel_tols(ODEDeriv<double> tol) {
        if (tol.size() != this->ode_.x_vars()) {
            throw std::invalid_argument("Incorrectly sized tolerance vector.");
        }
        for (Eigen::Index i = 0; i < tol.size(); ++i) {
            if (!(tol[i] >= 0.0)) {
                throw std::invalid_argument("rel_tols[" + std::to_string(i) +
                                            "] must be >= 0; got " + std::to_string(tol[i]));
            }
        }
        this->rel_tols_ = tol;
    }

    /// Joint tolerance invariant for the adaptive path: every component must
    /// satisfy abs_tols[i] + rel_tols[i] > 0. Otherwise scaled_residuals'
    /// denominator is 0 when the state component is 0, producing NaN error
    /// norms and infinite rejection loops. Checked at integrate entry rather
    /// than in the setters so set_abs_tol/set_rel_tol can be called in any
    /// order without transient validation failures.
    void validate_tolerances_for_adaptive() const {
        const Eigen::Index n = this->abs_tols_.size();
        if (this->rel_tols_.size() != n) {
            throw std::logic_error("Integrator internal error: abs_tols/rel_tols size mismatch.");
        }
        for (Eigen::Index i = 0; i < n; ++i) {
            if (!(this->abs_tols_[i] + this->rel_tols_[i] > 0.0)) {
                throw std::invalid_argument(
                    "Tolerance component " + std::to_string(i) +
                    " has abs_tol + rel_tol <= 0 (both vanish). Set at least one of "
                    "abs_tol or rel_tol to a positive value; otherwise the adaptive "
                    "error norm is undefined for zero state.");
            }
        }
    }

    ODEDeriv<double> get_abs_tols() const { return this->abs_tols_; }
    ODEDeriv<double> get_rel_tols() const { return this->rel_tols_; }

    void set_controller(IVPController kind) {
        switch (kind) {
        case IVPController::I:
            this->controller_variant_ = IController{};
            break;
        case IVPController::PI:
            this->controller_variant_ = PIController{};
            break;
        case IVPController::PID:
            this->controller_variant_ = PIDController{};
            break;
        }
        std::visit(
            [](auto &c) {
                c.validate();
                c.reset();
            },
            this->controller_variant_);
    }

    IVPController get_controller() const {
        if (std::holds_alternative<IController>(this->controller_variant_))
            return IVPController::I;
        if (std::holds_alternative<PIController>(this->controller_variant_))
            return IVPController::PI;
        return IVPController::PID;
    }

    void set_error_norm(ErrorNormType t) { this->error_norm_type_ = t; }
    ErrorNormType get_error_norm() const { return this->error_norm_type_; }

    int get_naccept() const { return this->naccept_; }
    int get_nreject() const { return this->nreject_; }

    /// Number of event crossings whose bisect+Newton refinement failed
    /// (both passes) during the most recent `integrate*` call that took
    /// events. The corresponding crossings ARE present in `eventtimes`
    /// (the bracketed pre-step values) but absent from the refined
    /// `eventstates` returned from `find_events`. Callers that care
    /// about full coverage should inspect this after each call. Reset
    /// to 0 at the start of every `find_events` invocation.
    int get_failed_event_count() const { return this->n_failed_event_refinements_; }

    void set_auto_initial_dt(bool on) { this->use_hairer_wanner_initdt_ = on; }
    bool get_auto_initial_dt() const { return this->use_hairer_wanner_initdt_; }

    /// Set the initial step size used when Hairer-Wanner auto-initdt is off
    /// (and as the fixed step when adaptive_ is false). Calling this flips HW
    /// auto-initdt off so the caller's value is respected — re-enable with
    /// set_auto_initial_dt(true) if desired.
    void set_initial_step_size(double h) {
        if (h <= 0.0) {
            throw ::std::invalid_argument(
                "Initial step size must be strictly positive (backward integration is driven "
                "by tf < t0, not by sign of h; zero would divide-by-zero in the fixed-step "
                "path).");
        }
        this->def_step_size_ = h;
        this->use_hairer_wanner_initdt_ = false;
    }

    /// Julia-style maxiters cap. Integration throws std::runtime_error if the
    /// adaptive loop takes more than `n` steps (accepted + rejected). Default
    /// is 1'000'000 — large enough to be invisible in normal use, small
    /// enough to bound a runaway rejection loop.
    void set_max_steps(int n) {
        if (n <= 0) {
            throw ::std::invalid_argument("max_steps must be positive.");
        }
        this->max_steps_ = n;
    }

    int get_max_steps() const { return this->max_steps_; }

    /////////////////////////////////////////////////////////////////////////////////////

  protected:
    static ControllerVariant default_controller_for(IVPAlg alg) {
        switch (alg) {
        case IVPAlg::DOPRI54: {
            PIController c;
            c.beta1 = 17.0 / 100.0; // DOPRI54 override: 1/order - 3·β₂/4 = 17/100
            c.beta2 = 4.0 / 100.0;
            c.gamma = 0.9;
            c.qmin = 1.0 / 5.0;
            c.qmax = 10.0;
            return c;
        }
        case IVPAlg::DOPRI87: {
            IController c;
            c.gamma = 0.9;
            c.qmin = 1.0 / 3.0; // DP8 override
            c.qmax = 6.0;
            return c;
        }
        case IVPAlg::Tsit5: {
            PIController c;
            c.beta1 = 7.0 / 50.0; // Julia generic: 7/(10·order), order=5
            c.beta2 = 2.0 / 25.0; // Julia generic: 2/(5·order),  order=5
            return c;
        }
        case IVPAlg::BS3: {
            PIController c;
            c.beta1 = 7.0 / 30.0; // order=3
            c.beta2 = 2.0 / 15.0;
            return c;
        }
        case IVPAlg::BS5: {
            PIController c;
            c.beta1 = 7.0 / 50.0; // order=5
            c.beta2 = 2.0 / 25.0;
            return c;
        }
        case IVPAlg::Vern7: {
            PIController c;
            c.beta1 = 1.0 / 10.0; // 7/(10·7)
            c.beta2 = 2.0 / 35.0; // 2/(5·7)
            return c;
        }
        case IVPAlg::Vern8: {
            PIController c;
            c.beta1 = 7.0 / 80.0; // order=8
            c.beta2 = 1.0 / 20.0; // 2/(5·8)
            return c;
        }
        case IVPAlg::Vern9: {
            PIController c;
            c.beta1 = 7.0 / 90.0; // order=9
            c.beta2 = 2.0 / 45.0; // 2/(5·9)
            return c;
        }
        default:
            throw std::logic_error("default_controller_for: algorithm not user-selectable");
        }
    }

    template <class State> void update_control(State &xtup) const {
        if constexpr (DODE::UV != 0) {
            if (this->use_controller_) {
                this->controller_.compute(xtup, xtup.template segment<DODE::UV>(
                                                    this->ode_.t_var() + 1, this->ode_.u_vars()));
            }
        }
    }

    /// Scalar integrate core.
    ///
    /// Per-call state lives in three caller-owned references:
    ///   - `controller`: seeded (and pre-reset) by the caller from
    ///     `this->controller_variant_`; mutated during the run.
    ///   - `naccept`, `nreject`: caller-owned counters; incremented by
    ///     this function. Caller zero-initializes before the call.
    /// The function never reads or writes the member-side
    /// `naccept_`, `nreject_`, or `controller_variant_` fields. This
    /// makes concurrent calls on the same `Integrator` instance safe
    /// from member-state races.
    Output<double> integrate_impl(const ODEState<double> &x, double tf,
                                  const std::vector<EventPack> &events,
                                  std::vector<std::vector<Eigen::Vector2d>> &eventtimes,
                                  bool storestates, bool storederivs, bool storemidpoints,
                                  std::vector<ODEState<double>> &states,
                                  std::vector<ODEDeriv<double>> &derivs,
                                  ControllerVariant &controller, int &naccept, int &nreject) const {

        integrators::AdaptiveConfig cfg{static_cast<int>(this->error_order_),
                                        this->error_norm_type_,
                                        this->def_step_size_,
                                        this->max_step_change_,
                                        this->max_steps_,
                                        this->adaptive_,
                                        this->use_hairer_wanner_initdt_};

        auto update_control_fn = [this](auto &s) { this->update_control(s); };

        return with_public_ivp_alg(this->rk_method_, [&](auto alg_tag) -> Output<double> {
            constexpr IVPAlg Alg = alg_tag.value;
            integrators::AdaptiveDriver<Alg, DODE, double> driver;
            return driver.integrate(this->ode_, x, tf, cfg, this->abs_tols_, this->rel_tols_,
                                    controller, naccept, nreject, events, eventtimes, storestates,
                                    storederivs, storemidpoints, states, derivs, update_control_fn);
        });
    }

    /// Vectorized integrate core — same per-call-state contract as
    /// `integrate_impl`, but expanded to **per-trajectory** controller
    /// variants and step counters. Each SIMD lane drives its own
    /// controller copy keyed by trajectory index, so a divergent batch
    /// (different ICs producing different step cadences) no longer
    /// shares accept/reject state across trajectories — preserving Julia
    /// per-integrator semantics for every lane.
    ///
    /// Caller contract: `controllers` and `nacc`/`nrej` must each have
    /// size == ntrajs and be pre-reset by the caller.
    std::vector<Output<double>>
    integrate_impl_vectorized(const std::vector<ODEState<double>> &xs, const Eigen::VectorXd &tfs,
                              const std::vector<EventPack> &events,
                              std::vector<std::vector<std::vector<Eigen::Vector2d>>> &eventtimes_s,
                              bool storestates, bool storederivs, bool storemidpoints,
                              std::vector<std::vector<ODEState<double>>> &states_s,
                              std::vector<std::vector<ODEDeriv<double>>> &derivs_s,
                              std::vector<ControllerVariant> &controllers, std::vector<int> &nacc,
                              std::vector<int> &nrej) const {

        integrators::AdaptiveConfig cfg{static_cast<int>(this->error_order_),
                                        this->error_norm_type_,
                                        this->def_step_size_,
                                        this->max_step_change_,
                                        this->max_steps_,
                                        this->adaptive_,
                                        this->use_hairer_wanner_initdt_};

        auto update_control_fn = [this](auto &s) { this->update_control(s); };

        return with_public_ivp_alg(
            this->rk_method_, [&](auto alg_tag) -> std::vector<Output<double>> {
                constexpr IVPAlg Alg = alg_tag.value;
                integrators::ParallelDriver<Alg, DODE> driver;
                return driver.integrate(this->ode_, xs, tfs, cfg, this->abs_tols_, this->rel_tols_,
                                        controllers, nacc, nrej, events, eventtimes_s, storestates,
                                        storederivs, storemidpoints, states_s, derivs_s,
                                        update_control_fn);
            });
    }

    /// \name Thread-safe `_core` helpers
    ///
    /// These mirror the public `integrate*` overloads that are reachable
    /// from parallel dispatchers but take controller + counter state by
    /// reference and never read or write `this->controller_variant_`,
    /// `this->naccept_`, or `this->nreject_`. Public wrappers create
    /// per-call locals, invoke the `_core`, and (single-threaded only)
    /// write counters back to the mutable members for `get_naccept()` /
    /// `get_nreject()` inspection. Parallel workers invoke `_core`
    /// directly and skip the writeback, so the members are never
    /// written from multiple threads.
    /// @{
    IntegRet integrate_core(const ODEState<double> &x0, double tf, ControllerVariant &controller,
                            int &naccept, int &nreject) const {
        ODEState<double> xf;
        std::vector<EventPack> events;
        std::vector<std::vector<Eigen::Vector2d>> eventtimes;
        bool storestates = false;
        bool storederivs = false;
        bool storemidpoints = false;
        std::vector<ODEState<double>> xs;
        std::vector<ODEDeriv<double>> d_xs;
        xf = this->integrate_impl(x0, tf, events, eventtimes, storestates, storederivs,
                                  storemidpoints, xs, d_xs, controller, naccept, nreject);
        return xf;
    }

    IntegEventRet integrate_core(const ODEState<double> &x0, double tf,
                                 const std::vector<EventPack> &events,
                                 ControllerVariant &controller, int &naccept, int &nreject) const {
        ODEState<double> xf;
        std::vector<std::vector<Eigen::Vector2d>> eventtimes;
        bool storestates = true;
        bool storederivs = true;
        bool storemidpoints = true;
        std::vector<ODEState<double>> xs;
        std::vector<ODEDeriv<double>> d_xs;
        xf = this->integrate_impl(x0, tf, events, eventtimes, storestates, storederivs,
                                  storemidpoints, xs, d_xs, controller, naccept, nreject);
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

    DenseRet integrate_dense_core(const ODEState<double> &x0, double tf,
                                  ControllerVariant &controller, int &naccept, int &nreject) const {
        ODEState<double> xf;
        std::vector<EventPack> events;
        std::vector<std::vector<Eigen::Vector2d>> eventtimes;
        bool storestates = true;
        bool storederivs = false;
        bool storemidpoints = false;
        std::vector<ODEState<double>> xs;
        std::vector<ODEDeriv<double>> d_xs;
        xf = this->integrate_impl(x0, tf, events, eventtimes, storestates, storederivs,
                                  storemidpoints, xs, d_xs, controller, naccept, nreject);
        return xs;
    }

    DenseEventRet integrate_dense_core(const ODEState<double> &x0, double tf,
                                       const std::vector<EventPack> &events, bool alloutput,
                                       ControllerVariant &controller, int &naccept,
                                       int &nreject) const {
        ODEState<double> xf;
        std::vector<std::vector<Eigen::Vector2d>> eventtimes;
        bool storestates = true;
        bool storederivs = true;
        bool storemidpoints = true;
        std::vector<ODEState<double>> xs;
        std::vector<ODEDeriv<double>> d_xs;
        xf = this->integrate_impl(x0, tf, events, eventtimes, storestates, storederivs,
                                  storemidpoints, xs, d_xs, controller, naccept, nreject);
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

    DenseEventRet integrate_dense_core(const ODEState<double> &x0, double tf, int n,
                                       const std::vector<EventPack> &events,
                                       ControllerVariant &controller, int &naccept,
                                       int &nreject) const {
        ODEState<double> xf;
        std::vector<std::vector<Eigen::Vector2d>> eventtimes;
        bool storestates = true;
        bool storederivs = true;
        bool storemidpoints = true;
        std::vector<ODEState<double>> xs;
        std::vector<ODEDeriv<double>> d_xs;
        xf = this->integrate_impl(x0, tf, events, eventtimes, storestates, storederivs,
                                  storemidpoints, xs, d_xs, controller, naccept, nreject);
        auto tab = this->make_table(xs, d_xs, true);
        std::vector<std::vector<ODEState<double>>> eventlocs =
            this->find_events(tab, events, eventtimes);
        Eigen::VectorXd ts;
        ts.setLinSpaced(n, xs[0][this->ode_.t_var()], xs.back()[this->ode_.t_var()]);
        std::vector<ODEState<double>> x_interp(n);
        for (int i = 0; i < n; i++) {
            x_interp[i].resize(this->ode_.input_rows());
            tab->interpolate_ref(ts[i], x_interp[i]);
        }
        return std::tuple{x_interp, eventlocs};
    }

    DenseRet integrate_dense_core(const ODEState<double> &x0, double tf, int n,
                                  ControllerVariant &controller, int &naccept, int &nreject) const {
        ODEState<double> xf;
        std::vector<std::vector<Eigen::Vector2d>> eventtimes;
        bool storestates = true;
        bool storederivs = true;
        bool storemidpoints = true;
        std::vector<ODEState<double>> xs;
        std::vector<ODEDeriv<double>> d_xs;
        std::vector<EventPack> events;
        xf = this->integrate_impl(x0, tf, events, eventtimes, storestates, storederivs,
                                  storemidpoints, xs, d_xs, controller, naccept, nreject);
        auto tab = this->make_table(xs, d_xs, true);
        Eigen::VectorXd ts;
        ts.setLinSpaced(n, xs[0][this->ode_.t_var()], xs.back()[this->ode_.t_var()]);
        std::vector<ODEState<double>> x_interp(n);
        for (int i = 0; i < n; i++) {
            x_interp[i].resize(this->ode_.input_rows());
            tab->interpolate_ref(ts[i], x_interp[i]);
        }
        return x_interp;
    }

    STMRet integrate_stm_core(const ODEState<double> &x0, double tf, ControllerVariant &controller,
                              int &naccept, int &nreject) const {
        auto xs = this->integrate_dense_core(x0, tf, controller, naccept, nreject);
        Jacobian<double> jx = this->calculate_jacobian(xs);
        return std::tuple{xs.back(), jx};
    }

    STMEventRet integrate_stm_core(const ODEState<double> &x0, double tf,
                                   const std::vector<EventPack> &events,
                                   ControllerVariant &controller, int &naccept,
                                   int &nreject) const {
        auto [xs, eventlocs] =
            this->integrate_dense_core(x0, tf, events, false, controller, naccept, nreject);
        Jacobian<double> jx = this->calculate_jacobian(xs);
        return std::tuple{xs.back(), jx, eventlocs};
    }
    /// @}

    std::vector<std::vector<ODEState<double>>>
    find_events(std::shared_ptr<LGLInterpTable> tab, const std::vector<EventPack> &events,
                const std::vector<std::vector<Eigen::Vector2d>> &eventtimes) const {
        // Reset per-call so get_failed_event_count() reflects only the
        // most recent integrate-with-events invocation.
        this->n_failed_event_refinements_ = 0;
        return EventHandler::refine_events<ODEState<double>>(
            tab, events, eventtimes, this->ode_.input_rows(), this->max_event_iters_,
            this->event_tol_, this->n_failed_event_refinements_);
    }

    std::shared_ptr<LGLInterpTable> make_table(const std::vector<ODEState<double>> &xs,
                                               bool fifthorder) const {
        std::vector<Eigen::VectorXd> xs_in;
        for (auto &X : xs) {
            xs_in.push_back(X);
        }

        GenericFunction<-1, -1> odetemp;
        if constexpr (DODE::IsGenericODE) {
            odetemp = this->ode_.func_;
        } else {
            odetemp = this->ode_;
        }
        TranscriptionModes m = fifthorder ? TranscriptionModes::LGL5 : TranscriptionModes::LGL3;
        std::shared_ptr<LGLInterpTable> tab = std::make_shared<LGLInterpTable>(
            odetemp, this->ode_.x_vars(), this->ode_.u_vars() + this->ode_.p_vars(), m);

        tab->load_exact_data(xs_in);

        return tab;
    }

    std::shared_ptr<LGLInterpTable> make_table(const std::vector<ODEState<double>> &xs,
                                               const std::vector<ODEDeriv<double>> &d_xs,
                                               bool fifthorder) const {

        GenericFunction<-1, -1> odetemp;
        if constexpr (DODE::IsGenericODE) {
            odetemp = this->ode_.func_;
        } else {
            odetemp = this->ode_;
        }
        TranscriptionModes m = fifthorder ? TranscriptionModes::LGL5 : TranscriptionModes::LGL3;
        std::shared_ptr<LGLInterpTable> tab = std::make_shared<LGLInterpTable>(
            odetemp, this->ode_.x_vars(), this->ode_.u_vars() + this->ode_.p_vars(), m);

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
        auto tmp = STMDriver::calculate_jacobians(this->stepper_, this->ode_, xs_s,
                                                  this->input_rows(), this->output_rows());
        std::vector<Jacobian<double>> out(tmp.size());
        for (std::size_t i = 0; i < tmp.size(); ++i)
            out[i] = tmp[i];
        return out;
    }

    Jacobian<double> calculate_jacobian(const std::vector<ODEState<double>> &xs) const {
        Jacobian<double> out;
        out = STMDriver::calculate_jacobian(this->stepper_, this->ode_, xs, this->input_rows(),
                                            this->output_rows(), this->enable_vectorization_);
        return out;
    }

    std::tuple<Jacobian<double>, Hessian<double>>
    calculate_jacobian_hessian(const std::vector<ODEState<double>> &xs,
                               const ODEState<double> &lf) const {
        if (xs.size() < 2) {
            throw std::invalid_argument(
                "calculate_jacobian_hessian requires at least 2 states (start + end of one "
                "integrator step); got xs.size() == " +
                std::to_string(xs.size()) + ".");
        }
        auto [jx_dyn, hx_dyn] = STMDriver::calculate_jacobian_hessian(
            this->stepper_, this->ode_, xs, lf, this->input_rows(), this->output_rows());
        Jacobian<double> jx;
        Hessian<double> hx;
        jx = jx_dyn;
        hx = hx_dyn;
        return {jx, hx};
    }

    std::tuple<std::vector<Jacobian<double>>, std::vector<Hessian<double>>>
    calculate_jacobians_hessians(const std::vector<std::vector<ODEState<double>>> &xs_s,
                                 const std::vector<ODEState<double>> &lf_s) const {
        auto [jxs_dyn, hxs_dyn] = STMDriver::calculate_jacobians_hessians(
            this->stepper_, this->ode_, xs_s, lf_s, this->input_rows(), this->output_rows());
        std::vector<Jacobian<double>> jxs(jxs_dyn.size());
        std::vector<Hessian<double>> hxs(hxs_dyn.size());
        for (std::size_t i = 0; i < jxs_dyn.size(); ++i) {
            jxs[i] = jxs_dyn[i];
            hxs[i] = hxs_dyn[i];
        }
        return {jxs, hxs};
    }

  public:
    ////////////////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////////////
    IntegRet integrate(const ODEState<double> &x0, double tf) const {
        ControllerVariant ctrl = this->make_worker_controller();
        int na = 0, nr = 0;
        auto xf = this->integrate_core(x0, tf, ctrl, na, nr);
        this->naccept_ = na;
        this->nreject_ = nr;
        return xf;
    }

    std::vector<ODEState<double>> integrate(const std::vector<ODEState<double>> &x0s,
                                            const Eigen::VectorXd &tfs) const {
        const int n = static_cast<int>(x0s.size());
        std::vector<ControllerVariant> ctrls(n, this->controller_variant_);
        for (auto &c : ctrls)
            std::visit([](auto &cc) { cc.reset(); }, c);
        std::vector<int> nacc(n, 0), nrej(n, 0);

        std::vector<ODEState<double>> result;
        if (!vectorize_batch_calls_) {
            std::vector<ODEState<double>> xfs(n);
            for (int i = 0; i < n; i++) {
                xfs[i] = this->integrate_core(x0s[i], tfs[i], ctrls[i], nacc[i], nrej[i]);
            }
            result = std::move(xfs);
        } else {
            std::vector<EventPack> events;
            std::vector<std::vector<std::vector<Eigen::Vector2d>>> eventtimes;
            bool storestates = false;
            bool storederivs = false;
            bool storemidpoints = false;
            std::vector<std::vector<ODEState<double>>> xs;
            std::vector<std::vector<ODEDeriv<double>>> d_xs;
            result =
                integrate_impl_vectorized(x0s, tfs, events, eventtimes, storestates, storederivs,
                                          storemidpoints, xs, d_xs, ctrls, nacc, nrej);
        }
        this->naccept_ = std::accumulate(nacc.begin(), nacc.end(), 0);
        this->nreject_ = std::accumulate(nrej.begin(), nrej.end(), 0);
        return result;
    }

    std::vector<STMRet> integrate_stm(const std::vector<ODEState<double>> &x0s,
                                      const Eigen::VectorXd &tfs) const {
        const int n = static_cast<int>(x0s.size());
        std::vector<ControllerVariant> ctrls(n, this->controller_variant_);
        for (auto &c : ctrls)
            std::visit([](auto &cc) { cc.reset(); }, c);
        std::vector<int> nacc(n, 0), nrej(n, 0);

        std::vector<STMRet> rets(n);
        if (!vectorize_batch_calls_) {
            for (int i = 0; i < n; i++) {
                rets[i] = this->integrate_stm_core(x0s[i], tfs[i], ctrls[i], nacc[i], nrej[i]);
            }
        } else {
            std::vector<EventPack> events;
            std::vector<std::vector<std::vector<Eigen::Vector2d>>> eventtimes;
            bool storestates = true;
            bool storederivs = false;
            bool storemidpoints = false;
            std::vector<std::vector<ODEState<double>>> xs(n);
            std::vector<std::vector<ODEDeriv<double>>> d_xs(n);

            auto x_finals =
                integrate_impl_vectorized(x0s, tfs, events, eventtimes, storestates, storederivs,
                                          storemidpoints, xs, d_xs, ctrls, nacc, nrej);

            auto jacs = this->calculate_jacobians(xs);
            for (int i = 0; i < n; i++) {
                rets[i] = std::tuple{x_finals[i], jacs[i]};
            }
        }
        this->naccept_ = std::accumulate(nacc.begin(), nacc.end(), 0);
        this->nreject_ = std::accumulate(nrej.begin(), nrej.end(), 0);
        return rets;
    }

    std::vector<std::tuple<ODEState<double>, Jacobian<double>, Hessian<double>>>
    integrate_stm2(const std::vector<ODEState<double>> &x0s, const Eigen::VectorXd &tfs,
                   const std::vector<ODEState<double>> &lfs) const {
        const int n = static_cast<int>(x0s.size());
        std::vector<ControllerVariant> ctrls(n, this->controller_variant_);
        for (auto &c : ctrls)
            std::visit([](auto &cc) { cc.reset(); }, c);
        std::vector<int> nacc(n, 0), nrej(n, 0);

        std::vector<std::tuple<ODEState<double>, Jacobian<double>, Hessian<double>>> rets(n);
        if (!vectorize_batch_calls_) {
            for (int i = 0; i < n; i++) {
                auto xs = this->integrate_dense_core(x0s[i], tfs[i], ctrls[i], nacc[i], nrej[i]);
                auto [J, H] = this->calculate_jacobian_hessian(xs, lfs[i]);
                rets[i] = std::tuple{xs.back(), J, H};
            }
        } else {
            std::vector<EventPack> events;
            std::vector<std::vector<std::vector<Eigen::Vector2d>>> eventtimes;
            bool storestates = true;
            bool storederivs = false;
            bool storemidpoints = false;
            std::vector<std::vector<ODEState<double>>> xs(n);
            std::vector<std::vector<ODEDeriv<double>>> d_xs(n);

            auto x_finals =
                integrate_impl_vectorized(x0s, tfs, events, eventtimes, storestates, storederivs,
                                          storemidpoints, xs, d_xs, ctrls, nacc, nrej);

            auto [js, hs_out] = this->calculate_jacobians_hessians(xs, lfs);
            for (int i = 0; i < n; i++) {
                rets[i] = std::tuple{x_finals[i], js[i], hs_out[i]};
            }
        }
        this->naccept_ = std::accumulate(nacc.begin(), nacc.end(), 0);
        this->nreject_ = std::accumulate(nrej.begin(), nrej.end(), 0);
        return rets;
    }

    IntegEventRet integrate(const ODEState<double> &x0, double tf,
                            const std::vector<EventPack> &events) const {
        ControllerVariant ctrl = this->make_worker_controller();
        int na = 0, nr = 0;
        auto r = this->integrate_core(x0, tf, events, ctrl, na, nr);
        this->naccept_ = na;
        this->nreject_ = nr;
        return r;
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
            ControllerVariant ctrl;
            int na = 0, nr = 0;
            for (int i = start; i < stop; i++) {
                ctrl = this->make_worker_controller();
                na = 0;
                nr = 0;
                results[i] = this->integrate_core(x0s[i], tfs[i], args..., ctrl, na, nr);
            }
        };

        tycho::utils::parallel_blocks(n, job, n_parts);
        return results;
    }

    /// Parallel batch integrate.
    /// Note: `get_naccept()` / `get_nreject()` are NOT updated by the
    /// parallel path (workers use per-iteration local counters to avoid
    /// cross-thread writes to member state). Values reflect whatever
    /// state the members held before this call.
    std::vector<IntegRet> integrate_parallel(const std::vector<ODEState<double>> &x0s,
                                             const Eigen::VectorXd &tfs, int n_parts) {
        return this->integrate_parallel_impl(x0s, tfs, n_parts);
    }
    /// Parallel batch integrate with events. Same `get_naccept()` /
    /// `get_nreject()` contract as the no-events overload.
    std::vector<IntegEventRet> integrate_parallel(const std::vector<ODEState<double>> &x0s,
                                                  const Eigen::VectorXd &tfs,
                                                  const std::vector<EventPack> &events,
                                                  int n_parts) {
        return this->integrate_parallel_impl(x0s, tfs, n_parts, events);
    }
    /////////////////////////////////////////////////////////////////////////////////////

    DenseRet integrate_dense(const ODEState<double> &x0, double tf) const {
        ControllerVariant ctrl = this->make_worker_controller();
        int na = 0, nr = 0;
        auto xs = this->integrate_dense_core(x0, tf, ctrl, na, nr);
        this->naccept_ = na;
        this->nreject_ = nr;
        return xs;
    }

    DenseEventRet integrate_dense(const ODEState<double> &x0, double tf,
                                  const std::vector<EventPack> &events, bool alloutput) const {
        ControllerVariant ctrl = this->make_worker_controller();
        int na = 0, nr = 0;
        auto r = this->integrate_dense_core(x0, tf, events, alloutput, ctrl, na, nr);
        this->naccept_ = na;
        this->nreject_ = nr;
        return r;
    }

    DenseEventRet integrate_dense(const ODEState<double> &x0, double tf, int n,
                                  const std::vector<EventPack> &events) const {
        ControllerVariant ctrl = this->make_worker_controller();
        int na = 0, nr = 0;
        auto r = this->integrate_dense_core(x0, tf, n, events, ctrl, na, nr);
        this->naccept_ = na;
        this->nreject_ = nr;
        return r;
    }

    DenseRet integrate_dense(const ODEState<double> &x0, double tf, int n) const {
        ControllerVariant ctrl = this->make_worker_controller();
        int na = 0, nr = 0;
        auto r = this->integrate_dense_core(x0, tf, n, ctrl, na, nr);
        this->naccept_ = na;
        this->nreject_ = nr;
        return r;
    }

    DenseRet integrate_dense(const ODEState<double> &x0, double tf, int num_states,
                             std::function<bool(ConstEigenRef<Eigen::VectorXd>)> exitfun) const {
        ControllerVariant ctrl = this->make_worker_controller();
        int na = 0, nr = 0;

        VectorX<double> ts = VectorX<double>::LinSpaced(num_states, x0[this->ode_.t_var()], tf);
        std::vector<ODEState<double>> xout;
        xout.reserve(num_states);
        xout.push_back(x0);
        for (int i = 1; i < num_states; i++) {
            xout.push_back(this->integrate_core(xout[i - 1], ts[i], ctrl, na, nr));
            if (exitfun(xout.back()))
                break;
        }
        this->naccept_ = na;
        this->nreject_ = nr;
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
            ControllerVariant ctrl;
            int na = 0, nr = 0;
            for (int i = start; i < stop; i++) {
                ctrl = this->make_worker_controller();
                na = 0;
                nr = 0;
                results[i] = this->integrate_dense_core(x0s[i], tfs[i], args..., ctrl, na, nr);
            }
        };

        tycho::utils::parallel_blocks(n, job, n_parts);
        return results;
    }

    /// Parallel batch dense integrate.
    /// Note: `get_naccept()` / `get_nreject()` are NOT updated by the
    /// parallel path. See `integrate_parallel` for the rationale.
    std::vector<DenseRet> integrate_dense_parallel(const std::vector<ODEState<double>> &x0s,
                                                   const Eigen::VectorXd &tfs, int n_parts) {
        return this->integrate_dense_parallel_impl(x0s, tfs, n_parts);
    }
    /// Parallel batch dense integrate with events. Same `get_naccept()` /
    /// `get_nreject()` contract as the no-events overload.
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
            ControllerVariant ctrl;
            int na = 0, nr = 0;
            for (int i = start; i < stop; i++) {
                ctrl = this->make_worker_controller();
                na = 0;
                nr = 0;
                results[i] =
                    this->integrate_dense_core(x0s[i], tfs[i], ns[i], args..., ctrl, na, nr);
            }
        };

        tycho::utils::parallel_blocks(n, job, n_parts);
        return results;
    }

    /// Parallel batch dense integrate with per-trajectory state counts.
    /// Note: `get_naccept()` / `get_nreject()` are NOT updated by the
    /// parallel path.
    std::vector<DenseEventRet> integrate_dense_parallel(const std::vector<ODEState<double>> &x0s,
                                                        const Eigen::VectorXd &tfs,
                                                        const std::vector<int> &ns,
                                                        const std::vector<EventPack> &events,
                                                        int n_parts) {
        return this->integrate_dense_parallel_impl_n(x0s, tfs, ns, n_parts, events);
    }
    /// Parallel batch dense integrate with per-trajectory state counts (no events).
    std::vector<DenseRet> integrate_dense_parallel(const std::vector<ODEState<double>> &x0s,
                                                   const Eigen::VectorXd &tfs,
                                                   const std::vector<int> &ns, int n_parts) {
        return this->integrate_dense_parallel_impl_n(x0s, tfs, ns, n_parts);
    }
    /////////////////////////////////////////////////////////////////////////////////////

    STMRet integrate_stm(const ODEState<double> &x0, double tf) const {
        ControllerVariant ctrl = this->make_worker_controller();
        int na = 0, nr = 0;
        auto r = this->integrate_stm_core(x0, tf, ctrl, na, nr);
        this->naccept_ = na;
        this->nreject_ = nr;
        return r;
    }
    STMEventRet integrate_stm(const ODEState<double> &x0, double tf,
                              const std::vector<EventPack> &events) const {
        ControllerVariant ctrl = this->make_worker_controller();
        int na = 0, nr = 0;
        auto r = this->integrate_stm_core(x0, tf, events, ctrl, na, nr);
        this->naccept_ = na;
        this->nreject_ = nr;
        return r;
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
            ControllerVariant ctrl;
            int na = 0, nr = 0;
            for (int i = start; i < stop; i++) {
                ctrl = this->make_worker_controller();
                na = 0;
                nr = 0;
                results[i] = this->integrate_stm_core(x0s[i], tfs[i], args..., ctrl, na, nr);
            }
        };

        tycho::utils::parallel_blocks(n, job, n_parts);
        return results;
    }

    /// Parallel batch STM integrate.
    /// Note: `get_naccept()` / `get_nreject()` are NOT updated by the
    /// parallel path.
    std::vector<STMRet> integrate_stm_parallel(const std::vector<ODEState<double>> &x0s,
                                               const Eigen::VectorXd &tfs, int n_parts) {
        return this->integrate_stm_parallel_impl(x0s, tfs, n_parts);
    }
    /// Parallel batch STM integrate with events. Same `get_naccept()` /
    /// `get_nreject()` contract as the no-events overload.
    std::vector<STMEventRet> integrate_stm_parallel(const std::vector<ODEState<double>> &x0s,
                                                    const Eigen::VectorXd &tfs,
                                                    const std::vector<EventPack> &events,
                                                    int n_parts) {
        return this->integrate_stm_parallel_impl(x0s, tfs, n_parts, events);
    }

    /// Segmented parallel STM over a single trajectory.
    /// Note: `get_naccept()` / `get_nreject()` are NOT updated by this
    /// path. Workers and the interleaved main-thread propagation each
    /// own private per-segment counters to avoid cross-thread writes
    /// to member state.
    STMRet integrate_stm_parallel(const ODEState<double> &x0, double tf, int n_parts) {
        VectorX<double> ts = VectorX<double>::LinSpaced(n_parts + 1, x0[this->ode_.t_var()], tf);
        std::vector<ODEState<double>> xs(n_parts + 1);
        xs[0] = x0;

        std::vector<std::future<STMRet>> results(n_parts);

        Eigen::MatrixXd jxall(this->input_rows(), this->input_rows());
        jxall.setIdentity();

        // Each stm_op invocation creates its own controller + counter
        // locals, so concurrent worker invocations are race-free.
        auto stm_op = [&](int i) {
            ControllerVariant ctrl = this->make_worker_controller();
            int na = 0, nr = 0;
            auto xi = xs[i];
            auto tf1 = ts[i + 1];
            return this->integrate_stm_core(xi, tf1, ctrl, na, nr);
        };

        if (n_parts > 1 && tycho::utils::use_thread_pool()) {
            // Main-thread propagation runs concurrently with workers;
            // keep its state strictly local (no member writes).
            ControllerVariant main_ctrl;
            int main_na = 0, main_nr = 0;
            int submitted = 0;
            try {
                for (int i = 0; i < n_parts; i++) {
                    results[i] =
                        tycho::utils::thread_pool().submit_task([&stm_op, i] { return stm_op(i); });
                    submitted = i + 1;
                    if (i < (n_parts - 1)) {
                        main_ctrl = this->make_worker_controller();
                        main_na = 0;
                        main_nr = 0;
                        xs[i + 1] =
                            this->integrate_core(xs[i], ts[i + 1], main_ctrl, main_na, main_nr);
                    }
                }
            } catch (...) {
                // If integrate() throws mid-loop, drain submitted futures to
                // prevent use-after-free of stack-captured references (&stm_op).
                // Mirror the good-path drain below: aggregate worker failures as
                // secondary context rather than silently dropping them. The
                // main-thread exception is the primary trigger, but a worker may
                // have faulted first with the real root cause (e.g., NaN on a
                // specific lane).
                std::string primary_msg;
                try {
                    throw; // rethrow in-flight to extract what()
                } catch (const std::exception &e) {
                    primary_msg = e.what();
                } catch (...) {
                    primary_msg = "<non-std::exception>";
                }
                std::vector<std::string> extra_msgs;
                for (int j = 0; j < submitted; j++) {
                    try {
                        results[j].get();
                    } catch (const std::exception &je) {
                        extra_msgs.emplace_back(je.what());
                    } catch (...) {
                        extra_msgs.emplace_back("<non-std::exception>");
                    }
                }
                if (extra_msgs.empty()) {
                    throw;
                }
                constexpr size_t kMaxExtras = 5;
                std::string joined =
                    "integrate_stm_parallel: main-thread segment failure: " + primary_msg +
                    "; worker failures (" + std::to_string(extra_msgs.size()) + "): ";
                size_t shown = std::min(extra_msgs.size(), kMaxExtras);
                for (size_t k = 0; k < shown; ++k) {
                    if (k)
                        joined += " | ";
                    joined += extra_msgs[k];
                }
                if (extra_msgs.size() > kMaxExtras) {
                    joined +=
                        " | ... and " + std::to_string(extra_msgs.size() - kMaxExtras) + " more";
                }
                throw std::runtime_error(joined);
            }
            // If .get() throws, drain remaining futures before rethrowing to
            // prevent use-after-free of stack-captured references (&stm_op, etc.).
            // Collect *all* failure messages so no root cause is lost; if more
            // than one segment failed, rethrow a composite runtime_error rather
            // than dropping secondaries to stderr.
            std::exception_ptr ex;
            std::string primary_msg;
            std::vector<std::string> extra_msgs;
            for (int i = 0; i < n_parts; i++) {
                try {
                    auto [xf, jx] = results[i].get();
                    jxall.topRows(this->output_rows()) = (jx * jxall).eval();
                    if (i == (n_parts - 1))
                        xs[i + 1] = xf;
                } catch (const std::exception &e) {
                    if (!ex) {
                        ex = std::current_exception();
                        primary_msg = e.what();
                    }
                    for (int j = i + 1; j < n_parts; j++) {
                        try {
                            results[j].get();
                        } catch (const std::exception &je) {
                            extra_msgs.emplace_back(je.what());
                        } catch (...) {
                            extra_msgs.emplace_back("<non-std::exception>");
                        }
                    }
                    break;
                } catch (...) {
                    if (!ex) {
                        ex = std::current_exception();
                        primary_msg = "<non-std::exception>";
                    }
                    for (int j = i + 1; j < n_parts; j++) {
                        try {
                            results[j].get();
                        } catch (const std::exception &je) {
                            extra_msgs.emplace_back(je.what());
                        } catch (...) {
                            extra_msgs.emplace_back("<non-std::exception>");
                        }
                    }
                    break;
                }
            }
            if (ex) {
                if (extra_msgs.empty()) {
                    std::rethrow_exception(ex);
                }
                // Cap secondary-failure detail at kMaxExtras to keep the
                // composite message bounded under large n_parts.
                constexpr size_t kMaxExtras = 5;
                std::string joined =
                    "integrate_stm_parallel: primary segment failure: " + primary_msg +
                    "; additional failures (" + std::to_string(extra_msgs.size()) + "): ";
                size_t shown = std::min(extra_msgs.size(), kMaxExtras);
                for (size_t k = 0; k < shown; ++k) {
                    if (k)
                        joined += " | ";
                    joined += extra_msgs[k];
                }
                if (extra_msgs.size() > kMaxExtras) {
                    joined +=
                        " | ... and " + std::to_string(extra_msgs.size() - kMaxExtras) + " more";
                }
                throw std::runtime_error(joined);
            }
        } else {
            // Non-threadpool fallback is single-threaded, but stay on
            // `_core` entries + local state for consistency — members
            // are never written by this path.
            ControllerVariant fallback_ctrl;
            int fallback_na = 0, fallback_nr = 0;
            for (int i = 0; i < n_parts; i++) {
                auto [xf, jx] = stm_op(i);
                jxall.topRows(this->output_rows()) = (jx * jxall).eval();
                if (i < n_parts - 1) {
                    fallback_ctrl = this->make_worker_controller();
                    fallback_na = 0;
                    fallback_nr = 0;
                    xs[i + 1] = this->integrate_core(xs[i], ts[i + 1], fallback_ctrl, fallback_na,
                                                     fallback_nr);
                } else {
                    xs[i + 1] = xf;
                }
            }
        }

        STMRet tup_final;
        std::get<0>(tup_final) = xs.back();
        std::get<1>(tup_final) = jxall.topRows(this->output_rows());
        return tup_final;
    }

    /////////////////////////////////////////////////////////////////////////////////////

    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();

        ODEState<Scalar> x0 = x.head(this->ode_.input_rows());
        Scalar tf = x[this->ode_.input_rows()];
        // compute_impl must stay const (VectorFunction::compute contract)
        // and is called from differentiation paths; avoid the member
        // writeback by going directly to the thread-safe core with
        // local state.
        ControllerVariant ctrl = this->make_worker_controller();
        int na = 0, nr = 0;
        fx = this->integrate_core(x0, tf, ctrl, na, nr);
    }
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(CVecRef<InType> x, CVecRef<OutType> fx_,
                                      CMatRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();

        ODEState<Scalar> x0 = x.head(this->ode_.input_rows());
        Scalar tf = x[this->ode_.input_rows()];
        ControllerVariant ctrl = this->make_worker_controller();
        int na = 0, nr = 0;
        auto xs = this->integrate_dense_core(x0, tf, ctrl, na, nr);
        fx = xs.back();
        jx = this->calculate_jacobian(xs);
    }
    template <class InType, class OutType, class JacType, class AdjGradType, class AdjHessType,
              class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        CVecRef<InType> x, CVecRef<OutType> fx_, CMatRef<JacType> jx_,
        CVecRef<AdjGradType> adjgrad_, CMatRef<AdjHessType> adjhess_,
        CVecRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        VecRef<OutType> fx = fx_.const_cast_derived();
        MatRef<JacType> jx = jx_.const_cast_derived();
        VecRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
        MatRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

        ODEState<Scalar> x0 = x.head(this->ode_.input_rows());
        ODEState<Scalar> lf = adjvars;
        Scalar tf = x[this->ode_.input_rows()];

        ControllerVariant ctrl = this->make_worker_controller();
        int na = 0, nr = 0;
        auto xs = this->integrate_dense_core(x0, tf, ctrl, na, nr);
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
