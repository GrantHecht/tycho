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
#include <cassert>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
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
#include "tycho/detail/utils/exception_what.h"
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

/// Dispatch `fn` with a compile-time IVPAlg tag matching the runtime `alg`,
/// so a runtime rk_method_ selector can drive templated code that needs a
/// compile-time Alg parameter (AdaptiveDriver<Alg,...>, ParallelDriver<Alg,...>).
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

    using EventPack = tycho::integrators::EventPack;
    using EventLocsType = std::vector<std::vector<std::optional<ODEState<double>>>>;

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
    // User-provided controller specification: the GenericFunction for the
    // control law and the input-variable locations it reads. Empty when
    // the integrator has no user controller. std::optional encodes the
    // "no controller" branch structurally, so the copy-ctor only touches
    // a populated GenericFunction (its default-state copy throws).
    std::optional<std::pair<ControllerType, Eigen::VectorXi>> controller_spec_;
    StepperWrapperType stepper_;
    IVPAlg rk_method_ = IVPAlg::DOPRI87;

  public:
    Integrator() { this->enable_vectorization_ = true; }

    Integrator(const DODE &dode, IVPAlg meth, double defstep) : Integrator() {
        // Use in_place_type to sidestep MSVC variant overload-resolution
        // ambiguity between the int and Eigen::VectorXi alternatives.
        ControlIndexType empty_ci{std::in_place_type<Eigen::VectorXi>};
        this->set_method(meth, dode, defstep, false, ControllerType{}, empty_ci);
        this->set_abs_tol(1.0e-12);
        this->set_rel_tol(0);
    }
    Integrator(const DODE &dode, double defstep) : Integrator(dode, IVPAlg::DOPRI87, defstep) {}
    Integrator(const DODE &dode, IVPAlg meth, double defstep, const ControllerType &ucon,
               const ControlIndexType &varlocs_t)
        : Integrator() {

        this->set_method(meth, dode, defstep, true, ucon, varlocs_t);
        this->set_abs_tol(1.0e-12);
        this->set_rel_tol(0);
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
        this->set_abs_tol(1.0e-12);
        this->set_rel_tol(0);
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
        this->set_abs_tol(1.0e-12);
        this->set_rel_tol(0);
    }
    Integrator(const DODE &dode, double defstep, std::shared_ptr<LGLInterpTable> tab,
               const Eigen::VectorXi &ulocs)
        : Integrator(dode, IVPAlg::DOPRI87, defstep, tab, ulocs) {}

    Integrator(const DODE &dode, IVPAlg meth, double defstep, std::shared_ptr<LGLInterpTable> tab)
        : Integrator() {

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
        this->set_abs_tol(1.0e-12);
        this->set_rel_tol(0);
    }
    Integrator(const DODE &dode, double defstep, std::shared_ptr<LGLInterpTable> tab)
        : Integrator(dode, IVPAlg::DOPRI87, defstep, tab) {}

    /// Reconfigures the stepper for a new RK method.
    ///
    /// Note: selecting a new method unconditionally resets the controller to
    /// that method's default kind (see `default_controller_for`). If you
    /// want a non-default controller, call `set_controller()` *after*
    /// `set_method()`. `get_controller()` can be used to inspect the
    /// current choice.
    void set_method(IVPAlg alg, const DODE &dode, double defstep, bool usecontrol,
                    const GenericFunction<-1, -1> &ucon, const ControlIndexType &varlocs_t) {

        if (!std::isfinite(defstep) || defstep <= 0.0) {
            throw ::std::invalid_argument(kInitialStepPositiveMsg);
        }
        this->def_step_size_ = defstep;
        // HW auto-initdt stays enabled — user can opt out via
        // set_initial_step_size() or set_auto_initial_dt(false).
        this->use_hairer_wanner_initdt_ = true;

        switch (alg) {
        case IVPAlg::DOPRI54:
            this->rk_method_ = IVPAlg::DOPRI54;
            this->error_order_ = 4;
            // DOPRI5 is the transcription-side tableau tag; DOPRI54 is the
            // runtime user-facing name.
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

        // default_controller_for returns a default-constructed controller —
        // its Params are known-valid by construction, so skip the redundant
        // validate_controller std::visit. If a future overload admits a
        // user-supplied controller, validate it at that call site instead.
        this->controller_variant_ = default_controller_for(alg);
        reset_controller(this->controller_variant_);
    }

    template <IVPAlg RKOp>
    void init_stepper_and_controller(const DODE &odet, bool usecontrol,
                                     const GenericFunction<-1, -1> &ucon,
                                     const ControlIndexType &varlocs_t) {

        this->ode_ = odet;
        Eigen::VectorXi varlocs = this->get_var_locs(varlocs_t);
        this->set_io_rows(this->ode_.input_rows() + 1, this->ode_.input_rows());

        this->use_controller_ = usecontrol;
        if (this->use_controller_) {
            this->controller_spec_.emplace(ucon, varlocs);
        } else {
            // No user controller — leave the optional empty. std::optional
            // copies cleanly without invoking GenericFunction's null-throw,
            // so the Integrator stays copyable.
            this->controller_spec_.reset();
        }

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

    int error_order_ = 7;
    double def_step_size_ = 0.1;
    double max_step_change_ = 10.0;
    // Julia-style maxiters cap — safety net against runaway rejection loops or
    // pathological dynamics. The adaptive controller handles step-size control;
    // there is no hard min/max on dt, only a hard cap on iterations.
    int max_steps_ = 1'000'000;
    bool adaptive_ = true;
    double event_tol_ = 1.0e-6;
    int max_event_iters_ = 10;
    bool vectorize_batch_calls_ = true;

    // Prototype controller. Read-only during integrate — workers clone via
    // make_worker_controller(). Mutated only by set_method / set_controller,
    // neither of which is concurrent-safe with an in-flight integrate.
    ControllerVariant controller_variant_;
    ErrorNormType error_norm_type_ = ErrorNormType::RMS;

    /// Build a per-call controller for a worker (or main thread): clone the
    /// prototype and reset internal state so it starts from first-step
    /// semantics. Centralizing avoids the "forgot the reset()" class of
    /// bug in per-lane controller construction.
    ControllerVariant make_worker_controller() const {
        ControllerVariant c = this->controller_variant_;
        reset_controller(c);
        return c;
    }

    /// Batch variant of make_worker_controller for SIMD/vectorized-batch paths
    /// that need one reset controller per trajectory.
    std::vector<ControllerVariant> make_worker_controllers(int n) const {
        std::vector<ControllerVariant> ctrls;
        ctrls.reserve(n);
        for (int i = 0; i < n; ++i)
            ctrls.push_back(this->make_worker_controller());
        return ctrls;
    }

    /// Compose and throw the aggregated segment-failure diagnostic.
    /// `primary_tag` identifies which side raised the primary failure
    /// (main-thread vs. worker segment); `extras` holds the drained-future
    /// what() strings. Caps at kMaxExtras to keep the composite message
    /// bounded under large n_parts.
    [[noreturn]] static void
    throw_aggregated_segment_failure(const char *primary_tag, const char *extras_tag,
                                     const std::string &primary_msg,
                                     const std::vector<std::string> &extras) {
        constexpr std::size_t kMaxExtras = 5;
        std::string joined = std::string("integrate_stm_parallel: ") + primary_tag + ": " +
                             primary_msg + "; " + extras_tag + " (" +
                             std::to_string(extras.size()) + "): ";
        std::size_t shown = std::min(extras.size(), kMaxExtras);
        for (std::size_t k = 0; k < shown; ++k) {
            if (k)
                joined += " | ";
            joined += extras[k];
        }
        if (extras.size() > kMaxExtras) {
            joined += " | ... and " + std::to_string(extras.size() - kMaxExtras) + " more";
        }
        throw std::runtime_error(joined);
    }

    /// Shared message for the two setter paths that reject a non-positive
    /// initial step (constructor path via set_method and the explicit
    /// set_initial_step_size). Kept in one place so they can't drift.
    static constexpr const char *kInitialStepPositiveMsg =
        "Initial step size must be strictly positive (backward integration is driven by tf < t0, "
        "not by sign of h; zero would divide-by-zero in the fixed-step path).";

    /// Wrap `f()` so any std::exception it raises is rethrown with a
    /// `"trajectory i: "` prefix; preserves the thrown type across the
    /// four std::exception subclasses nanobind maps specifically
    /// (invalid_argument, out_of_range, domain_error, logic_error);
    /// anything else collapses to std::runtime_error.
    template <class F> static auto decorate_trajectory(int i, F &&f) {
        auto decorate = [i](const char *w) { return "trajectory " + std::to_string(i) + ": " + w; };
        try {
            return std::forward<F>(f)();
        } catch (const std::invalid_argument &e) {
            throw std::invalid_argument(decorate(e.what()));
        } catch (const std::out_of_range &e) {
            throw std::out_of_range(decorate(e.what()));
        } catch (const std::domain_error &e) {
            throw std::domain_error(decorate(e.what()));
        } catch (const std::logic_error &e) {
            throw std::logic_error(decorate(e.what()));
        } catch (const std::exception &e) {
            throw std::runtime_error(decorate(e.what()));
        }
    }

    // `mutable` so const-qualified public wrappers can write back post-call
    // (VectorFunction::compute contract forces compute_impl to be const).
    // Parallel paths that can't safely write back document that via
    // get_naccept/get_nreject docstrings.
    // int64_t so batch aggregation across thousands of trajectories ×
    // millions of steps cannot silently wrap (the per-trajectory `int`
    // locals threaded through driver loops are bounded by `max_steps`,
    // but their sum on the BatchCounterWriteback writeback path is not).
    mutable int64_t naccept_ = 0;
    mutable int64_t nreject_ = 0;

    // Writes local na/nr into the member counters on scope exit (success
    // OR exception unwind). Without this, a failed integrate() would leave
    // `get_naccept()` returning the previous call's value — misleading when
    // debugging where a long run blew up. Mutable members allow a
    // const-Integrator reference to write through.
    struct CounterWriteback {
        const Integrator &integ;
        int &na;
        int &nr;
        ~CounterWriteback() noexcept {
            integ.naccept_ = na;
            integ.nreject_ = nr;
        }
    };

    // Vector analogue of CounterWriteback: sums per-trajectory counters
    // into the member counters on scope exit. On exception the members
    // reflect only trajectories that completed before the first failure;
    // unstarted lanes contribute zero.
    struct BatchCounterWriteback {
        const Integrator &integ;
        const std::vector<int> &nacc;
        const std::vector<int> &nrej;
        ~BatchCounterWriteback() noexcept {
            integ.naccept_ = std::accumulate(nacc.begin(), nacc.end(), int64_t{0});
            integ.nreject_ = std::accumulate(nrej.begin(), nrej.end(), int64_t{0});
        }
    };

    struct EventCounterWriteback {
        const Integrator &integ;
        int &nfailed;
        // Counter is exposed via get_failed_event_count(); callers that
        // care about silent nullopt slots should poll that instead.
        ~EventCounterWriteback() noexcept { integ.n_failed_event_refinements_ = nfailed; }
    };

    struct BatchEventCounterWriteback {
        const Integrator &integ;
        const std::vector<int> &nfailed;
        ~BatchEventCounterWriteback() noexcept {
            integ.n_failed_event_refinements_ =
                std::accumulate(nfailed.begin(), nfailed.end(), int64_t{0});
        }
    };

    // Count of event refinements that failed both bisect+Newton passes.
    // Such crossings occupy std::nullopt slots in the returned eventstates
    // vector (1:1 with eventtimes); the counter is a cheap aggregate for
    // callers that only want to know "did anything drop?". Published by
    // EventCounterWriteback / BatchEventCounterWriteback on scope exit
    // (success or unwind), so the value is always current after any
    // public find_events / integrate call.
    mutable int64_t n_failed_event_refinements_ = 0;

    // Flipped off by set_initial_step_size() so a caller-supplied initial
    // step is respected (principle of least surprise).
    bool use_hairer_wanner_initdt_ = true;

    ODEDeriv<double> abs_tols_;
    ODEDeriv<double> rel_tols_;

    // Tolerance setters size themselves from `ode_.x_vars()`, which is
    // populated by `set_method`. Constructors call `set_method` before any
    // tolerance setter for that reason.
    void set_abs_tol(double tol) {
        if (!std::isfinite(tol) || tol < 0.0) {
            throw std::invalid_argument("abs_tol must be finite and >= 0; got " +
                                        std::to_string(tol));
        }
        this->abs_tols_.setConstant(this->ode_.x_vars(), tol);
    }
    void set_rel_tol(double tol) {
        if (!std::isfinite(tol) || tol < 0.0) {
            throw std::invalid_argument("rel_tol must be finite and >= 0; got " +
                                        std::to_string(tol));
        }
        this->rel_tols_.setConstant(this->ode_.x_vars(), tol);
    }

    void set_abs_tols(ODEDeriv<double> tol) {
        if (tol.size() != this->ode_.x_vars()) {
            throw std::invalid_argument("Incorrectly sized tolerance vector.");
        }
        for (Eigen::Index i = 0; i < tol.size(); ++i) {
            if (!std::isfinite(tol[i]) || tol[i] < 0.0) {
                throw std::invalid_argument("abs_tols[" + std::to_string(i) +
                                            "] must be finite and >= 0; got " +
                                            std::to_string(tol[i]));
            }
        }
        this->abs_tols_ = tol;
    }
    void set_rel_tols(ODEDeriv<double> tol) {
        if (tol.size() != this->ode_.x_vars()) {
            throw std::invalid_argument("Incorrectly sized tolerance vector.");
        }
        for (Eigen::Index i = 0; i < tol.size(); ++i) {
            if (!std::isfinite(tol[i]) || tol[i] < 0.0) {
                throw std::invalid_argument("rel_tols[" + std::to_string(i) +
                                            "] must be finite and >= 0; got " +
                                            std::to_string(tol[i]));
            }
        }
        this->rel_tols_ = tol;
    }

    ODEDeriv<double> get_abs_tols() const { return this->abs_tols_; }
    ODEDeriv<double> get_rel_tols() const { return this->rel_tols_; }

    void set_controller(IVPController kind) {
        // Same reasoning as set_method: controller_defaults_for returns a
        // default-constructed controller whose Params are valid by design.
        // Skip validate_controller to avoid the extra std::visit per call.
        this->controller_variant_ = controller_defaults_for(this->rk_method_, kind);
        reset_controller(this->controller_variant_);
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

    int64_t get_naccept() const { return this->naccept_; }
    int64_t get_nreject() const { return this->nreject_; }

    /// Number of event crossings whose bisect+Newton refinement failed
    /// (both passes) during the most recent `integrate*` call that took
    /// events. `eventstates` from `find_events` keeps 1:1 alignment with
    /// `eventtimes` and marks each such crossing with `std::nullopt`; this
    /// counter is a cheap summary of how many nullopts are present. Reset
    /// to 0 at the start of every `find_events` invocation.
    [[nodiscard]] int64_t get_failed_event_count() const {
        return this->n_failed_event_refinements_;
    }

    void set_auto_initial_dt(bool on) { this->use_hairer_wanner_initdt_ = on; }
    bool get_auto_initial_dt() const { return this->use_hairer_wanner_initdt_; }

    /// Set the initial step size used when Hairer-Wanner auto-initdt is off
    /// (and as the fixed step when adaptive_ is false). Calling this flips HW
    /// auto-initdt off so the caller's value is respected — re-enable with
    /// set_auto_initial_dt(true) if desired.
    void set_initial_step_size(double h) {
        if (!std::isfinite(h) || h <= 0.0) {
            throw ::std::invalid_argument(kInitialStepPositiveMsg);
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

    /// Upper bound on per-step growth (dt_next / dt_current). Must be
    /// strictly greater than 1 — at or below 1 degenerates the clamp into
    /// a divide-or-shrink that drives dt to zero.
    void set_max_step_change(double v) {
        if (!(v > 1.0) || !std::isfinite(v)) {
            throw std::invalid_argument("max_step_change must be finite and > 1; got " +
                                        std::to_string(v));
        }
        this->max_step_change_ = v;
    }
    double get_max_step_change() const { return this->max_step_change_; }

    /// Bisect/Newton tolerance for event-crossing refinement. Must be
    /// strictly positive — zero or negative would make the refinement
    /// loop either never terminate or exit on the first iteration.
    void set_event_tol(double v) {
        if (!(v > 0.0) || !std::isfinite(v)) {
            throw std::invalid_argument("event_tol must be finite and > 0; got " +
                                        std::to_string(v));
        }
        this->event_tol_ = v;
    }
    double get_event_tol() const { return this->event_tol_; }

    /// Max Newton iterations for event refinement. Must be >= 1 — zero
    /// silently skips the Newton polish, leaving only the two-iter
    /// bisect bracket and degrading refinement quality without any
    /// surface signal.
    void set_max_event_iters(int n) {
        if (n < 1) {
            throw std::invalid_argument("max_event_iters must be >= 1; got " + std::to_string(n));
        }
        this->max_event_iters_ = n;
    }
    int get_max_event_iters() const { return this->max_event_iters_; }

    /// Active RK method. Exposed so other Integrator instantiations can copy
    /// settings across different DODE template parameters without needing to
    /// be declared a friend.
    IVPAlg get_method() const { return this->rk_method_; }

    /// Shallow-copy user-settable config from a source Integrator (possibly
    /// over a different DODE template parameter); ODE-structure state and
    /// per-call output counters are not copied.
    template <class OtherDODE> void copy_settings_from(const Integrator<OtherDODE> &src) {
        const bool target_uses_controller = this->use_controller_;
        // copy_settings_from preserves the target's own controller wiring —
        // varlocs and the user-controller function are ODE-structure state,
        // not user config, so they don't carry across from src. When the
        // target uses a controller, controller_spec_ has_value (invariant
        // tied to use_controller_); the no-controller branch ignores
        // ucon/varlocs entirely so the default-constructed placeholders
        // are never copied by set_method (which takes by const&).
        ControllerType target_controller_source;
        Eigen::VectorXi target_controller_varlocs;
        if (target_uses_controller) {
            target_controller_source = this->controller_spec_->first;
            target_controller_varlocs = this->controller_spec_->second;
        }

        this->set_method(src.get_method(), this->ode_, src.def_step_size_, target_uses_controller,
                         target_controller_source, target_controller_varlocs);

        this->adaptive_ = src.adaptive_;
        this->abs_tols_ = src.abs_tols_;
        this->rel_tols_ = src.rel_tols_;
        this->max_steps_ = src.get_max_steps();
        this->enable_vectorization_ = src.enable_vectorization_;
        this->vectorize_batch_calls_ = src.vectorize_batch_calls_;
        this->controller_variant_ = src.controller_variant_;
        this->error_norm_type_ = src.error_norm_type_;
        this->max_step_change_ = src.max_step_change_;
        this->event_tol_ = src.event_tol_;
        this->max_event_iters_ = src.max_event_iters_;
        this->use_hairer_wanner_initdt_ = src.use_hairer_wanner_initdt_;
        this->def_step_size_ = src.def_step_size_;
    }

    /////////////////////////////////////////////////////////////////////////////////////

  protected:
    static ControllerVariant default_controller_for(IVPAlg alg) {
        switch (alg) {
        case IVPAlg::DOPRI54: {
            PIController c;
            c.beta1_ = 17.0 / 100.0; // DOPRI54 override: 1/order - 3·β₂/4 = 17/100
            c.beta2_ = 4.0 / 100.0;
            c.gamma_ = 0.9;
            c.qmin_ = 1.0 / 5.0;
            c.qmax_ = 10.0;
            return c;
        }
        case IVPAlg::DOPRI87: {
            IController c;
            c.gamma_ = 0.9;
            c.qmin_ = 1.0 / 3.0; // DP8 override
            c.qmax_ = 6.0;
            return c;
        }
        case IVPAlg::Tsit5: {
            PIController c;
            c.beta1_ = 7.0 / 50.0; // Julia generic: 7/(10·order), order=5
            c.beta2_ = 2.0 / 25.0; // Julia generic: 2/(5·order),  order=5
            return c;
        }
        case IVPAlg::BS3: {
            PIController c;
            c.beta1_ = 7.0 / 30.0; // order=3
            c.beta2_ = 2.0 / 15.0;
            return c;
        }
        case IVPAlg::BS5: {
            PIController c;
            c.beta1_ = 7.0 / 50.0; // order=5
            c.beta2_ = 2.0 / 25.0;
            return c;
        }
        case IVPAlg::Vern7: {
            PIController c;
            c.beta1_ = 1.0 / 10.0; // 7/(10·7)
            c.beta2_ = 2.0 / 35.0; // 2/(5·7)
            return c;
        }
        case IVPAlg::Vern8: {
            PIController c;
            c.beta1_ = 7.0 / 80.0; // order=8
            c.beta2_ = 1.0 / 20.0; // 2/(5·8)
            return c;
        }
        case IVPAlg::Vern9: {
            PIController c;
            c.beta1_ = 7.0 / 90.0; // order=9
            c.beta2_ = 2.0 / 45.0; // 2/(5·9)
            return c;
        }
        default:
            throw std::logic_error("default_controller_for: algorithm not user-selectable");
        }
    }

    /// Build a controller of the requested `kind` with per-method tuned
    /// knobs when the method has them. When the method's preferred default
    /// kind differs from `kind` (e.g. user picks PI on DOPRI87, whose default
    /// is I), fall back to the struct's generic defaults for that kind —
    /// there is no Julia-documented method-tuning for the off-preference
    /// combinations. PID is never method-tuned and always returns generic.
    static ControllerVariant controller_defaults_for(IVPAlg alg, IVPController kind) {
        auto preferred = default_controller_for(alg);
        switch (kind) {
        case IVPController::I:
            if (std::holds_alternative<IController>(preferred))
                return preferred;
            return IController{};
        case IVPController::PI:
            if (std::holds_alternative<PIController>(preferred))
                return preferred;
            return PIController{};
        case IVPController::PID:
            return PIDController{};
        }
        // Unreachable — switch is exhaustive over IVPController.
        throw std::logic_error("controller_defaults_for: unknown IVPController kind");
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

        integrators::AdaptiveConfig cfg{this->error_order_,
                                        this->error_norm_type_,
                                        this->def_step_size_,
                                        this->max_step_change_,
                                        this->max_steps_,
                                        this->adaptive_,
                                        this->use_hairer_wanner_initdt_};

        auto update_control_fn = [this](auto &s) { this->update_control(s); };

        return with_public_ivp_alg(this->rk_method_, [&](auto alg_tag) -> Output<double> {
            constexpr IVPAlg Alg = alg_tag.value;
            using Driver = integrators::AdaptiveDriver<Alg, DODE, double>;
            Driver driver;
            typename Driver::IO io{naccept,     nreject,        events, eventtimes, storestates,
                                   storederivs, storemidpoints, states, derivs};
            return driver.integrate(this->ode_, x, tf, cfg, this->abs_tols_, this->rel_tols_,
                                    controller, io, update_control_fn);
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

        integrators::AdaptiveConfig cfg{this->error_order_,
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
                using Driver = integrators::ParallelDriver<Alg, DODE>;
                Driver driver;
                typename Driver::IO io{nacc,           nrej,        events,
                                       eventtimes_s,   storestates, storederivs,
                                       storemidpoints, states_s,    derivs_s};
                return driver.integrate(this->ode_, xs, tfs, cfg, this->abs_tols_, this->rel_tols_,
                                        controllers, io, update_control_fn);
            });
    }

    /// \name Thread-safe `_core` helpers
    /// Caller owns controller + counters by reference; function never
    /// touches `this->controller_variant_` / `naccept_` / `nreject_` —
    /// this is what makes concurrent worker invocations race-free.
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
        int nfailed = 0;
        return this->integrate_core(x0, tf, events, controller, naccept, nreject, nfailed);
    }

    IntegEventRet integrate_core(const ODEState<double> &x0, double tf,
                                 const std::vector<EventPack> &events,
                                 ControllerVariant &controller, int &naccept, int &nreject,
                                 int &nfailed_event_refinements) const {
        nfailed_event_refinements = 0;
        ODEState<double> xf;
        std::vector<std::vector<Eigen::Vector2d>> eventtimes;
        bool storestates = true;
        bool storederivs = true;
        bool storemidpoints = true;
        std::vector<ODEState<double>> xs;
        std::vector<ODEDeriv<double>> d_xs;
        xf = this->integrate_impl(x0, tf, events, eventtimes, storestates, storederivs,
                                  storemidpoints, xs, d_xs, controller, naccept, nreject);
        EventLocsType eventlocs(events.size());
        for (auto etimes : eventtimes) {
            if (etimes.size() > 0) {
                auto tab = this->make_table(xs, d_xs, false);
                eventlocs =
                    this->find_events_counted(tab, events, eventtimes, nfailed_event_refinements);
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
        int nfailed = 0;
        return this->integrate_dense_core(x0, tf, events, alloutput, controller, naccept, nreject,
                                          nfailed);
    }

    DenseEventRet integrate_dense_core(const ODEState<double> &x0, double tf,
                                       const std::vector<EventPack> &events, bool alloutput,
                                       ControllerVariant &controller, int &naccept, int &nreject,
                                       int &nfailed_event_refinements) const {
        nfailed_event_refinements = 0;
        ODEState<double> xf;
        std::vector<std::vector<Eigen::Vector2d>> eventtimes;
        bool storestates = true;
        bool storederivs = true;
        bool storemidpoints = true;
        std::vector<ODEState<double>> xs;
        std::vector<ODEDeriv<double>> d_xs;
        xf = this->integrate_impl(x0, tf, events, eventtimes, storestates, storederivs,
                                  storemidpoints, xs, d_xs, controller, naccept, nreject);
        EventLocsType eventlocs(events.size());
        for (auto etimes : eventtimes) {
            if (etimes.size() > 0) {
                auto tab = this->make_table(xs, d_xs, false);
                eventlocs =
                    this->find_events_counted(tab, events, eventtimes, nfailed_event_refinements);
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
        int nfailed = 0;
        return this->integrate_dense_core(x0, tf, n, events, controller, naccept, nreject, nfailed);
    }

    DenseEventRet integrate_dense_core(const ODEState<double> &x0, double tf, int n,
                                       const std::vector<EventPack> &events,
                                       ControllerVariant &controller, int &naccept, int &nreject,
                                       int &nfailed_event_refinements) const {
        nfailed_event_refinements = 0;
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
        EventLocsType eventlocs =
            this->find_events_counted(tab, events, eventtimes, nfailed_event_refinements);
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
                                   ControllerVariant &controller, int &naccept, int &nreject,
                                   int &nfailed_event_refinements) const {
        auto [xs, eventlocs] = this->integrate_dense_core(
            x0, tf, events, false, controller, naccept, nreject, nfailed_event_refinements);
        Jacobian<double> jx = this->calculate_jacobian(xs);
        return std::tuple{xs.back(), jx, eventlocs};
    }
    /// @}

  public:
    /// Refine bracketed event crossings in `eventtimes` against an
    /// interpolation table. Returns a vector aligned 1:1 with `eventtimes`:
    /// each entry holds the refined ODEState, or std::nullopt when neither
    /// the fast bisect+Newton pass nor the wider-bracket retry resolves the
    /// crossing. `n_failed_event_refinements` increments once per nullopt.
    /// Does NOT touch `n_failed_event_refinements_` on the integrator — pair
    /// with `EventCounterWriteback` (or use the sibling `find_events()`
    /// shorthand) when member-state publication into get_failed_event_count()
    /// is required.
    EventLocsType find_events_counted(std::shared_ptr<LGLInterpTable> tab,
                                      const std::vector<EventPack> &events,
                                      const std::vector<std::vector<Eigen::Vector2d>> &eventtimes,
                                      int &n_failed_event_refinements) const {
        n_failed_event_refinements = 0;
        return EventHandler::refine_events<ODEState<double>>(
            tab, events, eventtimes, this->ode_.input_rows(), this->max_event_iters_,
            this->event_tol_, n_failed_event_refinements);
    }

    /// Refine bracketed event crossings and publish the failure count to
    /// get_failed_event_count(). Parallel event paths use find_events_counted()
    /// with per-worker counters instead of this shared-member wrapper.
    EventLocsType find_events(std::shared_ptr<LGLInterpTable> tab,
                              const std::vector<EventPack> &events,
                              const std::vector<std::vector<Eigen::Vector2d>> &eventtimes) const {
        int nfailed = 0;
        EventCounterWriteback _writeback{*this, nfailed};
        return this->find_events_counted(tab, events, eventtimes, nfailed);
    }

    /// Build an LGLInterpTable from a dense state sequence for use with
    /// find_events without re-running a dense integrate.
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

  protected:
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
        CounterWriteback _writeback{*this, na, nr};
        return this->integrate_core(x0, tf, ctrl, na, nr);
    }

    std::vector<ODEState<double>> integrate(const std::vector<ODEState<double>> &x0s,
                                            const Eigen::VectorXd &tfs) const {
        // Empty-batch input is a no-op. Without this short-circuit the
        // vectorized branch routes to ParallelDriver, which rejects empty
        // input — making the API behavior depend on vectorize_batch_calls_.
        if (x0s.empty()) {
            return {};
        }
        const int n = static_cast<int>(x0s.size());
        std::vector<ControllerVariant> ctrls = this->make_worker_controllers(n);
        std::vector<int> nacc(n, 0), nrej(n, 0);
        BatchCounterWriteback _writeback{*this, nacc, nrej};

        if (!vectorize_batch_calls_) {
            std::vector<ODEState<double>> xfs(n);
            for (int i = 0; i < n; i++) {
                xfs[i] = this->integrate_core(x0s[i], tfs[i], ctrls[i], nacc[i], nrej[i]);
            }
            return xfs;
        }
        std::vector<EventPack> events;
        std::vector<std::vector<std::vector<Eigen::Vector2d>>> eventtimes;
        bool storestates = false;
        bool storederivs = false;
        bool storemidpoints = false;
        std::vector<std::vector<ODEState<double>>> xs;
        std::vector<std::vector<ODEDeriv<double>>> d_xs;
        return integrate_impl_vectorized(x0s, tfs, events, eventtimes, storestates, storederivs,
                                         storemidpoints, xs, d_xs, ctrls, nacc, nrej);
    }

    std::vector<STMRet> integrate_stm(const std::vector<ODEState<double>> &x0s,
                                      const Eigen::VectorXd &tfs) const {
        if (x0s.empty()) {
            return {};
        }
        const int n = static_cast<int>(x0s.size());
        std::vector<ControllerVariant> ctrls = this->make_worker_controllers(n);
        std::vector<int> nacc(n, 0), nrej(n, 0);
        BatchCounterWriteback _writeback{*this, nacc, nrej};

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
        return rets;
    }

    std::vector<std::tuple<ODEState<double>, Jacobian<double>, Hessian<double>>>
    integrate_stm2(const std::vector<ODEState<double>> &x0s, const Eigen::VectorXd &tfs,
                   const std::vector<ODEState<double>> &lfs) const {
        if (x0s.empty()) {
            return {};
        }
        const int n = static_cast<int>(x0s.size());
        std::vector<ControllerVariant> ctrls = this->make_worker_controllers(n);
        std::vector<int> nacc(n, 0), nrej(n, 0);
        BatchCounterWriteback _writeback{*this, nacc, nrej};

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
        return rets;
    }

    IntegEventRet integrate(const ODEState<double> &x0, double tf,
                            const std::vector<EventPack> &events) const {
        ControllerVariant ctrl = this->make_worker_controller();
        int na = 0, nr = 0;
        int nf = 0;
        CounterWriteback _writeback{*this, na, nr};
        EventCounterWriteback _event_writeback{*this, nf};
        return this->integrate_core(x0, tf, events, ctrl, na, nr, nf);
    }

    template <class... Args>
    auto integrate_parallel_impl(const std::vector<ODEState<double>> &x0s,
                                 const Eigen::VectorXd &tfs, int n_parts,
                                 std::vector<int> &nacc_out, std::vector<int> &nrej_out,
                                 Args &&...args)
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
                results[i] = decorate_trajectory(
                    i, [&] { return this->integrate_core(x0s[i], tfs[i], args..., ctrl, na, nr); });
                nacc_out[i] = na;
                nrej_out[i] = nr;
            }
        };

        tycho::utils::parallel_blocks(n, job, n_parts);
        return results;
    }

    std::vector<STMEventRet> integrate_stm_parallel_events_impl(
        const std::vector<ODEState<double>> &x0s, const Eigen::VectorXd &tfs, int n_parts,
        std::vector<int> &nacc_out, std::vector<int> &nrej_out, std::vector<int> &nfailed_out,
        const std::vector<EventPack> &events) {
        if (x0s.size() != tfs.size()) {
            throw std::invalid_argument(
                "List of initial states and final times must be the same size");
        }

        int n = x0s.size();
        std::vector<STMEventRet> results(n);

        auto job = [&](int start, int stop) {
            ControllerVariant ctrl;
            int na = 0, nr = 0, nf = 0;
            for (int i = start; i < stop; i++) {
                ctrl = this->make_worker_controller();
                na = 0;
                nr = 0;
                nf = 0;
                results[i] = decorate_trajectory(i, [&] {
                    return this->integrate_stm_core(x0s[i], tfs[i], events, ctrl, na, nr, nf);
                });
                nacc_out[i] = na;
                nrej_out[i] = nr;
                nfailed_out[i] = nf;
            }
        };

        tycho::utils::parallel_blocks(n, job, n_parts);
        return results;
    }

    std::vector<DenseEventRet> integrate_dense_parallel_events_impl_n(
        const std::vector<ODEState<double>> &x0s, const Eigen::VectorXd &tfs,
        const std::vector<int> &ns, int n_parts, std::vector<int> &nacc_out,
        std::vector<int> &nrej_out, std::vector<int> &nfailed_out,
        const std::vector<EventPack> &events) {
        if (x0s.size() != tfs.size()) {
            throw std::invalid_argument(
                "List of initial states and final times must be the same size");
        }
        if (x0s.size() != ns.size()) {
            throw std::invalid_argument(
                "List of initial states and state numbers must be the same size");
        }

        int n = x0s.size();
        std::vector<DenseEventRet> results(n);

        auto job = [&](int start, int stop) {
            ControllerVariant ctrl;
            int na = 0, nr = 0, nf = 0;
            for (int i = start; i < stop; i++) {
                ctrl = this->make_worker_controller();
                na = 0;
                nr = 0;
                nf = 0;
                results[i] = decorate_trajectory(i, [&] {
                    return this->integrate_dense_core(x0s[i], tfs[i], ns[i], events, ctrl, na, nr,
                                                      nf);
                });
                nacc_out[i] = na;
                nrej_out[i] = nr;
                nfailed_out[i] = nf;
            }
        };

        tycho::utils::parallel_blocks(n, job, n_parts);
        return results;
    }

    std::vector<DenseEventRet> integrate_dense_parallel_events_impl(
        const std::vector<ODEState<double>> &x0s, const Eigen::VectorXd &tfs, int n_parts,
        std::vector<int> &nacc_out, std::vector<int> &nrej_out, std::vector<int> &nfailed_out,
        const std::vector<EventPack> &events, bool alloutput) {
        if (x0s.size() != tfs.size()) {
            throw std::invalid_argument(
                "List of initial states and final times must be the same size");
        }

        int n = x0s.size();
        std::vector<DenseEventRet> results(n);

        auto job = [&](int start, int stop) {
            ControllerVariant ctrl;
            int na = 0, nr = 0, nf = 0;
            for (int i = start; i < stop; i++) {
                ctrl = this->make_worker_controller();
                na = 0;
                nr = 0;
                nf = 0;
                results[i] = decorate_trajectory(i, [&] {
                    return this->integrate_dense_core(x0s[i], tfs[i], events, alloutput, ctrl, na,
                                                      nr, nf);
                });
                nacc_out[i] = na;
                nrej_out[i] = nr;
                nfailed_out[i] = nf;
            }
        };

        tycho::utils::parallel_blocks(n, job, n_parts);
        return results;
    }

    std::vector<IntegEventRet> integrate_parallel_events_impl(
        const std::vector<ODEState<double>> &x0s, const Eigen::VectorXd &tfs, int n_parts,
        std::vector<int> &nacc_out, std::vector<int> &nrej_out, std::vector<int> &nfailed_out,
        const std::vector<EventPack> &events) {
        if (x0s.size() != tfs.size()) {
            throw std::invalid_argument(
                "List of initial states and final times must be the same size");
        }

        int n = x0s.size();
        std::vector<IntegEventRet> results(n);

        auto job = [&](int start, int stop) {
            ControllerVariant ctrl;
            int na = 0, nr = 0, nf = 0;
            for (int i = start; i < stop; i++) {
                ctrl = this->make_worker_controller();
                na = 0;
                nr = 0;
                nf = 0;
                results[i] = decorate_trajectory(i, [&] {
                    return this->integrate_core(x0s[i], tfs[i], events, ctrl, na, nr, nf);
                });
                nacc_out[i] = na;
                nrej_out[i] = nr;
                nfailed_out[i] = nf;
            }
        };

        tycho::utils::parallel_blocks(n, job, n_parts);
        return results;
    }

    /// Parallel batch integrate. Accept/reject counters are summed across
    /// lanes into `get_naccept()` / `get_nreject()` on scope exit via
    /// BatchCounterWriteback (RAII-safe on unwind).
    std::vector<IntegRet> integrate_parallel(const std::vector<ODEState<double>> &x0s,
                                             const Eigen::VectorXd &tfs, int n_parts) {
        const int n = static_cast<int>(x0s.size());
        std::vector<int> nacc(n, 0), nrej(n, 0);
        BatchCounterWriteback _writeback{*this, nacc, nrej};
        return this->integrate_parallel_impl(x0s, tfs, n_parts, nacc, nrej);
    }
    /// Parallel batch integrate with events. @see integrate_parallel.
    std::vector<IntegEventRet> integrate_parallel(const std::vector<ODEState<double>> &x0s,
                                                  const Eigen::VectorXd &tfs,
                                                  const std::vector<EventPack> &events,
                                                  int n_parts) {
        const int n = static_cast<int>(x0s.size());
        std::vector<int> nacc(n, 0), nrej(n, 0);
        std::vector<int> nfailed(n, 0);
        BatchCounterWriteback _writeback{*this, nacc, nrej};
        BatchEventCounterWriteback _event_writeback{*this, nfailed};
        return this->integrate_parallel_events_impl(x0s, tfs, n_parts, nacc, nrej, nfailed, events);
    }
    /////////////////////////////////////////////////////////////////////////////////////

    DenseRet integrate_dense(const ODEState<double> &x0, double tf) const {
        ControllerVariant ctrl = this->make_worker_controller();
        int na = 0, nr = 0;
        CounterWriteback _writeback{*this, na, nr};
        return this->integrate_dense_core(x0, tf, ctrl, na, nr);
    }

    DenseEventRet integrate_dense(const ODEState<double> &x0, double tf,
                                  const std::vector<EventPack> &events, bool alloutput) const {
        ControllerVariant ctrl = this->make_worker_controller();
        int na = 0, nr = 0;
        int nf = 0;
        CounterWriteback _writeback{*this, na, nr};
        EventCounterWriteback _event_writeback{*this, nf};
        return this->integrate_dense_core(x0, tf, events, alloutput, ctrl, na, nr, nf);
    }

    DenseEventRet integrate_dense(const ODEState<double> &x0, double tf, int n,
                                  const std::vector<EventPack> &events) const {
        ControllerVariant ctrl = this->make_worker_controller();
        int na = 0, nr = 0;
        int nf = 0;
        CounterWriteback _writeback{*this, na, nr};
        EventCounterWriteback _event_writeback{*this, nf};
        return this->integrate_dense_core(x0, tf, n, events, ctrl, na, nr, nf);
    }

    DenseRet integrate_dense(const ODEState<double> &x0, double tf, int n) const {
        ControllerVariant ctrl = this->make_worker_controller();
        int na = 0, nr = 0;
        CounterWriteback _writeback{*this, na, nr};
        return this->integrate_dense_core(x0, tf, n, ctrl, na, nr);
    }

    DenseRet integrate_dense(const ODEState<double> &x0, double tf, int num_states,
                             std::function<bool(ConstEigenRef<Eigen::VectorXd>)> exitfun) const {
        ControllerVariant ctrl = this->make_worker_controller();
        int na = 0, nr = 0;
        CounterWriteback _writeback{*this, na, nr};

        VectorX<double> ts = VectorX<double>::LinSpaced(num_states, x0[this->ode_.t_var()], tf);
        std::vector<ODEState<double>> xout;
        xout.reserve(num_states);
        xout.push_back(x0);
        for (int i = 1; i < num_states; i++) {
            xout.push_back(this->integrate_core(xout[i - 1], ts[i], ctrl, na, nr));
            if (exitfun(xout.back()))
                break;
        }
        return xout;
    }

    template <class... Args>
    auto integrate_dense_parallel_impl(const std::vector<ODEState<double>> &x0s,
                                       const Eigen::VectorXd &tfs, int n_parts,
                                       std::vector<int> &nacc_out, std::vector<int> &nrej_out,
                                       Args &&...args)
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
                results[i] = decorate_trajectory(i, [&] {
                    return this->integrate_dense_core(x0s[i], tfs[i], args..., ctrl, na, nr);
                });
                nacc_out[i] = na;
                nrej_out[i] = nr;
            }
        };

        tycho::utils::parallel_blocks(n, job, n_parts);
        return results;
    }

    /// Parallel batch dense integrate. @see integrate_parallel for counters.
    std::vector<DenseRet> integrate_dense_parallel(const std::vector<ODEState<double>> &x0s,
                                                   const Eigen::VectorXd &tfs, int n_parts) {
        const int n = static_cast<int>(x0s.size());
        std::vector<int> nacc(n, 0), nrej(n, 0);
        BatchCounterWriteback _writeback{*this, nacc, nrej};
        return this->integrate_dense_parallel_impl(x0s, tfs, n_parts, nacc, nrej);
    }
    /// Parallel batch dense integrate with events. @see integrate_parallel.
    std::vector<DenseEventRet> integrate_dense_parallel(const std::vector<ODEState<double>> &x0s,
                                                        const Eigen::VectorXd &tfs,
                                                        const std::vector<EventPack> &events,
                                                        int n_parts) {
        const int n = static_cast<int>(x0s.size());
        std::vector<int> nacc(n, 0), nrej(n, 0);
        std::vector<int> nfailed(n, 0);
        BatchCounterWriteback _writeback{*this, nacc, nrej};
        BatchEventCounterWriteback _event_writeback{*this, nfailed};
        return this->integrate_dense_parallel_events_impl(x0s, tfs, n_parts, nacc, nrej, nfailed,
                                                          events, false);
    }
    /////////////////////////////////////////////////////////////////////////////////////

    template <class... Args>
    auto integrate_dense_parallel_impl_n(const std::vector<ODEState<double>> &x0s,
                                         const Eigen::VectorXd &tfs, const std::vector<int> &ns,
                                         int n_parts, std::vector<int> &nacc_out,
                                         std::vector<int> &nrej_out, Args &&...args)
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
                results[i] = decorate_trajectory(i, [&] {
                    return this->integrate_dense_core(x0s[i], tfs[i], ns[i], args..., ctrl, na, nr);
                });
                nacc_out[i] = na;
                nrej_out[i] = nr;
            }
        };

        tycho::utils::parallel_blocks(n, job, n_parts);
        return results;
    }

    /// Parallel batch dense integrate with per-trajectory state counts.
    /// @see integrate_parallel for counters.
    std::vector<DenseEventRet> integrate_dense_parallel(const std::vector<ODEState<double>> &x0s,
                                                        const Eigen::VectorXd &tfs,
                                                        const std::vector<int> &ns,
                                                        const std::vector<EventPack> &events,
                                                        int n_parts) {
        const int n = static_cast<int>(x0s.size());
        std::vector<int> nacc(n, 0), nrej(n, 0);
        std::vector<int> nfailed(n, 0);
        BatchCounterWriteback _writeback{*this, nacc, nrej};
        BatchEventCounterWriteback _event_writeback{*this, nfailed};
        return this->integrate_dense_parallel_events_impl_n(x0s, tfs, ns, n_parts, nacc, nrej,
                                                            nfailed, events);
    }
    /// Parallel batch dense integrate with per-trajectory state counts (no events).
    std::vector<DenseRet> integrate_dense_parallel(const std::vector<ODEState<double>> &x0s,
                                                   const Eigen::VectorXd &tfs,
                                                   const std::vector<int> &ns, int n_parts) {
        const int n = static_cast<int>(x0s.size());
        std::vector<int> nacc(n, 0), nrej(n, 0);
        BatchCounterWriteback _writeback{*this, nacc, nrej};
        return this->integrate_dense_parallel_impl_n(x0s, tfs, ns, n_parts, nacc, nrej);
    }
    /////////////////////////////////////////////////////////////////////////////////////

    STMRet integrate_stm(const ODEState<double> &x0, double tf) const {
        ControllerVariant ctrl = this->make_worker_controller();
        int na = 0, nr = 0;
        CounterWriteback _writeback{*this, na, nr};
        return this->integrate_stm_core(x0, tf, ctrl, na, nr);
    }
    STMEventRet integrate_stm(const ODEState<double> &x0, double tf,
                              const std::vector<EventPack> &events) const {
        ControllerVariant ctrl = this->make_worker_controller();
        int na = 0, nr = 0;
        int nf = 0;
        CounterWriteback _writeback{*this, na, nr};
        EventCounterWriteback _event_writeback{*this, nf};
        return this->integrate_stm_core(x0, tf, events, ctrl, na, nr, nf);
    }

    std::vector<STMRet> integrate_stm_parallel_impl(const std::vector<ODEState<double>> &x0s,
                                                    const Eigen::VectorXd &tfs, int n_parts,
                                                    std::vector<int> &nacc_out,
                                                    std::vector<int> &nrej_out) {
        if (x0s.size() != tfs.size()) {
            throw std::invalid_argument(
                "List of initial states and final times must be the same size");
        }

        int n = x0s.size();
        std::vector<STMRet> results(n);

        auto job = [&](int start, int stop) {
            ControllerVariant ctrl;
            int na = 0, nr = 0;
            for (int i = start; i < stop; i++) {
                ctrl = this->make_worker_controller();
                na = 0;
                nr = 0;
                results[i] = decorate_trajectory(
                    i, [&] { return this->integrate_stm_core(x0s[i], tfs[i], ctrl, na, nr); });
                nacc_out[i] = na;
                nrej_out[i] = nr;
            }
        };

        tycho::utils::parallel_blocks(n, job, n_parts);
        return results;
    }

    /// Parallel batch STM integrate. @see integrate_parallel for the
    /// `get_naccept()` / `get_nreject()` contract.
    std::vector<STMRet> integrate_stm_parallel(const std::vector<ODEState<double>> &x0s,
                                               const Eigen::VectorXd &tfs, int n_parts) {
        const int n = static_cast<int>(x0s.size());
        std::vector<int> nacc(n, 0), nrej(n, 0);
        BatchCounterWriteback _writeback{*this, nacc, nrej};
        return this->integrate_stm_parallel_impl(x0s, tfs, n_parts, nacc, nrej);
    }
    /// Parallel batch STM integrate with events. Same `get_naccept()` /
    /// `get_nreject()` contract as the no-events overload.
    std::vector<STMEventRet> integrate_stm_parallel(const std::vector<ODEState<double>> &x0s,
                                                    const Eigen::VectorXd &tfs,
                                                    const std::vector<EventPack> &events,
                                                    int n_parts) {
        const int n = static_cast<int>(x0s.size());
        std::vector<int> nacc(n, 0), nrej(n, 0);
        std::vector<int> nfailed(n, 0);
        BatchCounterWriteback _writeback{*this, nacc, nrej};
        BatchEventCounterWriteback _event_writeback{*this, nfailed};
        return this->integrate_stm_parallel_events_impl(x0s, tfs, n_parts, nacc, nrej, nfailed,
                                                        events);
    }

    /// Segmented parallel STM over a single trajectory.
    /// Threadpool branch: `get_naccept()` / `get_nreject()` are NOT updated
    /// (workers own private per-segment counters to avoid cross-thread
    /// writes to member state). Non-threadpool fallback IS safe to write
    /// back — runs single-threaded and accumulates into the members.
    STMRet integrate_stm_parallel(const ODEState<double> &x0, double tf, int n_parts) {
        if (n_parts < 0) {
            throw std::invalid_argument("integrate_stm_parallel: n_parts must be >= 0; got " +
                                        std::to_string(n_parts));
        }
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
                        // Use integrate_stm_core (discarding the STM) so the
                        // main thread exercises the same FP arithmetic as the
                        // workers. xs[i+1] becomes worker (i+1)'s starting
                        // state; if it diverged from worker i's xf in FP, the
                        // chain-rule product in jxall would reflect sensitivity
                        // at a mismatched linearization point. Identical
                        // codepaths ⇒ bit-identical xs ⇒ exact chaining.
                        auto stm_result =
                            this->integrate_stm_core(xs[i], ts[i + 1], main_ctrl, main_na, main_nr);
                        xs[i + 1] = std::get<0>(stm_result);
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
                const std::string primary_msg =
                    tycho::utils::exception_what(std::current_exception());
                std::vector<std::string> extra_msgs;
                for (int j = 0; j < submitted; j++) {
                    try {
                        results[j].get();
                    } catch (...) {
                        extra_msgs.emplace_back(
                            tycho::utils::exception_what(std::current_exception()));
                    }
                }
                if (extra_msgs.empty()) {
                    throw;
                }
                throw_aggregated_segment_failure("main-thread segment failure", "worker failures",
                                                 primary_msg, extra_msgs);
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
                } catch (...) {
                    ex = std::current_exception();
                    primary_msg = tycho::utils::exception_what(ex);
                    for (int j = i + 1; j < n_parts; j++) {
                        try {
                            results[j].get();
                        } catch (...) {
                            extra_msgs.emplace_back(
                                tycho::utils::exception_what(std::current_exception()));
                        }
                    }
                    break;
                }
            }
            if (ex) {
                if (extra_msgs.empty()) {
                    std::rethrow_exception(ex);
                }
                throw_aggregated_segment_failure("primary segment failure", "additional failures",
                                                 primary_msg, extra_msgs);
            }
        } else {
            // Non-threadpool fallback is single-threaded, so it can safely
            // accumulate counters into the members (no cross-thread writes).
            // No second propagation is needed — each segment's `xf` feeds
            // directly into the next segment's start state.
            int total_na = 0, total_nr = 0;
            CounterWriteback _writeback{*this, total_na, total_nr};
            for (int i = 0; i < n_parts; i++) {
                ControllerVariant seg_ctrl = this->make_worker_controller();
                int seg_na = 0, seg_nr = 0;
                auto [xf, jx] =
                    this->integrate_stm_core(xs[i], ts[i + 1], seg_ctrl, seg_na, seg_nr);
                total_na += seg_na;
                total_nr += seg_nr;

                jxall.topRows(this->output_rows()) = (jx * jxall).eval();
                xs[i + 1] = xf;
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
        // jx and adjhess are guarded inside STMDriver via check_stm_finite_or_throw,
        // but adjgrad = jx^T · adjvars can still produce NaN if the caller
        // supplied non-finite adjvars. Localize that failure here rather than
        // letting it propagate into the solver as a silent bad gradient.
        if (!adjgrad.allFinite()) {
            throw std::runtime_error("Non-finite adjoint gradient in "
                                     "compute_jacobian_adjointgradient_adjointhessian_impl; "
                                     "adjvars contained NaN/Inf.");
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////
};

} // namespace tycho::integrators
