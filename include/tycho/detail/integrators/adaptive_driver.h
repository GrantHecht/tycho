// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================

#pragma once

#include "tycho/detail/integrators/error_norm.h"
#include "tycho/detail/integrators/event_handler.h"
#include "tycho/detail/integrators/initial_dt.h"
#include "tycho/detail/integrators/step_controller.h"
#include "tycho/detail/integrators/stepper.h"
#include "tycho/detail/typedefs/eigen_types.h"

#include <Eigen/Core>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace tycho::integrators {

/// Run-time configuration passed to AdaptiveDriver::integrate — carries the
/// Integrator-level settings that control the adaptive loop's behavior.
/// Ownership stays with the caller; the driver reads only.
struct AdaptiveConfig {
    int error_order = 4;
    ErrorNormType error_norm_type = ErrorNormType::RMS;
    double def_step_size = 0.01;
    double max_step_change = 10.0;
    int max_steps = 1'000'000;
    bool adaptive = true;
    bool use_hairer_wanner_initdt = true;

    /// Cheap sanity check — throws `std::invalid_argument` on any value
    /// that would lead to a divide-by-zero, infinite loop, or otherwise
    /// undefined behavior inside the step loop. Called by drivers at the
    /// top of `integrate()` so a bad config fails fast with a descriptive
    /// message rather than producing NaNs or hanging.
    void validate() const {
        if (max_steps < 1) {
            throw std::invalid_argument("AdaptiveConfig::max_steps must be >= 1; got " +
                                        std::to_string(max_steps));
        }
        if (!(max_step_change > 1.0)) {
            throw std::invalid_argument("AdaptiveConfig::max_step_change must be strictly > 1.0 "
                                        "(clamp on |dt_new/dt_old|); "
                                        "got " +
                                        std::to_string(max_step_change));
        }
        if (error_order <= 0) {
            throw std::invalid_argument(
                "AdaptiveConfig::error_order must be positive (enters the q=err^(1/p) exponent); "
                "got " +
                std::to_string(error_order));
        }
        if (!std::isfinite(def_step_size) || def_step_size <= 0.0) {
            throw std::invalid_argument(
                "AdaptiveConfig::def_step_size must be finite and > 0; got " +
                std::to_string(def_step_size));
        }
    }
};

/// Adaptive step-size integration driver.
///
/// Composes Stepper<Alg,DODE,Scalar> + a runtime-dispatched controller
/// (via ControllerVariant) + EventHandler::check_crossings to perform a
/// full adaptive integration from x(t0) to x(tf).
///
/// Template parameters:
///   Alg    — IVPAlg enum selecting the Butcher tableau (compile-time)
///   DODE   — ODE type
///   Scalar — numeric type (double by default; SuperScalar for batched use)
///
/// State held by the driver is minimal: just the Stepper<Alg> which maintains
/// the FSAL cache across consecutive integrate() calls. All other inputs
/// (tolerances, controller, counters, events, storage flags) are passed per
/// call so the driver can be reused without copying Integrator-level config.
template <IVPAlg Alg, class DODE, class Scalar = double> class AdaptiveDriver {
  public:
    using ODEState = typename DODE::template Input<Scalar>;
    using ODEDeriv = typename DODE::template Output<Scalar>;
    using EventPack = typename EventHandler::EventPack;

    /// Output-side references and per-call storage flags. Groups the 9
    /// borrow-by-reference parameters that `integrate()` needs to write
    /// into caller-owned storage, reducing the public signature from 17
    /// positional args to 9. The caller owns every referent; the driver
    /// only reads flags and writes through the references.
    struct IO {
        int &naccept;
        int &nreject;
        const std::vector<EventPack> &events;
        std::vector<std::vector<Eigen::Vector2d>> &eventtimes;
        bool storestates;
        bool storederivs;
        bool storemidpoints;
        std::vector<ODEState> &states;
        std::vector<ODEDeriv> &derivs;
    };

    /// Reset the stepper's FSAL cache. Callers must invoke this before the
    /// first integrate() call following a state change that invalidates the
    /// cached f(x_prev) (e.g., a fresh starting state unrelated to the last
    /// step's output).
    void reset_fsal() { stepper_.reset_fsal(); }

    /// Perform adaptive integration from x(t0) to x(tf).
    ///
    /// Contract:
    ///   - `controller` is mutated across the run; caller must pass a
    ///     ControllerVariant that has already been reset() or is
    ///     freshly constructed.
    ///   - `naccept`, `nreject` are zeroed by the caller and incremented
    ///     by the driver; the driver never reads this->member counters.
    ///   - `abs_tols` and `rel_tols` must satisfy abs[i] + rel[i] > 0 per
    ///     component in adaptive mode; violation throws.
    ///   - On H == 0 (zero interval), returns the input state after
    ///     populating outputs with a single (x, ode.compute(x)) triple.
    ///   - On NaN/Inf in either the state or the error norm, throws with a
    ///     diagnostic message — never runs to max_steps on such inputs.
    ///   - `ControlFn` is callable with (ODEState&) and invoked exactly
    ///     where Integrator's update_control was: at each stage tuple
    ///     construction in the stepper plus at final / midpoint assembly.
    template <class ControlFn>
    ODEState integrate(const DODE &ode, const ODEState &x, Scalar tf, const AdaptiveConfig &cfg,
                       const ODEDeriv &abs_tols, const ODEDeriv &rel_tols,
                       ControllerVariant &controller, IO io, ControlFn &&update_control) {

        int &naccept = io.naccept;
        int &nreject = io.nreject;
        const std::vector<EventPack> &events = io.events;
        std::vector<std::vector<Eigen::Vector2d>> &eventtimes = io.eventtimes;
        const bool storestates = io.storestates;
        const bool storederivs = io.storederivs;
        const bool storemidpoints = io.storemidpoints;
        std::vector<ODEState> &states = io.states;
        std::vector<ODEDeriv> &derivs = io.derivs;

        // Validate inputs first so failure paths leave the stepper unchanged.
        cfg.validate();
        validate_controller(controller);

        if (x.size() != ode.input_rows()) {
            throw std::invalid_argument("AdaptiveDriver: incorrectly sized input state.");
        }

        // After validation succeeds, invalidate any FSAL state left over from
        // a prior integrate() (including one that threw mid-step). Stale
        // k_fsal_ would otherwise be reused as stage 0 on the first step.
        stepper_.reset_fsal();

        // Joint tolerance invariant in adaptive mode: abs[i] + rel[i] > 0.
        if (cfg.adaptive) {
            if (rel_tols.size() != abs_tols.size()) {
                throw std::logic_error("AdaptiveDriver: abs_tols/rel_tols size mismatch.");
            }
            for (Eigen::Index i = 0; i < abs_tols.size(); ++i) {
                if (!(abs_tols[i] + rel_tols[i] > 0.0)) {
                    throw std::invalid_argument(
                        "AdaptiveDriver: tolerance component " + std::to_string(i) +
                        " has abs_tol + rel_tol <= 0. Set at least one positive; otherwise "
                        "the adaptive error norm is undefined for zero state.");
                }
            }
        }

        Scalar t0 = x[ode.t_var()];
        Scalar H = tf - t0;

        // Zero-interval short-circuit: no step to take. Populate outputs with
        // the endpoint state + derivative and return. Skipping stepper.step
        // avoids divide-by-zero in the FSAL / midpoint derivative
        // reconstruction that would otherwise poison `derivs` with NaN.
        if (H == Scalar(0.0)) {
            ODEState xi0 = x;
            update_control(xi0);
            eventtimes.resize(events.size());
            for (auto &v : eventtimes)
                v.clear();
            for (std::size_t j = 0; j < events.size(); ++j) {
                if (events[j].vf.input_rows() != ode.input_rows()) {
                    throw std::invalid_argument(
                        "AdaptiveDriver: event function input size mismatch.");
                }
            }
            if (storestates) {
                states.resize(0);
                derivs.resize(0);
                ODEDeriv xdoti0(ode.output_rows());
                xdoti0.setZero();
                ode.compute(xi0, xdoti0);
                states.push_back(xi0);
                if (storederivs)
                    derivs.push_back(xdoti0);
                if (storemidpoints) {
                    states.push_back(xi0);
                    if (storederivs)
                        derivs.push_back(xdoti0);
                }
                states.push_back(xi0);
                if (storederivs)
                    derivs.push_back(xdoti0);
            }
            return xi0;
        }

        // Initial step size.
        Scalar h;
        int numsteps;
        if (cfg.adaptive && cfg.use_hairer_wanner_initdt) {
            h = estimate_initial_dt(ode, x, tf, abs_tols, rel_tols, cfg.error_order,
                                    cfg.error_norm_type);
            numsteps = std::max(1, int(std::abs(H / (h == Scalar(0.0) ? Scalar(1.0) : h))));
        } else {
            numsteps = int(std::abs(H / cfg.def_step_size)) + 1;
            h = Scalar(0.9) * (H / Scalar(numsteps));
        }

        ODEState xi = x;
        update_control(xi);

        ODEState xnext = xi;
        ODEState xnext_est = xi;
        ODEState xnext_mid = xi;

        ODEDeriv xdoti(ode.output_rows());
        xdoti.setZero();
        ode.compute(xi, xdoti);

        // Event state.
        std::vector<Vector1<double>> prev_event_vals(events.size());
        std::vector<Vector1<double>> next_event_vals(events.size());
        for (std::size_t j = 0; j < events.size(); ++j) {
            prev_event_vals[j].setZero();
            next_event_vals[j].setZero();
            if (events[j].vf.input_rows() != ode.input_rows()) {
                throw std::invalid_argument("AdaptiveDriver: event function input size mismatch.");
            }
            events[j].vf.compute(xi, prev_event_vals[j]);
        }
        eventtimes.resize(events.size());
        // Clear any stale crossings from a reused buffer — the driver contract
        // treats `eventtimes` as caller-owned, so a reuse without this clear
        // would poison find_events_counted with old data.
        for (auto &v : eventtimes)
            v.clear();

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

        bool continueloop = true;
        while (continueloop) {
            if (static_cast<long long>(naccept) + nreject >= cfg.max_steps) {
                throw std::runtime_error(
                    "AdaptiveDriver exceeded max_steps (" + std::to_string(cfg.max_steps) +
                    ") before reaching tf; the adaptive controller may be stuck in a "
                    "rejection loop. Raise via max_steps or loosen tolerances.");
            }
            Scalar tnext = xi[ode.t_var()] + h;
            if (H > Scalar(0.0)) {
                if ((tnext - tf) >= Scalar(0.0)) {
                    h = tf - xi[ode.t_var()];
                    tnext = tf;
                    continueloop = false;
                }
            } else {
                if ((tnext - tf) <= Scalar(0.0)) {
                    h = tf - xi[ode.t_var()];
                    tnext = tf;
                    continueloop = false;
                }
            }

            xnext.setZero();
            xnext_est.setZero();
            xnext_mid.setZero();

            // Snapshot FSAL cache so we can restore on reject — stepper.step()
            // unconditionally writes k_fsal_ at end-of-step, but the retry
            // must read f(xi) (not the stale f(xnext_rejected)) as its first
            // stage.
            auto fsal_saved = stepper_.snapshot_fsal();

            if (storemidpoints || storederivs) {
                stepper_.template step<true>(ode, xi, tnext, xnext, xnext_est, xnext_mid,
                                             update_control);
            } else {
                stepper_.template step<false>(ode, xi, tnext, xnext, xnext_est, xnext_mid,
                                              update_control);
            }

            if (cfg.adaptive) {
                auto u_x_vars = ode.x_vars();
                auto utilde = xnext.head(u_x_vars) - xnext_est.head(u_x_vars);
                auto res = scaled_residuals(utilde, xi.head(u_x_vars), xnext.head(u_x_vars),
                                            abs_tols, rel_tols);
                double err_norm = error_norm(res, cfg.error_norm_type);

                // Single-chokepoint finite guard. A non-finite err_norm catches
                // BOTH xnext NaN (NaN flows into the numerator) and xnext_est
                // NaN (NaN flows into utilde), so one scalar isfinite subsumes
                // both xnext and err_norm checks. Fixed-step path handles xnext
                // separately below since it doesn't compute err_norm.
                if (!std::isfinite(err_norm)) {
                    throw std::runtime_error(
                        "Non-finite error norm from AdaptiveDriver (" + std::to_string(err_norm) +
                        ") at t=" + std::to_string(static_cast<double>(xi[ode.t_var()])) +
                        " (h=" + std::to_string(static_cast<double>(h)) +
                        "); state or embedded estimate produced NaN/Inf. Check ODE dynamics "
                        "and intermediate-stage evaluations.");
                }

                // err_norm catches NaN that flows through xnext or xnext_est, but
                // xnext_mid is computed from a different stage combination
                // (extra-stage interpolant on Vern7/8/9) and can carry NaN while
                // the embedded pair stays finite. Guard the midpoint slot
                // separately when storemidpoints will push it into user storage.
                if (storemidpoints) {
                    check_state_finite_or_throw(xnext_mid.head(ode.x_vars()), xi[ode.t_var()], h,
                                                "AdaptiveDriver::stepper.step (midpoint)");
                }

                auto outcome = update_controller(controller, static_cast<double>(h), err_norm,
                                                 cfg.error_order, naccept);
                double hnext = outcome.dt_new;

                if (hnext / static_cast<double>(h) > cfg.max_step_change)
                    h *= Scalar(cfg.max_step_change);
                else if (hnext / static_cast<double>(h) < 1.0 / cfg.max_step_change)
                    h /= Scalar(cfg.max_step_change);
                else
                    h = Scalar(hnext);

                if (!outcome.accepted) {
                    stepper_.restore_fsal(fsal_saved);
                    nreject++;
                    continueloop = true;
                    continue;
                }
                naccept++;
            } else {
                // Fixed-step path: err_norm is not computed, so the adaptive
                // chokepoint above doesn't fire. Check xnext directly.
                check_state_finite_or_throw(xnext.head(ode.x_vars()), xi[ode.t_var()], h,
                                            "AdaptiveDriver::stepper.step");
                // xnext_mid is consumed by user storage when storemidpoints,
                // so a NaN there silently corrupts output. The post-step xnext
                // check above does not cover the midpoint slot.
                if (storemidpoints) {
                    check_state_finite_or_throw(xnext_mid.head(ode.x_vars()), xi[ode.t_var()], h,
                                                "AdaptiveDriver::stepper.step (midpoint)");
                }
            }

            bool eventbreak = false;
            if (!events.empty()) {
                eventbreak = EventHandler::check_crossings(events, prev_event_vals, next_event_vals,
                                                           xnext, ode.t_var(), eventtimes,
                                                           static_cast<double>(xi[ode.t_var()]));
            }

            xi = xnext;
            // Only copy the FSAL cache out when a caller actually wants
            // per-step derivatives stored — otherwise it's a wasted copy
            // of a buffer whose lifetime is the next step(). The reference
            // is invalidated by the next step() call (see Stepper::peek_fsal
            // docstring).
            if (storederivs) {
                // peek_fsal requires the preceding step() to have set
                // compute_midpoint=true. The (storemidpoints||storederivs)
                // disjunction at the step() call site above guarantees that
                // today; decoupling compute_midpoint from storederivs would
                // break this branch.
                xdoti = stepper_.peek_fsal();
            }
            prev_event_vals = next_event_vals;

            if (storestates) {
                if (storemidpoints) {
                    states.push_back(xnext_mid);
                    if (storederivs) {
                        ODEDeriv xdot_mid(ode.output_rows());
                        xdot_mid.setZero();
                        ode.compute(xnext_mid, xdot_mid);
                        // The ODE may be singular at the midpoint state even
                        // when xnext_mid itself is finite (e.g. 1/r dynamics
                        // whose midpoint reconstruction lands near r=0). Guard
                        // before pushing into the user's deriv buffer.
                        check_state_finite_or_throw(
                            xdot_mid.head(ode.x_vars()), xi[ode.t_var()], h,
                            "AdaptiveDriver::stepper.step (midpoint deriv)");
                        derivs.push_back(xdot_mid);
                    }
                }
                states.push_back(xi);
                if (storederivs)
                    derivs.push_back(xdoti);
            }

            if (eventbreak)
                break;
        }
        return xi;
    }

  private:
    Stepper<Alg, DODE, Scalar> stepper_;
};

} // namespace tycho::integrators
