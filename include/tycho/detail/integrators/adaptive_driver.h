// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================

#pragma once

#include "tycho/detail/integrators/error_norm.h"
#include "tycho/detail/integrators/event_handler.h"
#include "tycho/detail/integrators/initial_dt.h"
#include "tycho/detail/integrators/step_controller.h"
#include "tycho/detail/integrators/stepper.h"
#include "tycho/detail/typedefs/eigen_types.h"

#include <Eigen/Core>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace tycho::integrators {

/// Run-time configuration passed to AdaptiveDriver::integrate — carries the
/// Integrator-level settings that control the adaptive loop's behavior.
/// Ownership stays with the caller; the driver reads only.
struct AdaptiveConfig {
    /// Error estimator order p; enters the step-size exponent q=err^(1/(p+1)).
    int error_order = 4;
    /// Error norm variant (RMS or infinity norm).
    ErrorNormType error_norm_type = ErrorNormType::RMS;
    /// Initial/fixed step size when Hairer-Wanner auto-init is off.
    double def_step_size = 0.01;
    /// Maximum ratio |dt_new/dt_old| allowed per step (must be > 1). This is a
    /// Tycho extension with no OrdinaryDiffEq analog — Julia bounds step growth
    /// solely through the in-controller qmin/qmax clamp. Applied on every step
    /// EXCEPT the first accepted one, which the drivers exempt to match
    /// OrdinaryDiffEq's qmax_first_step (a conservative Hairer-Wanner initial dt
    /// is allowed to grow quickly on step one). On steady-state steps at the
    /// default 10.0: the shrink side is inert (the shipped controllers' qmin=1/5
    /// never shrinks past 1/10), while the growth side ties the steady-state
    /// qmax=10 exactly — so it never binds *tighter* than the controller and
    /// only diverges from OrdinaryDiffEq if a user sets it below the
    /// controller's qmax.
    double max_step_change = 10.0;
    /// Hard cap on total steps (accepted + rejected) before throwing.
    int max_steps = 1'000'000;
    /// When false, use fixed step-size integration.
    bool adaptive = true;
    /// When true, estimate initial dt via Hairer-Wanner algorithm.
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
            throw std::invalid_argument("AdaptiveConfig::error_order must be positive "
                                        "(enters the q=err^(1/(p+1)) exponent); got " +
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
///   Scalar — numeric type. This driver is the SCALAR path (double). The
///            batched SuperScalar path is ParallelDriver; AdaptiveDriver's body
///            (1/h FSAL scaling, scalar comparisons) does not compile with a
///            SuperScalar Scalar.
///
/// State held by the driver is minimal: just the Stepper<Alg> which maintains
/// the FSAL cache across consecutive integrate() calls. All other inputs
/// (tolerances, controller, counters, events, storage flags) are passed per
/// call so the driver can be reused without copying Integrator-level config.
template <IVPAlg Alg, class DODE, class Scalar = double> class AdaptiveDriver {
  public:
    /// ODE state vector type for the given Scalar.
    using ODEState = typename DODE::template Input<Scalar>;
    /// ODE derivative vector type for the given Scalar.
    using ODEDeriv = typename DODE::template Output<Scalar>;
    /// Alias for the event specification type.
    using EventPack = typename EventHandler::EventPack;

    /// Output-side references and per-call storage flags. Groups the 9
    /// borrow-by-reference parameters that `integrate()` needs to write
    /// into caller-owned storage, reducing the public signature from 17
    /// positional args to 9. The caller owns every referent; the driver
    /// only reads flags and writes through the references.
    struct IO {
        /// Reference to the caller's accepted-step counter.
        int &naccept;
        /// Reference to the caller's rejected-step counter.
        int &nreject;
        /// Events to monitor during integration.
        const std::vector<EventPack> &events;
        /// Output: bracketed crossing intervals [t_prev, t_next].
        std::vector<std::vector<Eigen::Vector2d>> &eventtimes;
        /// When true, collect intermediate states into states.
        bool storestates;
        /// When true, collect intermediate derivatives into derivs.
        bool storederivs;
        /// When true, also collect midpoint states/derivs.
        bool storemidpoints;
        /// Output: collected trajectory states (when storestates).
        std::vector<ODEState> &states;
        /// Output: collected trajectory derivatives (when storederivs).
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

        // Reject non-finite tf / x0 up front (INTEGRATORS_REVIEW §1.5). A NaN
        // tf makes every (tnext - tf) comparison false, so the loop would
        // otherwise run to max_steps before throwing a misleading "stuck"
        // diagnostic; a non-finite x0 poisons the first derivative.
        if constexpr (std::is_floating_point_v<Scalar>) {
            if (!std::isfinite(static_cast<double>(tf))) {
                throw std::invalid_argument("AdaptiveDriver: tf is not finite (tf=" +
                                            std::to_string(static_cast<double>(tf)) + ").");
            }
            for (Eigen::Index i = 0; i < x.size(); ++i) {
                if (!std::isfinite(static_cast<double>(x[i]))) {
                    throw std::invalid_argument("AdaptiveDriver: initial state component " +
                                                std::to_string(i) + " is not finite.");
                }
            }
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

        // Initial step size. Compute the nominal step count in double and clamp
        // to max_steps before any int cast / reserve, so a huge H/step ratio can
        // neither overflow int (UB) nor drive a multi-GB reserve ahead of the
        // max_steps guard (INTEGRATORS_REVIEW §1.4).
        Scalar h;
        int numsteps;
        {
            double nominal;
            if (cfg.adaptive && cfg.use_hairer_wanner_initdt) {
                h = estimate_initial_dt(ode, x, tf, abs_tols, rel_tols, cfg.error_order,
                                        cfg.error_norm_type);
                double hh = std::abs(static_cast<double>(h));
                nominal = std::abs(static_cast<double>(H)) / (hh == 0.0 ? 1.0 : hh);
            } else {
                nominal =
                    std::abs(static_cast<double>(H) / static_cast<double>(cfg.def_step_size)) + 1.0;
            }
            double clamped = std::min(nominal, static_cast<double>(cfg.max_steps));
            numsteps = std::max(1, static_cast<int>(clamped));
            if (!(cfg.adaptive && cfg.use_hairer_wanner_initdt)) {
                // Fixed-step h is computed from the UNCLAMPED nominal so the step
                // size still matches def_step_size; only the reserve count
                // (numsteps) is clamped. Using the clamped count here would
                // silently coarsen a fixed-step run whose nominal count exceeds
                // max_steps instead of letting the in-loop max_steps guard throw.
                h = Scalar(0.9) * (H / Scalar(nominal));
            }
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
                    "rejection loop, or a fixed-step run requested more steps than the cap "
                    "allows. Raise via max_steps, loosen tolerances, or enlarge the step size.");
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

            // Zero-progress stall: `h` has shrunk below the representable spacing
            // of t, so `tnext = t + h` rounds back to t and the step's internal
            // (tnext - t0) would be exactly 0. Two causes reach here: (1) at large
            // |t| the controller shrinks h below ULP(t); (2) the state repeatedly
            // produced non-finite derivatives, so the non-finite reject-and-shrink
            // path drove h to underflow (a genuine singularity that does not
            // resolve under step reduction). Route both to a specific diagnostic
            // here — otherwise Stepper::step would run with an effective h == 0,
            // which its §3.3 guard rejects with a generic invalid_argument. This
            // keeps the direct-caller guard for its intended use (tf == t0).
            // Fires only on genuine underflow (h-resolvable steps advance with
            // tnext != t).
            if (continueloop && tnext == xi[ode.t_var()]) {
                throw std::runtime_error(
                    "AdaptiveDriver: step size underflowed to zero (tnext rounds back to t — no "
                    "progress possible). Either |t| is too large for the step resolution, or the "
                    "state repeatedly produced non-finite (NaN/Inf) derivatives that did not "
                    "resolve under step reduction (a singularity in the dynamics). max_steps=" +
                    std::to_string(cfg.max_steps) +
                    ". Rescale the independent variable, loosen tolerances, or check the dynamics "
                    "for singularities.");
            }

            xnext.setZero();
            xnext_est.setZero();
            xnext_mid.setZero();

            // Snapshot FSAL cache so we can restore on reject — stepper.step()
            // unconditionally writes k_fsal_ at end-of-step, but the retry
            // must read f(xi) (not the stale f(xnext_rejected)) as its first
            // stage. Fixed-step mode takes no rejections (every
            // restore_fsal() call site below is reached only from inside
            // `if (cfg.adaptive)` branches), so the snapshot -- an
            // ODEDeriv-sized copy of the FSAL cache, a heap allocation for
            // dynamic-size ODEs -- is pure waste there every step
            // (INTEGRATORS §2.4). Default-constructed (empty) when
            // !cfg.adaptive; never read on that path.
            using FsalSnap = typename Stepper<Alg, DODE, Scalar>::FsalSnapshot;
            FsalSnap fsal_saved{};
            if (cfg.adaptive) {
                fsal_saved = stepper_.snapshot_fsal();
            }

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

                // Non-finite err_norm (NaN/Inf from xnext, xnext_est, or the
                // residual — one scalar isfinite subsumes both xnext and
                // xnext_est): reject the step and shrink h, then retry — do NOT
                // throw. This mirrors OrdinaryDiffEq, where EEst == NaN fails the
                // `EEst <= 1` accept test and the step is rejected; an over-large
                // step into a stiff/near-singular region can often be resolved at
                // smaller h, so a multiple-shooting / PSIOPT iterate recovers
                // internally instead of aborting the whole solve. The controller
                // cannot derive a growth factor from a non-finite EEst, so we
                // shrink by the fixed kNonfiniteStepShrink. A genuinely
                // unintegrable state still fails — the zero-progress stall guard
                // (tnext == t, above) or the max_steps cap terminates the loop
                // with a diagnostic after a bounded number of retries.
                if (!std::isfinite(err_norm)) {
                    stepper_.restore_fsal(fsal_saved);
                    h *= Scalar(kNonfiniteStepShrink);
                    nreject++;
                    continueloop = true;
                    continue;
                }

                // err_norm catches NaN that flows through xnext or xnext_est, but
                // xnext_mid is computed from a different stage combination
                // (extra-stage interpolant on Vern7/8/9) and can carry NaN while
                // the embedded pair stays finite. Treat a non-finite midpoint the
                // same way — reject and shrink — when storemidpoints will push it
                // into user storage.
                if (storemidpoints && !xnext_mid.head(ode.x_vars()).allFinite()) {
                    stepper_.restore_fsal(fsal_saved);
                    h *= Scalar(kNonfiniteStepShrink);
                    nreject++;
                    continueloop = true;
                    continue;
                }

                auto outcome = update_controller(controller, static_cast<double>(h), err_norm,
                                                 cfg.error_order, naccept);
                double hnext = outcome.dt_new;

                // First accepted step is exempt from the max_step_change clamp —
                // OrdinaryDiffEq's qmax_first_step deliberately lets a conservative
                // Hairer-Wanner initial dt grow quickly, and this Tycho-only clamp
                // would otherwise cap that first-step growth (naccept is the count of
                // *previously* accepted steps, so == 0 is the first step).
                if (naccept == 0)
                    h = Scalar(hnext);
                else if (hnext / static_cast<double>(h) > cfg.max_step_change)
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
                // Fixed-step mode takes no rejections, but each accepted step
                // must still count toward max_steps so the cap actually bounds
                // the loop (an oversized fixed-step request throws promptly at
                // the guard above instead of grinding out its full nominal count
                // — the reserve clamp at §1.4 relies on this).
                naccept++;
            }

            bool eventbreak = false;
            if (!events.empty()) {
                eventbreak = EventHandler::check_crossings(events, prev_event_vals, next_event_vals,
                                                           xnext, ode.t_var(), eventtimes,
                                                           static_cast<double>(xi[ode.t_var()]));
            }

            xi = xnext;
            // INTEGRATORS §2.2 f(xf) reuse. Every reject branch above
            // `continue`s before reaching this line, so getting here means
            // the step was accepted and `xi` (just assigned xnext) really is
            // this step's xf — exactly the "coordinated caller passes
            // xi == prev xf" contract Stepper::step's docstring requires
            // before a peeked f(xf) may be promoted to a FSAL seed.
            //
            // stepper_.peek_fresh() is true here iff step<true>() ran this
            // iteration, i.e. storemidpoints||storederivs (loop-invariant
            // for the whole integrate() call — see the step<true>/
            // step<false> dispatch above), which is the same condition that
            // already gated the old storederivs-only peek_fsal() read. For
            // FSAL/LastStageIsFxf methods (DOPRI54, Tsit5, BS3, BS5)
            // peek_fresh_ is unconditionally true after every step() and
            // fsal_valid_ is already set inside step() itself (see
            // stepper.h's k_fsal_ update block), so seed_fsal below just
            // re-seeds the identical value — a no-op in effect. The actual
            // gap this closes is DOPRI87/Vern7/8/9 (!LastStageIsFxf): step()
            // deliberately leaves fsal_valid_ false there (a bare Stepper
            // caller issuing step() against an unrelated x0 must not
            // silently reuse a stale f(xf_prev) as the next stage-0), but
            // AdaptiveDriver is exactly the coordinated caller that can
            // safely promote the peek to a seed here, saving one ODE eval on
            // the next step.
            if (stepper_.peek_fresh()) {
                // Copy out (not alias) before seed_fsal() below overwrites
                // k_fsal_ and clears peek_fresh_ — storederivs needs the
                // value read here.
                const ODEDeriv fxf = stepper_.peek_fsal();
                if (storederivs) {
                    xdoti = fxf;
                }
                stepper_.seed_fsal(fxf);
            }
            // next_event_vals is unconditionally overwritten element-by-
            // element (setZero() + vf.compute()) at the top of the next
            // check_crossings() call, so its post-swap stale contents are
            // never read -- std::swap avoids the copy-assignment
            // (INTEGRATORS §2.6).
            std::swap(prev_event_vals, next_event_vals);

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
