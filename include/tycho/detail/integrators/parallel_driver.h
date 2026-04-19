// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================

#pragma once

#include "tycho/detail/integrators/adaptive_driver.h"
#include "tycho/detail/integrators/error_norm.h"
#include "tycho/detail/integrators/event_handler.h"
#include "tycho/detail/integrators/initial_dt.h"
#include "tycho/detail/integrators/step_controller.h"
#include "tycho/detail/integrators/stepper.h"
#include "tycho/detail/typedefs/eigen_types.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace tycho::integrators {

/// SIMD batch integration driver.
///
/// Mirrors AdaptiveDriver but drives `N` trajectories through a single
/// SuperScalar stepper, packing lanes into a width-`vsize` SIMD register.
/// Each lane carries its own ControllerVariant, accept/reject counters,
/// eventtimes, and storage vectors.
///
/// Template parameters:
///   Alg  — IVPAlg enum selecting the Butcher tableau
///   DODE — ODE type
///
/// Scalar is fixed to tycho::DefaultSuperScalar; the driver extracts per-lane
/// results at the scalar-double level for event detection, controller
/// evaluation, and storage. The internal stepper operates on SuperScalar.
///
/// Per-lane FSAL is threaded through the stepper's k_fsal_ cache: before each
/// step the cached k_fsal_ is seeded from xdotis[i] packed into SuperScalar;
/// after the step, k_fsal_ is extracted per-lane back to xdotis[i] for
/// accepted lanes only (rejected lanes keep their pre-step xdotis, so the
/// retry sees the correct f(xi)).
template <IVPAlg Alg, class DODE> struct ParallelDriver {

    using SuperScalar = tycho::DefaultSuperScalar;
    using ScalarState = typename DODE::template Input<double>;
    using ScalarDeriv = typename DODE::template Output<double>;
    using SSState = typename DODE::template Input<SuperScalar>;
    using SSDeriv = typename DODE::template Output<SuperScalar>;
    using EventPack = typename EventHandler::EventPack;

    Stepper<Alg, DODE, SuperScalar> stepper_;

    /// Reset the SIMD stepper's FSAL cache. Call before the first integrate()
    /// following a state change that invalidates the cached f(x_prev).
    void reset_fsal() { stepper_.reset_fsal(); }

    /// Perform SIMD batch adaptive integration.
    ///
    /// Contract:
    ///   - `controllers`, `nacc`, `nrej`, `eventtimes_s`, `states_s`,
    ///     `derivs_s` must each have size == xs.size() and be pre-reset
    ///     by the caller.
    ///   - `abs_tols` and `rel_tols` share across all lanes (they apply
    ///     to the ODE state, not per-trajectory).
    ///   - Any lane with H == 0 is skipped (outputs populated up front)
    ///     and marked complete before the SIMD loop starts.
    ///   - NaN/Inf state or err_norm on any lane throws with the
    ///     trajectory index embedded in the diagnostic.
    template <class ControlFn>
    std::vector<ScalarState>
    integrate(const DODE &ode, const std::vector<ScalarState> &xs, const Eigen::VectorXd &tfs,
              const AdaptiveConfig &cfg, const ScalarDeriv &abs_tols, const ScalarDeriv &rel_tols,
              std::vector<ControllerVariant> &controllers, std::vector<int> &nacc,
              std::vector<int> &nrej, const std::vector<EventPack> &events,
              std::vector<std::vector<std::vector<Eigen::Vector2d>>> &eventtimes_s,
              bool storestates, bool storederivs, bool storemidpoints,
              std::vector<std::vector<ScalarState>> &states_s,
              std::vector<std::vector<ScalarDeriv>> &derivs_s, ControlFn &&update_control) {

        if (xs.size() != static_cast<std::size_t>(tfs.size())) {
            throw std::invalid_argument(
                "ParallelDriver: initial-state and final-time vectors must match in size.");
        }
        if (xs.empty()) {
            throw std::invalid_argument("ParallelDriver: must supply at least one initial state.");
        }

        const int ntrajs = static_cast<int>(xs.size());

        if (static_cast<int>(controllers.size()) != ntrajs)
            throw std::invalid_argument(
                "ParallelDriver: controllers vector size must equal number of trajectories.");
        if (static_cast<int>(nacc.size()) != ntrajs || static_cast<int>(nrej.size()) != ntrajs)
            throw std::invalid_argument(
                "ParallelDriver: nacc/nrej vector size must equal number of trajectories.");

        // Joint tolerance invariant.
        if (cfg.adaptive) {
            if (rel_tols.size() != abs_tols.size()) {
                throw std::logic_error("ParallelDriver: abs_tols/rel_tols size mismatch.");
            }
            for (Eigen::Index i = 0; i < abs_tols.size(); ++i) {
                if (!(abs_tols[i] + rel_tols[i] > 0.0)) {
                    throw std::invalid_argument("ParallelDriver: tolerance component " +
                                                std::to_string(i) + " has abs_tol + rel_tol <= 0.");
                }
            }
        }

        Eigen::VectorXd hs(ntrajs);
        Eigen::VectorXd h_spans(ntrajs);
        std::vector<ScalarState> xis = xs;
        std::vector<ScalarDeriv> xdotis(ntrajs);

        std::vector<std::vector<Vector1<double>>> prev_event_vals_s(ntrajs);
        std::vector<std::vector<Vector1<double>>> next_event_vals_s(ntrajs);

        // Per-lane initialization: initial-dt, tolerance validation, event
        // value seeding, storestates preamble, zero-interval short-circuit.
        for (int i = 0; i < ntrajs; ++i) {
            if (xis[i].size() != ode.input_rows()) {
                throw std::invalid_argument(
                    "ParallelDriver: incorrectly sized input state on lane " + std::to_string(i));
            }
            update_control(xis[i]);

            double t0 = xis[i][ode.t_var()];
            h_spans[i] = tfs[i] - t0;

            int numsteps;
            if (h_spans[i] == 0.0) {
                hs[i] = 0.0;
                numsteps = 1;
            } else if (cfg.adaptive && cfg.use_hairer_wanner_initdt) {
                hs[i] = estimate_initial_dt(ode, xis[i], tfs[i], abs_tols, rel_tols,
                                            cfg.error_order, cfg.error_norm_type);
                numsteps = std::max(1, int(std::abs(h_spans[i] / (hs[i] == 0.0 ? 1.0 : hs[i]))));
            } else {
                numsteps = int(std::abs(h_spans[i] / cfg.def_step_size)) + 1;
                hs[i] = 0.9 * (h_spans[i] / double(numsteps));
            }

            xdotis[i].resize(ode.output_rows());
            xdotis[i].setZero();
            ode.compute(xis[i], xdotis[i]);

            prev_event_vals_s[i].resize(events.size());
            next_event_vals_s[i].resize(events.size());
            for (std::size_t j = 0; j < events.size(); ++j) {
                prev_event_vals_s[i][j].setZero();
                next_event_vals_s[i][j].setZero();
                if (std::get<0>(events[j]).input_rows() != ode.input_rows()) {
                    throw std::invalid_argument(
                        "ParallelDriver: event function input size mismatch.");
                }
                std::get<0>(events[j]).compute(xis[i], prev_event_vals_s[i][j]);
            }
            if (!events.empty()) {
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

                // Zero-interval lane: emulate the loop's push pattern here
                // and mark the lane complete before the SIMD loop starts.
                if (h_spans[i] == 0.0) {
                    if (storemidpoints) {
                        states_s[i].push_back(xis[i]);
                        if (storederivs)
                            derivs_s[i].push_back(xdotis[i]);
                    }
                    states_s[i].push_back(xis[i]);
                    if (storederivs)
                        derivs_s[i].push_back(xdotis[i]);
                }
            }
        }

        // Continueloops: true until lane finishes (or H==0 up front).
        std::vector<bool> continueloops(ntrajs, true);
        for (int i = 0; i < ntrajs; ++i) {
            if (h_spans[i] == 0.0)
                continueloops[i] = false;
        }

        // SIMD working buffers.
        SSState xi_ss(ode.input_rows());
        SSState xnext_ss(ode.input_rows());
        SSState xnext_est_ss(ode.input_rows());
        SSState xnext_mid_ss(ode.input_rows());
        SSDeriv xdotnext_ss(ode.output_rows());
        SuperScalar tnext_ss;

        ScalarState xnext(ode.input_rows());
        ScalarState xnext_mid(ode.input_rows());
        ScalarDeriv xdotnext(ode.output_rows());
        ScalarState xnext_est_lane(ode.input_rows());

        ScalarDeriv abs_error(ode.x_vars());
        SSDeriv abs_error_ss(ode.x_vars());

        int numrunning = 0;
        int lastrunning = -1;
        for (int i = 0; i < ntrajs; ++i) {
            if (continueloops[i]) {
                numrunning++;
                lastrunning = i;
            }
        }

        // Pad unused SIMD slots with lane-0 data (harmless — only V<Vmax is
        // read out). Matches the production path's padding convention.
        if (ntrajs < SuperScalar::SizeAtCompileTime) {
            for (int V = 0; V < SuperScalar::SizeAtCompileTime; ++V) {
                for (int k = 0; k < ode.input_rows(); ++k) {
                    xi_ss[k][V] = xis[0][k];
                }
                for (int k = 0; k < ode.output_rows(); ++k) {
                    xdotnext_ss[k][V] = xdotis[0][k];
                }
            }
        }

        while (numrunning > 0) {
            // Per-lane max_steps guard — name the offending trajectory so
            // users can localize a runaway lane without waiting for the whole
            // batch to exhaust its budget.
            for (int i = 0; i < ntrajs; ++i) {
                if (continueloops[i] && nacc[i] + nrej[i] >= cfg.max_steps) {
                    throw std::runtime_error(
                        "ParallelDriver: trajectory " + std::to_string(i) +
                        " exceeded max_steps (" + std::to_string(cfg.max_steps) +
                        ") before reaching tf; the adaptive controller for this lane may "
                        "be stuck in a rejection loop.");
                }
            }

            std::array<int, SuperScalar::SizeAtCompileTime> idxs;
            int V = 0;

            for (int i = 0; i < ntrajs; ++i) {
                if (!continueloops[i])
                    continue;

                double tnext = xis[i][ode.t_var()] + hs[i];
                if (h_spans[i] > 0.0) {
                    if ((tnext - tfs[i]) >= 0.0) {
                        hs[i] = tfs[i] - xis[i][ode.t_var()];
                        tnext = tfs[i];
                        continueloops[i] = false;
                    }
                } else {
                    if ((tnext - tfs[i]) <= 0.0) {
                        hs[i] = tfs[i] - xis[i][ode.t_var()];
                        tnext = tfs[i];
                        continueloops[i] = false;
                    }
                }

                for (int k = 0; k < ode.input_rows(); ++k) {
                    xi_ss[k][V] = xis[i][k];
                }
                for (int k = 0; k < ode.output_rows(); ++k) {
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

                    // Seed FSAL only for methods where Stepper actually updates
                    // k_fsal_ post-step (FSAL or LastStageIsFxf). For methods
                    // without either — Vern7/8/9 — seed-based reuse would
                    // silently use stale f(x_prev) on subsequent steps because
                    // Stepper leaves k_fsal_ unchanged when compute_midpoint
                    // is false. Force fresh compute in that case, matching
                    // the legacy integrate_impl_vectorized behavior.
                    constexpr bool method_does_fsal =
                        RKCoeffs<Alg>::FSAL || RKCoeffs<Alg>::LastStageIsFxf;
                    if constexpr (method_does_fsal) {
                        stepper_.k_fsal_ = xdotnext_ss;
                        stepper_.fsal_valid_ = true;
                    } else {
                        stepper_.fsal_valid_ = false;
                    }

                    stepper_.step(ode, xi_ss, tnext_ss, xnext_ss, xnext_est_ss,
                                  storemidpoints || storederivs, xnext_mid_ss, update_control);

                    // Extract k_fsal_ back to xdotnext_ss for FSAL methods so
                    // accepted lanes propagate f(xf) to the next step. For
                    // non-FSAL methods, xdotnext_ss is unchanged — the next
                    // step will recompute fresh regardless.
                    if constexpr (method_does_fsal) {
                        xdotnext_ss = stepper_.k_fsal_;
                    }

                    abs_error_ss =
                        (xnext_ss.head(ode.x_vars()) - xnext_est_ss.head(ode.x_vars())).cwiseAbs();

                    for (int V2 = 0; V2 < Vmax; ++V2) {
                        int itmp = idxs[V2];

                        for (int k = 0; k < ode.input_rows(); ++k) {
                            xnext[k] = xnext_ss[k][V2];
                            xnext_mid[k] = xnext_mid_ss[k][V2];
                            xnext_est_lane[k] = xnext_est_ss[k][V2];
                        }
                        for (int k = 0; k < ode.output_rows(); ++k) {
                            xdotnext[k] = xdotnext_ss[k][V2];
                            abs_error[k] = abs_error_ss[k][V2];
                        }

                        if (cfg.adaptive) {
                            double h_lane = hs[itmp];
                            auto u_x_vars = ode.x_vars();
                            auto utilde = xnext.head(u_x_vars) - xnext_est_lane.head(u_x_vars);
                            auto res = scaled_residuals(utilde, xis[itmp].head(u_x_vars),
                                                        xnext.head(u_x_vars), abs_tols, rel_tols);
                            double err_norm = error_norm(res, cfg.error_norm_type);

                            // Single-chokepoint per-lane finite guard.
                            // Non-finite err_norm subsumes both xnext and
                            // xnext_est NaN checks (NaN in either flows into
                            // utilde/res/norm). One scalar isfinite per lane
                            // replaces the previous full allFinite scan on
                            // xnext. Fixed-step path checks xnext separately
                            // since err_norm isn't computed there.
                            if (!std::isfinite(err_norm)) {
                                throw std::runtime_error(
                                    "Non-finite error norm from ParallelDriver (" +
                                    std::to_string(err_norm) + ") on trajectory " +
                                    std::to_string(itmp) +
                                    " at t=" + std::to_string(xis[itmp][ode.t_var()]) +
                                    " (h=" + std::to_string(h_lane) +
                                    "); state or embedded estimate produced NaN/Inf.");
                            }

                            auto outcome = std::visit(
                                [&](auto &c) {
                                    return c.update(h_lane, err_norm, cfg.error_order, nacc[itmp]);
                                },
                                controllers[itmp]);
                            double hnext = outcome.dt_new;

                            if (hnext / h_lane > cfg.max_step_change)
                                h_lane *= cfg.max_step_change;
                            else if (hnext / h_lane < 1.0 / cfg.max_step_change)
                                h_lane /= cfg.max_step_change;
                            else
                                h_lane = hnext;
                            hs[itmp] = h_lane;

                            if (!outcome.accepted) {
                                nrej[itmp]++;
                                continueloops[itmp] = true;
                                continue;
                            }
                            nacc[itmp]++;
                        } else {
                            // Fixed-step path: no err_norm, so check xnext
                            // directly for this lane.
                            check_state_finite_or_throw(xnext.head(ode.x_vars()),
                                                        xis[itmp][ode.t_var()], hs[itmp],
                                                        "ParallelDriver::stepper.step", itmp);
                        }

                        // Event detection (per lane).
                        bool eventbreak = false;
                        if (!events.empty()) {
                            eventbreak = EventHandler::check_crossings(
                                events, prev_event_vals_s[itmp], next_event_vals_s[itmp], xnext,
                                ode.t_var(), eventtimes_s[itmp], xis[itmp][ode.t_var()]);
                        }

                        xis[itmp] = xnext;
                        xdotis[itmp] = xdotnext;
                        prev_event_vals_s[itmp] = next_event_vals_s[itmp];

                        if (storestates) {
                            if (storemidpoints) {
                                states_s[itmp].push_back(xnext_mid);
                                if (storederivs) {
                                    ScalarDeriv xdot_mid(ode.output_rows());
                                    xdot_mid.setZero();
                                    ode.compute(xnext_mid, xdot_mid);
                                    derivs_s[itmp].push_back(xdot_mid);
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

            numrunning = 0;
            lastrunning = -1;
            for (int i = 0; i < ntrajs; ++i) {
                if (continueloops[i]) {
                    numrunning++;
                    lastrunning = i;
                }
            }
        }

        return xis;
    }
};

} // namespace tycho::integrators
