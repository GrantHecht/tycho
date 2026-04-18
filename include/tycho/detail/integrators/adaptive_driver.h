// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================

#pragma once

#include "tycho/detail/integrators/error_norm.h"
#include "tycho/detail/integrators/event_handler.h"
#include "tycho/detail/integrators/step_controller.h"
#include "tycho/detail/integrators/stepper.h"

#include <Eigen/Core>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace tycho::integrators {

/// Adaptive step-size integration driver.
///
/// Composes Stepper + IController + EventHandler to perform a full
/// adaptive integration from x(t0) to x(tf). Extracted from the scalar
/// path of Integrator::integrate_impl.
///
/// Template parameters:
///   Alg        — IVPAlg enum (DOPRI54, DOPRI87)
///   Controller — step-size controller (IController or compatible)
///   DODE       — ODE type
///   Scalar     — numeric type (typically double)
template <IVPAlg Alg, class Controller, class DODE, class Scalar = double>
struct AdaptiveDriver {

    using ODEState = typename DODE::template Input<Scalar>;
    using ODEDeriv = typename DODE::template Output<Scalar>;
    using EventPack = typename EventHandler::EventPack;

    Stepper<Alg, DODE, Scalar> stepper_;
    Controller controller_;

    // Tolerances and step bounds
    ODEDeriv abs_tols_;
    ODEDeriv rel_tols_;
    Scalar def_step_size_ = 0.01;
    Scalar max_step_change_ = 10.0;
    int max_steps_ = 1'000'000;
    bool adaptive_ = true;
    int error_order_ = RKCoeffs<Alg>::ErrorOrder;
    ErrorNormType error_norm_type_ = ErrorNormType::RMS;

    // Step statistics
    int naccept_count_ = 0;
    int nreject_count_ = 0;

    // Event parameters
    int max_event_iters_ = 20;
    Scalar event_tol_ = 1e-12;

    /// Perform adaptive integration from x(t0) to x(tf).
    ///
    /// ControlFn must be callable with (ODEState&).
    /// Results are stored in `states` and `derivs` vectors if requested.
    template <class ControlFn>
    ODEState integrate(const DODE &ode, const ODEState &x, Scalar tf,
                       const std::vector<EventPack> &events,
                       std::vector<std::vector<Eigen::Vector2d>> &eventtimes, bool storestates,
                       bool storederivs, bool storemidpoints, std::vector<ODEState> &states,
                       std::vector<ODEDeriv> &derivs, ControlFn &&update_control) {

        if (x.size() != ode.input_rows()) {
            throw std::invalid_argument("Incorrectly sized input state.");
        }

        Scalar t0 = x[ode.t_var()];
        Scalar H = tf - t0;
        int numsteps = int(std::abs(H / def_step_size_)) + 1;
        Scalar h = Scalar(0.9) * (H / Scalar(numsteps));

        ODEState xi = x;
        update_control(xi);

        ODEState xnext = xi;
        ODEState xnext_est = xi;
        ODEState xnext_mid = xi;

        ODEDeriv xdoti(ode.output_rows());
        xdoti.setZero();
        ode.compute(xi, xdoti);
        ODEDeriv xdotnext = xdoti;

        // Initialize FSAL with the derivative at x0
        stepper_.k_fsal_ = xdoti;
        stepper_.fsal_valid_ = true;

        // Event state
        using Vector1 = Eigen::Matrix<double, 1, 1>;
        std::vector<Vector1> prev_event_vals(events.size());
        std::vector<Vector1> next_event_vals(events.size());

        for (int j = 0; j < static_cast<int>(events.size()); j++) {
            prev_event_vals[j].setZero();
            next_event_vals[j].setZero();

            if (std::get<0>(events[j]).input_rows() != ode.input_rows()) {
                throw std::invalid_argument(
                    "Input size of event function must equal input size of ode_.");
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

        bool continueloop = true;

        while (continueloop) {
            if (naccept_count_ + nreject_count_ >= max_steps_) {
                throw std::runtime_error(
                    "AdaptiveDriver exceeded max_steps (" + std::to_string(max_steps_) +
                    "); raise via max_steps_ or loosen tolerances.");
            }
            Scalar tnext = xi[ode.t_var()] + h;

            if (H > 0.0) {
                if ((tnext - tf) >= 0.0) {
                    h = tf - xi[ode.t_var()];
                    tnext = tf;
                    continueloop = false;
                }
            } else {
                if ((tnext - tf) <= 0.0) {
                    h = tf - xi[ode.t_var()];
                    tnext = tf;
                    continueloop = false;
                }
            }

            xnext.setZero();
            xnext_est.setZero();
            xnext_mid.setZero();

            stepper_.step(ode, xi, tnext, xnext, xnext_est, storemidpoints || storederivs,
                          xnext_mid, update_control);

            if (adaptive_) {
                auto u_x_vars = ode.x_vars();
                auto utilde = xnext.head(u_x_vars) - xnext_est.head(u_x_vars);
                auto res = scaled_residuals(utilde, xi.head(u_x_vars),
                                            xnext.head(u_x_vars), abs_tols_,
                                            rel_tols_);
                double err_norm = error_norm(res, error_norm_type_);
                auto outcome = controller_.update(h, err_norm, error_order_,
                                                  naccept_count_);
                double hnext = outcome.dt_new;

                if (hnext / h > max_step_change_)
                    h *= max_step_change_;
                else if (hnext / h < 1.0 / max_step_change_)
                    h /= max_step_change_;
                else
                    h = hnext;

                if (!outcome.accepted) {
                    nreject_count_++;
                    continueloop = true;
                    continue;
                }
                naccept_count_++;
            }

            // Event detection
            bool eventbreak = false;
            if (!events.empty()) {
                eventbreak = EventHandler::check_crossings(
                    events, prev_event_vals, next_event_vals, xnext, ode.t_var(), eventtimes,
                    xi[ode.t_var()]);
            }

            xi = xnext;
            xdoti = stepper_.k_fsal_; // derivative at xnext (from FSAL or midpoint)
            prev_event_vals = next_event_vals;

            if (storestates) {
                if (storemidpoints) {
                    states.push_back(xnext_mid);
                    if (storederivs) {
                        xdotnext.setZero();
                        ode.compute(xnext_mid, xdotnext);
                        derivs.push_back(xdotnext);
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
};

} // namespace tycho::integrators
