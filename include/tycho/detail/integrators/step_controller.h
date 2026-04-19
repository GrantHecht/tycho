// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace tycho::integrators {

/// Result of one controller step decision.
///
/// Fields:
///   accepted — whether the current step should be accepted.
///   dt_new   — proposed next dt. The outer loop may further clamp to
///              [min_step_size, max_step_size] and the max_step_change_ ratio;
///              controllers only express their own qmin/qmax. dt_new preserves
///              the sign of h (forward or backward integration).
///   q        — reciprocal of the raw step-growth factor (Julia convention:
///              dt_new = dt / q). Exposed for diagnostics/tests.
struct ControllerOutput {
    bool accepted;
    double dt_new;
    double q;
};

/// Runtime selector for controller kind. User-facing enum.
enum class IVPController {
    I,   // Standard integral controller (Julia: IController)
    PI,  // Proportional-integral controller (Julia: PIController)
    PID, // Proportional-integral-derivative controller (Julia: PIDController)
};

namespace detail {

/// Build a "field must satisfy CONDITION; got VALUE" message.
inline std::string controller_invariant_msg(const char *controller, const char *field,
                                            const char *condition, double value) {
    std::string msg = controller;
    msg += ": ";
    msg += field;
    msg += " must ";
    msg += condition;
    msg += "; got ";
    msg += std::to_string(value);
    return msg;
}

} // namespace detail

// -----------------------------------------------------------------------------
// I controller — Julia form
// -----------------------------------------------------------------------------
/// Standard integral step-size controller, Julia formulation.
///
///     q   = EEst^(1/k) / γ,   clipped to [1/qmax, 1/qmin]
///     dt' = dt / q
///
/// where k = order + 1 (the order passed in is the adaptive_order of the
/// algorithm). On the first step, `qmax_first_step` replaces `qmax` to allow
/// a very large dt increase after initial-dt estimation. On reject, the
/// same formula runs but the step is not committed; the outer loop retries.
///
/// Tuning knobs and internal state both follow the project's
/// `snake_case_` trailing-underscore convention for publicly mutable
/// member fields. Accidental mid-integration mutation of `qold_` will
/// tear the controller's tracking — call reset() to clear cleanly.
/// Misconfigured tuning knobs are caught by validate() at install time
/// rather than producing silently-bad step cadence inside the loop.
///
/// Julia reference:
///   ~/.julia/packages/OrdinaryDiffEqCore/.../controllers.jl:171-199 (legacy)
///   ~/.julia/packages/OrdinaryDiffEqCore/.../controllers.jl:244-278 (new)
struct IController {
    double gamma = 0.9;
    double qmin = 1.0 / 5.0;
    double qmax = 10.0;
    double qmax_first_step = 10000.0;
    double qsteady_min = 1.0;
    double qsteady_max = 1.0;

    // Internal state
    double qold_ = 1.0;

    /// Throws std::invalid_argument if any tuning knob violates its
    /// invariant. Called by Integrator::set_controller at install time so
    /// misconfiguration is surfaced before the integration loop starts
    /// (rather than producing silently-wrong step cadence). Cheap one-shot
    /// check — not on the hot path.
    void validate() const {
        if (!(gamma > 0.0 && gamma <= 1.0))
            throw std::invalid_argument(detail::controller_invariant_msg(
                "IController", "gamma", "lie in (0, 1]", gamma));
        if (!(qmin > 0.0 && qmin < 1.0))
            throw std::invalid_argument(
                detail::controller_invariant_msg("IController", "qmin", "lie in (0, 1)", qmin));
        if (!(qmax > 1.0))
            throw std::invalid_argument(
                detail::controller_invariant_msg("IController", "qmax", "be > 1", qmax));
        if (!(qmax_first_step >= qmax))
            throw std::invalid_argument(detail::controller_invariant_msg(
                "IController", "qmax_first_step", "be >= qmax", qmax_first_step));
        if (!(qsteady_min <= qsteady_max))
            throw std::invalid_argument(detail::controller_invariant_msg(
                "IController", "qsteady_min", "be <= qsteady_max", qsteady_min));
    }

    ControllerOutput update(double h, double err_norm, int order, int naccept) {
        double eff_qmax = (naccept == 0) ? qmax_first_step : qmax;
        double q;
        if (err_norm == 0.0) {
            q = 1.0 / eff_qmax;
        } else {
            double expo = 1.0 / (static_cast<double>(order) + 1.0);
            double qtmp = std::pow(err_norm, expo) / gamma;
            q = std::max(1.0 / eff_qmax, std::min(1.0 / qmin, qtmp));
        }
        bool accepted = (err_norm <= 1.0);
        double q_applied = q;
        if (accepted && qsteady_min <= q && q <= qsteady_max) {
            q_applied = 1.0;
        }
        qold_ = q;
        // `q` field reflects the clipped, pre-deadband value so diagnostics
        // can recover the raw growth factor; `dt_new` reflects what we
        // actually apply (including deadband snap).
        return {accepted, h / q_applied, q};
    }

    void reset() { qold_ = 1.0; }
};

// -----------------------------------------------------------------------------
// PI controller — Julia legacy form
// -----------------------------------------------------------------------------
/// Proportional-integral step-size controller, Julia legacy form.
///
///     q₁₁ = EEst^β₁
///     q   = q₁₁ / errold^β₂,   with q/γ clipped to [1/qmax, 1/qmin]
///     dt' = dt / q (accept) or dt' = dt / min(1/qmin, q₁₁/γ) (reject)
///
/// After accept: errold := max(EEst, qoldinit).
///
/// Beta values are pre-scaled by method order per Julia's PIController
/// convention (β₁ and β₂ absorb the 1/order factor). The `order` parameter
/// to update() is unused here (kept for uniform signature with IController).
///
/// Julia reference:
///   ~/.julia/packages/OrdinaryDiffEqCore/.../controllers.jl:317-366 (legacy)
///   ~/.julia/packages/OrdinaryDiffEqCore/.../controllers.jl:401-467 (new)
struct PIController {
    double beta1 = 7.0 / 50.0;
    double beta2 = 2.0 / 25.0;
    double gamma = 0.9;
    double qmin = 1.0 / 5.0;
    double qmax = 10.0;
    double qmax_first_step = 10000.0;
    double qsteady_min = 1.0;
    double qsteady_max = 1.0;
    double qoldinit = 1.0e-4;

    // Internal state
    double errold_ = 1.0e-4;
    double q11_ = 1.0;

    void validate() const {
        if (!(beta1 >= 0.0))
            throw std::invalid_argument(
                detail::controller_invariant_msg("PIController", "beta1", "be >= 0", beta1));
        if (!(beta2 >= 0.0))
            throw std::invalid_argument(
                detail::controller_invariant_msg("PIController", "beta2", "be >= 0", beta2));
        if (!(gamma > 0.0 && gamma <= 1.0))
            throw std::invalid_argument(detail::controller_invariant_msg(
                "PIController", "gamma", "lie in (0, 1]", gamma));
        if (!(qmin > 0.0 && qmin < 1.0))
            throw std::invalid_argument(
                detail::controller_invariant_msg("PIController", "qmin", "lie in (0, 1)", qmin));
        if (!(qmax > 1.0))
            throw std::invalid_argument(
                detail::controller_invariant_msg("PIController", "qmax", "be > 1", qmax));
        if (!(qmax_first_step >= qmax))
            throw std::invalid_argument(detail::controller_invariant_msg(
                "PIController", "qmax_first_step", "be >= qmax", qmax_first_step));
        if (!(qsteady_min <= qsteady_max))
            throw std::invalid_argument(detail::controller_invariant_msg(
                "PIController", "qsteady_min", "be <= qsteady_max", qsteady_min));
        if (!(qoldinit > 0.0))
            throw std::invalid_argument(detail::controller_invariant_msg(
                "PIController", "qoldinit", "be > 0", qoldinit));
    }

    ControllerOutput update(double h, double err_norm, int /*order*/, int naccept) {
        double eff_qmax = (naccept == 0) ? qmax_first_step : qmax;
        double q;
        if (err_norm == 0.0) {
            q = 1.0 / eff_qmax;
            q11_ = 0.0;
        } else {
            q11_ = std::pow(err_norm, beta1);
            double qtmp = q11_ / std::pow(errold_, beta2);
            q = std::max(1.0 / eff_qmax, std::min(1.0 / qmin, qtmp / gamma));
        }

        bool accepted = (err_norm <= 1.0);
        if (accepted) {
            double q_applied = q;
            if (qsteady_min <= q && q <= qsteady_max)
                q_applied = 1.0;
            errold_ = std::max(err_norm, qoldinit);
            return {true, h / q_applied, q};
        }
        // Reject: dt shrink uses q11 (cached EEst^beta1), not q.
        double q_reject = std::min(1.0 / qmin, q11_ / gamma);
        // Report the q we actually used for dt_new so diagnostics stay truthful.
        return {false, h / q_reject, q_reject};
    }

    void reset() {
        errold_ = qoldinit;
        q11_ = 1.0;
    }
};

// -----------------------------------------------------------------------------
// PID controller — Julia legacy form
// -----------------------------------------------------------------------------
/// Proportional-integral-derivative step-size controller with Söderlind limiter.
///
///     dt_factor = ε₁^(β₁/k) · ε₂^(β₂/k) · ε₃^(β₃/k)
///     dt_factor := limiter(dt_factor)   where limiter(x) = 1 + atan(x - 1)
///     accept    := dt_factor >= accept_safety
///     dt'       = dt · dt_factor (accept) or dt · qold (reject)
///
/// where ε_i = 1/EEst_i and k = min(order, adaptive_order) + 1. Beta values
/// are NOT pre-scaled — the formula divides by k internally.
///
/// Julia reference:
///   ~/.julia/packages/OrdinaryDiffEqCore/.../controllers.jl:535-637
struct PIDController {
    double beta1 = 1.0;
    double beta2 = 0.0;
    double beta3 = 0.0;
    double accept_safety = 0.81;
    double qsteady_min = 1.0;
    double qsteady_max = 1.0;

    // Internal state: error history ε₁..ε₃ (= 1/EEst), and previous dt_factor
    std::array<double, 3> err_ = {1.0, 1.0, 1.0};
    double qold_ = 1.0;

    void validate() const {
        if (!(beta1 >= 0.0))
            throw std::invalid_argument(
                detail::controller_invariant_msg("PIDController", "beta1", "be >= 0", beta1));
        if (!(beta2 >= 0.0))
            throw std::invalid_argument(
                detail::controller_invariant_msg("PIDController", "beta2", "be >= 0", beta2));
        if (!(beta3 >= 0.0))
            throw std::invalid_argument(
                detail::controller_invariant_msg("PIDController", "beta3", "be >= 0", beta3));
        if (!(accept_safety > 0.0 && accept_safety <= 1.0))
            throw std::invalid_argument(detail::controller_invariant_msg(
                "PIDController", "accept_safety", "lie in (0, 1]", accept_safety));
        if (!(qsteady_min <= qsteady_max))
            throw std::invalid_argument(detail::controller_invariant_msg(
                "PIDController", "qsteady_min", "be <= qsteady_max", qsteady_min));
    }

    static double default_limiter(double x) { return 1.0 + std::atan(x - 1.0); }

    ControllerOutput update(double h, double err_norm, int order, int /*naccept*/) {
        constexpr double eps_min = 2.220446049250313e-16; // eps(Float64)
        double eest = std::max(err_norm, eps_min);
        err_[0] = 1.0 / eest;

        double k = static_cast<double>(order) + 1.0;
        double dt_factor = std::pow(err_[0], beta1 / k) * std::pow(err_[1], beta2 / k) *
                           std::pow(err_[2], beta3 / k);
        dt_factor = default_limiter(dt_factor);

        bool accepted = (dt_factor >= accept_safety);
        qold_ = dt_factor;

        if (accepted) {
            double dt_factor_applied = dt_factor;
            double inv_df = 1.0 / dt_factor;
            if (qsteady_min <= inv_df && inv_df <= qsteady_max)
                dt_factor_applied = 1.0;
            // Shift history on accept
            err_[2] = err_[1];
            err_[1] = err_[0];
            return {true, h * dt_factor_applied, 1.0 / dt_factor};
        }
        // Reject: use the same (rejected, < accept_safety) factor to shrink dt.
        return {false, h * dt_factor, 1.0 / dt_factor};
    }

    void reset() {
        err_ = {1.0, 1.0, 1.0};
        qold_ = 1.0;
    }
};

} // namespace tycho::integrators
