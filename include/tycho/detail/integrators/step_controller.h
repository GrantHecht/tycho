// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <variant>

namespace tycho::integrators {

/// Result of one controller step decision.
///
/// Fields:
///   accepted — whether the current step should be accepted.
///   dt_new   — proposed next dt. The outer loop may further clamp via the
///              max_step_change_ ratio; controllers only express their own
///              qmin/qmax. dt_new preserves the sign of h (forward or
///              backward integration). There is no hard min/max step size
///              cap — the only backstop against runaway h is max_steps.
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
///     q   = EEst^(1/k) / γ,   clipped to [1/qmax_, 1/qmin_]
///     dt' = dt / q
///
/// where k = order + 1 (the `order` parameter passed to update() is the
/// algorithm's `ErrorOrder` — the order of the embedded error estimator).
/// On the first step, `qmax_first_step_` replaces `qmax_` to allow a very
/// large dt increase after initial-dt estimation.
///
/// Matches OrdinaryDiffEqCore.jl `IController`.
struct IController {
    double gamma_ = 0.9;
    double qmin_ = 1.0 / 5.0;
    double qmax_ = 10.0;
    double qmax_first_step_ = 10000.0;
    double qsteady_min_ = 1.0;
    double qsteady_max_ = 1.0;

    // Internal state
    double qold_ = 1.0;

    void validate() const {
        if (!(gamma_ > 0.0 && gamma_ <= 1.0))
            throw std::invalid_argument(
                detail::controller_invariant_msg("IController", "gamma_", "lie in (0, 1]", gamma_));
        if (!(qmin_ > 0.0 && qmin_ < 1.0))
            throw std::invalid_argument(
                detail::controller_invariant_msg("IController", "qmin_", "lie in (0, 1)", qmin_));
        if (!(qmax_ > 1.0))
            throw std::invalid_argument(
                detail::controller_invariant_msg("IController", "qmax_", "be > 1", qmax_));
        if (!(qmax_first_step_ >= qmax_))
            throw std::invalid_argument(detail::controller_invariant_msg(
                "IController", "qmax_first_step_", "be >= qmax_", qmax_first_step_));
        if (!(qsteady_min_ <= qsteady_max_))
            throw std::invalid_argument(detail::controller_invariant_msg(
                "IController", "qsteady_min_", "be <= qsteady_max_", qsteady_min_));
    }

    ControllerOutput update(double h, double err_norm, int order, int naccept) {
        double eff_qmax = (naccept == 0) ? qmax_first_step_ : qmax_;
        double q;
        if (err_norm == 0.0) {
            q = 1.0 / eff_qmax;
        } else {
            double expo = 1.0 / (static_cast<double>(order) + 1.0);
            double qtmp = std::pow(err_norm, expo) / gamma_;
            q = std::max(1.0 / eff_qmax, std::min(1.0 / qmin_, qtmp));
        }
        bool accepted = (err_norm <= 1.0);
        double q_applied = q;
        if (accepted && qsteady_min_ <= q && q <= qsteady_max_) {
            q_applied = 1.0;
        }
        qold_ = q;
        // `q` field reflects the clipped-but-not-deadbanded growth factor
        // (clamped to [1/qmax_, 1/qmin_]); `dt_new` reflects what we actually
        // apply, including the deadband snap to 1.0.
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
///     q   = q₁₁ / errold^β₂,   with q/γ clipped to [1/qmax_, 1/qmin_]
///     dt' = dt / q (accept) or dt' = dt / min(1/qmin_, q₁₁/γ) (reject)
///
/// After accept: errold := max(EEst, qoldinit_).
///
/// Beta values are pre-scaled by method order per Julia's PIController
/// convention (β₁ and β₂ absorb the 1/order factor). The `order` parameter
/// to update() is unused here (kept for uniform signature with IController).
///
/// Matches OrdinaryDiffEqCore.jl `PIController`.
struct PIController {
    double beta1_ = 7.0 / 50.0;
    double beta2_ = 2.0 / 25.0;
    double gamma_ = 0.9;
    double qmin_ = 1.0 / 5.0;
    double qmax_ = 10.0;
    double qmax_first_step_ = 10000.0;
    double qsteady_min_ = 1.0;
    double qsteady_max_ = 1.0;
    double qoldinit_ = 1.0e-4;

    // Internal state
    double errold_ = 1.0e-4;
    double q11_ = 1.0;

    void validate() const {
        if (!(beta1_ >= 0.0))
            throw std::invalid_argument(
                detail::controller_invariant_msg("PIController", "beta1_", "be >= 0", beta1_));
        if (!(beta2_ >= 0.0))
            throw std::invalid_argument(
                detail::controller_invariant_msg("PIController", "beta2_", "be >= 0", beta2_));
        if (!(gamma_ > 0.0 && gamma_ <= 1.0))
            throw std::invalid_argument(detail::controller_invariant_msg("PIController", "gamma_",
                                                                         "lie in (0, 1]", gamma_));
        if (!(qmin_ > 0.0 && qmin_ < 1.0))
            throw std::invalid_argument(
                detail::controller_invariant_msg("PIController", "qmin_", "lie in (0, 1)", qmin_));
        if (!(qmax_ > 1.0))
            throw std::invalid_argument(
                detail::controller_invariant_msg("PIController", "qmax_", "be > 1", qmax_));
        if (!(qmax_first_step_ >= qmax_))
            throw std::invalid_argument(detail::controller_invariant_msg(
                "PIController", "qmax_first_step_", "be >= qmax_", qmax_first_step_));
        if (!(qsteady_min_ <= qsteady_max_))
            throw std::invalid_argument(detail::controller_invariant_msg(
                "PIController", "qsteady_min_", "be <= qsteady_max_", qsteady_min_));
        if (!(qoldinit_ > 0.0))
            throw std::invalid_argument(
                detail::controller_invariant_msg("PIController", "qoldinit_", "be > 0", qoldinit_));
    }

    ControllerOutput update(double h, double err_norm, int /*order*/, int naccept) {
        double eff_qmax = (naccept == 0) ? qmax_first_step_ : qmax_;
        double q;
        if (err_norm == 0.0) {
            q = 1.0 / eff_qmax;
            q11_ = 0.0;
        } else {
            q11_ = std::pow(err_norm, beta1_);
            double qtmp = q11_ / std::pow(errold_, beta2_);
            q = std::max(1.0 / eff_qmax, std::min(1.0 / qmin_, qtmp / gamma_));
        }

        bool accepted = (err_norm <= 1.0);
        if (accepted) {
            double q_applied = q;
            if (qsteady_min_ <= q && q <= qsteady_max_)
                q_applied = 1.0;
            errold_ = std::max(err_norm, qoldinit_);
            return {true, h / q_applied, q};
        }
        // Reject: dt shrink uses q11 (cached EEst^beta1), not q.
        double q_reject = std::min(1.0 / qmin_, q11_ / gamma_);
        return {false, h / q_reject, q_reject};
    }

    void reset() {
        errold_ = qoldinit_;
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
///     accept    := dt_factor >= accept_safety_
///     dt'       = dt · dt_factor on both accept and reject.
///
/// where ε_i = 1/EEst_i and k = order + 1 (the `order` parameter is the
/// algorithm's `ErrorOrder` — order of the embedded error estimator).
/// Beta values are NOT pre-scaled — the formula divides by k internally.
///
/// `qold_` is diagnostic-only (last-computed dt_factor, readable by tests);
/// it is not consumed by the control law itself.
///
/// Matches OrdinaryDiffEqCore.jl `PIDController`.
struct PIDController {
    double beta1_ = 1.0;
    double beta2_ = 0.0;
    double beta3_ = 0.0;
    double accept_safety_ = 0.81;
    double qsteady_min_ = 1.0;
    double qsteady_max_ = 1.0;

    // Internal state: error history ε₁..ε₃ (= 1/EEst), and previous dt_factor
    std::array<double, 3> err_ = {1.0, 1.0, 1.0};
    double qold_ = 1.0;

    void validate() const {
        if (!(beta1_ >= 0.0))
            throw std::invalid_argument(
                detail::controller_invariant_msg("PIDController", "beta1_", "be >= 0", beta1_));
        if (!(beta2_ >= 0.0))
            throw std::invalid_argument(
                detail::controller_invariant_msg("PIDController", "beta2_", "be >= 0", beta2_));
        if (!(beta3_ >= 0.0))
            throw std::invalid_argument(
                detail::controller_invariant_msg("PIDController", "beta3_", "be >= 0", beta3_));
        if (!(accept_safety_ > 0.0 && accept_safety_ <= 1.0))
            throw std::invalid_argument(detail::controller_invariant_msg(
                "PIDController", "accept_safety_", "lie in (0, 1]", accept_safety_));
        if (!(qsteady_min_ <= qsteady_max_))
            throw std::invalid_argument(detail::controller_invariant_msg(
                "PIDController", "qsteady_min_", "be <= qsteady_max_", qsteady_min_));
    }

    static double default_limiter(double x) { return 1.0 + std::atan(x - 1.0); }

    ControllerOutput update(double h, double err_norm, int order, int /*naccept*/) {
        constexpr double eps_min = 2.220446049250313e-16; // eps(Float64)
        double eest = std::max(err_norm, eps_min);
        err_[0] = 1.0 / eest;

        double k = static_cast<double>(order) + 1.0;
        double dt_factor = std::pow(err_[0], beta1_ / k) * std::pow(err_[1], beta2_ / k) *
                           std::pow(err_[2], beta3_ / k);
        dt_factor = default_limiter(dt_factor);

        bool accepted = (dt_factor >= accept_safety_);
        qold_ = dt_factor;

        if (accepted) {
            double dt_factor_applied = dt_factor;
            double inv_df = 1.0 / dt_factor;
            if (qsteady_min_ <= inv_df && inv_df <= qsteady_max_)
                dt_factor_applied = 1.0;
            // Shift history on accept
            err_[2] = err_[1];
            err_[1] = err_[0];
            return {true, h * dt_factor_applied, 1.0 / dt_factor};
        }
        // Reject: use the same (rejected, < accept_safety_) factor to shrink dt.
        return {false, h * dt_factor, 1.0 / dt_factor};
    }

    void reset() {
        err_ = {1.0, 1.0, 1.0};
        qold_ = 1.0;
    }
};

/// Runtime-dispatched controller state. Integrators and extracted drivers
/// hold this as the concrete storage for "the current controller", then
/// dispatch through `update_controller` / `reset_controller` below.
using ControllerVariant = std::variant<IController, PIController, PIDController>;

/// Minimum surface every ControllerVariant alternative must provide. A
/// future controller kind that forgets `update()` or typos its signature
/// fails here at compile time with a single readable message, instead of
/// producing cryptic `std::visit` lambda errors at every call site.
template <class C>
concept Controller = requires(C &c, double h, double err_norm, int order, int naccept) {
    { c.update(h, err_norm, order, naccept) } -> std::same_as<ControllerOutput>;
    { c.reset() } -> std::same_as<void>;
    { c.validate() } -> std::same_as<void>;
};

static_assert(Controller<IController>);
static_assert(Controller<PIController>);
static_assert(Controller<PIDController>);

/// Forward an update call to whichever controller is active in the variant.
/// Centralizes the boilerplate `std::visit([&](auto& c) { return c.update(…); }, v)`
/// at every step-loop site. `[[gnu::always_inline]]` because this is on the
/// adaptive hot path once per accepted step — without it, Clang LTO can
/// leave a real call frame here and cost double-digit-% on high-order
/// methods where per-step overhead dominates (e.g. Vern7/8/9 PI). Benefits:
/// single point of dispatch, concept constrains what alternatives may carry.
[[gnu::always_inline]] inline ControllerOutput
update_controller(ControllerVariant &v, double h, double err_norm, int order, int naccept) {
    return std::visit([&](auto &c) { return c.update(h, err_norm, order, naccept); }, v);
}

/// Reset whichever controller is active in the variant to its first-step
/// state (err history, qold_, etc.) so a cloned prototype starts clean.
[[gnu::always_inline]] inline void reset_controller(ControllerVariant &v) {
    std::visit([](auto &c) { c.reset(); }, v);
}

/// Validate whichever controller is active in the variant — checks the
/// per-controller invariants described by `IController::validate()` etc.
/// Not hot path; `inline` only.
inline void validate_controller(const ControllerVariant &v) {
    std::visit([](const auto &c) { c.validate(); }, v);
}

} // namespace tycho::integrators
