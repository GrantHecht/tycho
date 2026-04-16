// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================

#pragma once

#include "tycho/detail/integrators/rk_coeffs.h"
#include "tycho/detail/utils/memory_management.h"
#include <Eigen/Core>

namespace tycho::integrators {

/// Stateful IVP stepper that performs one RK step using the BumpAllocator
/// for stage buffers. Maintains FSAL cache between consecutive steps.
///
/// Template parameters:
///   Alg   — IVPAlg enum selecting the Butcher tableau
///   DODE  — ODE type (provides Input<Scalar>, Output<Scalar>, XV, compute)
///   Scalar — numeric type (double, SuperScalar, etc.)
template <IVPAlg Alg, class DODE, class Scalar> struct Stepper {

    using RKData = RKCoeffs<Alg>;
    static constexpr int Stages = RKData::Stages;
    static constexpr int Stgsm1 = Stages - 1;

    using ODEState = typename DODE::template Input<Scalar>;
    using ODEDeriv = typename DODE::template Output<Scalar>;

    // FSAL cache
    ODEDeriv k_fsal_;
    bool fsal_valid_ = false;

    /// Perform one RK step from state x to time tf.
    ///
    /// On output: xf = next state, xf_est = embedded estimate,
    /// xf_mid = midpoint state (if compute_midpoint=true).
    ///
    /// ControlFn must be callable with (ODEState<Scalar>&).
    // TODO(SP2): pre-compose control function instead of per-stage mutation
    template <class ControlFn>
    void step(const DODE &ode, const ODEState &x, Scalar tf, ODEState &xf, ODEState &xf_est,
              bool compute_midpoint, ODEState &xf_mid, ControlFn &&update_control) {

        auto Impl = [&](auto &k_vals, auto &xtup) {
            xtup = x;
            Scalar t0 = xtup[ode.t_var()];
            Scalar h = tf - t0;

            if (fsal_valid_) {
                k_vals[0] = k_fsal_ * h;
            } else {
                update_control(xtup);
                ode.compute(xtup, k_vals[0]);
                k_vals[0] *= h;
            }

            for (int i = 0; i < Stgsm1; i++) {
                Scalar ti = t0 + RKData::C[i] * h;
                xtup = x;
                xtup[ode.t_var()] = ti;
                const int ip1 = i + 1;
                for (int j = 0; j < ip1; j++) {
                    xtup.template segment<DODE::XV>(0, ode.x_vars()) +=
                        Scalar(RKData::A[i][j]) * k_vals[j];
                }
                update_control(xtup);
                ode.compute(xtup, k_vals[ip1]);
                k_vals[ip1] *= h;
            }

            // Compute xf = x + h * sum(B[i] * k[i])
            xtup = x;
            xtup[ode.t_var()] = tf;
            for (int i = 0; i < Stages; i++) {
                xtup.template segment<DODE::XV>(0, ode.x_vars()) +=
                    Scalar(RKData::B[i]) * k_vals[i];
            }
            update_control(xtup);
            xf = xtup;

            // Update FSAL cache (always update for FSAL methods)
            if constexpr (RKData::FSAL) {
                k_fsal_ = k_vals.back() * (1.0 / h);
                fsal_valid_ = true;
            }

            // Compute xf_est = x + h * sum(Bhat[i] * k[i])
            xtup = x;
            xtup[ode.t_var()] = tf;
            for (int i = 0; i < Stages; i++) {
                xtup.template segment<DODE::XV>(0, ode.x_vars()) +=
                    Scalar(RKData::Bhat[i]) * k_vals[i];
            }
            xf_est = xtup;

            // Compute midpoint
            if (compute_midpoint) {
                xtup = x;
                xtup[ode.t_var()] = t0 + h / 2.0;
                for (int i = 0; i < Stages; i++) {
                    xtup.template segment<DODE::XV>(0, ode.x_vars()) +=
                        Scalar(RKData::Bmid[i] / 2.0) * k_vals[i];
                }
                if constexpr (!RKData::FSAL) {
                    k_vals.back().setZero();
                    ode.compute(xf, k_vals.back());
                    xtup.template segment<DODE::XV>(0, ode.x_vars()) +=
                        Scalar(RKData::Bmid.back() / 2.0) * k_vals.back() * h;
                    k_fsal_ = k_vals.back();
                }
                update_control(xtup);
                xf_mid = xtup;
            }
        };

        utils::BumpAllocator::allocate_run(
            Impl, utils::ArrayOfTempSpecs<ODEDeriv, Stages>(ode.output_rows(), 1),
            utils::TempSpec<ODEState>(ode.input_rows(), 1));
    }

    void reset_fsal() { fsal_valid_ = false; }

    static constexpr int error_order() { return RKData::ErrorOrder; }
};

} // namespace tycho::integrators
