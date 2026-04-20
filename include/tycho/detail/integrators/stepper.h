// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================

#pragma once

#include "tycho/detail/integrators/rk_coeffs.h"
#include "tycho/detail/utils/memory_management.h"
#include <Eigen/Core>

namespace tycho::integrators {

/// Stateful IVP stepper that performs one RK step using the BumpAllocator
/// for stage buffers. Maintains FSAL cache between consecutive steps and
/// supports the interpolant extra-stage path used by BS5/Tsit5/Vern7/8/9
/// for dense-output midpoint reconstruction.
///
/// Template parameters:
///   Alg   — IVPAlg enum selecting the Butcher tableau
///   DODE  — ODE type (provides Input<Scalar>, Output<Scalar>, XV, compute)
///   Scalar — numeric type (double, SuperScalar, etc.)
template <IVPAlg Alg, class DODE, class Scalar> struct Stepper {

    using RKData = RKCoeffs<Alg>;
    static constexpr int Stages = RKData::Stages;
    static constexpr int Stgsm1 = Stages - 1;

    // k_vals_extra layout (matches the production RKCoeffs dense-output schema):
    //   [0]                 — f(xf)·h when !LastStageIsFxf (DOPRI87 / Vern*)
    //   [offset..offset+e]  — interpolation extra stages (BS5/Tsit5/Vern*)
    //   offset = LastStageIsFxf ? 0 : 1
    static constexpr int ExtraOffset = RKData::LastStageIsFxf ? 0 : 1;
    static constexpr int ExtraCount = RKData::InterpStages + ExtraOffset;

    using ODEState = typename DODE::template Input<Scalar>;
    using ODEDeriv = typename DODE::template Output<Scalar>;

    /// Snapshot of the FSAL cache for reject-rollback in adaptive drivers.
    /// AdaptiveDriver takes this before each step and restores on reject so
    /// the retry reads f(xi), not the unconditionally-overwritten f(xnext).
    struct FsalSnapshot {
        ODEDeriv k;
        bool valid;
    };

    /// Seed f(xi) into the FSAL cache and mark it valid. Used by batch/SIMD
    /// callers (ParallelDriver) that compute the per-lane derivative outside
    /// the stepper and want the next step() to reuse it as stage 0.
    void seed_fsal(const ODEDeriv &k) {
        k_fsal_ = k;
        fsal_valid_ = true;
    }

    /// Snapshot k_fsal_ + fsal_valid_ for later restore_fsal().
    FsalSnapshot snapshot_fsal() const { return {k_fsal_, fsal_valid_}; }

    /// Restore a previously-snapshotted FSAL state. Used by AdaptiveDriver on
    /// reject so the retry's stage 0 reads the pre-step f(xi).
    void restore_fsal(const FsalSnapshot &s) {
        k_fsal_ = s.k;
        fsal_valid_ = s.valid;
    }

    /// Invalidate the FSAL cache without touching k_fsal_'s contents.
    void reset_fsal() { fsal_valid_ = false; }

    /// Read-only access to the FSAL cache. Callers that read this post-step
    /// to recover f(xnext) must gate on fsal_valid() — the Vern* midpoint
    /// branch populates k_fsal_ but intentionally leaves fsal_valid_=false.
    ///
    /// Aliasing contract: the returned reference aliases the internal buffer
    /// and is invalidated by any subsequent call to step(), seed_fsal(), or
    /// reset_fsal(). Do not hold the reference across those calls — copy
    /// the value out first.
    const ODEDeriv &peek_fsal() const { return k_fsal_; }
    bool fsal_valid() const { return fsal_valid_; }

  private:
    // FSAL cache. Holds f(xf) at Scalar units (no h factor) after a step,
    // for any method where k_vals.back() contains f(xf)·h (FSAL or
    // LastStageIsFxf=true), or where the interpolant path's extra f(xf)
    // evaluation has been performed (LastStageIsFxf=false + compute_midpoint).
    ODEDeriv k_fsal_;
    bool fsal_valid_ = false;

  public:

    /// Perform one RK step from state x to time tf.
    ///
    /// On output: xf = next state, xf_est = embedded estimate,
    /// xf_mid = midpoint state (only written if compute_midpoint=true).
    /// k_fsal_ is updated to f(xf) when the method/compute_midpoint combination
    /// makes it available.
    ///
    /// ControlFn must be callable with (ODEState&).
    template <class ControlFn>
    void step(const DODE &ode, const ODEState &x, Scalar tf, ODEState &xf, ODEState &xf_est,
              bool compute_midpoint, ODEState &xf_mid, ControlFn &&update_control) {

        auto Impl = [&](auto &k_vals, auto &xtup, auto &k_vals_extra) {
            xtup = x;
            Scalar t0 = xtup[ode.t_var()];
            Scalar h = tf - t0;

            // Seed k_vals[0]. Reuse cached k_fsal_ = f(xi) (= f(prev xf)) if
            // the caller has signalled a valid FSAL state; otherwise compute
            // f(x)·h fresh.
            if (fsal_valid_) {
                k_vals[0] = k_fsal_ * h;
            } else {
                update_control(xtup);
                ode.compute(xtup, k_vals[0]);
                k_vals[0] *= h;
            }

            // Main RK stages.
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

            // xf = x + h · Σ B[i]·k[i]
            xtup = x;
            xtup[ode.t_var()] = tf;
            for (int i = 0; i < Stages; i++) {
                xtup.template segment<DODE::XV>(0, ode.x_vars()) +=
                    Scalar(RKData::B[i]) * k_vals[i];
            }
            update_control(xtup);
            xf = xtup;

            // Update FSAL cache for methods where k_vals.back() = f(xf)·h —
            // i.e., FSAL (DOPRI54, Tsit5, BS3, BS5) or LastStageIsFxf=true
            // (DOPRI87). For LastStageIsFxf=false (Vern7/8/9) the extra f(xf)
            // evaluation happens in the midpoint branch below.
            if constexpr (RKData::FSAL || RKData::LastStageIsFxf) {
                k_fsal_ = k_vals.back() * (Scalar(1.0) / h);
                fsal_valid_ = true;
            }

            // xf_est = x + h · Σ Bhat[i]·k[i]
            xtup = x;
            xtup[ode.t_var()] = tf;
            for (int i = 0; i < Stages; i++) {
                xtup.template segment<DODE::XV>(0, ode.x_vars()) +=
                    Scalar(RKData::Bhat[i]) * k_vals[i];
            }
            xf_est = xtup;

            // Midpoint with interpolant extra stages (BS5/Tsit5/Vern7/8/9).
            if (compute_midpoint) {
                // For LastStageIsFxf=false methods (Vern7/8/9), f(xf) is not
                // in k_vals.back(); evaluate it into k_vals_extra[0] so the
                // Bmid sum can pick it up, and seed k_fsal_ with f(xf).
                if constexpr (!RKData::LastStageIsFxf) {
                    ode.compute(xf, k_vals_extra[0]);
                    k_vals_extra[0] *= h;
                    k_fsal_ = k_vals_extra[0] * (Scalar(1.0) / h);
                    // Intentionally do NOT set fsal_valid_=true here. Although
                    // k_fsal_ now holds f(xf), promoting it to a FSAL seed
                    // would silently corrupt direct Stepper callers that issue
                    // the next step() call against a different x0 (the stale
                    // f(xf_prev) would be used as k[0]). Coordinated callers
                    // such as AdaptiveDriver — which always pass xi == prev xf
                    // — can opt in by setting stepper.fsal_valid_ themselves.
                }

                // Compute interpolation extra stages. Each extra stage e
                // builds x_e = x + h·(Σ_main ExtraA[e][j]·k_j +
                // Σ_extra ExtraA[e][Stages+j]·k_extra_j), evaluates f(t_e,x_e)·h,
                // and stores into k_vals_extra[ExtraOffset + e].
                if constexpr (RKData::InterpStages > 0) {
                    for (int e = 0; e < RKData::InterpStages; e++) {
                        xtup = x;
                        xtup[ode.t_var()] = t0 + RKData::ExtraC[e] * h;
                        for (int j = 0; j < Stages; j++) {
                            xtup.template segment<DODE::XV>(0, ode.x_vars()) +=
                                Scalar(RKData::ExtraA[e][j]) * k_vals[j];
                        }
                        for (int j = 0; j < e; j++) {
                            xtup.template segment<DODE::XV>(0, ode.x_vars()) +=
                                Scalar(RKData::ExtraA[e][Stages + j]) *
                                k_vals_extra[ExtraOffset + j];
                        }
                        update_control(xtup);
                        ode.compute(xtup, k_vals_extra[ExtraOffset + e]);
                        k_vals_extra[ExtraOffset + e] *= h;
                    }
                }

                // Assemble xf_mid = x + h·(Σ_main (Bmid[i]/2)·k_i +
                // (Bmid[Stages]/2)·k_extra[0] if !LastStageIsFxf +
                // Σ_interp (Bmid[Stages+ExtraOffset+e]/2)·k_extra[ExtraOffset+e]).
                xtup = x;
                xtup[ode.t_var()] = t0 + h / Scalar(2.0);
                for (int i = 0; i < Stages; i++) {
                    xtup.template segment<DODE::XV>(0, ode.x_vars()) +=
                        Scalar(RKData::Bmid[i] / 2.0) * k_vals[i];
                }
                if constexpr (!RKData::LastStageIsFxf) {
                    xtup.template segment<DODE::XV>(0, ode.x_vars()) +=
                        Scalar(RKData::Bmid[Stages] / 2.0) * k_vals_extra[0];
                }
                if constexpr (RKData::InterpStages > 0) {
                    for (int e = 0; e < RKData::InterpStages; e++) {
                        xtup.template segment<DODE::XV>(0, ode.x_vars()) +=
                            Scalar(RKData::Bmid[Stages + ExtraOffset + e] / 2.0) *
                            k_vals_extra[ExtraOffset + e];
                    }
                }
                update_control(xtup);
                xf_mid = xtup;
            }
        };

        utils::BumpAllocator::allocate_run(
            Impl, utils::ArrayOfTempSpecs<ODEDeriv, Stages>(ode.output_rows(), 1),
            utils::TempSpec<ODEState>(ode.input_rows(), 1),
            utils::ArrayOfTempSpecs<ODEDeriv, ExtraCount>(ode.output_rows(), 1));
    }

    static constexpr int error_order() { return RKData::ErrorOrder; }
};

} // namespace tycho::integrators
