// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt).
// =============================================================================

#include "tycho/detail/integrators/error_norm.h"
#include "tycho/detail/integrators/rk_coeffs.h"
#include "tycho/detail/integrators/step_controller.h"

namespace tycho {
using namespace tycho::vf;
using namespace tycho::integrators;

void RKFlagsBuild(nb::module_ &m) {
    nb::enum_<IVPAlg>(
        m, "IVPAlg",
        R"doc(Adaptive Runge-Kutta algorithm used by an integrator or mesh-error estimator.

Pass as the first argument to ``integrator(alg, dt)`` (or its string name,
e.g. ``"DOPRI87"``) to select the RK scheme. Only the eight values listed here
are runtime-selectable; the remaining internal variants are used by the
transcription and collocation subsystems.
)doc")
        .value("DOPRI54", IVPAlg::DOPRI54,
               "Dormand-Prince 5(4) — 7 stages, FSAL, adaptive (5th-order solution, "
               "4th-order error estimate).")
        .value("DOPRI87", IVPAlg::DOPRI87,
               "Dormand-Prince 8(7) — 13 stages, adaptive (8th-order solution, "
               "7th-order error estimate). Default algorithm.")
        .value("Tsit5", IVPAlg::Tsit5,
               "Tsitouras 5(4) — 7 stages, FSAL, adaptive (5th-order solution, "
               "4th-order error estimate).")
        .value("BS3", IVPAlg::BS3,
               "Bogacki-Shampine 3(2) — 4 stages, FSAL, adaptive (3rd-order solution, "
               "2nd-order error estimate).")
        .value("BS5", IVPAlg::BS5,
               "Bogacki-Shampine 5(4) — 8 stages, adaptive with lazy interpolant "
               "(5th-order solution, 4th-order error estimate).")
        .value("Vern7", IVPAlg::Vern7,
               "Verner 7(6) — 10 stages, adaptive with lazy interpolant "
               "(7th-order solution, 6th-order error estimate).")
        .value("Vern8", IVPAlg::Vern8,
               "Verner 8(7) — 13 stages, adaptive with lazy interpolant "
               "(8th-order solution, 7th-order error estimate).")
        .value("Vern9", IVPAlg::Vern9,
               "Verner 9(8) — 16 stages, adaptive with lazy interpolant "
               "(9th-order solution, 8th-order error estimate).");

    nb::enum_<IVPController>(m, "IVPController",
                             R"doc(Adaptive step-size controller used by the integrator.

Controls how the step size is grown or shrunk after each accepted or rejected
step based on the local error estimate. Select via ``integrator.set_controller``
or an equivalent constructor argument.
)doc")
        .value("I", IVPController::I,
               "Integral (I) controller — step size scaled by EEst^(1/k) / gamma. "
               "Simple and robust; default for most methods.")
        .value("PI", IVPController::PI,
               "Proportional-integral (PI) controller — uses current and previous "
               "error estimates for smoother step growth.")
        .value("PID", IVPController::PID,
               "Proportional-integral-derivative (PID) controller with Soderlind limiter — "
               "uses three successive error estimates for the smoothest adaptation.");

    nb::enum_<ErrorNormType>(
        m, "ErrorNormType",
        R"doc(Norm used to reduce per-component local errors to a scalar for step control.

The scalar error estimate (EEst) is compared against 1.0 to decide
accept/reject and to scale the next step size.
)doc")
        .value("RMS", ErrorNormType::RMS,
               "Root-mean-square of scaled per-component errors. "
               "Matches Julia OrdinaryDiffEq default norm; default choice.")
        .value("MAX", ErrorNormType::MAX,
               "Maximum (L-infinity) of scaled per-component errors. "
               "More conservative; useful for stiff or sensitivity-critical problems.");
}

} // namespace tycho
