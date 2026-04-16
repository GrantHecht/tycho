// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================

#pragma once
#include <cmath>

namespace tycho::integrators {

struct IController {
    double safety = 0.9;
    double exponent_bias = 1.0;
    double prev_err_norm_ = 0.0;

    double propose_step(double h, double err_norm, int order) const {
        return safety * h * std::pow(1.0 / err_norm, 1.0 / (order + exponent_bias));
    }

    void accept(double err_norm) { prev_err_norm_ = err_norm; }
    void reject(double err_norm) { prev_err_norm_ = err_norm; }
    void reset() { prev_err_norm_ = 0.0; }
};

} // namespace tycho::integrators
