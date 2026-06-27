#pragma once

/// @file
/// @brief Runge-Kutta integrators — umbrella header for the integrators subsystem.
///
/// Includes rk_coeffs.h (Butcher tableaux), rk_steppers.h (collocation stepper),
/// and integrator.h (the main Integrator class with adaptive integration and
/// event detection).

// Tycho — Runge-Kutta steppers and coefficients

#include "tycho/vector_functions.h"

#include "tycho/detail/integrators/rk_coeffs.h"
#include "tycho/detail/integrators/rk_steppers.h"
#include "tycho/detail/integrators/integrator.h"
