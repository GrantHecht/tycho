#pragma once

/// @file astro.h
/// @brief Tycho Astrodynamics subsystem umbrella header.
///
/// Including this header brings in all public astrodynamics types:
/// Kepler two-body propagation, CR3BP dynamics, MEE/Cartesian
/// coordinate-conversion VectorFunctions, J2 perturbations, solar-sail
/// thruster models, and Lambert solvers.

#include "tycho/vector_functions.h"

#include "tycho/detail/astro/kepler/kepler_model.h"
#include "tycho/detail/astro/kepler/kepler_propagator.h"
#include "tycho/detail/astro/conversions/kepler_utils.h"
#include "tycho/detail/astro/kepler/kepler_propagation.h"
#include "tycho/detail/astro/dynamics/cartesian_dynamics.h"
#include "tycho/detail/astro/dynamics/crtbp_dynamics.h"
#include "tycho/detail/astro/dynamics/j2.h"
#include "tycho/detail/astro/dynamics/mee_dynamics.h"
#include "tycho/detail/astro/conversions/cartesian_to_mee.h"
#include "tycho/detail/astro/conversions/mee_to_cartesian.h"
#include "tycho/detail/astro/kepler/lambert_solvers.h"
#include "tycho/detail/astro/dynamics/thruster_models.h"
