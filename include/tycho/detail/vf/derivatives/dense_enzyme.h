// =============================================================================
// Tycho — Copyright 2026-present Grant R. Hecht. Apache 2.0.
// EnzymeAD specializations for DenseFirstDerivatives / DenseSecondDerivatives.
// Self-guarded on TYCHO_HAS_ENZYME_AD; safe to include unconditionally.
// =============================================================================
#pragma once

#include "tycho/detail/vf/derivatives/dense_derivatives.h"

namespace tycho::vf {

#ifndef TYCHO_HAS_ENZYME_AD

namespace detail {
template <class> struct enzyme_dependent_false : std::false_type {};
} // namespace detail

template <class Derived, int IR, int OR>
struct DenseFirstDerivatives<Derived, IR, OR, DenseDerivativeMode::EnzymeAD>
    : DenseFunctionBase<Derived, IR, OR> {
    static_assert(detail::enzyme_dependent_false<Derived>::value,
        "DenseDerivativeMode::EnzymeAD requires CMake option ENABLE_ENZYME_AD=ON "
        "and the Enzyme Clang plugin installed. See CLAUDE.md for the override "
        "invocation.");
};

template <class Derived, int IR, int OR, DenseDerivativeMode JMode>
struct DenseSecondDerivatives<Derived, IR, OR, JMode, DenseDerivativeMode::EnzymeAD>
    : DenseFirstDerivatives<Derived, IR, OR, JMode> {
    static_assert(detail::enzyme_dependent_false<Derived>::value,
        "DenseDerivativeMode::EnzymeAD Hessian requires CMake option ENABLE_ENZYME_AD=ON "
        "and the Enzyme Clang plugin installed. Note: EnzymeAD Hessian also requires "
        "Phase 2 of the Enzyme rollout to have landed; before Phase 2, pair "
        "<EnzymeAD, AutodiffFwd> instead.");
};

#else // TYCHO_HAS_ENZYME_AD

// Active specializations land in Task 1.7 (Phase 1 Jacobian) and Task 2.x (Phase 2 Hessian).
// Until 1.7 is complete, this branch is empty so an attempted instantiation under
// TYCHO_HAS_ENZYME_AD will fail at the primary template (no specialization match).

#endif // TYCHO_HAS_ENZYME_AD

} // namespace tycho::vf
