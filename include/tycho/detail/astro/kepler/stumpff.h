// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
//
// Stumpff aggregate: a 4-tuple (U0, U1, U2, U3) of universal-variable
// Stumpff functions.  The LCD kernel produces these together as the
// converged universal-variable state; the IFT layer also passes them
// in groups (∂U/∂X and ∂U/∂α are each 4-tuples).  Bundling them as a
// named aggregate replaces the loose `Scalar U0, U1, U2, U3` field
// layout flagged by the type-design review (S7).
// =============================================================================
#pragma once

#include <Eigen/Core>

namespace tycho::astro::detail {

template <class Scalar>
struct Stumpff {
    Scalar U0;
    Scalar U1;
    Scalar U2;
    Scalar U3;

    // Elementwise mask-blend used by the SIMD bailout snapshot/restore
    // pattern in kepler_lcd_iterate_simd_ellipse.  The Mask type is the
    // SS-typed Eigen::Array<bool, W, 1> — the underlying per-field
    // .select(...) member is provided by Eigen::Array.
    template <class Mask>
    Stumpff select(const Mask &m, const Stumpff &other) const {
        return {m.select(U0, other.U0), m.select(U1, other.U1),
                m.select(U2, other.U2), m.select(U3, other.U3)};
    }
};

} // namespace tycho::astro::detail
