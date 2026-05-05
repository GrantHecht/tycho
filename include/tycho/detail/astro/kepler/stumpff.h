// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
//
// Stumpff aggregate: a 4-tuple (U0, U1, U2, U3) of universal-variable
// Stumpff functions.  The LCD kernel produces these together as the
// converged universal-variable state; the IFT layer also passes them
// in groups (∂U/∂X and ∂U/∂α are each 4-tuples).  Bundling them as a
// named aggregate replaces the loose `Scalar U0, U1, U2, U3` field
// layout — the named accessors expose the natural row-by-Stumpff-index
// grouping that downstream consumers read in groups.
// =============================================================================
#pragma once

namespace tycho::astro::detail {

template <class Scalar> struct Stumpff {
    Scalar U0;
    Scalar U1;
    Scalar U2;
    Scalar U3;
};

} // namespace tycho::astro::detail
