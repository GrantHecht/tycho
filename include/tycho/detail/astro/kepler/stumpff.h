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

/// @internal
/// @brief Four-tuple of universal-variable Stumpff functions (U0, U1, U2, U3).
///
/// The LCD Kepler kernel produces all four together as the converged
/// universal-variable state.  Bundling them as a named aggregate replaces
/// loose scalar fields and exposes the natural Stumpff-index grouping that
/// downstream consumers read in groups.
/// @tparam Scalar Numeric scalar type (double or Eigen::Array SuperScalar).
/// @endinternal
template <class Scalar> struct Stumpff {
    Scalar U0; ///< @internal U0 universal Stumpff function value. @endinternal
    Scalar U1; ///< @internal U1 universal Stumpff function value. @endinternal
    Scalar U2; ///< @internal U2 universal Stumpff function value. @endinternal
    Scalar U3; ///< @internal U3 universal Stumpff function value. @endinternal
};

} // namespace tycho::astro::detail
