// MUST NOT COMPILE.
//
// The tycho-side half of the visibility guarantee: a function type reaching
// ConstraintInterface with no SolverInterfaceAdapter specialization in scope
// is a compile error with an authored message.
//
// A composed expression is the honest way to probe this, because it is the
// case a translation unit can actually reach. The type below,
// Scaled<Arguments<3>>, is built in place and has no name to hang a
// registration on -- and it is exactly the shape of the two entry sites the
// seam caught in ode_phase_base.cpp, which had been erased twice and paying a
// second virtual dispatch per solver call with nothing to say so. It is also
// indistinguishable, to the compiler, from a registered family whose
// registration this TU failed to include: neither has an adapter in scope, and
// that is precisely why the registration must live in the file that defines
// the type.
//
// The fix at a site like this is one of the two sanctioned routes: register
// the family beside its definition if it is a named, reused type, or wrap the
// expression in a GenericFunction (which stores the same payload at the same
// single dispatch). Neither is applied here -- refusing is the point.
//
// Expected: the authored "hven constraint adapter: no SolverInterfaceAdapter<T>
// specialization" static_assert, and NOT an error-severity "no member named"
// beside it -- the primary template's install_constraint asserts and emplaces
// nothing.

#include <tycho/tycho.h>

tycho::solvers::ConstraintInterface probe_unregistered_expression() {
    auto args = tycho::vf::Arguments<3>();
    auto expr = 2.0 * args;

    return tycho::solvers::ConstraintInterface(expr);
}
