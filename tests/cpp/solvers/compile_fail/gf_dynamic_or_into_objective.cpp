// MUST NOT COMPILE.
//
// A GenericFunction whose output size is not statically 1 cannot be an
// objective: the pack hook it would need, pack_into_objective_interface, only
// exists on GFConcept<IR, 1>. Before the seam this failed by accident -- the
// generic constructor was selected and ObjectiveModel then failed to
// instantiate somewhere inside itself -- so the diagnostic was whatever the
// Model happened to emit. The adapter in
// include/tycho/detail/vf/type_erasure/gf_type_erasure.h authors it instead,
// and guards the pack call with the same `if constexpr (OR == 1)` condition it
// asserts on, so nothing is instantiated behind the failed check.
//
// Expected: the authored "hven objective adapter: only GenericFunction<IR, 1>"
// static_assert, and NOT an error-severity "no member named" beside it.

#include <tycho/tycho.h>

tycho::solvers::ObjectiveInterface probe_gf_dynamic_or_into_objective() {
    auto args = tycho::vf::Arguments<3>();
    auto expr = 2.0 * args;

    // Output size is dynamic here, not statically 1.
    tycho::vf::GenericFunction<-1, -1> gf(expr);

    return tycho::solvers::ObjectiveInterface(gf);
}
