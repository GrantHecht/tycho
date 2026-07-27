///////////////////////////////////////////////////////////////////////////////
// Shared ProgressMeasures literal builder for acceptance/restoration tests.
//
// Deliberately includes ONLY globalization/progress_measures.h -- NOT
// solver_test_utils.h, which pulls the full tycho/tycho.h umbrella. Several
// of this helper's call sites live in leaf-header test TUs (they test a
// single globalization component in isolation and include nothing heavier
// than that component's own header); routing pm() through solver_test_utils.h
// would move those TUs onto the 4-7 GB heavy-template-instantiation path for
// no reason -- pm() only ever touches the plain-data ProgressMeasures type.
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "tycho/detail/solvers/globalization/progress_measures.h"

namespace TychoTest {

/// @brief Build a ProgressMeasures(infeasibility, objective, auxiliary) triple.
/// `obj` and `aux` default to 0.0 so callers that only care about theta (e.g.
/// the classic-merit restoration exit test, which reads only
/// settings_.econ_tol_ / infeasibility) can write `pm(theta)`.
inline tycho::solvers::ProgressMeasures pm(double inf, double obj = 0.0, double aux = 0.0) {
    tycho::solvers::ProgressMeasures p;
    p.infeasibility = inf;
    p.objective = obj;
    p.auxiliary = aux;
    return p;
}

} // namespace TychoTest
