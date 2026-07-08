///////////////////////////////////////////////////////////////////////////////
// set_units() retranscription test (OC review §1.12)
//
// ODEPhase::set_units() (include/tycho/detail/optimal_control/phase/ode_phase.h)
// rebuilds ode_scaled_ but pre-fix did not mark the phase as needing
// retranscription. A phase that had already been solved would silently keep
// its stale NLP -- built against the OLD units -- across the next
// solve_optimize(), rather than rebuilding it against the new ones.
//
// Reproduced here by solving a phase once, calling set_units() with non-unit
// units, and solving again: the re-solved result must match a fresh phase
// that was configured with those units from the start.
///////////////////////////////////////////////////////////////////////////////

#include "oc_test_utils.h"
#include <gtest/gtest.h>
#include <tycho/tycho.h>

using namespace tycho;
using namespace TychoTest;

class SetUnitsRetranscription : public OptimalControlTest {};

TEST_F(SetUnitsRetranscription, ResolveMatchesFreshPhase) {
    Eigen::VectorXd units = brach_nonunit_units();

    auto reused = make_brach_phase(20);
    reused->optimizer_->set_print_level(0);
    auto reused_status1 = reused->solve_optimize();
    EXPECT_LE(reused_status1, PSIOPT::ConvergenceFlags::ACCEPTABLE);

    reused->set_units(units);
    auto reused_status2 = reused->solve_optimize();
    EXPECT_LE(reused_status2, PSIOPT::ConvergenceFlags::ACCEPTABLE);

    auto fresh = make_brach_phase(20);
    fresh->optimizer_->set_print_level(0);
    fresh->set_units(units);
    auto fresh_status = fresh->solve_optimize();
    EXPECT_LE(fresh_status, PSIOPT::ConvergenceFlags::ACCEPTABLE);

    EXPECT_NEAR(reused->return_traj().back()[reused->t_var()],
                fresh->return_traj().back()[fresh->t_var()], 1e-6);
}
