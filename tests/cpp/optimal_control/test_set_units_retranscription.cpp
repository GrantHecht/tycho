///////////////////////////////////////////////////////////////////////////////
// set_units() retranscription test (OC review §1.12)
//
// ODEPhase::set_units() (include/tycho/detail/optimal_control/phase/ode_phase.h)
// rebuilds ode_scaled_ but pre-fix did not mark the phase as needing
// retranscription. Under auto-scaling (set_auto_scaling(true), the production
// pairing -- see examples/python_examples/GoddardRocket.py:116-117) the units
// are baked into the transcribed NLP: the defect constraints use ode_scaled_
// (transcribe_dynamics) and user constraints get IOScaled input/output scales
// computed from xtup_units_ (transcribe_basic_funcs / calc_auto_scales in
// ode_phase_base.cpp). Meanwhile make_solver_input()/collect_solver_output()
// (ode_phase_base.h) read xtup_units_ live at every solve. So pre-fix, a
// solved phase that then called set_units() kept the stale NLP built against
// the OLD units while packing/unpacking with the NEW ones -- an
// inconsistently-scaled problem -- on the next solve_optimize().
//
// Exercised here by enabling auto-scaling, solving once, calling set_units()
// with non-unit units, and solving again: the re-solved result must match a
// fresh auto-scaled phase configured with those units from the start (and
// both must hit the known Brachistochrone optimum, tf ~ 1.8013 s).
//
// Note calc_auto_scales() only derives per-function output scales FROM
// xtup_units_ (via get_input_scale); it never overwrites the explicitly-set
// units, so the set_units() values stay load-bearing across repeated solves.
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
    reused->set_auto_scaling(true);
    reused->optimizer_->set_print_level(0);
    auto reused_status1 = reused->solve_optimize();
    EXPECT_LE(reused_status1, tycho::ConvergenceFlags::ACCEPTABLE);

    reused->set_units(units);
    auto reused_status2 = reused->solve_optimize();
    EXPECT_LE(reused_status2, tycho::ConvergenceFlags::ACCEPTABLE);

    auto fresh = make_brach_phase(20);
    fresh->set_auto_scaling(true);
    fresh->optimizer_->set_print_level(0);
    fresh->set_units(units);
    auto fresh_status = fresh->solve_optimize();
    EXPECT_LE(fresh_status, tycho::ConvergenceFlags::ACCEPTABLE);

    double reused_tf = reused->return_traj().back()[reused->t_var()];
    double fresh_tf = fresh->return_traj().back()[fresh->t_var()];

    // Suite-precedent tolerances for solved-trajectory comparison at
    // ACCEPTABLE-level convergence (see test_control_modes.cpp).
    EXPECT_NEAR(reused_tf, 1.8013, 0.02);
    EXPECT_NEAR(fresh_tf, 1.8013, 0.02);
    EXPECT_NEAR(reused_tf, fresh_tf, 0.02);
}
