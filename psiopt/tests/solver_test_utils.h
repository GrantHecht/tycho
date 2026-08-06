///////////////////////////////////////////////////////////////////////////////
// InertSolverContext -- an owning stand-in for globalization-component unit
// tests that only need a SolverContext-satisfying signature, not a live solve.
//
// This is the VF-free half of what used to be a single
// tests/cpp/solvers/solver_test_utils.h. The other half -- the SolverTest
// fixture and make_brach_solver_phase(), which build a real
// ODEPhase<BrachODE> and therefore need the full tycho/tycho.h umbrella --
// stays on the Tycho side under its original name and path
// (tests/cpp/solvers/solver_test_utils.h), since psiopt has no VectorFunction
// dependency to build that against.
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "tycho/detail/solvers/globalization/solver_context.h"
#include "tycho/detail/solvers/jet.h"

namespace TychoTest {

using namespace tycho::solvers;

///////////////////////////////////////////////////////////////////////////////
// Helper: an inert (all-zero-by-default) owning SolverContext for
// globalization-component unit tests that only need a context satisfying a
// signature, not a live solve.
//
// SolverContext (solver_context.h) is references-only, so nothing can hand
// one back by value on its own -- every referenced object must outlive every
// read through the context. InertSolverContext instead OWNS the storage
// (Settings, KKT solver, scratch vector, dimension ints) as members and
// ctx() returns a SolverContext borrowing from THIS object.
//
// CRITICAL LIFETIME RULE: keep the InertSolverContext instance alive (a
// named local) for as long as the SolverContext returned by ctx() is used.
// `InertSolverContext().ctx()` is a dangling-reference bug -- the temporary
// is destroyed at the end of the full expression, taking its members (and
// every reference ctx() handed out into them) with it. The correct pattern
// is:
//   TychoTest::InertSolverContext inert;
//   inert.settings_.econ_tol_ = 1e-6;   // mutate before calling ctx()
//   SomeComponent c(inert.ctx());       // `inert` outlives `c`'s use of it
//
// Dimensions default to zero (nlp_ stays null, restoration_/eval_errors_
// stay null) -- the common case where the context is never dereferenced for
// real work, only threaded through a signature. Tests that need a specific
// KKT layout set primal_vars_/slack_vars_/equal_cons_/inequal_cons_/kkt_dim_
// directly before calling ctx() (see e.g.
// NestedRestorationMonotoneSchedule in test_feasibility_switch.cpp, on the
// Tycho side).
///////////////////////////////////////////////////////////////////////////////

struct InertSolverContext {
    PSIOPT::Settings settings_;
    KktSolverType kkt_solver_;
    Eigen::VectorXd scratch_;
    int primal_vars_ = 0;
    int slack_vars_ = 0;
    int equal_cons_ = 0;
    int inequal_cons_ = 0;
    int kkt_dim_ = 0;
    const RestorationStrategy *restoration_ = nullptr;
    EvalErrorLog *eval_errors_ = nullptr;

    SolverContext ctx() {
        // clang-format off
        return SolverContext{
            nullptr,      kkt_solver_,   settings_,
            primal_vars_, slack_vars_,   equal_cons_, inequal_cons_, kkt_dim_,
            scratch_,     restoration_,  eval_errors_,
        };
        // clang-format on
    }
};

} // namespace TychoTest
