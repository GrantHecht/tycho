///////////////////////////////////////////////////////////////////////////////
// Shared utilities for solver tests
//
// Provides the SolverTest fixture and a Brachistochrone phase builder
// pre-configured for solver/Jet tests (silent output: print_level is inverted,
// 0 is full output and 3+ is fully silent -- see PSIOPT::Settings::print_level_).
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "tycho/detail/solvers/globalization/solver_context.h"
#include "tycho/detail/solvers/jet.h"
#include <tycho/tycho.h>
#include "oc_test_utils.h"
#include "test_utils.h"
#include <cmath>
#include <functional>
#include <gtest/gtest.h>
#include <memory>

namespace TychoTest {

using namespace tycho;
using namespace tycho::solvers;

///////////////////////////////////////////////////////////////////////////////
// Test fixture
///////////////////////////////////////////////////////////////////////////////

class SolverTest : public VectorFunctionFixture {};

///////////////////////////////////////////////////////////////////////////////
// Helper: build a Brachistochrone phase for solver/Jet tests
///////////////////////////////////////////////////////////////////////////////

inline std::shared_ptr<ODEPhase<BrachODE>> make_brach_solver_phase(int n_segs = 16) {
    auto phase = make_brach_phase(n_segs * 3 + 1, n_segs);
    phase->optimizer_->set_print_level(3);
    return phase;
}

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
// NestedRestorationMonotoneSchedule in test_feasibility_switch.cpp).
///////////////////////////////////////////////////////////////////////////////

class InertSolverContext {
  public:
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
