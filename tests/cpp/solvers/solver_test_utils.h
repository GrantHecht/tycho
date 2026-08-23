///////////////////////////////////////////////////////////////////////////////
// Shared utilities for solver tests
//
// Provides the SolverTest fixture and a Brachistochrone phase builder
// pre-configured for solver/Jet tests (silent output: print_level is inverted,
// 0 is full output and 3+ is fully silent -- see InteriorPointSolver::Settings::print_level_).
//
// This is the VF-heavy half of what used to be a single
// tests/cpp/solvers/solver_test_utils.h. The other half -- InertSolverContext,
// which needs only hven/detail/globalization/solver_context.h +
// hven/detail/interior/jet.h and no VectorFunction dependency -- moved into
// the hven project (hven/tests/interior/solver_test_utils.h) alongside the
// solver-internal tests that use nothing else.
//
// InertSolverContext is kept here too (duplicated, not cross-included from
// hven/tests/): test_feasibility_switch.cpp and test_interior_point_solver_native_bounds.cpp
// stayed on the Tycho side (they build real VF phases via make_brach_solver_phase)
// but also use InertSolverContext directly. A tycho-side test including an
// hven/tests/ header would run the cross-boundary include the split was
// meant to avoid in the other direction, so the ~15-line struct is duplicated
// instead of shared. Keep both copies in sync if SolverContext's shape changes.
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "oc_test_utils.h"
#include "test_utils.h"
#include "tycho/detail/hven_namespaces.h"
#include <hven/detail/globalization/solver_context.h>
#include <hven/detail/interior/jet.h>
#include <cmath>
#include <functional>
#include <gtest/gtest.h>
#include <memory>
#include <tycho/tycho.h>

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
// signature, not a live solve. See psiopt/tests/solver_test_utils.h for the
// full contract/lifetime-rule documentation -- identical here.
///////////////////////////////////////////////////////////////////////////////

struct InertSolverContext {
    InteriorPointSolver::Settings settings_;
    KktSolverType kkt_solver_;
    Eigen::VectorXd scratch_;
    Eigen::VectorXd declaration_primals_scratch_;
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
            nullptr,      kkt_solver_,                  settings_,
            primal_vars_, slack_vars_,                  equal_cons_, inequal_cons_, kkt_dim_,
            scratch_,     declaration_primals_scratch_, restoration_, eval_errors_,
        };
        // clang-format on
    }
};

} // namespace TychoTest
