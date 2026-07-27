///////////////////////////////////////////////////////////////////////////////
// Shared utilities for solver tests
//
// Provides the SolverTest fixture and a Brachistochrone phase builder
// pre-configured for solver/Jet tests (silent output: print_level is inverted,
// 0 is full output and 3+ is fully silent -- see PSIOPT::Settings::print_level_).
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "tycho/detail/solvers/jet.h"
#include <tycho/tycho.h>
#include "oc_test_utils.h"
#include "test_utils.h"
#include <cmath>
#include <functional>
#include <gtest/gtest.h>
#include <memory>

using namespace tycho::solvers;

namespace TychoTest {

using namespace tycho;

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

} // namespace TychoTest
