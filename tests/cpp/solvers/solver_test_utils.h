///////////////////////////////////////////////////////////////////////////////
// Shared utilities for solver tests
//
// Provides the SolverTest fixture and a Brachistochrone phase builder
// pre-configured for solver/Jet tests (silent output).
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Solvers/Jet.h"
#include "Tycho.h"
#include "oc_test_utils.h"
#include "test_utils.h"
#include <cmath>
#include <functional>
#include <gtest/gtest.h>
#include <memory>

namespace TychoTest {

using namespace Tycho;

///////////////////////////////////////////////////////////////////////////////
// Test fixture
///////////////////////////////////////////////////////////////////////////////

class SolverTest : public VectorFunctionFixture {};

///////////////////////////////////////////////////////////////////////////////
// Helper: build a Brachistochrone phase for solver/Jet tests
///////////////////////////////////////////////////////////////////////////////

inline std::shared_ptr<ODEPhase<BrachODE>> make_brach_solver_phase(int n_segs = 16) {
    auto phase = make_brach_phase(n_segs * 3 + 1, n_segs);
    phase->optimizer->PrintLevel = 0;
    return phase;
}

} // namespace TychoTest
