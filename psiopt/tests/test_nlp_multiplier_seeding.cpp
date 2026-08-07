// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// PSIOPT::set_initial_multipliers / apply_staged_multipliers -- the opt-in
// constraint-multiplier seeding entry consumed by NLPSolver::
// apply_starting_multipliers. Problem structs here are deliberately distinct
// from (though structurally similar to) the ones in test_nlp_solver.cpp: the
// unity build merges test TUs, so file-scope names must not collide.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>

#include "tycho/solvers/nlp_solver.h"

namespace {
constexpr double kSeedSolverInf = std::numeric_limits<double>::infinity();
} // namespace

using tycho::ConstEigenRef;
using tycho::solvers::NLPProblem;
using tycho::solvers::NLPSolver;

// The canonical Ipopt HS071 example: n=4, one lower-bounded product row, one
// equality sphere row, dense Jacobian and Hessian. Same problem as
// test_nlp_solver.cpp's Hs071Problem, duplicated under a distinct name to
// avoid a unity-build symbol collision.
struct SeedHs071Problem : NLPProblem {
    int num_vars() const override { return 4; }
    int num_cons() const override { return 2; }
    int num_jac_nonzeros() const override { return 8; }
    int num_hess_nonzeros() const override { return 10; }

    void bounds(Eigen::Ref<Eigen::VectorXd> xl, Eigen::Ref<Eigen::VectorXd> xu,
                Eigen::Ref<Eigen::VectorXd> gl, Eigen::Ref<Eigen::VectorXd> gu) const override {
        xl << 1.0, 1.0, 1.0, 1.0;
        xu << 5.0, 5.0, 5.0, 5.0;
        gl << 25.0, 40.0;
        gu << kSeedSolverInf, 40.0;
    }
    void eval_f(ConstEigenRef<Eigen::VectorXd> x, double &f) const override {
        f = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
    }
    void eval_grad_f(ConstEigenRef<Eigen::VectorXd> x,
                     Eigen::Ref<Eigen::VectorXd> g) const override {
        g[0] = x[3] * (2.0 * x[0] + x[1] + x[2]);
        g[1] = x[0] * x[3];
        g[2] = x[0] * x[3] + 1.0;
        g[3] = x[0] * (x[0] + x[1] + x[2]);
    }
    void eval_g(ConstEigenRef<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> g) const override {
        g[0] = x[0] * x[1] * x[2] * x[3];
        g[1] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
    }
    void jac_structure(Eigen::Ref<Eigen::VectorXi> r,
                       Eigen::Ref<Eigen::VectorXi> c) const override {
        r << 0, 0, 0, 0, 1, 1, 1, 1;
        c << 0, 1, 2, 3, 0, 1, 2, 3;
    }
    void hess_structure(Eigen::Ref<Eigen::VectorXi> r,
                        Eigen::Ref<Eigen::VectorXi> c) const override {
        r << 0, 1, 1, 2, 2, 2, 3, 3, 3, 3;
        c << 0, 0, 1, 0, 1, 2, 0, 1, 2, 3;
    }
    void eval_jac(ConstEigenRef<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> v) const override {
        v[0] = x[1] * x[2] * x[3];
        v[1] = x[0] * x[2] * x[3];
        v[2] = x[0] * x[1] * x[3];
        v[3] = x[0] * x[1] * x[2];
        v[4] = 2.0 * x[0];
        v[5] = 2.0 * x[1];
        v[6] = 2.0 * x[2];
        v[7] = 2.0 * x[3];
    }
    void eval_hess(ConstEigenRef<Eigen::VectorXd> x, double obj_factor,
                   ConstEigenRef<Eigen::VectorXd> lambda,
                   Eigen::Ref<Eigen::VectorXd> v) const override {
        const double x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3];
        v[0] = obj_factor * 2 * x3 + lambda[1] * 2;
        v[1] = obj_factor * x3 + lambda[0] * x2 * x3;
        v[2] = lambda[1] * 2;
        v[3] = obj_factor * x3 + lambda[0] * x1 * x3;
        v[4] = lambda[0] * x0 * x3;
        v[5] = lambda[1] * 2;
        v[6] = obj_factor * (2 * x0 + x1 + x2) + lambda[0] * x1 * x2;
        v[7] = obj_factor * x0 + lambda[0] * x0 * x2;
        v[8] = obj_factor * x0 + lambda[0] * x0 * x1;
        v[9] = lambda[1] * 2;
    }
    std::string name() const override { return "SeedHs071Problem"; }
};

// SeedHs071Problem plus a starting_multipliers() override seeding a modest,
// merely-plausible-sign guess (not the exact solution multiplier -- unlike
// EqualityMultiplierHasIpoptSign-style tests, this is only checking that a
// seed which is in the right ballpark does not perturb the converged
// optimum).
struct SeededSeedHs071Problem : SeedHs071Problem {
    bool starting_multipliers(Eigen::Ref<Eigen::VectorXd> lambda) const override {
        lambda[0] = -1.0; // row 0: active lower bound -> negative Ipopt sign
        lambda[1] = 0.5;  // row 1: equality, sign unconstrained
        return true;
    }
    std::string name() const override { return "SeededSeedHs071Problem"; }
};

TEST(NLPMultiplierSeedingTest, SeededSolveMatchesUnseededSolution) {
    Eigen::VectorXd x0(4);
    x0 << 1.0, 5.0, 5.0, 1.0;

    NLPSolver unseeded(std::make_shared<SeedHs071Problem>());
    unseeded.optimizer_->set_print_level(10);
    ASSERT_EQ(unseeded.optimize(x0), tycho::ConvergenceFlags::CONVERGED);

    NLPSolver seeded(std::make_shared<SeededSeedHs071Problem>());
    seeded.optimizer_->set_print_level(10);
    ASSERT_EQ(seeded.optimize(x0), tycho::ConvergenceFlags::CONVERGED);

    EXPECT_LT((seeded.return_x() - unseeded.return_x()).lpNorm<Eigen::Infinity>(), 1e-6);
}

// f = x0^2 + x1^2 subject to x0 + x1 = 2 -- optimum (1, 1), a single equality
// row and no inequality rows. Used below to probe set_initial_multipliers
// directly (rather than through starting_multipliers()) with deliberately
// wrong-sized vectors.
struct SeedEqOnlyProblem : NLPProblem {
    int num_vars() const override { return 2; }
    int num_cons() const override { return 1; }
    int num_jac_nonzeros() const override { return 2; }
    int num_hess_nonzeros() const override { return 2; }

    void bounds(Eigen::Ref<Eigen::VectorXd> xl, Eigen::Ref<Eigen::VectorXd> xu,
                Eigen::Ref<Eigen::VectorXd> gl, Eigen::Ref<Eigen::VectorXd> gu) const override {
        xl << -kSeedSolverInf, -kSeedSolverInf;
        xu << kSeedSolverInf, kSeedSolverInf;
        gl << 2.0;
        gu << 2.0;
    }
    void eval_f(ConstEigenRef<Eigen::VectorXd> x, double &f) const override {
        f = x[0] * x[0] + x[1] * x[1];
    }
    void eval_grad_f(ConstEigenRef<Eigen::VectorXd> x,
                     Eigen::Ref<Eigen::VectorXd> g) const override {
        g[0] = 2.0 * x[0];
        g[1] = 2.0 * x[1];
    }
    void eval_g(ConstEigenRef<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> g) const override {
        g[0] = x[0] + x[1];
    }
    void jac_structure(Eigen::Ref<Eigen::VectorXi> r,
                       Eigen::Ref<Eigen::VectorXi> c) const override {
        r << 0, 0;
        c << 0, 1;
    }
    void hess_structure(Eigen::Ref<Eigen::VectorXi> r,
                        Eigen::Ref<Eigen::VectorXi> c) const override {
        r << 0, 1;
        c << 0, 1;
    }
    void eval_jac(ConstEigenRef<Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd> v) const override {
        v[0] = 1.0;
        v[1] = 1.0;
    }
    void eval_hess(ConstEigenRef<Eigen::VectorXd>, double obj_factor,
                   ConstEigenRef<Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd> v) const override {
        v[0] = 2.0 * obj_factor;
        v[1] = 2.0 * obj_factor;
    }
    std::string name() const override { return "SeedEqOnlyProblem"; }
};

TEST(NLPMultiplierSeedingTest, SeedSizeMismatchThrowsAndIsConsumed) {
    NLPSolver solver(std::make_shared<SeedEqOnlyProblem>());
    solver.optimizer_->set_print_level(10);
    solver.transcribe();

    // SeedEqOnlyProblem has 1 equality row and 0 inequality rows; stage sizes
    // that match neither.
    Eigen::VectorXd bad_eq(2);
    bad_eq << -2.0, 0.0;
    Eigen::VectorXd bad_iq(1);
    bad_iq << 1.0;
    solver.optimizer_->set_initial_multipliers(bad_eq, bad_iq);

    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(2);
    // Probing PSIOPT's own optimizer_ directly, not NLPSolver::optimize():
    // SeedEqOnlyProblem's starting_multipliers() returns false, so going
    // through NLPSolver would itself clear this staging before PSIOPT ever
    // saw it (see DecliningProblemClearsStaleStaging) -- that is the correct
    // behavior for a problem that never asked to be seeded, but it means
    // this size-mismatch probe has to bypass it. The throw now fires from
    // validate_staged_multipliers, right after variable-treatment
    // reconfiguration and before the entry init_impl/factorization -- earlier
    // than the original install-site throw, but still inside this one
    // optimize() call either way.
    EXPECT_THROW(solver.optimizer_->optimize(x0), std::invalid_argument);

    // The bad staging must have been consumed (cleared) on the throw path --
    // a second, unseeded optimize() call must converge normally rather than
    // re-throwing or silently reusing the stale bad seed.
    EXPECT_FALSE(solver.optimizer_->mults_staged_);
    Eigen::VectorXd x = solver.optimizer_->optimize(x0);
    ASSERT_EQ(solver.optimizer_->result().converge_flag_, tycho::ConvergenceFlags::CONVERGED);
    Eigen::VectorXd expect(2);
    expect << 1.0, 1.0;
    EXPECT_LT((x - expect).lpNorm<Eigen::Infinity>(), 1e-6);
}

// f = x0^2 subject to x0 >= 1 -- optimum x0 = 1, a single active lower-bound
// inequality row and no equality rows.
struct SeedLowerBoundProblem : NLPProblem {
    int num_vars() const override { return 1; }
    int num_cons() const override { return 1; }
    int num_jac_nonzeros() const override { return 1; }
    int num_hess_nonzeros() const override { return 1; }

    void bounds(Eigen::Ref<Eigen::VectorXd> xl, Eigen::Ref<Eigen::VectorXd> xu,
                Eigen::Ref<Eigen::VectorXd> gl, Eigen::Ref<Eigen::VectorXd> gu) const override {
        xl << -kSeedSolverInf;
        xu << kSeedSolverInf;
        gl << 1.0;
        gu << kSeedSolverInf;
    }
    void eval_f(ConstEigenRef<Eigen::VectorXd> x, double &f) const override { f = x[0] * x[0]; }
    void eval_grad_f(ConstEigenRef<Eigen::VectorXd> x,
                     Eigen::Ref<Eigen::VectorXd> g) const override {
        g[0] = 2.0 * x[0];
    }
    void eval_g(ConstEigenRef<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> g) const override {
        g[0] = x[0];
    }
    void jac_structure(Eigen::Ref<Eigen::VectorXi> r,
                       Eigen::Ref<Eigen::VectorXi> c) const override {
        r << 0;
        c << 0;
    }
    void hess_structure(Eigen::Ref<Eigen::VectorXi> r,
                        Eigen::Ref<Eigen::VectorXi> c) const override {
        r << 0;
        c << 0;
    }
    void eval_jac(ConstEigenRef<Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd> v) const override {
        v[0] = 1.0;
    }
    void eval_hess(ConstEigenRef<Eigen::VectorXd>, double obj_factor,
                   ConstEigenRef<Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd> v) const override {
        v[0] = 2.0 * obj_factor;
    }
    std::string name() const override { return "SeedLowerBoundProblem"; }
};

// starting_multipliers() returns a positive value for the active lower-bound
// row, which is the WRONG Ipopt sign there (see
// LowerBoundedRowActiveWithNegativeIpoptMultiplier in test_nlp_solver.cpp --
// the correct sign is negative). apply_starting_multipliers's
// LowerBounded-row mapping negates it (iqm = -lam), so this deliberately
// produces a negative seed on the PSIOPT-internal inequality multiplier,
// which apply_staged_multipliers must clamp to kSeededIqMultFloor rather than
// installing it verbatim.
struct SeededSeedLowerBoundProblem : SeedLowerBoundProblem {
    bool starting_multipliers(Eigen::Ref<Eigen::VectorXd> lambda) const override {
        lambda[0] = 100.0;
        return true;
    }
    std::string name() const override { return "SeededSeedLowerBoundProblem"; }
};

TEST(NLPMultiplierSeedingTest, NegativeIqSeedIsClamped) {
    NLPSolver solver(std::make_shared<SeededSeedLowerBoundProblem>());
    solver.optimizer_->set_print_level(10);
    Eigen::VectorXd x0(1);
    x0 << 3.0;
    ASSERT_EQ(solver.optimize(x0), tycho::ConvergenceFlags::CONVERGED);
    EXPECT_NEAR(solver.return_x()[0], 1.0, 1e-5);
}

TEST(NLPMultiplierSeedingTest, UnseededPathDoesNotConsultStaging) {
    NLPSolver solver(std::make_shared<SeedEqOnlyProblem>());
    solver.optimizer_->set_print_level(10);
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(2);

    EXPECT_FALSE(solver.optimizer_->mults_staged_);
    ASSERT_EQ(solver.optimize(x0), tycho::ConvergenceFlags::CONVERGED);
    EXPECT_FALSE(solver.optimizer_->mults_staged_);
}

// -----------------------------------------------------------------------------
// Fix round 1: (1) MakeConstraint + seeding, (2) staleness/ipopt-backend
// staging hygiene, (3) NaN rejection + magnitude cap, (4) solve-first phase
// sequencing (SOE ignores the seed; it must reach the first OPT/OPTNO phase).
// -----------------------------------------------------------------------------

// SeededPhaseEntryEqOnlyProblem's seed reaching the OPT phase's very first
// iteration (before any Newton step) is observed through PSIOPT's early
// (per-iteration) callback: XSL's KKTVector layout is documented as
// [primals | slacks | eq_lmults | iq_lmults] (kkt_vector.h). For
// SeedEqOnlyProblem specifically -- 2 unbounded primal vars (so reduced ==
// full, no fixed-variable elimination), 0 inequality rows (no slacks), 1
// equality row -- that puts the single equality multiplier at XSL[2].
struct SeededPhaseEntryEqOnlyProblem : SeedEqOnlyProblem {
    bool starting_multipliers(Eigen::Ref<Eigen::VectorXd> lambda) const override {
        // Deliberately NOT the true KKT multiplier (-2.0, see the EqOnly-style
        // tests elsewhere in this series): SeedEqOnlyProblem is exactly
        // quadratic/linear, so its SOE phase alone already lands exactly on
        // the true optimum with the true multiplier, and init_impl's own
        // fresh KKT solve at the inter-phase re-init would independently
        // re-derive that same -2.0 regardless of any seed. Seeding a value
        // that is otherwise impossible for PSIOPT to produce here is what
        // makes "the callback observed the seed" distinguishable from "the
        // callback observed PSIOPT's own correct answer by coincidence".
        lambda[0] = 7.0;
        return true;
    }
    std::string name() const override { return "SeededPhaseEntryEqOnlyProblem"; }
};

TEST(NLPMultiplierSeedingTest, SeededSolveOptimizeReachesOptPhase) {
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(2);

    // solve_optimize() runs SOE then OPT. The callback fires once per
    // iteration of EVERY phase and `i` resets to 0 at the start of each
    // phase's own iteration loop, so capturing (and overwriting) on every
    // i==0 leaves the LAST captured value as the OPT phase's entry value --
    // the one this test cares about -- regardless of how many iterations SOE
    // itself takes.
    double unseeded_opt_entry_eq_mult = std::numeric_limits<double>::quiet_NaN();
    {
        NLPSolver solver(std::make_shared<SeedEqOnlyProblem>());
        solver.optimizer_->set_print_level(10);
        solver.optimizer_->set_early_callback(
            [&](int i, double, Eigen::Ref<Eigen::VectorXd> XSL, double, Eigen::Ref<Eigen::VectorXd>,
                Eigen::Ref<Eigen::VectorXd>,
                Eigen::SparseMatrix<double, Eigen::RowMajor> &) -> int {
                if (i == 0) {
                    unseeded_opt_entry_eq_mult = XSL[2];
                }
                return 0;
            });
        ASSERT_EQ(solver.solve_optimize(x0), tycho::ConvergenceFlags::CONVERGED);
    }
    ASSERT_FALSE(std::isnan(unseeded_opt_entry_eq_mult));

    double seeded_opt_entry_eq_mult = std::numeric_limits<double>::quiet_NaN();
    {
        NLPSolver solver(std::make_shared<SeededPhaseEntryEqOnlyProblem>());
        solver.optimizer_->set_print_level(10);
        solver.optimizer_->set_early_callback(
            [&](int i, double, Eigen::Ref<Eigen::VectorXd> XSL, double, Eigen::Ref<Eigen::VectorXd>,
                Eigen::Ref<Eigen::VectorXd>,
                Eigen::SparseMatrix<double, Eigen::RowMajor> &) -> int {
                if (i == 0) {
                    seeded_opt_entry_eq_mult = XSL[2];
                }
                return 0;
            });
        ASSERT_EQ(solver.solve_optimize(x0), tycho::ConvergenceFlags::CONVERGED);
    }

    EXPECT_DOUBLE_EQ(unseeded_opt_entry_eq_mult, -2.0); // PSIOPT's own correct answer
    EXPECT_DOUBLE_EQ(seeded_opt_entry_eq_mult, 7.0);    // the seed, verbatim
    EXPECT_NE(seeded_opt_entry_eq_mult, unseeded_opt_entry_eq_mult);
}

// Same objective/constraint as SeedEqOnlyProblem (x0^2 + x1^2 s.t. x0+x1=2),
// but x1 is FIXED (xl==xu==1.0). Under the MakeConstraint fixed-variable
// treatment PSIOPT keeps x1 as a solver variable and adds one internal
// equality row x1-1=0 on top of the problem's own single row -- growing
// equal_cons_ to 2 while user_equal_cons_ (what starting_multipliers()/
// NLPSolver see) stays at 1. That mismatch is exactly what finding 1's fix
// targets: a seed sized to the 1 user row must still be accepted, and the
// internal row zero-padded, not rejected as a size mismatch against 2.
struct SeedFixedVarEqProblem : NLPProblem {
    int num_vars() const override { return 2; }
    int num_cons() const override { return 1; }
    int num_jac_nonzeros() const override { return 2; }
    int num_hess_nonzeros() const override { return 2; }

    void bounds(Eigen::Ref<Eigen::VectorXd> xl, Eigen::Ref<Eigen::VectorXd> xu,
                Eigen::Ref<Eigen::VectorXd> gl, Eigen::Ref<Eigen::VectorXd> gu) const override {
        xl << -kSeedSolverInf, 1.0;
        xu << kSeedSolverInf, 1.0;
        gl << 2.0;
        gu << 2.0;
    }
    void eval_f(ConstEigenRef<Eigen::VectorXd> x, double &f) const override {
        f = x[0] * x[0] + x[1] * x[1];
    }
    void eval_grad_f(ConstEigenRef<Eigen::VectorXd> x,
                     Eigen::Ref<Eigen::VectorXd> g) const override {
        g[0] = 2.0 * x[0];
        g[1] = 2.0 * x[1];
    }
    void eval_g(ConstEigenRef<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> g) const override {
        g[0] = x[0] + x[1];
    }
    void jac_structure(Eigen::Ref<Eigen::VectorXi> r,
                       Eigen::Ref<Eigen::VectorXi> c) const override {
        r << 0, 0;
        c << 0, 1;
    }
    void hess_structure(Eigen::Ref<Eigen::VectorXi> r,
                        Eigen::Ref<Eigen::VectorXi> c) const override {
        r << 0, 1;
        c << 0, 1;
    }
    void eval_jac(ConstEigenRef<Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd> v) const override {
        v[0] = 1.0;
        v[1] = 1.0;
    }
    void eval_hess(ConstEigenRef<Eigen::VectorXd>, double obj_factor,
                   ConstEigenRef<Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd> v) const override {
        v[0] = 2.0 * obj_factor;
        v[1] = 2.0 * obj_factor;
    }
    std::string name() const override { return "SeedFixedVarEqProblem"; }
};

struct SeededSeedFixedVarEqProblem : SeedFixedVarEqProblem {
    bool starting_multipliers(Eigen::Ref<Eigen::VectorXd> lambda) const override {
        lambda[0] = -2.0;
        return true;
    }
    std::string name() const override { return "SeededSeedFixedVarEqProblem"; }
};

TEST(NLPMultiplierSeedingTest, SeededSolveWithMakeConstraintFixedVarConverges) {
    NLPSolver solver(std::make_shared<SeededSeedFixedVarEqProblem>());
    solver.optimizer_->set_print_level(10);
    solver.optimizer_->set_fixed_variable_treatment(
        tycho::solvers::FixedVariableTreatments::MakeConstraint);
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(2);
    ASSERT_EQ(solver.optimize(x0), tycho::ConvergenceFlags::CONVERGED);
    Eigen::VectorXd x = solver.return_x();
    EXPECT_NEAR(x[0], 1.0, 1e-6);
    EXPECT_NEAR(x[1], 1.0, 1e-6);
}

TEST(NLPMultiplierSeedingTest, DecliningProblemClearsStaleStaging) {
    NLPSolver solver(std::make_shared<SeedEqOnlyProblem>());
    solver.optimizer_->set_print_level(10);
    solver.transcribe();

    // Arm a deliberately poisoned stale seed directly, bypassing
    // starting_multipliers() entirely -- simulating state left behind by,
    // e.g., an earlier direct optimizer_ call. SeedEqOnlyProblem's own
    // starting_multipliers() returns false, so this solve never asked to be
    // seeded. If apply_starting_multipliers's early-return path failed to
    // clear the stale staging, this NaN would reach
    // PSIOPT::apply_staged_multipliers and throw -- so a clean CONVERGED
    // result below is itself proof the stale seed was never applied.
    Eigen::VectorXd stale_eq(1);
    stale_eq << std::numeric_limits<double>::quiet_NaN();
    Eigen::VectorXd stale_iq(0);
    solver.optimizer_->set_initial_multipliers(stale_eq, stale_iq);
    ASSERT_TRUE(solver.optimizer_->mults_staged_);

    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(2);
    ASSERT_EQ(solver.optimize(x0), tycho::ConvergenceFlags::CONVERGED);
    EXPECT_FALSE(solver.optimizer_->mults_staged_);

    Eigen::VectorXd x = solver.return_x();
    Eigen::VectorXd expect(2);
    expect << 1.0, 1.0;
    EXPECT_LT((x - expect).lpNorm<Eigen::Infinity>(), 1e-6);
}

TEST(NLPMultiplierSeedingTest, NaNSeedThrows) {
    NLPSolver solver(std::make_shared<SeedEqOnlyProblem>());
    solver.optimizer_->set_print_level(10);
    solver.transcribe();

    Eigen::VectorXd eq(1);
    eq << std::numeric_limits<double>::quiet_NaN();
    Eigen::VectorXd iq(0);
    solver.optimizer_->set_initial_multipliers(eq, iq);

    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(2);
    // Direct optimizer_ call, not NLPSolver::optimize() -- see the note in
    // SeedSizeMismatchThrowsAndIsConsumed above: SeedEqOnlyProblem declines
    // to seed, so going through NLPSolver would clear this staging first.
    // Probing PSIOPT's own validation this way is also the point: it must
    // reject a non-finite seed even from a caller that bypasses NLPSolver's
    // allFinite() guard entirely.
    EXPECT_THROW(solver.optimizer_->optimize(x0), std::invalid_argument);
    EXPECT_FALSE(solver.optimizer_->mults_staged_);
}

// A seed of 1e12 into the single equality row -- SeedEqOnlyProblem is
// exactly quadratic/linear, so a plain "does it still converge" check on the
// final solution is vacuous: the first Newton step lands exactly on the
// solution regardless of the seeded multiplier, clamp deleted or not. Making
// this assertive means observing the CAPPED value directly, the same way
// SeededSolveOptimizeReachesOptPhase does.
struct SeededOversizedEqOnlyProblem : SeedEqOnlyProblem {
    bool starting_multipliers(Eigen::Ref<Eigen::VectorXd> lambda) const override {
        lambda[0] = 1.0e12;
        return true;
    }
    std::string name() const override { return "SeededOversizedEqOnlyProblem"; }
};

TEST(NLPMultiplierSeedingTest, OversizedSeedIsCapped) {
    NLPSolver solver(std::make_shared<SeededOversizedEqOnlyProblem>());
    solver.optimizer_->set_print_level(10);

    double captured_eq_mult = std::numeric_limits<double>::quiet_NaN();
    solver.optimizer_->set_early_callback(
        [&](int i, double, Eigen::Ref<Eigen::VectorXd> XSL, double, Eigen::Ref<Eigen::VectorXd>,
            Eigen::Ref<Eigen::VectorXd>, Eigen::SparseMatrix<double, Eigen::RowMajor> &) -> int {
            if (i == 0) {
                captured_eq_mult = XSL[2]; // see SeededSolveOptimizeReachesOptPhase for the layout
            }
            return 0;
        });

    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(2);
    ASSERT_EQ(solver.optimize(x0), tycho::ConvergenceFlags::CONVERGED);
    EXPECT_DOUBLE_EQ(captured_eq_mult, 1.0e6); // kSeededMultInitMax, not the raw 1e12 seed
}

// Same idea on the inequality side: SeedLowerBoundProblem (1 primal var, 1
// slack for the single active lower-bound row, 0 eq rows) puts the iq
// multiplier at XSL[2] (primal(1)+slack(1)+eq(0)). starting_multipliers()'s
// LowerBounded-row mapping negates lam (iqm = -lam), so seeding lam=-1e12
// arrives at PSIOPT as a +1e12 iq seed, which the upper clamp (not the floor
// -- that only bites negative seeds) must cap.
struct SeededOversizedLowerBoundProblem : SeedLowerBoundProblem {
    bool starting_multipliers(Eigen::Ref<Eigen::VectorXd> lambda) const override {
        lambda[0] = -1.0e12;
        return true;
    }
    std::string name() const override { return "SeededOversizedLowerBoundProblem"; }
};

TEST(NLPMultiplierSeedingTest, OversizedIqSeedIsCapped) {
    NLPSolver solver(std::make_shared<SeededOversizedLowerBoundProblem>());
    solver.optimizer_->set_print_level(10);

    double captured_iq_mult = std::numeric_limits<double>::quiet_NaN();
    solver.optimizer_->set_early_callback(
        [&](int i, double, Eigen::Ref<Eigen::VectorXd> XSL, double, Eigen::Ref<Eigen::VectorXd>,
            Eigen::Ref<Eigen::VectorXd>, Eigen::SparseMatrix<double, Eigen::RowMajor> &) -> int {
            if (i == 0) {
                captured_iq_mult = XSL[2];
            }
            return 0;
        });

    Eigen::VectorXd x0(1);
    x0 << 3.0;
    ASSERT_EQ(solver.optimize(x0), tycho::ConvergenceFlags::CONVERGED);
    EXPECT_DOUBLE_EQ(captured_iq_mult, 1.0e6);
}

// Not in the required list, but directly covers finding 2b (otherwise
// untested): a problem that requests seeding while nlp_solver_ is set to the
// ipopt backend must fail loudly rather than silently dropping the seed (the
// ipopt backend's run_nlp_solver path never reaches
// PSIOPT::run_phase_sequence, so a staged seed would sit unconsumed forever).
// This throws purely from the nlp_solver_ == ipopt check in
// apply_starting_multipliers, before run_nlp_solver/ipopt_backend::solve is
// ever reached, so it does not require an ENABLE_IPOPT build.
struct SeededIpoptRejectEqOnlyProblem : SeedEqOnlyProblem {
    bool starting_multipliers(Eigen::Ref<Eigen::VectorXd> lambda) const override {
        lambda[0] = -2.0;
        return true;
    }
    std::string name() const override { return "SeededIpoptRejectEqOnlyProblem"; }
};

TEST(NLPMultiplierSeedingTest, SeededProblemRejectsIpoptBackend) {
    NLPSolver solver(std::make_shared<SeededIpoptRejectEqOnlyProblem>());
    solver.optimizer_->set_print_level(10);
    solver.nlp_solver_ = tycho::solvers::NLPSolvers::ipopt;

    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(2);
    EXPECT_THROW(solver.optimize(x0), std::invalid_argument);
    EXPECT_FALSE(solver.optimizer_->mults_staged_);
}
