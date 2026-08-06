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
    EXPECT_THROW(solver.optimize(x0), std::invalid_argument);

    // The bad staging must have been consumed (cleared) on the throw path --
    // a second, unseeded optimize() call must converge normally rather than
    // re-throwing or silently reusing the stale bad seed.
    EXPECT_FALSE(solver.optimizer_->mults_staged_);
    ASSERT_EQ(solver.optimize(x0), tycho::ConvergenceFlags::CONVERGED);
    Eigen::VectorXd x = solver.return_x();
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
