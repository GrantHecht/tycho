#include <gtest/gtest.h>

#include <limits>
#include <memory>

#include "tycho/detail/solvers/nlp_adapter.h"

namespace {
constexpr double kInf = std::numeric_limits<double>::infinity();
} // namespace

using tycho::ConstEigenRef;
using tycho::solvers::NLPAdapterCore;
using tycho::solvers::NLPProblem;
using tycho::solvers::NLPRowClassification;
using tycho::solvers::NLPRowKind;

// Minimal configurable problem for validation tests. Field defaults describe a
// valid 2-var, 2-con problem; individual tests break one field at a time.
struct AdapterValProblem : NLPProblem {
    int n_ = 2, m_ = 2, jnnz_ = 4, hnnz_ = 2;
    Eigen::VectorXd xl_{{-kInf, -kInf}}, xu_{{kInf, kInf}};
    Eigen::VectorXd gl_{{0.0, -kInf}}, gu_{{0.0, 1.0}};
    Eigen::VectorXi jr_{{0, 0, 1, 1}}, jc_{{0, 1, 0, 1}};
    Eigen::VectorXi hr_{{0, 1}}, hc_{{0, 1}};

    int num_vars() const override { return n_; }
    int num_cons() const override { return m_; }
    int num_jac_nonzeros() const override { return jnnz_; }
    int num_hess_nonzeros() const override { return hnnz_; }
    void bounds(Eigen::Ref<Eigen::VectorXd> xl, Eigen::Ref<Eigen::VectorXd> xu,
                Eigen::Ref<Eigen::VectorXd> gl, Eigen::Ref<Eigen::VectorXd> gu) const override {
        xl = xl_;
        xu = xu_;
        gl = gl_;
        gu = gu_;
    }
    void eval_f(ConstEigenRef<Eigen::VectorXd>, double &f) const override { f = 0.0; }
    void eval_grad_f(ConstEigenRef<Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd> g) const override {
        g.setZero();
    }
    void eval_g(ConstEigenRef<Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd> g) const override {
        g.setZero();
    }
    void jac_structure(Eigen::Ref<Eigen::VectorXi> r,
                       Eigen::Ref<Eigen::VectorXi> c) const override {
        r = jr_;
        c = jc_;
    }
    void hess_structure(Eigen::Ref<Eigen::VectorXi> r,
                        Eigen::Ref<Eigen::VectorXi> c) const override {
        r = hr_;
        c = hc_;
    }
    void eval_jac(ConstEigenRef<Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd> v) const override {
        v.setZero();
    }
    void eval_hess(ConstEigenRef<Eigen::VectorXd>, double, ConstEigenRef<Eigen::VectorXd>,
                   Eigen::Ref<Eigen::VectorXd> v) const override {
        v.setZero();
    }
};

TEST(NLPRowClassificationTest, ClassifiesAllRowKindsAndCounts) {
    Eigen::VectorXd gl(5), gu(5);
    gl << 3.0, -kInf, 1.0, 1.0, -kInf;
    gu << 3.0, 2.0, kInf, 4.0, kInf;
    auto rc = NLPRowClassification::classify(gl, gu);
    EXPECT_EQ(rc.kinds_[0], NLPRowKind::Equality);
    EXPECT_EQ(rc.kinds_[1], NLPRowKind::UpperBounded);
    EXPECT_EQ(rc.kinds_[2], NLPRowKind::LowerBounded);
    EXPECT_EQ(rc.kinds_[3], NLPRowKind::Range);
    EXPECT_EQ(rc.kinds_[4], NLPRowKind::Free);
    EXPECT_EQ(rc.num_eq_, 1);
    EXPECT_EQ(rc.num_iq_, 4); // upper + lower + range(2)
    EXPECT_EQ(rc.eq_row_[0], 0);
    EXPECT_EQ(rc.iq_upper_row_[1], 0);
    EXPECT_EQ(rc.iq_lower_row_[2], 1);
    EXPECT_EQ(rc.iq_upper_row_[3], 2); // range: upper first,
    EXPECT_EQ(rc.iq_lower_row_[3], 3); // then negated lower
    EXPECT_EQ(rc.eq_row_[1], -1);
    EXPECT_EQ(rc.iq_upper_row_[0], -1);
}

TEST(NLPRowClassificationTest, RejectsInvertedAndNaNAndInfiniteEqualityBounds) {
    Eigen::VectorXd gl(1), gu(1);
    gl << 2.0;
    gu << 1.0;
    EXPECT_THROW(NLPRowClassification::classify(gl, gu), std::invalid_argument);
    gl << std::numeric_limits<double>::quiet_NaN();
    gu << 1.0;
    EXPECT_THROW(NLPRowClassification::classify(gl, gu), std::invalid_argument);
    gl << kInf;
    gu << kInf; // "equality at infinity"
    EXPECT_THROW(NLPRowClassification::classify(gl, gu), std::invalid_argument);
}

TEST(NLPAdapterCoreTest, ValidSetupPopulatesTables) {
    auto core = std::make_shared<NLPAdapterCore>(std::make_shared<AdapterValProblem>());
    EXPECT_EQ(core->n_, 2);
    EXPECT_EQ(core->rows_.num_eq_, 1);
    EXPECT_EQ(core->rows_.num_iq_, 1);
    EXPECT_EQ(core->eq_jac_.size(), 2u); // row 0 slots
    EXPECT_EQ(core->iq_jac_.size(), 2u); // row 1 slots
    EXPECT_EQ(core->hess_owner_, NLPAdapterCore::HessOwner::IqPiece);
}

TEST(NLPAdapterCoreTest, RejectsBadSizesAndStructures) {
    auto bad = [](auto mutate) {
        auto p = std::make_shared<AdapterValProblem>();
        mutate(*p);
        EXPECT_THROW(NLPAdapterCore{p}, std::invalid_argument);
    };
    bad([](AdapterValProblem &p) { p.n_ = 0; });
    bad([](AdapterValProblem &p) { p.m_ = -1; });
    bad([](AdapterValProblem &p) { p.m_ = 0; });      // jac_nnz stays 4: nonzeros without rows
    bad([](AdapterValProblem &p) { p.jr_[0] = 2; });  // jac row out of range
    bad([](AdapterValProblem &p) { p.jc_[0] = -1; }); // jac col out of range
    bad([](AdapterValProblem &p) {
        p.hr_[0] = 0;
        p.hc_[0] = 1;
    });                                              // upper-triangle entry
    bad([](AdapterValProblem &p) { p.hr_[0] = 5; }); // hess row out of range
    bad([](AdapterValProblem &p) {
        p.xl_[0] = 1.0;
        p.xu_[0] = 0.0;
    }); // inverted var bound
    bad([](AdapterValProblem &p) { p.xl_[0] = std::numeric_limits<double>::quiet_NaN(); });
    EXPECT_THROW(NLPAdapterCore{nullptr}, std::invalid_argument);
}

TEST(NLPAdapterCoreTest, HessianOwnerFallsBackToEqThenObjective) {
    auto p = std::make_shared<AdapterValProblem>();
    p->gl_ << 0.0, 0.0;
    p->gu_ << 0.0, 0.0; // both rows equality
    EXPECT_EQ(NLPAdapterCore{p}.hess_owner_, NLPAdapterCore::HessOwner::EqPiece);
    auto q = std::make_shared<AdapterValProblem>();
    q->m_ = 0;
    q->jnnz_ = 0;
    q->gl_.resize(0);
    q->gu_.resize(0);
    q->jr_.resize(0);
    q->jc_.resize(0);
    EXPECT_EQ(NLPAdapterCore{q}.hess_owner_, NLPAdapterCore::HessOwner::Objective);
}

// Assembly test problem (n=2): f = x0^2 + x1^2; rows: x0 + 2*x1 = 4 (equality),
// x0^2 + x1 <= 9 (upper-bounded), x0*x1 >= 1 (lower-bounded, negated by the
// adapter). The Hessian structure carries a duplicate (0, 0) slot to prove
// duplicate slots are summed rather than overwritten.
struct AsmTestProblem : NLPProblem {
    mutable int n_eval_g_ = 0, n_eval_jac_ = 0, n_eval_hess_ = 0;

    int num_vars() const override { return 2; }
    int num_cons() const override { return 3; }
    int num_jac_nonzeros() const override { return 6; }
    int num_hess_nonzeros() const override { return 4; }

    void bounds(Eigen::Ref<Eigen::VectorXd> xl, Eigen::Ref<Eigen::VectorXd> xu,
                Eigen::Ref<Eigen::VectorXd> gl, Eigen::Ref<Eigen::VectorXd> gu) const override {
        xl << -kInf, -kInf;
        xu << kInf, kInf;
        gl << 4.0, -kInf, 1.0;
        gu << 4.0, 9.0, kInf;
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
        n_eval_g_++;
        g[0] = x[0] + 2.0 * x[1];
        g[1] = x[0] * x[0] + x[1];
        g[2] = x[0] * x[1];
    }
    void jac_structure(Eigen::Ref<Eigen::VectorXi> r,
                       Eigen::Ref<Eigen::VectorXi> c) const override {
        r << 0, 0, 1, 1, 2, 2;
        c << 0, 1, 0, 1, 0, 1;
    }
    void hess_structure(Eigen::Ref<Eigen::VectorXi> r,
                        Eigen::Ref<Eigen::VectorXi> c) const override {
        r << 0, 1, 1, 0;
        c << 0, 1, 0, 0;
    }
    void eval_jac(ConstEigenRef<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> v) const override {
        n_eval_jac_++;
        v[0] = 1.0;
        v[1] = 2.0;
        v[2] = 2.0 * x[0];
        v[3] = 1.0;
        v[4] = x[1];
        v[5] = x[0];
    }
    void eval_hess(ConstEigenRef<Eigen::VectorXd>, double obj_factor,
                   ConstEigenRef<Eigen::VectorXd> lambda,
                   Eigen::Ref<Eigen::VectorXd> v) const override {
        n_eval_hess_++;
        v[0] = 1.5 * obj_factor + 2.0 * lambda[1];
        v[1] = 2.0 * obj_factor;
        v[2] = lambda[2];
        v[3] = 0.5 * obj_factor;
    }
};

TEST(NLPAdapterAssemblyTest, KktMatchesDenseReferenceAndCallbacksAreCounted) {
    auto prob = std::make_shared<AsmTestProblem>();
    auto core = std::make_shared<NLPAdapterCore>(prob);
    auto nlp = tycho::solvers::make_nlp_program(core);

    Eigen::SparseMatrix<double, Eigen::RowMajor> kkt(nlp->kkt_dim_, nlp->kkt_dim_);
    nlp->analyze_sparsity(kkt);
    // analyze_sparsity establishes the sparsity pattern by summing dummy 1.0
    // triplets at every physical slot (duplicate-slot claims sum, so a slot
    // two claims share reads back as 2.0) -- it is not a zeroed matrix, so a
    // fresh eval_kkt scatter must start from an explicit zero.
    Eigen::Map<Eigen::VectorXd>(kkt.valuePtr(), kkt.nonZeros()).setZero();

    Eigen::VectorXd X(2);
    X << 1.3, 0.7;
    Eigen::VectorXd LE(1), LI(2);
    LE << 0.4;
    LI << 0.9, 0.2;
    const double sigma = 2.0;

    double val = 0.0;
    Eigen::VectorXd PGX = Eigen::VectorXd::Zero(nlp->primal_vars_);
    Eigen::VectorXd AGX = Eigen::VectorXd::Zero(nlp->primal_vars_);
    Eigen::VectorXd FXE = Eigen::VectorXd::Zero(nlp->equal_cons_);
    Eigen::VectorXd FXI = Eigen::VectorXd::Zero(nlp->inequal_cons_);
    nlp->eval_kkt(sigma, X, LE, LI, val, PGX, AGX, FXE, FXI, kkt);

    const double x0 = X[0], x1 = X[1];
    EXPECT_NEAR(val, sigma * (x0 * x0 + x1 * x1), 1e-14);
    EXPECT_NEAR(FXE[0], x0 + 2 * x1 - 4.0, 1e-14);
    EXPECT_NEAR(FXI[0], x0 * x0 + x1 - 9.0, 1e-14);
    EXPECT_NEAR(FXI[1], 1.0 - x0 * x1, 1e-14);
    EXPECT_NEAR(PGX[0], sigma * 2 * x0, 1e-14);
    EXPECT_NEAR(PGX[1], sigma * 2 * x1, 1e-14);
    // Adjoint gradient: J_eq^T LE + J_iq^T LI over the SOLVER rows (lower negated).
    EXPECT_NEAR(AGX[0], LE[0] * 1.0 + LI[0] * 2 * x0 + LI[1] * (-x1), 1e-14);
    EXPECT_NEAR(AGX[1], LE[0] * 2.0 + LI[0] * 1.0 + LI[1] * (-x0), 1e-14);

    // User-space lambda: eq passes through, upper passes through, lower negates.
    const double lam1 = LI[0], lam2 = -LI[1];
    // NonLinearProgram::analyze_sparsity stores every claimed (row, col) pair
    // physically at (min(row, col), max(row, col)) -- PSIOPT's sparse backends
    // want the upper triangle of the symmetric KKT system filled, so a claim
    // like (constraint_row, variable_col), with constraint_row always the
    // larger index, is read back via kkt.coeff(variable_col, constraint_row).
    // The Hessian block's own diagonal entries need no such reordering.
    EXPECT_NEAR(kkt.coeff(0, 0), sigma * 2 + 2 * lam1, 1e-14); // duplicate slots summed
    EXPECT_NEAR(kkt.coeff(1, 1), sigma * 2, 1e-14);
    EXPECT_NEAR(kkt.coeff(0, 1), lam2, 1e-14);
    // Jacobian block: constraint rows sit after primals and slacks.
    const int con0 = nlp->primal_vars_ + nlp->slack_vars_;
    EXPECT_NEAR(kkt.coeff(0, con0 + 0), 1.0, 1e-14);
    EXPECT_NEAR(kkt.coeff(1, con0 + 0), 2.0, 1e-14);
    const int iq0 = con0 + nlp->equal_cons_;
    EXPECT_NEAR(kkt.coeff(0, iq0 + 0), 2 * x0, 1e-14);
    EXPECT_NEAR(kkt.coeff(1, iq0 + 0), 1.0, 1e-14);
    EXPECT_NEAR(kkt.coeff(0, iq0 + 1), -x1, 1e-14);
    EXPECT_NEAR(kkt.coeff(1, iq0 + 1), -x0, 1e-14);

    // Exactly one user callback of each kind for the whole assembly.
    EXPECT_EQ(prob->n_eval_g_, 1);
    EXPECT_EQ(prob->n_eval_jac_, 1);
    EXPECT_EQ(prob->n_eval_hess_, 1);
}

TEST(NLPAdapterAssemblyTest, NoObjectiveChainUsesZeroObjFactorConsumeOnce) {
    auto prob = std::make_shared<AsmTestProblem>();
    auto core = std::make_shared<NLPAdapterCore>(prob);
    auto nlp = tycho::solvers::make_nlp_program(core);
    Eigen::SparseMatrix<double, Eigen::RowMajor> kkt(nlp->kkt_dim_, nlp->kkt_dim_);
    nlp->analyze_sparsity(kkt);

    Eigen::VectorXd X(2), LE(1), LI(2);
    X << 1.3, 0.7;
    LE << 0.4;
    LI << 0.9, 0.2;
    double val = 0.0;
    Eigen::VectorXd PGX = Eigen::VectorXd::Zero(nlp->primal_vars_);
    Eigen::VectorXd AGX = Eigen::VectorXd::Zero(nlp->primal_vars_);
    Eigen::VectorXd FXE = Eigen::VectorXd::Zero(nlp->equal_cons_);
    Eigen::VectorXd FXI = Eigen::VectorXd::Zero(nlp->inequal_cons_);
    auto zero_kkt = [&] { Eigen::Map<Eigen::VectorXd>(kkt.valuePtr(), kkt.nonZeros()).setZero(); };

    // A full assembly first: leaves NO stale objective-scale record behind...
    nlp->eval_kkt(2.0, X, LE, LI, val, PGX, AGX, FXE, FXI, kkt);
    // ...so the no-objective assembly gets a pure-constraint Hessian.
    zero_kkt();
    nlp->eval_kkt_no(2.0, X, LE, LI, val, PGX, AGX, FXE, FXI, kkt);
    EXPECT_NEAR(kkt.coeff(1, 1), 0.0, 1e-14);       // the 2*sigma term is gone
    EXPECT_NEAR(kkt.coeff(0, 0), 2 * LI[0], 1e-14); // constraint curvature stays
    // Physically stored at (min, max) -- see the comment in the previous test.
    EXPECT_NEAR(kkt.coeff(0, 1), -LI[1], 1e-14);
    // And a full assembly afterwards has the objective curvature back.
    zero_kkt();
    nlp->eval_kkt(2.0, X, LE, LI, val, PGX, AGX, FXE, FXI, kkt);
    EXPECT_NEAR(kkt.coeff(1, 1), 2.0 * 2.0, 1e-14);
}

// AsmTestProblem whose eval_jac throws once armed -- models a user callback
// that aborts an in-flight KKT assembly partway through (e.g. the equality
// piece's own refresh_jac, after the objective piece already ran).
struct ThrowingJacAsmTestProblem : AsmTestProblem {
    bool armed_ = false;

    void eval_jac(ConstEigenRef<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> v) const override {
        if (armed_) {
            throw std::runtime_error("ThrowingJacAsmTestProblem: eval_jac armed to throw");
        }
        AsmTestProblem::eval_jac(x, v);
    }
};

TEST(NLPAdapterAssemblyTest, AbortedAssemblyDoesNotLeakStalePendingRecords) {
    auto prob = std::make_shared<ThrowingJacAsmTestProblem>();
    auto core = std::make_shared<NLPAdapterCore>(prob);
    auto nlp = tycho::solvers::make_nlp_program(core);

    Eigen::SparseMatrix<double, Eigen::RowMajor> kkt(nlp->kkt_dim_, nlp->kkt_dim_);
    nlp->analyze_sparsity(kkt);
    auto zero_kkt = [&] { Eigen::Map<Eigen::VectorXd>(kkt.valuePtr(), kkt.nonZeros()).setZero(); };
    zero_kkt();

    Eigen::VectorXd X1(2), X2(2), LE(1), LI(2);
    X1 << 1.3, 0.7;
    X2 << 2.1, -0.4;
    LE << 0.4;
    LI << 0.9, 0.2;
    double val = 0.0;
    Eigen::VectorXd PGX = Eigen::VectorXd::Zero(nlp->primal_vars_);
    Eigen::VectorXd AGX = Eigen::VectorXd::Zero(nlp->primal_vars_);
    Eigen::VectorXd FXE = Eigen::VectorXd::Zero(nlp->equal_cons_);
    Eigen::VectorXd FXI = Eigen::VectorXd::Zero(nlp->inequal_cons_);

    // (1) A full assembly at x1 succeeds -- the objective piece's Hessian-
    // bearing method runs and records, and the IQ owner consumes and resets.
    nlp->eval_kkt(2.0, X1, LE, LI, val, PGX, AGX, FXE, FXI, kkt);

    // (2) Arm the throw and run a second full assembly at x2. The objective
    // piece runs first and records pending_obj_scale_ = 2.0; the equality
    // piece's own refresh_jac then throws, aborting the chain before the IQ
    // owner ever runs to consume that record.
    prob->armed_ = true;
    EXPECT_THROW(nlp->eval_kkt(2.0, X2, LE, LI, val, PGX, AGX, FXE, FXI, kkt), std::runtime_error);
    prob->armed_ = false;

    // (3) A no-objective assembly (PSIOPT's restoration entry point) at the
    // now-cached x1 iterate must see obj_factor == 0: without the fix, the
    // stale pending_obj_scale_ left behind by the aborted assembly in (2)
    // would leak sigma's curvature into this constraint-only Hessian.
    zero_kkt();
    nlp->eval_kkt_no(2.0, X1, LE, LI, val, PGX, AGX, FXE, FXI, kkt);
    EXPECT_NEAR(kkt.coeff(1, 1), 0.0, 1e-14);
}

// Range-row assembly test problem (n=1): f = x0^2 is irrelevant to the
// assertions below (eval_hess is wired to lambda alone, so the test isolates
// the Range row's own composed multiplier); the single user row 1 <= x0 <= 3
// is Range-kind and splits into two solver inequality rows sharing the one
// declared Jacobian slot (upper: x0 - 3, lower: 1 - x0).
struct RangeAsmTestProblem : NLPProblem {
    int num_vars() const override { return 1; }
    int num_cons() const override { return 1; }
    int num_jac_nonzeros() const override { return 1; }
    int num_hess_nonzeros() const override { return 1; }

    void bounds(Eigen::Ref<Eigen::VectorXd> xl, Eigen::Ref<Eigen::VectorXd> xu,
                Eigen::Ref<Eigen::VectorXd> gl, Eigen::Ref<Eigen::VectorXd> gu) const override {
        xl << -kInf;
        xu << kInf;
        gl << 1.0;
        gu << 3.0;
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
    void eval_hess(ConstEigenRef<Eigen::VectorXd>, double, ConstEigenRef<Eigen::VectorXd> lambda,
                   Eigen::Ref<Eigen::VectorXd> v) const override {
        v[0] = lambda[0]; // isolates lambda_user_ = li_upper - li_lower
    }
};

TEST(NLPAdapterAssemblyTest, RangeRowScattersDuplicatedJacobianAndComposedLambdaHessian) {
    auto prob = std::make_shared<RangeAsmTestProblem>();
    auto core = std::make_shared<NLPAdapterCore>(prob);
    ASSERT_EQ(core->rows_.kinds_[0], NLPRowKind::Range);
    auto nlp = tycho::solvers::make_nlp_program(core);

    Eigen::SparseMatrix<double, Eigen::RowMajor> kkt(nlp->kkt_dim_, nlp->kkt_dim_);
    nlp->analyze_sparsity(kkt);
    // As in the assembly tests above, analyze_sparsity leaves placeholder 1.0
    // sums at every physical slot, so a fresh scatter must start from zero.
    Eigen::Map<Eigen::VectorXd>(kkt.valuePtr(), kkt.nonZeros()).setZero();

    Eigen::VectorXd X(1);
    X << 1.7;
    Eigen::VectorXd LE(0), LI(2);
    LI << 0.9, 0.4; // li_upper, li_lower
    const double sigma = 1.0;

    double val = 0.0;
    Eigen::VectorXd PGX = Eigen::VectorXd::Zero(nlp->primal_vars_);
    Eigen::VectorXd AGX = Eigen::VectorXd::Zero(nlp->primal_vars_);
    Eigen::VectorXd FXE = Eigen::VectorXd::Zero(nlp->equal_cons_);
    Eigen::VectorXd FXI = Eigen::VectorXd::Zero(nlp->inequal_cons_);
    nlp->eval_kkt(sigma, X, LE, LI, val, PGX, AGX, FXE, FXI, kkt);

    // (a) both inequality residuals land in FXI: upper is g-gu, lower is gl-g.
    EXPECT_NEAR(FXI[0], X[0] - 3.0, 1e-14);
    EXPECT_NEAR(FXI[1], 1.0 - X[0], 1e-14);

    // (b) the duplicated Jacobian claims scatter +J into the upper row and -J
    // into the lower row. Physically stored at (min(row, col), max(row, col)) --
    // see the comment on the dense assembly test above.
    const int con0 = nlp->primal_vars_ + nlp->slack_vars_ + nlp->equal_cons_;
    EXPECT_NEAR(kkt.coeff(0, con0 + 0), 1.0, 1e-14);
    EXPECT_NEAR(kkt.coeff(0, con0 + 1), -1.0, 1e-14);

    // (c) the Hessian slot uses lambda_user_ = li_upper - li_lower.
    EXPECT_NEAR(kkt.coeff(0, 0), LI[0] - LI[1], 1e-14);
}
