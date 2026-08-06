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
