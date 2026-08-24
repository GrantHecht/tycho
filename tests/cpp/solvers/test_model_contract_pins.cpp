///////////////////////////////////////////////////////////////////////////////
// Boundary pins for the model contract, as the interior-point door sees it.
//
// The conversion-equivalence suite next door says the three doors agree on the
// answer. This file pins the parts of the contract that are not about the
// answer at all -- what is evaluated and WHEN, which multiplier block shapes
// are legal and which are refused and where, what a declared bound means at
// the extremes, and what a caller may assume about a destination an aborted
// assembly wrote into.
//
// Everything here is stated against the published surfaces (the model
// contract, its triplet-shaped convenience form, the host that carries a model
// onto the solver's program) and nothing here relies on solver internals or on
// any previously recorded trace of them.
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/hven_namespaces.h"

#include <hven/detail/model/nlp_adapter.h>
#include <hven/drivers/interior_point_solver.h>
#include <hven/model/nlp_problem.h>
#include <hven/model/nlp_problem_model.h>
#include <hven/model/nlp_solver.h>
#include <hven/model/non_linear_program.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

using hven::solvers::FixedVariableTreatments;
using hven::solvers::make_nlp_program;
using hven::solvers::NLPAdapterCore;
using hven::solvers::NlpModel;
using hven::solvers::NLPProblem;
using hven::solvers::NlpProblemModel;
using hven::solvers::NLPSolver;
using hven::solvers::NonLinearProgram;
using tycho::ConstEigenRef;
using Vec = Eigen::VectorXd;
using SpMatRM = Eigen::SparseMatrix<double, Eigen::RowMajor>;

namespace {

constexpr double kPinsInf = std::numeric_limits<double>::infinity();

/// A small, well behaved model used wherever the pin is about the plumbing
/// rather than about the mathematics:
///
///   min  (x0 - 1)^2 + (x1 - 2)^2
///   s.t. x0 + x1 - 1 = 0        (one equality row)
///        x0 - x1 - 3 <= 0       (one inequality row)
///
/// Counts every call it receives, so a test can say what was evaluated and how
/// often. The counters are mutable because the contract's methods are const:
/// counting is an observation of the model, not a change to it.
struct PinsCountingModel : NlpModel {
    Vec lower_ = Vec::Constant(2, -kPinsInf);
    Vec upper_ = Vec::Constant(2, kPinsInf);

    mutable int f_ = 0, grad_ = 0, ce_ = 0, ci_ = 0;
    mutable int jac_e_ = 0, jac_i_ = 0, hess_ = 0;
    mutable int start_point_ = 0;

    void reset_counts() const {
        f_ = grad_ = ce_ = ci_ = jac_e_ = jac_i_ = hess_ = start_point_ = 0;
    }

    hven::Index n() const override { return 2; }
    hven::Index me() const override { return 1; }
    hven::Index mi() const override { return 1; }

    double eval_f(const Vec &x) const override {
        f_++;
        const double a = x[0] - 1.0, b = x[1] - 2.0;
        return a * a + b * b;
    }
    Vec eval_grad(const Vec &x) const override {
        grad_++;
        return (Vec(2) << 2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.0)).finished();
    }
    Vec eval_ce(const Vec &x) const override {
        ce_++;
        return (Vec(1) << x[0] + x[1] - 1.0).finished();
    }
    Vec eval_ci(const Vec &x) const override {
        ci_++;
        return (Vec(1) << x[0] - x[1] - 3.0).finished();
    }
    SpMatRM eval_jac_e(const Vec &) const override {
        jac_e_++;
        SpMatRM j(1, 2);
        std::vector<Eigen::Triplet<double>> t{{0, 0, 1.0}, {0, 1, 1.0}};
        j.setFromTriplets(t.begin(), t.end());
        j.makeCompressed();
        return j;
    }
    SpMatRM eval_jac_i(const Vec &) const override {
        jac_i_++;
        SpMatRM j(1, 2);
        std::vector<Eigen::Triplet<double>> t{{0, 0, 1.0}, {0, 1, -1.0}};
        j.setFromTriplets(t.begin(), t.end());
        j.makeCompressed();
        return j;
    }
    SpMatRM eval_hess(const Vec &, double obj_scale, const Vec &, const Vec &) const override {
        hess_++;
        SpMatRM h(2, 2);
        std::vector<Eigen::Triplet<double>> t{{0, 0, 2.0 * obj_scale}, {1, 1, 2.0 * obj_scale}};
        h.setFromTriplets(t.begin(), t.end());
        h.makeCompressed();
        return h;
    }
    const Vec &lower() const override { return lower_; }
    const Vec &upper() const override { return upper_; }
    Vec start_point() const override {
        start_point_++;
        return Vec::Zero(2);
    }
};

/// The same problem in the triplet shape, counting its own callbacks.
struct PinsCountingProblem : NLPProblem {
    mutable int f_ = 0, grad_f_ = 0, g_ = 0, jac_ = 0, hess_ = 0;
    mutable int jac_structure_ = 0, hess_structure_ = 0, bounds_ = 0;

    int num_vars() const override { return 2; }
    int num_cons() const override { return 2; }
    int num_jac_nonzeros() const override { return 4; }
    int num_hess_nonzeros() const override { return 2; }

    void bounds(Eigen::Ref<Eigen::VectorXd> xl, Eigen::Ref<Eigen::VectorXd> xu,
                Eigen::Ref<Eigen::VectorXd> gl, Eigen::Ref<Eigen::VectorXd> gu) const override {
        bounds_++;
        xl << -kPinsInf, -kPinsInf;
        xu << kPinsInf, kPinsInf;
        gl << 1.0, -kPinsInf;
        gu << 1.0, 3.0;
    }
    void eval_f(ConstEigenRef<Eigen::VectorXd> x, double &f) const override {
        f_++;
        const double a = x[0] - 1.0, b = x[1] - 2.0;
        f = a * a + b * b;
    }
    void eval_grad_f(ConstEigenRef<Eigen::VectorXd> x,
                     Eigen::Ref<Eigen::VectorXd> g) const override {
        grad_f_++;
        g[0] = 2.0 * (x[0] - 1.0);
        g[1] = 2.0 * (x[1] - 2.0);
    }
    void eval_g(ConstEigenRef<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> g) const override {
        g_++;
        g[0] = x[0] + x[1];
        g[1] = x[0] - x[1];
    }
    void jac_structure(Eigen::Ref<Eigen::VectorXi> r,
                       Eigen::Ref<Eigen::VectorXi> c) const override {
        jac_structure_++;
        r << 0, 0, 1, 1;
        c << 0, 1, 0, 1;
    }
    void hess_structure(Eigen::Ref<Eigen::VectorXi> r,
                        Eigen::Ref<Eigen::VectorXi> c) const override {
        hess_structure_++;
        r << 0, 1;
        c << 0, 1;
    }
    void eval_jac(ConstEigenRef<Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd> v) const override {
        jac_++;
        v[0] = 1.0;
        v[1] = 1.0;
        v[2] = 1.0;
        v[3] = -1.0;
    }
    void eval_hess(ConstEigenRef<Eigen::VectorXd>, double obj_factor,
                   ConstEigenRef<Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd> v) const override {
        hess_++;
        v[0] = 2.0 * obj_factor;
        v[1] = 2.0 * obj_factor;
    }
    std::string name() const override { return "PinsCountingProblem"; }
};

/// One assembly harness: a program laid over a host, plus the destinations an
/// evaluation writes into, seeded before every call the way a consumer seeds
/// them.
struct PinsAssembly {
    std::shared_ptr<NLPAdapterCore> core_;
    std::shared_ptr<NonLinearProgram> nlp_;
    SpMatRM kkt_;
    double val_ = 0.0;
    Eigen::VectorXd pgx_, agx_, fxe_, fxi_;

    explicit PinsAssembly(const std::shared_ptr<NlpModel> &model, std::string name = "NLP model")
        : core_(std::make_shared<NLPAdapterCore>(model, std::move(name))) {
        nlp_ = make_nlp_program(core_);
        kkt_ = SpMatRM(nlp_->kkt_dim_, nlp_->kkt_dim_);
        nlp_->analyze_sparsity(kkt_);
        this->seed();
    }

    /// Zeroes every destination. This is the consumer's own responsibility on
    /// every assembly, aborted or not, which is what makes a retry after a
    /// throw well defined.
    void seed() {
        Eigen::Map<Eigen::VectorXd>(kkt_.valuePtr(), kkt_.nonZeros()).setZero();
        val_ = 0.0;
        pgx_ = Eigen::VectorXd::Zero(nlp_->primal_vars_);
        agx_ = Eigen::VectorXd::Zero(nlp_->primal_vars_);
        fxe_ = Eigen::VectorXd::Zero(nlp_->equal_cons_);
        fxi_ = Eigen::VectorXd::Zero(nlp_->inequal_cons_);
    }

    void eval_kkt(double obj_scale, const Eigen::VectorXd &x, const Eigen::VectorXd &le,
                  const Eigen::VectorXd &li) {
        nlp_->eval_kkt(obj_scale, x, le, li, val_, pgx_, agx_, fxe_, fxi_, kkt_);
    }

    Eigen::VectorXd kkt_values() const {
        return Eigen::Map<const Eigen::VectorXd>(kkt_.valuePtr(), kkt_.nonZeros());
    }
};

Eigen::VectorXd pins_point() { return (Eigen::VectorXd(2) << 1.3, 0.7).finished(); }

} // namespace

///////////////////////////////////////////////////////////////////////////////
// What transcription evaluates, and when.
//
// Setting a problem up is not a purely structural act: the host has to learn
// the sparsity pattern of the three matrices, and the only way to learn it is
// to evaluate them. That evaluation happens once, at the model's own start
// point, before any solve iterate exists -- so it is part of what a caller
// observes, and any count of a model's callbacks has to account for it.
///////////////////////////////////////////////////////////////////////////////

TEST(ModelContractPins, TranscriptionEvaluatesTheDerivativesOnceAtTheStartPoint) {
    auto model = std::make_shared<PinsCountingModel>();
    NLPAdapterCore core(model, "PinsCountingModel");

    // Exactly one of each derivative, at the start point. The values are
    // discarded; the pattern is what is kept.
    EXPECT_EQ(model->hess_, 1);
    EXPECT_EQ(model->jac_e_, 1);
    EXPECT_EQ(model->jac_i_, 1);
    EXPECT_EQ(model->start_point_, 1);

    // And nothing else: the objective and the residuals are not part of
    // learning a pattern, so a model whose eval_f is expensive or undefined at
    // the start point pays nothing here.
    EXPECT_EQ(model->f_, 0);
    EXPECT_EQ(model->grad_, 0);
    EXPECT_EQ(model->ce_, 0);
    EXPECT_EQ(model->ci_, 0);

    EXPECT_EQ(core.n_, 2);
    EXPECT_EQ(core.num_eq_, 1);
    EXPECT_EQ(core.num_iq_, 1);
}

TEST(ModelContractPins, TripletTranscriptionEvaluatesTheJacobianAndHessianOnceEach) {
    auto problem = std::make_shared<PinsCountingProblem>();
    NLPSolver solver(problem);
    solver.optimizer_->set_print_level(3);
    solver.transcribe();

    // One Jacobian call, not two: the conversion caches per iterate, so the
    // equality and the inequality blocks of one pattern walk share it.
    EXPECT_EQ(problem->jac_, 1);
    EXPECT_EQ(problem->hess_, 1);
    EXPECT_EQ(problem->f_, 0);
    EXPECT_EQ(problem->grad_f_, 0);
    EXPECT_EQ(problem->g_, 0);

    // The structures are queried once and never again, which is what lets the
    // conversion lay its patterns at construction and only write values after.
    EXPECT_EQ(problem->jac_structure_, 1);
    EXPECT_EQ(problem->hess_structure_, 1);
}

TEST(ModelContractPins, AnEvaluationAfterTranscriptionAddsToThoseCounts) {
    auto model = std::make_shared<PinsCountingModel>();
    PinsAssembly asm_(model, "PinsCountingModel");
    model->reset_counts();

    const Eigen::VectorXd x = pins_point();
    const Eigen::VectorXd le = (Eigen::VectorXd(1) << 0.5).finished();
    const Eigen::VectorXd li = (Eigen::VectorXd(1) << 0.25).finished();
    asm_.eval_kkt(1.0, x, le, li);

    // One full assembly is one of everything. The counts start from zero here
    // only because they were reset after transcription -- an unreset count
    // carries transcription's derivative calls as well, which is the point of
    // the pin above.
    EXPECT_EQ(model->f_, 1);
    EXPECT_EQ(model->grad_, 1);
    EXPECT_EQ(model->ce_, 1);
    EXPECT_EQ(model->ci_, 1);
    EXPECT_EQ(model->jac_e_, 1);
    EXPECT_EQ(model->jac_i_, 1);
    EXPECT_EQ(model->hess_, 1);
}

///////////////////////////////////////////////////////////////////////////////
// Multiplier block shapes: what an EMPTY block means, and where.
//
// THE RULE, stated in one place because two layers read a block and they read
// it under different premises:
//
//   At the MODEL surface -- eval_hess on the contract itself, and on the
//   conversion that implements it -- an empty multiplier block is a legal,
//   compact spelling of an all-zero block. There is no chain behind the call,
//   so the length carries no other information, and a caller probing the
//   structure at zero multipliers should not have to materialize a zero
//   vector to do it.
//
//   At the CHAIN surface -- the host that carries a model onto the solver's
//   program -- a block SHORTER than the rows the host laid, empty included,
//   is refused. There the length is a statement about the assembled row
//   space: the engine hands down a block at least as long as the rows it
//   laid (it may be longer, since the engine appends rows of its own after
//   the model's, and the host's are the head), so a short one is a defect in
//   the chain and reading it as zeros would hide it. An empty block is the
//   RIGHT length there exactly when the host laid no rows of that kind.
//
// Both host entry points that read a block -- the record the equality piece
// leaves for the Hessian owner, and the owner's own read -- follow the chain
// rule, and both are pinned below, each by the site its refusal names.
///////////////////////////////////////////////////////////////////////////////

TEST(ModelContractPins, TheModelSurfaceReadsAnEmptyMultiplierBlockAsAllZero) {
    NlpProblemModel model(std::make_shared<PinsCountingProblem>());
    ASSERT_EQ(model.me(), 1);
    ASSERT_EQ(model.mi(), 1);

    const Eigen::VectorXd x = pins_point();
    const Eigen::VectorXd zero_e = Eigen::VectorXd::Zero(1);
    const Eigen::VectorXd zero_i = Eigen::VectorXd::Zero(1);

    const SpMatRM from_zeros = model.eval_hess(x, 1.0, zero_e, zero_i);
    const SpMatRM from_empty = model.eval_hess(x, 1.0, Eigen::VectorXd(0), Eigen::VectorXd(0));

    ASSERT_EQ(from_zeros.nonZeros(), from_empty.nonZeros());
    for (int k = 0; k < from_zeros.nonZeros(); k++) {
        EXPECT_EQ(from_zeros.valuePtr()[k], from_empty.valuePtr()[k]) << "entry " << k;
    }

    // The composition onto the declared row space reads an empty block the
    // same way, for the same reason.
    const Eigen::VectorXd composed =
        model.compose_user_multipliers(Eigen::VectorXd(0), Eigen::VectorXd(0));
    EXPECT_EQ(composed.size(), 2);
    EXPECT_TRUE(composed.isZero(0.0));
}

TEST(ModelContractPins, TheChainSurfaceRefusesAnEmptyBlockWhereItHostsRows) {
    auto model = std::make_shared<PinsCountingModel>();
    NLPAdapterCore core(model, "PinsCountingModel");
    ASSERT_EQ(core.num_eq_, 1);
    ASSERT_EQ(core.num_iq_, 1);

    // The record the equality piece leaves behind.
    try {
        core.record_equality_multipliers(Eigen::VectorXd(0));
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument &e) {
        const std::string message(e.what());
        EXPECT_NE(message.find("0 equality multipliers"), std::string::npos) << message;
        EXPECT_NE(message.find("hosts 1 equality rows"), std::string::npos) << message;
        EXPECT_NE(message.find("at the record"), std::string::npos) << message;
    }

    // The Hessian owner's own read, refused for either block.
    const Eigen::VectorXd x = pins_point();
    const Eigen::VectorXd one_e = (Eigen::VectorXd(1) << 0.5).finished();
    const Eigen::VectorXd one_i = (Eigen::VectorXd(1) << 0.25).finished();
    try {
        core.eval_hessian_values(x, 1.0, Eigen::VectorXd(0), one_i);
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument &e) {
        const std::string message(e.what());
        EXPECT_NE(message.find("0 equality multipliers"), std::string::npos) << message;
        EXPECT_NE(message.find("at the Hessian owner"), std::string::npos) << message;
    }
    try {
        core.eval_hessian_values(x, 1.0, one_e, Eigen::VectorXd(0));
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument &e) {
        const std::string message(e.what());
        EXPECT_NE(message.find("0 inequality multipliers"), std::string::npos) << message;
        EXPECT_NE(message.find("at the Hessian owner"), std::string::npos) << message;
    }

    // A block LONGER than the host's rows is fine: the host's own block is the
    // head of what the engine hands down.
    const Eigen::VectorXd long_e = (Eigen::VectorXd(3) << 0.5, 9.0, 9.0).finished();
    EXPECT_NO_THROW(core.record_equality_multipliers(long_e));
    EXPECT_NO_THROW(core.eval_hessian_values(x, 1.0, long_e, one_i));
}

namespace {

/// A model with no rows at all, for the other half of the chain rule: where a
/// host laid no rows of a kind, the empty block IS the right length.
struct PinsUnconstrainedModel : NlpModel {
    Vec lower_ = Vec::Constant(2, -kPinsInf);
    Vec upper_ = Vec::Constant(2, kPinsInf);

    hven::Index n() const override { return 2; }
    hven::Index me() const override { return 0; }
    hven::Index mi() const override { return 0; }

    double eval_f(const Vec &x) const override { return x[0] * x[0] + x[1] * x[1]; }
    Vec eval_grad(const Vec &x) const override {
        return (Vec(2) << 2.0 * x[0], 2.0 * x[1]).finished();
    }
    Vec eval_ce(const Vec &) const override { return Vec(0); }
    Vec eval_ci(const Vec &) const override { return Vec(0); }
    SpMatRM eval_jac_e(const Vec &) const override {
        SpMatRM j(0, 2);
        j.makeCompressed();
        return j;
    }
    SpMatRM eval_jac_i(const Vec &) const override {
        SpMatRM j(0, 2);
        j.makeCompressed();
        return j;
    }
    SpMatRM eval_hess(const Vec &, double obj_scale, const Vec &, const Vec &) const override {
        SpMatRM h(2, 2);
        std::vector<Eigen::Triplet<double>> t{{0, 0, 2.0 * obj_scale}, {1, 1, 2.0 * obj_scale}};
        h.setFromTriplets(t.begin(), t.end());
        h.makeCompressed();
        return h;
    }
    const Vec &lower() const override { return lower_; }
    const Vec &upper() const override { return upper_; }
    Vec start_point() const override { return Vec::Zero(2); }
};

} // namespace

TEST(ModelContractPins, TheChainSurfaceAcceptsAnEmptyBlockWhereItHostsNoRows) {
    auto model = std::make_shared<PinsUnconstrainedModel>();
    NLPAdapterCore core(model, "PinsUnconstrainedModel");
    ASSERT_EQ(core.num_eq_, 0);
    ASSERT_EQ(core.num_iq_, 0);

    // Not "an empty block means zeros" -- an empty block is the whole block,
    // because there are no rows for it to be the head of.
    EXPECT_NO_THROW(
        core.eval_hessian_values(pins_point(), 1.0, Eigen::VectorXd(0), Eigen::VectorXd(0)));
    EXPECT_NO_THROW(core.record_equality_multipliers(Eigen::VectorXd(0)));
}

// A short block has to be caught at the piece's FIRST READ of it, not at the
// later record: the read indexes into the block, so reaching the record means
// the fill has already indexed past its end -- which under NDEBUG is silent.
// The two guards therefore name different sites in the same message, and this
// pin asserts which one fired rather than only that something did. The message
// text and both sites live in the solver library's adapter host
// (dep/hven/src/model/nlp_adapter.cpp, refuse_short_multiplier_block).
TEST(ModelContractPins, AShortMultiplierBlockIsRefusedAtTheFirstReadNotAtTheRecord) {
    auto model = std::make_shared<PinsCountingModel>();
    PinsAssembly asm_(model, "PinsCountingModel");
    ASSERT_EQ(asm_.core_->num_eq_, 1);

    const Eigen::VectorXd x = pins_point();
    const Eigen::VectorXd empty_e(0);
    const Eigen::VectorXd li = (Eigen::VectorXd(1) << 0.25).finished();
    try {
        asm_.eval_kkt(2.0, x, empty_e, li);
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument &e) {
        const std::string message(e.what());
        EXPECT_NE(message.find("0 equality multipliers"), std::string::npos) << message;
        EXPECT_NE(message.find("hosts 1 equality rows"), std::string::npos) << message;
        EXPECT_NE(message.find("PinsCountingModel"), std::string::npos) << message;
        EXPECT_NE(message.find("at the piece's first read"), std::string::npos) << message;
        EXPECT_EQ(message.find("at the record"), std::string::npos) << message;
        EXPECT_EQ(message.find("at the Hessian owner"), std::string::npos) << message;
    }
}

///////////////////////////////////////////////////////////////////////////////
// What a declared bound means.
///////////////////////////////////////////////////////////////////////////////

namespace {

/// A model carrying a finite LOWER bound of magnitude 1e20, to pin that this
/// door reads it as the bound it is.
///
///   min  (x0 * 1e-20 - 0.5)^2 + (x1 - 2)^2   s.t.  x0 >= 1e20
///
/// The first term alone is minimized at x0 = 0.5e20, BELOW the bound, so the
/// bound binds and the answer is x0 = 1e20. A door that read a bound of that
/// magnitude as "unbounded" would return 0.5e20 instead -- a factor of two,
/// not a rounding question. The 1e-20 scaling inside the objective is what
/// keeps the residuals of ordinary size while the bound itself is enormous,
/// so the solve converges on its own merits and the pin is about the bound.
struct PinsLargeBoundModel : NlpModel {
    static constexpr double kScale = 1.0e-20;
    Vec lower_ = (Vec(2) << 1.0e20, -kPinsInf).finished();
    Vec upper_ = Vec::Constant(2, kPinsInf);

    hven::Index n() const override { return 2; }
    hven::Index me() const override { return 0; }
    hven::Index mi() const override { return 0; }

    double eval_f(const Vec &x) const override {
        const double a = x[0] * kScale - 0.5, b = x[1] - 2.0;
        return a * a + b * b;
    }
    Vec eval_grad(const Vec &x) const override {
        return (Vec(2) << 2.0 * (x[0] * kScale - 0.5) * kScale, 2.0 * (x[1] - 2.0)).finished();
    }
    Vec eval_ce(const Vec &) const override { return Vec(0); }
    Vec eval_ci(const Vec &) const override { return Vec(0); }
    SpMatRM eval_jac_e(const Vec &) const override {
        SpMatRM j(0, 2);
        j.makeCompressed();
        return j;
    }
    SpMatRM eval_jac_i(const Vec &) const override {
        SpMatRM j(0, 2);
        j.makeCompressed();
        return j;
    }
    SpMatRM eval_hess(const Vec &, double obj_scale, const Vec &, const Vec &) const override {
        SpMatRM h(2, 2);
        std::vector<Eigen::Triplet<double>> t{{0, 0, 2.0 * kScale * kScale * obj_scale},
                                              {1, 1, 2.0 * obj_scale}};
        h.setFromTriplets(t.begin(), t.end());
        h.makeCompressed();
        return h;
    }
    const Vec &lower() const override { return lower_; }
    const Vec &upper() const override { return upper_; }
    Vec start_point() const override {
        Vec x0(2);
        for (int i = 0; i < 2; i++) {
            x0[i] = std::min(std::max(0.0, lower_[i]), upper_[i]);
        }
        return x0;
    }
};

} // namespace

// Every finite value is a real bound however large: this door applies no
// cutoff past which a number starts meaning "unbounded". The bound is carried
// verbatim through the host, and reaches the solver's own bound set -- the
// list the barrier terms are built from -- as a bound rather than as an
// absence.
//
// SCOPE. This says nothing about magnitudes at or above 1e20 anywhere else.
// The library's SSN/QP route deliberately treats a bound of that magnitude as
// absent, so a genuine bound in [1e20, infinity) is honored here and not
// there. That divergence is a registered, intentional difference between the
// two routes; no cross-route equivalence is asserted in that band, here or
// anywhere in this suite.
TEST(ModelContractPins, AFiniteBoundOfVeryLargeMagnitudeIsARealBoundAtThisDoor) {
    auto model = std::make_shared<PinsLargeBoundModel>();
    auto core = std::make_shared<NLPAdapterCore>(model, "PinsLargeBoundModel");

    // Verbatim through the host: the value itself, not an infinity and not a
    // rescaled stand-in.
    EXPECT_EQ(core->x_lower_[0], 1.0e20);
    EXPECT_TRUE(std::isfinite(core->x_lower_[0]));
    EXPECT_EQ(core->x_upper_[0], kPinsInf);

    // And through to the layout: the projected start point sits ON the bound,
    // which it only can if the bound is there at all.
    EXPECT_EQ(model->start_point()[0], 1.0e20);

    auto nlp = make_nlp_program(core);
    ASSERT_TRUE(nlp->has_variable_bounds());
    // The return says whether the layout had to be rebuilt, which it does not
    // here -- nothing is fixed, so nothing is eliminated. The classification
    // still runs, and the bound set below is what it produced.
    nlp->configure_variable_treatment(FixedVariableTreatments::MakeParameter, 0.0);

    // The bound set is what the barrier divides by. A variable whose bound had
    // been read as "absent" would not be in it at all.
    const auto &bounds = nlp->variable_bound_set();
    ASSERT_TRUE(bounds.any());
    ASSERT_EQ(bounds.lower_idx_.size(), 1);
    EXPECT_EQ(bounds.lower_idx_[0], 0);
    EXPECT_EQ(bounds.lower_val_[0], 1.0e20); // relax factor 0.0: recorded verbatim
    EXPECT_EQ(bounds.upper_idx_.size(), 0);
    EXPECT_FALSE(nlp->is_reduced());
}

// The same bound, end to end. Without it the answer is 0.5e20; with it the
// answer is the bound. The solve is asked to distinguish those, which it can
// only do if the bound is a bound.
TEST(ModelContractPins, ASolveHonorsAFiniteBoundOfVeryLargeMagnitude) {
    auto model = std::make_shared<PinsLargeBoundModel>();
    auto core = std::make_shared<NLPAdapterCore>(model, "PinsLargeBoundModel");
    auto nlp = make_nlp_program(core);

    hven::solvers::InteriorPointSolver opt;
    opt.set_print_level(3);
    // Every finite bound is widened by the relax factor before the barrier
    // sees it, and on a bound of this magnitude the default widening is itself
    // enormous (1e-8 of 1e20 is 1e12). Zeroing it here is not part of the pin;
    // it is what lets the pin be stated against the declared value rather than
    // against the declared value less a term about the relaxation.
    opt.set_bound_relax_factor(0.0);
    opt.set_nlp(nlp);
    const Eigen::VectorXd x = opt.optimize((Eigen::VectorXd(2) << 2.0e20, 0.0).finished());

    ASSERT_EQ(x.size(), 2);
    EXPECT_LE(opt.result().converge_flag_, tycho::ConvergenceFlags::ACCEPTABLE);
    EXPECT_GE(x[0], 1.0e20) << "the declared lower bound was not honored";
    EXPECT_LE(x[0], 1.0e20 * (1.0 + 1.0e-6)) << "the point did not settle on the bound";
    // The unbounded coordinate lands where its own term wants it, which says
    // the large bound did not disturb the rest of the problem.
    EXPECT_NEAR(x[1], 2.0, 1.0e-5);
}

namespace {

/// A model whose lower bound is NaN, for the rejection pin below.
struct PinsNaNBoundModel : PinsLargeBoundModel {
    PinsNaNBoundModel() { lower_[1] = std::numeric_limits<double>::quiet_NaN(); }
};

/// A model whose bound pair is inverted.
struct PinsInvertedBoundModel : PinsLargeBoundModel {
    PinsInvertedBoundModel() {
        lower_[1] = 2.0;
        upper_[1] = 1.0;
    }
};

/// A model fixing a variable at an infinity, which is not a way to fix it.
struct PinsInfiniteFixModel : PinsLargeBoundModel {
    PinsInfiniteFixModel() {
        lower_[1] = kPinsInf;
        upper_[1] = kPinsInf;
    }
};

} // namespace

// A NaN bound is rejected where the model is adopted, by name, rather than
// carried into a layout and divided by later. Same for the two other
// unusable pairs the host screens.
TEST(ModelContractPins, UnusableVariableBoundsAreRefusedWhenTheModelIsAdopted) {
    try {
        NLPAdapterCore core(std::make_shared<PinsNaNBoundModel>(), "PinsNaNBoundModel");
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument &e) {
        const std::string message(e.what());
        EXPECT_NE(message.find("variable 1"), std::string::npos) << message;
        EXPECT_NE(message.find("NaN"), std::string::npos) << message;
        EXPECT_NE(message.find("PinsNaNBoundModel"), std::string::npos) << message;
    }

    try {
        NLPAdapterCore core(std::make_shared<PinsInvertedBoundModel>(), "PinsInvertedBoundModel");
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument &e) {
        const std::string message(e.what());
        EXPECT_NE(message.find("variable 1"), std::string::npos) << message;
        EXPECT_NE(message.find("exceeds"), std::string::npos) << message;
    }

    try {
        NLPAdapterCore core(std::make_shared<PinsInfiniteFixModel>(), "PinsInfiniteFixModel");
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument &e) {
        const std::string message(e.what());
        EXPECT_NE(message.find("variable 1"), std::string::npos) << message;
        EXPECT_NE(message.find("fixed at infinity"), std::string::npos) << message;
    }
}

///////////////////////////////////////////////////////////////////////////////
// What an aborted assembly leaves behind.
//
// A model callback can throw part way through an assembly, after some of the
// destination has already been written. THE RULE: the destination then holds
// an UNSPECIFIED PREFIX -- the consumer may not assume it is complete, and may
// not assume which part of it is missing -- and a retry RE-SEEDS rather than
// resuming. Seeding is the consumer's own step before every assembly, so a
// retry that follows the ordinary protocol is already correct; what the host
// owes is that nothing survives the abort to corrupt the retry, which is why
// its per-assembly consume-once records are cleared on the way out through any
// exception.
//
// The two assertions below are exactly those two halves. Nothing here asserts
// WHICH entries the aborted assembly had written, because that is the part the
// rule leaves unspecified.
///////////////////////////////////////////////////////////////////////////////

namespace {

/// PinsCountingModel with a Hessian that throws for a bounded number of calls
/// and then heals. The Hessian is the LAST thing an assembly evaluates, so a
/// throw there aborts with the objective and both constraint blocks already
/// written -- the deepest partial fill an assembly can produce.
struct PinsThrowingHessianModel : PinsCountingModel {
    mutable int throw_budget_ = 1;

    SpMatRM eval_hess(const Vec &x, double obj_scale, const Vec &le, const Vec &li) const override {
        if (throw_budget_ > 0) {
            throw_budget_--;
            throw std::runtime_error("PinsThrowingHessianModel: refusing this Hessian");
        }
        return PinsCountingModel::eval_hess(x, obj_scale, le, li);
    }
};

} // namespace

TEST(ModelContractPins, AnAssemblyAbortedMidFillLeavesAnIncompleteDestination) {
    auto model = std::make_shared<PinsThrowingHessianModel>();
    // Transcription evaluates the Hessian once, so the budget has to cover
    // that call as well as the one the assembly below makes.
    model->throw_budget_ = 0;
    PinsAssembly asm_(model, "PinsThrowingHessianModel");

    const Eigen::VectorXd x = pins_point();
    const Eigen::VectorXd le = (Eigen::VectorXd(1) << 0.5).finished();
    const Eigen::VectorXd li = (Eigen::VectorXd(1) << 0.25).finished();

    // A complete assembly, for something to compare against.
    asm_.seed();
    ASSERT_NO_THROW(asm_.eval_kkt(2.0, x, le, li));
    const Eigen::VectorXd complete = asm_.kkt_values();
    ASSERT_GT(complete.size(), 0);
    ASSERT_FALSE(complete.isZero(0.0));

    // The same assembly, aborted at the Hessian.
    asm_.seed();
    model->throw_budget_ = 1;
    EXPECT_THROW(asm_.eval_kkt(2.0, x, le, li), std::runtime_error);

    // The destination is NOT the destination of a completed assembly. Which
    // entries are missing is deliberately not asserted -- the rule is that the
    // caller may not use it, not that it stopped at a particular place.
    const Eigen::VectorXd aborted = asm_.kkt_values();
    ASSERT_EQ(aborted.size(), complete.size());
    EXPECT_FALSE(aborted.isApprox(complete, 0.0))
        << "an aborted assembly must not be mistakable for a completed one";
}

TEST(ModelContractPins, ARetryAfterAnAbortedAssemblyReSeedsAndReproducesTheCompleteFill) {
    auto model = std::make_shared<PinsThrowingHessianModel>();
    model->throw_budget_ = 0;
    PinsAssembly asm_(model, "PinsThrowingHessianModel");

    const Eigen::VectorXd x = pins_point();
    const Eigen::VectorXd le = (Eigen::VectorXd(1) << 0.5).finished();
    const Eigen::VectorXd li = (Eigen::VectorXd(1) << 0.25).finished();

    // The reference, from a host that never aborted anything.
    auto clean_model = std::make_shared<PinsThrowingHessianModel>();
    clean_model->throw_budget_ = 0;
    PinsAssembly clean(clean_model, "PinsThrowingHessianModel");
    clean.seed();
    ASSERT_NO_THROW(clean.eval_kkt(2.0, x, le, li));
    const Eigen::VectorXd reference = clean.kkt_values();

    // Abort, then re-seed and retry against a healed model. The retry has to
    // land exactly on the reference: nothing the aborted assembly wrote may
    // survive its re-seeding, and no record it left may be picked up.
    asm_.seed();
    model->throw_budget_ = 1;
    ASSERT_THROW(asm_.eval_kkt(2.0, x, le, li), std::runtime_error);

    asm_.seed();
    ASSERT_NO_THROW(asm_.eval_kkt(2.0, x, le, li));
    const Eigen::VectorXd retried = asm_.kkt_values();
    ASSERT_EQ(retried.size(), reference.size());
    for (Eigen::Index k = 0; k < retried.size(); k++) {
        EXPECT_EQ(retried[k], reference[k]) << "KKT value " << k;
    }
    EXPECT_EQ(asm_.val_, clean.val_);
    for (Eigen::Index k = 0; k < asm_.agx_.size(); k++) {
        EXPECT_EQ(asm_.agx_[k], clean.agx_[k]) << "adjoint gradient " << k;
    }
}

// The mechanism behind the retry: the host's per-assembly records are consumed
// once, and an assembly that throws clears them on the way out. A record left
// behind would be read by the NEXT chain -- which may be a different one
// entirely, since a caller may retry, and the solver may run a different
// evaluation shape next -- as if it were that chain's own.
TEST(ModelContractPins, AnAbortedAssemblyLeavesNoConsumeOnceRecordBehind) {
    auto model = std::make_shared<PinsThrowingHessianModel>();
    model->throw_budget_ = 0;
    PinsAssembly asm_(model, "PinsThrowingHessianModel");

    const Eigen::VectorXd x = pins_point();
    const Eigen::VectorXd le = (Eigen::VectorXd(1) << 0.5).finished();
    const Eigen::VectorXd li = (Eigen::VectorXd(1) << 0.25).finished();

    asm_.seed();
    model->throw_budget_ = 1;
    ASSERT_THROW(asm_.eval_kkt(2.0, x, le, li), std::runtime_error);

    EXPECT_FALSE(asm_.core_->pending_obj_scale_.has_value())
        << "the objective scale this chain recorded must not outlive it";
    EXPECT_FALSE(asm_.core_->le_recorded_)
        << "the equality multipliers this chain recorded must not outlive it";
}
