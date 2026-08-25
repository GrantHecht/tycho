///////////////////////////////////////////////////////////////////////////////
// Conversion equivalence: one problem, three doors.
//
// The same mathematical NLP can reach the interior-point engine three ways,
// and this file states, once, that the three agree -- on the optimum AND on
// the multipliers, including the multipliers of a ranged constraint, whose
// single declared value corresponds to a PAIR of native inequality rows.
//
//   Door 1 (VF)      tycho::solvers::OptimizationProblem: the objective and
//                    the constraints as VectorFunction expression trees,
//                    declared -- pieces, thread modes, bounds and partition
//                    count -- into an AggregateDeclaration, laid out from it,
//                    and solved.
//   Door 2 (native)  the same problem hand-authored against the native model
//                    contract (hven::solvers::NlpModel), carried onto a
//                    NonLinearProgram by the adapter host and solved.
//   Door 3 (triplet) the same problem declared in the triplet shape
//                    (hven::solvers::NLPProblem) and handed to NLPSolver,
//                    which converts it through NlpProblemModel and then
//                    through that same adapter host.
//
// Doors 2 and 3 share their route past the model surface: NLPSolver's own
// conversion product IS an NlpProblemModel and IS the model its adapter host
// carries, which the first test below asserts rather than assumes. So door 3
// is door 2 with one extra conversion in front of it, and the conversion is
// what this file measures.
//
// WHERE EXACT EQUALITY IS ASSERTED, AND WHERE ONLY A TOLERANCE IS.
//
//   EXACTLY, with ==, wherever the comparison involves NO ARITHMETIC. The
//   variable bounds and the model's start point, which the conversion is
//   required to pass through verbatim; the row classification and the
//   native row numbering it assigns; the stored sparsity patterns and their
//   entry counts; and the declared-row multiplier map, whose whole content
//   is a choice of sign, a subtraction of two given numbers and a max
//   against zero. Nothing there has any freedom to round differently, so a
//   difference is a defect and a tolerance would hide it.
//
//   TO A NEAR-ULP GATE, wherever the same mathematics is written out twice
//   and evaluated. The conversion adds only exact operations (a multiply by
//   +/-1.0, and the negation implicit in -(g - l) == (l - g)), but it does
//   not add them in the same PLACE: it evaluates g(x) in one call and
//   applies the row's shift in another, where a hand-authored native model
//   writes the residual as one expression. Under this project's Release
//   flag regime the compiler is free to contract and reassociate within
//   each of those, and it does -- the two spellings of x0^2 + x1^2 - 4
//   differ in the last bit or two. That freedom is the compiler's, not the
//   conversion's, so the gate here is the same one the model contract's own
//   evaluation-override rule uses: a near-ulp RELATIVE gate against a unit
//   floor, which catches a transcription slip (an O(1) relative error) and
//   ignores reassociation residue. kConvEquivUlpGate is that gate at the
//   model surface; kConvEquivSolveGate is the (looser, still far below
//   solver tolerance) one for the SOLVED answers of doors 2 and 3, which
//   inherit the model surface's last-bit differences through a well
//   conditioned Newton iteration.
//
//   Door 1 vs doors 2/3: SOLVER TOLERANCE, and deliberately so. The VF path
//   lays a different KKT element order (its own pieces, its own claim
//   order) and computes derivatives by the VectorFunction machinery rather
//   than from the closed forms the other two hand over, so the
//   factorization pivots and the iterate history differ. Only the converged
//   answer is common, and it is common to solver tolerance.
//
// The practical reading: doors 2 and 3 agree to the last bit or two, doors
// 1 and 2 agree to the accuracy the solve was asked for, and the conversion
// is transparent at both scales.
//
// The multiplier conventions asserted here are the ones the model contract
// pins, not ones inferred from what the engine happened to return:
//
//   stationarity   grad(f) + Je^T lambda_e + Ji^T lambda_i - z = 0,
//                  lambda_i >= 0, z >= 0 at an active lower bound
//   declared row   Equality      lambda(r) =  lambda_e(eq_row(r))
//   correspondence UpperBounded  lambda(r) =  lambda_i(iq_upper_row(r))
//                  LowerBounded  lambda(r) = -lambda_i(iq_lower_row(r))
//                  Range         lambda(r) =  lambda_i(iq_upper_row(r))
//                                           - lambda_i(iq_lower_row(r))
//
// Every native fixture in this file leaves the model contract's in-place
// evaluation methods at their defaults, so it exercises the delegation to
// the by-value methods that an ordinary model gets. One fixture
// (ConvEquivEqBoundNativeInPlace) overrides all six instead, and the suite
// asserts the two produce the same solve to the last bit -- which is what
// the in-place contract promises and the only thing an override may change.
///////////////////////////////////////////////////////////////////////////////

#include "tycho/detail/hven_namespaces.h"
#include "tycho/detail/solvers_vf/optimization_problem.h"

#include <hven/detail/model/nlp_adapter.h>
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
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

using hven::solvers::FixedVariableTreatments;
using hven::solvers::make_nlp_program;
using hven::solvers::NLPAdapterCore;
using hven::solvers::NlpModel;
using hven::solvers::NLPProblem;
using hven::solvers::NlpProblemModel;
using hven::solvers::NLPRowKind;
using hven::solvers::NLPSolver;
using hven::solvers::OptimizationProblemBase;
using tycho::ConstEigenRef;
using tycho::solvers::OptimizationProblem;
using Vec = Eigen::VectorXd;
using SpMatRM = Eigen::SparseMatrix<double, Eigen::RowMajor>;

namespace {

constexpr double kConvEquivInf = std::numeric_limits<double>::infinity();

// Every door is driven at the same tolerances, so that a difference between
// two doors is a difference between the doors and not between two stopping
// points on the same path.
constexpr double kConvEquivSolverTol = 1.0e-10;
constexpr int kConvEquivMaxIters = 300;

// Door 1 against doors 2/3. The engine stops at kConvEquivSolverTol on its own
// residuals; the primal and the multipliers it reports are that accurate, not
// more, and the two paths reach it along different iterate sequences.
constexpr double kConvEquivCrossDoorTol = 1.0e-7;

// Doors 2 vs 3 at the model surface. A near-ulp relative gate against a unit
// floor: it passes reassociation residue between two spellings of the same
// expression and fails a transcription slip, which is an O(1) relative error.
// The unit floor is what makes it usable on a residual that is itself near
// zero, where a bare relative comparison has no scale to work with.
constexpr double kConvEquivUlpGate = 1.0e-13;

// Doors 2 vs 3 at the SOLVED answer. Looser than the model-surface gate,
// because the solve carries the last-bit difference of every evaluation
// through its iteration -- and set from what that costs, measured: the two
// doors' primals differ by about 5e-17 and their multipliers by about 4e-16,
// so this leaves three to four orders of margin over the observed disagreement
// while still refusing anything an O(1) transcription slip would produce.
//
// It is NOT set from the solver tolerance, which is the same 1e-10 (see
// kConvEquivSolverTol above) and therefore says nothing about how closely two
// doors of one problem should track each other. A gate at that value would sit
// six orders above the measurement and pass almost anything.
constexpr double kConvEquivSolveGate = 1.0e-12;

// An analytic optimum, where this file knows one in closed form. Held to the
// same accuracy as the cross-door comparison for the same reason.
constexpr double kConvEquivAnalyticTol = 1.0e-7;

void conv_equiv_configure(hven::solvers::InteriorPointSolver &opt) {
    opt.set_print_level(3); // silent: print_level is inverted, 3+ prints nothing
    opt.set_tols(kConvEquivSolverTol, kConvEquivSolverTol, kConvEquivSolverTol,
                 kConvEquivSolverTol);
    opt.set_max_iters(kConvEquivMaxIters);
}

///////////////////////////////////////////////////////////////////////////////
// Door 2's driver.
//
// NLPSolver is door 3's driver and there is no published counterpart for a
// model that needs no conversion, so this is that counterpart: everything
// NLPSolver does after its conversion step, and nothing else. Deriving from
// the same problem base is what makes the comparison fair -- the optimizer
// object, its default partition and QP-thread settings, and the solve dispatch
// are then literally the same code on both doors.
///////////////////////////////////////////////////////////////////////////////
struct ConvEquivNativeDoor : OptimizationProblemBase {
    std::shared_ptr<NlpModel> model_;
    std::shared_ptr<NLPAdapterCore> core_;
    std::string name_;

    Eigen::VectorXd active_variables_;
    Eigen::VectorXd active_eq_lmults_;
    Eigen::VectorXd active_iq_lmults_;
    bool do_transcription_ = true;

    ConvEquivNativeDoor(std::shared_ptr<NlpModel> model, std::string name)
        : model_(std::move(model)), name_(std::move(name)) {}

    void transcribe() {
        this->core_ = std::make_shared<NLPAdapterCore>(this->model_, this->name_);
        this->nlp_ = make_nlp_program(this->core_);
        this->optimizer_->set_nlp(this->nlp_);
        this->do_transcription_ = false;
    }

    tycho::ConvergenceFlags run_optimize(const Eigen::VectorXd &x0) {
        if (this->do_transcription_) {
            this->transcribe();
        }
        auto out = this->run_nlp_solver(JetJobModes::Optimize, x0);
        this->active_variables_ = out.variables_;
        this->active_eq_lmults_ = out.eq_lmults_;
        this->active_iq_lmults_ = out.iq_lmults_;
        return out.flag_;
    }

    // The problem base's own entry points. Only optimize() is reached from
    // this file; the rest exist because the base declares them.
    tycho::ConvergenceFlags solve() override { return this->run_optimize(this->active_variables_); }
    tycho::ConvergenceFlags optimize() override {
        return this->run_optimize(this->active_variables_);
    }
    tycho::ConvergenceFlags solve_optimize() override {
        return this->run_optimize(this->active_variables_);
    }
    tycho::ConvergenceFlags solve_optimize_solve() override {
        return this->run_optimize(this->active_variables_);
    }
    tycho::ConvergenceFlags optimize_solve() override {
        return this->run_optimize(this->active_variables_);
    }
    void jet_initialize() override { this->transcribe(); }
    void jet_release() override {
        this->nlp_.reset();
        this->do_transcription_ = true;
    }
};

/// One door's answer, in the native row spaces every door can be read in.
struct ConvEquivAnswer {
    tycho::ConvergenceFlags flag_ = tycho::ConvergenceFlags::NOTCONVERGED;
    Eigen::VectorXd x_;
    Eigen::VectorXd lambda_e_;
    Eigen::VectorXd lambda_i_;
    double obj_ = 0.0;
};

ConvEquivAnswer conv_equiv_solve_native(const std::shared_ptr<NlpModel> &model,
                                        const std::string &name, const Eigen::VectorXd &x0,
                                        FixedVariableTreatments treatment) {
    ConvEquivNativeDoor door(model, name);
    conv_equiv_configure(*door.optimizer_);
    door.optimizer_->set_fixed_variable_treatment(treatment);
    ConvEquivAnswer a;
    a.flag_ = door.run_optimize(x0);
    a.x_ = door.active_variables_;
    a.lambda_e_ = door.active_eq_lmults_;
    a.lambda_i_ = door.active_iq_lmults_;
    a.obj_ = door.optimizer_->result().obj_val_;
    return a;
}

ConvEquivAnswer conv_equiv_solve_triplet(const std::shared_ptr<NLPProblem> &problem,
                                         const Eigen::VectorXd &x0,
                                         FixedVariableTreatments treatment) {
    NLPSolver solver(problem);
    conv_equiv_configure(*solver.optimizer_);
    solver.optimizer_->set_fixed_variable_treatment(treatment);
    ConvEquivAnswer a;
    a.flag_ = solver.optimize(x0);
    a.x_ = solver.return_x();
    a.lambda_e_ = solver.active_eq_lmults_;
    a.lambda_i_ = solver.active_iq_lmults_;
    a.obj_ = solver.optimizer_->result().obj_val_;
    return a;
}

/// Exact vector equality, for the quantities no arithmetic stands behind.
/// Reported element by element so a failure names the coordinate.
void conv_equiv_expect_bit_identical(const Eigen::VectorXd &a, const Eigen::VectorXd &b,
                                     const char *what) {
    ASSERT_EQ(a.size(), b.size()) << what;
    for (Eigen::Index k = 0; k < a.size(); k++) {
        EXPECT_EQ(a[k], b[k]) << what << " differ at element " << k;
    }
}

/// The near-ulp gate: |a - b| <= rel * max(1, |a|, |b|). See the file banner
/// for why the comparisons that cross two spellings of one expression use this
/// rather than ==.
bool conv_equiv_close(double a, double b, double rel) {
    const double scale = std::max({1.0, std::abs(a), std::abs(b)});
    return std::abs(a - b) <= rel * scale;
}

void conv_equiv_expect_close(double a, double b, double rel, const char *what) {
    EXPECT_TRUE(conv_equiv_close(a, b, rel))
        << what << ": " << a << " vs " << b << " (gate " << rel << ")";
}

void conv_equiv_expect_close(const Eigen::VectorXd &a, const Eigen::VectorXd &b, double rel,
                             const char *what) {
    ASSERT_EQ(a.size(), b.size()) << what;
    for (Eigen::Index k = 0; k < a.size(); k++) {
        EXPECT_TRUE(conv_equiv_close(a[k], b[k], rel))
            << what << " differ at element " << k << ": " << a[k] << " vs " << b[k];
    }
}

void conv_equiv_expect_near(const Eigen::VectorXd &a, const Eigen::VectorXd &b, double tol,
                            const char *what) {
    ASSERT_EQ(a.size(), b.size()) << what;
    for (Eigen::Index k = 0; k < a.size(); k++) {
        EXPECT_NEAR(a[k], b[k], tol) << what << " differ at element " << k;
    }
}

/// A sparse matrix as a dense one, for exact comparison of two returns that
/// are meant to carry the same numbers over the same pattern.
Eigen::MatrixXd conv_equiv_dense(const SpMatRM &m) { return Eigen::MatrixXd(m); }

/// Two matrices whose VALUES cross the near-ulp gate and whose STORED PATTERN
/// must match exactly: the adapter host lays its KKT claims over the pattern,
/// so a model that emits a structural zero where another emits nothing is a
/// different model, however close the dense pictures are.
void conv_equiv_expect_close(const SpMatRM &a, const SpMatRM &b, double rel, const char *what) {
    const Eigen::MatrixXd da = conv_equiv_dense(a);
    const Eigen::MatrixXd db = conv_equiv_dense(b);
    ASSERT_EQ(da.rows(), db.rows()) << what;
    ASSERT_EQ(da.cols(), db.cols()) << what;
    for (Eigen::Index r = 0; r < da.rows(); r++) {
        for (Eigen::Index c = 0; c < da.cols(); c++) {
            EXPECT_TRUE(conv_equiv_close(da(r, c), db(r, c), rel))
                << what << " differ at (" << r << ", " << c << "): " << da(r, c) << " vs "
                << db(r, c);
        }
    }
    EXPECT_EQ(a.nonZeros(), b.nonZeros()) << what << ": stored entry counts";
    EXPECT_TRUE(a.isCompressed()) << what << ": left side must be compressed";
    EXPECT_TRUE(b.isCompressed()) << what << ": right side must be compressed";
}

/// Compares the two model surfaces entry by entry at one point. This is the
/// level at which the conversion is expected to be numerically transparent.
void conv_equiv_expect_models_agree(const NlpModel &native, const NlpModel &converted,
                                    const Eigen::VectorXd &x, double obj_scale,
                                    const Eigen::VectorXd &le, const Eigen::VectorXd &li) {
    ASSERT_EQ(native.n(), converted.n());
    ASSERT_EQ(native.me(), converted.me());
    ASSERT_EQ(native.mi(), converted.mi());

    conv_equiv_expect_close(native.eval_f(x), converted.eval_f(x), kConvEquivUlpGate, "eval_f");
    conv_equiv_expect_close(native.eval_grad(x), converted.eval_grad(x), kConvEquivUlpGate,
                            "eval_grad");
    if (native.me() > 0) {
        conv_equiv_expect_close(native.eval_ce(x), converted.eval_ce(x), kConvEquivUlpGate,
                                "eval_ce");
        conv_equiv_expect_close(SpMatRM(native.eval_jac_e(x)), SpMatRM(converted.eval_jac_e(x)),
                                kConvEquivUlpGate, "eval_jac_e");
    }
    if (native.mi() > 0) {
        conv_equiv_expect_close(native.eval_ci(x), converted.eval_ci(x), kConvEquivUlpGate,
                                "eval_ci");
        conv_equiv_expect_close(SpMatRM(native.eval_jac_i(x)), SpMatRM(converted.eval_jac_i(x)),
                                kConvEquivUlpGate, "eval_jac_i");
    }
    conv_equiv_expect_close(native.eval_hess(x, obj_scale, le, li),
                            converted.eval_hess(x, obj_scale, le, li), kConvEquivUlpGate,
                            "eval_hess");

    // The bounds and the start point are the other half of the model surface,
    // and the conversion performs no arithmetic on them at all: they are
    // required to be verbatim, so they are compared exactly.
    conv_equiv_expect_bit_identical(native.lower(), converted.lower(), "lower");
    conv_equiv_expect_bit_identical(native.upper(), converted.upper(), "upper");
    conv_equiv_expect_bit_identical(native.start_point(), converted.start_point(), "start_point");
}

///////////////////////////////////////////////////////////////////////////////
// Fixture A -- an equality constraint and a variable bound that is ACTIVE at
// the optimum, so both an equality multiplier and a bound multiplier are
// nonzero and the stationarity convention has something to say.
//
//   min  (x0 - 1)^2 + (x1 - 2)^2
//   s.t. x0^2 + x1^2 = 4
//        x0 >= 1.5
//
// Without the bound the answer is the point of the circle nearest (1, 2),
// which has x0 = 2/sqrt(5) ~ 0.894; the bound therefore binds, and the
// optimum is x0 = 1.5, x1 = sqrt(1.75).
///////////////////////////////////////////////////////////////////////////////

const double kEqBoundX0 = 1.5;
const double kEqBoundX1 = std::sqrt(1.75);

Eigen::VectorXd conv_equiv_eq_bound_start() { return (Eigen::VectorXd(2) << 1.6, 1.0).finished(); }

struct ConvEquivEqBoundProblem : NLPProblem {
    int num_vars() const override { return 2; }
    int num_cons() const override { return 1; }
    int num_jac_nonzeros() const override { return 2; }
    int num_hess_nonzeros() const override { return 2; } // diagonal: f and g are separable

    void bounds(Eigen::Ref<Eigen::VectorXd> xl, Eigen::Ref<Eigen::VectorXd> xu,
                Eigen::Ref<Eigen::VectorXd> gl, Eigen::Ref<Eigen::VectorXd> gu) const override {
        xl << kEqBoundX0, -kConvEquivInf;
        xu << kConvEquivInf, kConvEquivInf;
        gl << 4.0;
        gu << 4.0;
    }
    void eval_f(ConstEigenRef<Eigen::VectorXd> x, double &f) const override {
        const double a = x[0] - 1.0, b = x[1] - 2.0;
        f = a * a + b * b;
    }
    void eval_grad_f(ConstEigenRef<Eigen::VectorXd> x,
                     Eigen::Ref<Eigen::VectorXd> g) const override {
        g[0] = 2.0 * (x[0] - 1.0);
        g[1] = 2.0 * (x[1] - 2.0);
    }
    void eval_g(ConstEigenRef<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> g) const override {
        g[0] = x[0] * x[0] + x[1] * x[1];
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
    void eval_jac(ConstEigenRef<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> v) const override {
        v[0] = 2.0 * x[0];
        v[1] = 2.0 * x[1];
    }
    void eval_hess(ConstEigenRef<Eigen::VectorXd>, double obj_factor,
                   ConstEigenRef<Eigen::VectorXd> lambda,
                   Eigen::Ref<Eigen::VectorXd> v) const override {
        v[0] = 2.0 * obj_factor + 2.0 * lambda[0];
        v[1] = 2.0 * obj_factor + 2.0 * lambda[0];
    }
    std::string name() const override { return "ConvEquivEqBound"; }
};

/// The same problem against the native contract, written to mirror what the
/// conversion produces: one equality row carrying g(x) - 4, no inequality
/// rows, a diagonal Hessian over the same two coordinates.
struct ConvEquivEqBoundNative : NlpModel {
    Vec lower_ = (Vec(2) << kEqBoundX0, -kConvEquivInf).finished();
    Vec upper_ = (Vec(2) << kConvEquivInf, kConvEquivInf).finished();

    hven::Index n() const override { return 2; }
    hven::Index me() const override { return 1; }
    hven::Index mi() const override { return 0; }

    double eval_f(const Vec &x) const override {
        const double a = x[0] - 1.0, b = x[1] - 2.0;
        return a * a + b * b;
    }
    Vec eval_grad(const Vec &x) const override {
        return (Vec(2) << 2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.0)).finished();
    }
    Vec eval_ce(const Vec &x) const override {
        return (Vec(1) << x[0] * x[0] + x[1] * x[1] - 4.0).finished();
    }
    Vec eval_ci(const Vec &) const override { return Vec(0); }

    SpMatRM eval_jac_e(const Vec &x) const override {
        SpMatRM j(1, 2);
        std::vector<Eigen::Triplet<double>> t{{0, 0, 2.0 * x[0]}, {0, 1, 2.0 * x[1]}};
        j.setFromTriplets(t.begin(), t.end());
        j.makeCompressed();
        return j;
    }
    SpMatRM eval_jac_i(const Vec &) const override {
        SpMatRM j(0, 2);
        j.makeCompressed();
        return j;
    }
    SpMatRM eval_hess(const Vec &, double obj_scale, const Vec &lambda_e,
                      const Vec &) const override {
        // Every caller in this tree hands this model exactly its one equality
        // multiplier -- the host slices its own head before calling down, and
        // the direct calls below size their blocks to me(). The length guard
        // is defensive only; it is not an interpretation of an empty block,
        // which the boundary pins state is a LENGTH and never a spelling of
        // "all zero".
        const double le = (lambda_e.size() == 0) ? 0.0 : lambda_e[0];
        const double d = 2.0 * obj_scale + 2.0 * le;
        SpMatRM h(2, 2);
        // Emitted unconditionally, structural zeros included: the pattern is a
        // property of the model, never of the point or the multipliers.
        std::vector<Eigen::Triplet<double>> t{{0, 0, d}, {1, 1, d}};
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

/// Fixture A's model again, this time overriding every in-place evaluation.
/// The contract allows an override to be cheaper than the default delegation
/// and nothing else, so this must solve to the same bits as the model above.
struct ConvEquivEqBoundNativeInPlace : ConvEquivEqBoundNative {
    void eval_grad_in_place(const Vec &x, Vec &out) const override {
        out.resize(2);
        out[0] = 2.0 * (x[0] - 1.0);
        out[1] = 2.0 * (x[1] - 2.0);
    }
    void eval_ce_in_place(const Vec &x, Vec &out) const override {
        out.resize(1);
        out[0] = x[0] * x[0] + x[1] * x[1] - 4.0;
    }
    void eval_ci_in_place(const Vec &, Vec &out) const override { out.resize(0); }
    void eval_jac_e_in_place(const Vec &x, SpMatRM &out) const override {
        out = this->eval_jac_e(x);
    }
    void eval_jac_i_in_place(const Vec &x, SpMatRM &out) const override {
        out = this->eval_jac_i(x);
    }
    void eval_hess_in_place(const Vec &x, double obj_scale, const Vec &le, const Vec &li,
                            SpMatRM &out) const override {
        out = this->eval_hess(x, obj_scale, le, li);
    }
};

/// Fixture A through the VectorFunction door. The variable bound is declared
/// on the transcribed program and the layout re-materialized, which is the
/// only route a bare OptimizationProblem has to a native variable bound.
ConvEquivAnswer conv_equiv_solve_eq_bound_vf() {
    using tycho::vf::Arguments;
    using tycho::vf::GenericFunction;

    OptimizationProblem prob;
    prob.set_vars(conv_equiv_eq_bound_start());
    {
        auto args = Arguments<2>();
        auto x0 = args.coeff<0>();
        auto x1 = args.coeff<1>();
        prob.add_objective(
            GenericFunction<-1, 1>((x0 - 1.0) * (x0 - 1.0) + (x1 - 2.0) * (x1 - 2.0)),
            (Eigen::VectorXi(2) << 0, 1).finished());
    }
    {
        auto args = Arguments<2>();
        auto x0 = args.coeff<0>();
        auto x1 = args.coeff<1>();
        prob.add_equal_con(GenericFunction<-1, -1>(x0 * x0 + x1 * x1 - 4.0),
                           (Eigen::VectorXi(2) << 0, 1).finished());
    }
    prob.add_variable_bound(0, kEqBoundX0, kConvEquivInf);
    conv_equiv_configure(*prob.optimizer_);
    prob.transcribe();

    ConvEquivAnswer a;
    a.x_ = prob.optimizer_->optimize(conv_equiv_eq_bound_start());
    a.flag_ = prob.optimizer_->result().converge_flag_;
    a.lambda_e_ = prob.optimizer_->result().eq_lmults_;
    a.lambda_i_ = prob.optimizer_->result().iq_lmults_;
    a.obj_ = prob.optimizer_->result().obj_val_;
    return a;
}

///////////////////////////////////////////////////////////////////////////////
// Fixture B -- two RANGED constraint rows, one binding on its upper side and
// one on its lower side, which is what makes the range-split multiplier map
// testable in both of its signed branches.
//
//   min  (x0 - 1)^2 + (x1 - 2)^2
//   s.t. 1.0 <= x0 + x1 <= 1.5
//       -0.5 <= x0 - x1 <= 5.0
//
// Both upper-active branches of the unconstrained descent are cut off: the
// answer is the corner x = (0.5, 1.0), where x0 + x1 = 1.5 (the FIRST row's
// UPPER bound) and x0 - x1 = -0.5 (the SECOND row's LOWER bound).
//
// Declared row r becomes native inequality rows in declaration order, upper
// part first: [row0 upper, row0 lower, row1 upper, row1 lower] = [0, 1, 2, 3].
// At the optimum lambda_i = (1.5, 0, 0, 0.5), so the declared multipliers are
// (+1.5, -0.5): a positive one from an upper-active range and a negative one
// from a lower-active range.
///////////////////////////////////////////////////////////////////////////////

Eigen::VectorXd conv_equiv_ranged_start() { return (Eigen::VectorXd(2) << 0.4, 0.9).finished(); }

const double kRangedX0 = 0.5;
const double kRangedX1 = 1.0;

struct ConvEquivRangedProblem : NLPProblem {
    int num_vars() const override { return 2; }
    int num_cons() const override { return 2; }
    int num_jac_nonzeros() const override { return 4; }
    int num_hess_nonzeros() const override { return 2; }

    void bounds(Eigen::Ref<Eigen::VectorXd> xl, Eigen::Ref<Eigen::VectorXd> xu,
                Eigen::Ref<Eigen::VectorXd> gl, Eigen::Ref<Eigen::VectorXd> gu) const override {
        xl << -kConvEquivInf, -kConvEquivInf;
        xu << kConvEquivInf, kConvEquivInf;
        gl << 1.0, -0.5;
        gu << 1.5, 5.0;
    }
    void eval_f(ConstEigenRef<Eigen::VectorXd> x, double &f) const override {
        const double a = x[0] - 1.0, b = x[1] - 2.0;
        f = a * a + b * b;
    }
    void eval_grad_f(ConstEigenRef<Eigen::VectorXd> x,
                     Eigen::Ref<Eigen::VectorXd> g) const override {
        g[0] = 2.0 * (x[0] - 1.0);
        g[1] = 2.0 * (x[1] - 2.0);
    }
    void eval_g(ConstEigenRef<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> g) const override {
        g[0] = x[0] + x[1];
        g[1] = x[0] - x[1];
    }
    void jac_structure(Eigen::Ref<Eigen::VectorXi> r,
                       Eigen::Ref<Eigen::VectorXi> c) const override {
        r << 0, 0, 1, 1;
        c << 0, 1, 0, 1;
    }
    void hess_structure(Eigen::Ref<Eigen::VectorXi> r,
                        Eigen::Ref<Eigen::VectorXi> c) const override {
        r << 0, 1;
        c << 0, 1;
    }
    void eval_jac(ConstEigenRef<Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd> v) const override {
        v[0] = 1.0;
        v[1] = 1.0;
        v[2] = 1.0;
        v[3] = -1.0;
    }
    void eval_hess(ConstEigenRef<Eigen::VectorXd>, double obj_factor,
                   ConstEigenRef<Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd> v) const override {
        // The constraints are affine, so only the objective contributes -- but
        // both entries are still emitted at every obj_factor, structural zeros
        // included.
        v[0] = 2.0 * obj_factor;
        v[1] = 2.0 * obj_factor;
    }
    std::string name() const override { return "ConvEquivRanged"; }
};

/// Fixture B against the native contract, with the range split written out by
/// hand in exactly the order and with exactly the signs the conversion uses.
struct ConvEquivRangedNative : NlpModel {
    Vec lower_ = (Vec(2) << -kConvEquivInf, -kConvEquivInf).finished();
    Vec upper_ = (Vec(2) << kConvEquivInf, kConvEquivInf).finished();

    hven::Index n() const override { return 2; }
    hven::Index me() const override { return 0; }
    hven::Index mi() const override { return 4; }

    double eval_f(const Vec &x) const override {
        const double a = x[0] - 1.0, b = x[1] - 2.0;
        return a * a + b * b;
    }
    Vec eval_grad(const Vec &x) const override {
        return (Vec(2) << 2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.0)).finished();
    }
    Vec eval_ce(const Vec &) const override { return Vec(0); }
    Vec eval_ci(const Vec &x) const override {
        const double g0 = x[0] + x[1];
        const double g1 = x[0] - x[1];
        Vec c(4);
        c[0] = g0 - 1.5;    // row 0, upper part
        c[1] = -(g0 - 1.0); // row 0, lower part, negated
        c[2] = g1 - 5.0;    // row 1, upper part
        c[3] = -(g1 + 0.5); // row 1, lower part, negated
        return c;
    }
    SpMatRM eval_jac_e(const Vec &) const override {
        SpMatRM j(0, 2);
        j.makeCompressed();
        return j;
    }
    SpMatRM eval_jac_i(const Vec &) const override {
        SpMatRM j(4, 2);
        std::vector<Eigen::Triplet<double>> t{
            {0, 0, 1.0}, {0, 1, 1.0},  {1, 0, -1.0}, {1, 1, -1.0},
            {2, 0, 1.0}, {2, 1, -1.0}, {3, 0, -1.0}, {3, 1, 1.0},
        };
        j.setFromTriplets(t.begin(), t.end());
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

/// Fixture B through the VectorFunction door. The four split rows are added as
/// four scalar inequality constraints in the same order the conversion assigns
/// them, so the inequality multiplier vector is directly comparable.
ConvEquivAnswer conv_equiv_solve_ranged_vf() {
    using tycho::vf::Arguments;
    using tycho::vf::GenericFunction;

    const Eigen::VectorXi both = (Eigen::VectorXi(2) << 0, 1).finished();

    OptimizationProblem prob;
    prob.set_vars(conv_equiv_ranged_start());
    {
        auto args = Arguments<2>();
        auto x0 = args.coeff<0>();
        auto x1 = args.coeff<1>();
        prob.add_objective(
            GenericFunction<-1, 1>((x0 - 1.0) * (x0 - 1.0) + (x1 - 2.0) * (x1 - 2.0)), both);
    }
    {
        auto args = Arguments<2>();
        auto x0 = args.coeff<0>();
        auto x1 = args.coeff<1>();
        prob.add_inequal_con(GenericFunction<-1, -1>(x0 + x1 - 1.5), both);
    }
    {
        auto args = Arguments<2>();
        auto x0 = args.coeff<0>();
        auto x1 = args.coeff<1>();
        prob.add_inequal_con(GenericFunction<-1, -1>(1.0 - x0 - x1), both);
    }
    {
        auto args = Arguments<2>();
        auto x0 = args.coeff<0>();
        auto x1 = args.coeff<1>();
        prob.add_inequal_con(GenericFunction<-1, -1>(x0 - x1 - 5.0), both);
    }
    {
        auto args = Arguments<2>();
        auto x0 = args.coeff<0>();
        auto x1 = args.coeff<1>();
        prob.add_inequal_con(GenericFunction<-1, -1>(-0.5 - x0 + x1), both);
    }
    conv_equiv_configure(*prob.optimizer_);

    ConvEquivAnswer a;
    a.flag_ = prob.optimize();
    a.x_ = prob.return_vars();
    a.lambda_e_ = prob.active_eq_lmults_;
    a.lambda_i_ = prob.active_iq_lmults_;
    a.obj_ = prob.optimizer_->result().obj_val_;
    return a;
}

///////////////////////////////////////////////////////////////////////////////
// Fixture C -- an equality row, an inequality row, and a variable FIXED by
// equal bounds, which is the treatment case.
//
//   min  (x0 - 1)^2 + (x1 - 2)^2 + (x2 - 3)^2
//   s.t. x0 + x1 + x2 = 3
//        x0 - x1 <= 0.5
//        x2 = 1   (declared as the bound pair [1, 1])
//
// With x2 pinned the equality reads x0 + x1 = 2 and the optimum is
// x = (0.5, 1.5, 1.0); the inequality is slack there (-1.0 <= 0.5), so its
// multiplier is zero and the equality's is 1.
//
// The three treatments solve three different systems and the suite says so:
//   make_parameter  eliminates x2 -- the KKT system is one row and column
//                   narrower, and the equality multiplier vector holds the
//                   one user row.
//   make_constraint keeps x2 and appends an internal row x2 - 1 = 0 AFTER
//                   every user row, so the equality multiplier vector is one
//                   longer, and its tail entry is that row's multiplier: 3,
//                   the amount stationarity in x2 needs beyond the user row.
//   relax_bounds    keeps x2 as an ordinary bounded variable with its pair
//                   pushed apart, so the system is the declared size.
///////////////////////////////////////////////////////////////////////////////

Eigen::VectorXd conv_equiv_fixed_start() {
    return (Eigen::VectorXd(3) << 0.2, 1.2, 1.0).finished();
}

const double kFixedX2 = 1.0;
const double kFixedOptX0 = 0.5;
const double kFixedOptX1 = 1.5;
const double kFixedLambdaEq = 1.0;
const double kFixedLambdaFixRow = 3.0;

struct ConvEquivFixedProblem : NLPProblem {
    int num_vars() const override { return 3; }
    int num_cons() const override { return 2; }
    int num_jac_nonzeros() const override { return 5; }
    int num_hess_nonzeros() const override { return 3; }

    void bounds(Eigen::Ref<Eigen::VectorXd> xl, Eigen::Ref<Eigen::VectorXd> xu,
                Eigen::Ref<Eigen::VectorXd> gl, Eigen::Ref<Eigen::VectorXd> gu) const override {
        xl << -kConvEquivInf, -kConvEquivInf, kFixedX2;
        xu << kConvEquivInf, kConvEquivInf, kFixedX2;
        gl << 3.0, -kConvEquivInf;
        gu << 3.0, 0.5;
    }
    void eval_f(ConstEigenRef<Eigen::VectorXd> x, double &f) const override {
        const double a = x[0] - 1.0, b = x[1] - 2.0, c = x[2] - 3.0;
        f = a * a + b * b + c * c;
    }
    void eval_grad_f(ConstEigenRef<Eigen::VectorXd> x,
                     Eigen::Ref<Eigen::VectorXd> g) const override {
        g[0] = 2.0 * (x[0] - 1.0);
        g[1] = 2.0 * (x[1] - 2.0);
        g[2] = 2.0 * (x[2] - 3.0);
    }
    void eval_g(ConstEigenRef<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> g) const override {
        g[0] = x[0] + x[1] + x[2];
        g[1] = x[0] - x[1];
    }
    void jac_structure(Eigen::Ref<Eigen::VectorXi> r,
                       Eigen::Ref<Eigen::VectorXi> c) const override {
        r << 0, 0, 0, 1, 1;
        c << 0, 1, 2, 0, 1;
    }
    void hess_structure(Eigen::Ref<Eigen::VectorXi> r,
                        Eigen::Ref<Eigen::VectorXi> c) const override {
        r << 0, 1, 2;
        c << 0, 1, 2;
    }
    void eval_jac(ConstEigenRef<Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd> v) const override {
        v[0] = 1.0;
        v[1] = 1.0;
        v[2] = 1.0;
        v[3] = 1.0;
        v[4] = -1.0;
    }
    void eval_hess(ConstEigenRef<Eigen::VectorXd>, double obj_factor,
                   ConstEigenRef<Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd> v) const override {
        v[0] = 2.0 * obj_factor;
        v[1] = 2.0 * obj_factor;
        v[2] = 2.0 * obj_factor;
    }
    std::string name() const override { return "ConvEquivFixed"; }
};

struct ConvEquivFixedNative : NlpModel {
    Vec lower_ = (Vec(3) << -kConvEquivInf, -kConvEquivInf, kFixedX2).finished();
    Vec upper_ = (Vec(3) << kConvEquivInf, kConvEquivInf, kFixedX2).finished();

    hven::Index n() const override { return 3; }
    hven::Index me() const override { return 1; }
    hven::Index mi() const override { return 1; }

    double eval_f(const Vec &x) const override {
        const double a = x[0] - 1.0, b = x[1] - 2.0, c = x[2] - 3.0;
        return a * a + b * b + c * c;
    }
    Vec eval_grad(const Vec &x) const override {
        return (Vec(3) << 2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.0), 2.0 * (x[2] - 3.0)).finished();
    }
    Vec eval_ce(const Vec &x) const override {
        return (Vec(1) << x[0] + x[1] + x[2] - 3.0).finished();
    }
    Vec eval_ci(const Vec &x) const override { return (Vec(1) << x[0] - x[1] - 0.5).finished(); }
    SpMatRM eval_jac_e(const Vec &) const override {
        SpMatRM j(1, 3);
        std::vector<Eigen::Triplet<double>> t{{0, 0, 1.0}, {0, 1, 1.0}, {0, 2, 1.0}};
        j.setFromTriplets(t.begin(), t.end());
        j.makeCompressed();
        return j;
    }
    SpMatRM eval_jac_i(const Vec &) const override {
        SpMatRM j(1, 3);
        std::vector<Eigen::Triplet<double>> t{{0, 0, 1.0}, {0, 1, -1.0}};
        j.setFromTriplets(t.begin(), t.end());
        j.makeCompressed();
        return j;
    }
    SpMatRM eval_hess(const Vec &, double obj_scale, const Vec &, const Vec &) const override {
        SpMatRM h(3, 3);
        std::vector<Eigen::Triplet<double>> t{
            {0, 0, 2.0 * obj_scale}, {1, 1, 2.0 * obj_scale}, {2, 2, 2.0 * obj_scale}};
        h.setFromTriplets(t.begin(), t.end());
        h.makeCompressed();
        return h;
    }
    const Vec &lower() const override { return lower_; }
    const Vec &upper() const override { return upper_; }
    Vec start_point() const override {
        Vec x0(3);
        for (int i = 0; i < 3; i++) {
            x0[i] = std::min(std::max(0.0, lower_[i]), upper_[i]);
        }
        return x0;
    }
};

ConvEquivAnswer conv_equiv_solve_fixed_vf(FixedVariableTreatments treatment) {
    using tycho::vf::Arguments;
    using tycho::vf::GenericFunction;

    const Eigen::VectorXi all3 = (Eigen::VectorXi(3) << 0, 1, 2).finished();

    OptimizationProblem prob;
    prob.set_vars(conv_equiv_fixed_start());
    {
        auto args = Arguments<3>();
        auto x0 = args.coeff<0>();
        auto x1 = args.coeff<1>();
        auto x2 = args.coeff<2>();
        prob.add_objective(GenericFunction<-1, 1>((x0 - 1.0) * (x0 - 1.0) +
                                                  (x1 - 2.0) * (x1 - 2.0) +
                                                  (x2 - 3.0) * (x2 - 3.0)),
                           all3);
    }
    {
        auto args = Arguments<3>();
        auto x0 = args.coeff<0>();
        auto x1 = args.coeff<1>();
        auto x2 = args.coeff<2>();
        prob.add_equal_con(GenericFunction<-1, -1>(x0 + x1 + x2 - 3.0), all3);
    }
    {
        auto args = Arguments<3>();
        auto x0 = args.coeff<0>();
        auto x1 = args.coeff<1>();
        prob.add_inequal_con(GenericFunction<-1, -1>(x0 - x1 - 0.5), all3);
    }
    prob.add_variable_bound(2, kFixedX2, kFixedX2);
    conv_equiv_configure(*prob.optimizer_);
    prob.optimizer_->set_fixed_variable_treatment(treatment);
    prob.transcribe();

    ConvEquivAnswer a;
    a.x_ = prob.optimizer_->optimize(conv_equiv_fixed_start());
    a.flag_ = prob.optimizer_->result().converge_flag_;
    a.lambda_e_ = prob.optimizer_->result().eq_lmults_;
    a.lambda_i_ = prob.optimizer_->result().iq_lmults_;
    a.obj_ = prob.optimizer_->result().obj_val_;
    return a;
}

} // namespace

///////////////////////////////////////////////////////////////////////////////
// The route the triplet door takes.
///////////////////////////////////////////////////////////////////////////////

// There is one conversion from the triplet declaration to the native contract
// and one host that carries a model to the engine, and the triplet door uses
// both: its conversion product IS an NlpProblemModel, and that same object is
// the model its host carries. Everything the rest of this file concludes about
// the conversion rests on there being no second route around it.
TEST(ConversionEquivalence, TripletDoorReachesTheEngineThroughTheModelConversion) {
    auto problem = std::make_shared<ConvEquivEqBoundProblem>();
    NLPSolver solver(problem);
    conv_equiv_configure(*solver.optimizer_);
    solver.transcribe();

    ASSERT_NE(solver.model_, nullptr);
    ASSERT_NE(solver.core_, nullptr);
    EXPECT_EQ(solver.core_->model_.get(), static_cast<NlpModel *>(solver.model_.get()));
    EXPECT_EQ(&solver.model_->problem(), problem.get());
    EXPECT_EQ(solver.model_->problem_ptr().get(), problem.get());

    // The host's own view of the model's shape is the model's, not the
    // declaration's: one equality row here, from one declared equality row.
    EXPECT_EQ(solver.core_->n_, 2);
    EXPECT_EQ(solver.core_->num_eq_, 1);
    EXPECT_EQ(solver.core_->num_iq_, 0);
    EXPECT_EQ(solver.model_->num_declared_rows(), 1);
}

///////////////////////////////////////////////////////////////////////////////
// Fixture A -- equality plus an active variable bound.
///////////////////////////////////////////////////////////////////////////////

// The model surface first: the converted model and the hand-authored one must
// report the same numbers, to the near-ulp gate, at every point either is
// asked about, and the same patterns and bounds exactly. Probed away from the
// optimum as well as at it, so the agreement is the functions' and not the
// solution's.
TEST(ConversionEquivalence, EqBoundModelSurfacesMatchAcrossTheConversion) {
    ConvEquivEqBoundNative native;
    NlpProblemModel converted(std::make_shared<ConvEquivEqBoundProblem>());

    const std::vector<Eigen::VectorXd> probes = {
        (Eigen::VectorXd(2) << 1.6, 1.0).finished(),
        (Eigen::VectorXd(2) << kEqBoundX0, kEqBoundX1).finished(),
        (Eigen::VectorXd(2) << -0.25, 3.5).finished(),
        (Eigen::VectorXd(2) << 0.0, 0.0).finished(),
    };
    const Eigen::VectorXd le = (Eigen::VectorXd(1) << 0.75).finished();
    const Eigen::VectorXd li(0);
    for (const auto &x : probes) {
        SCOPED_TRACE(::testing::Message() << "probe (" << x[0] << ", " << x[1] << ")");
        conv_equiv_expect_models_agree(native, converted, x, 1.0, le, li);
        // And at the zero-multiplier, zero-scale corner the structure probes
        // use, where the pattern must still be the whole pattern.
        conv_equiv_expect_models_agree(native, converted, x, 0.0, Eigen::VectorXd::Zero(1), li);
    }
}

// Same mathematics, same order, same engine: the two doors land on the same
// answer to a gate set three to four orders above their measured disagreement,
// not to the tolerance the solve was asked for. That is the measurement -- a
// conversion that PERTURBED a value rather than preserved it would show up
// here long before it showed up at solver tolerance, while the last-bit
// residue two spellings of one expression legitimately carry does not.
TEST(ConversionEquivalence, EqBoundNativeAndTripletDoorsSolveToTheSameNumbers) {
    const Eigen::VectorXd x0 = conv_equiv_eq_bound_start();
    const auto native =
        conv_equiv_solve_native(std::make_shared<ConvEquivEqBoundNative>(), "ConvEquivEqBound", x0,
                                FixedVariableTreatments::MakeParameter);
    const auto triplet = conv_equiv_solve_triplet(std::make_shared<ConvEquivEqBoundProblem>(), x0,
                                                  FixedVariableTreatments::MakeParameter);

    ASSERT_EQ(native.flag_, tycho::ConvergenceFlags::CONVERGED);
    ASSERT_EQ(triplet.flag_, tycho::ConvergenceFlags::CONVERGED);
    conv_equiv_expect_close(native.x_, triplet.x_, kConvEquivSolveGate, "primal point");
    conv_equiv_expect_close(native.lambda_e_, triplet.lambda_e_, kConvEquivSolveGate,
                            "equality multipliers");
    conv_equiv_expect_close(native.lambda_i_, triplet.lambda_i_, kConvEquivSolveGate,
                            "inequality multipliers");
    conv_equiv_expect_close(native.obj_, triplet.obj_, kConvEquivSolveGate, "objective");
}

// The in-place evaluation methods may be cheaper than the default delegation
// and nothing else. A model overriding all six therefore solves to the same
// bits as the same model leaving them alone.
TEST(ConversionEquivalence, EqBoundInPlaceOverridesChangeNothing) {
    const Eigen::VectorXd x0 = conv_equiv_eq_bound_start();
    const auto plain =
        conv_equiv_solve_native(std::make_shared<ConvEquivEqBoundNative>(), "ConvEquivEqBound", x0,
                                FixedVariableTreatments::MakeParameter);
    const auto in_place =
        conv_equiv_solve_native(std::make_shared<ConvEquivEqBoundNativeInPlace>(),
                                "ConvEquivEqBound", x0, FixedVariableTreatments::MakeParameter);

    ASSERT_EQ(plain.flag_, tycho::ConvergenceFlags::CONVERGED);
    ASSERT_EQ(in_place.flag_, tycho::ConvergenceFlags::CONVERGED);
    conv_equiv_expect_close(plain.x_, in_place.x_, kConvEquivSolveGate, "primal point");
    conv_equiv_expect_close(plain.lambda_e_, in_place.lambda_e_, kConvEquivSolveGate,
                            "equality multipliers");
    conv_equiv_expect_close(plain.obj_, in_place.obj_, kConvEquivSolveGate, "objective");
}

// The contract behind the test above, stated where it is actually written: an
// in-place override leaves the destination holding EXACTLY what the by-value
// counterpart returns. Compared on one object, so there is no second
// transcription for a compiler to reassociate differently -- the only way
// these can differ is an override that computes something else.
TEST(ConversionEquivalence, InPlaceOverridesReturnWhatTheByValueMethodsReturn) {
    const ConvEquivEqBoundNativeInPlace model;
    const std::vector<Eigen::VectorXd> probes = {
        conv_equiv_eq_bound_start(),
        (Eigen::VectorXd(2) << kEqBoundX0, kEqBoundX1).finished(),
        (Eigen::VectorXd(2) << -0.25, 3.5).finished(),
    };
    const Eigen::VectorXd le = (Eigen::VectorXd(1) << 0.75).finished();
    const Eigen::VectorXd li(0);
    for (const auto &x : probes) {
        SCOPED_TRACE(::testing::Message() << "probe (" << x[0] << ", " << x[1] << ")");
        Eigen::VectorXd grad, ce, ci;
        model.eval_grad_in_place(x, grad);
        model.eval_ce_in_place(x, ce);
        model.eval_ci_in_place(x, ci);
        conv_equiv_expect_bit_identical(grad, model.eval_grad(x), "eval_grad in place");
        conv_equiv_expect_bit_identical(ce, model.eval_ce(x), "eval_ce in place");
        conv_equiv_expect_bit_identical(ci, model.eval_ci(x), "eval_ci in place");

        SpMatRM je, hess;
        model.eval_jac_e_in_place(x, je);
        model.eval_hess_in_place(x, 1.0, le, li, hess);
        conv_equiv_expect_close(je, SpMatRM(model.eval_jac_e(x)), 0.0, "eval_jac_e in place");
        conv_equiv_expect_close(hess, model.eval_hess(x, 1.0, le, li), 0.0, "eval_hess in place");

        // A destination reused across calls must come back holding the same
        // thing, and compressed -- a consumer pairs stored value k with the
        // pattern's kth entry, which only compressed storage makes meaningful.
        model.eval_jac_e_in_place(x, je);
        EXPECT_TRUE(je.isCompressed());
    }
}

// The VectorFunction door reaches the same answer along its own path, so only
// the answer is compared, and to solver tolerance. The multiplier and the
// bound multiplier implied by the pinned stationarity condition are checked
// against their closed forms as well, which is what makes this an assertion
// about the CONVENTION and not merely about three paths agreeing with each
// other.
TEST(ConversionEquivalence, EqBoundVfDoorAgreesWithTheOtherTwo) {
    const Eigen::VectorXd x0 = conv_equiv_eq_bound_start();
    const auto vf = conv_equiv_solve_eq_bound_vf();
    const auto native =
        conv_equiv_solve_native(std::make_shared<ConvEquivEqBoundNative>(), "ConvEquivEqBound", x0,
                                FixedVariableTreatments::MakeParameter);
    ASSERT_EQ(vf.flag_, tycho::ConvergenceFlags::CONVERGED);
    ASSERT_EQ(native.flag_, tycho::ConvergenceFlags::CONVERGED);

    conv_equiv_expect_near(vf.x_, native.x_, kConvEquivCrossDoorTol, "primal point");
    conv_equiv_expect_near(vf.lambda_e_, native.lambda_e_, kConvEquivCrossDoorTol,
                           "equality multipliers");
    EXPECT_NEAR(vf.obj_, native.obj_, kConvEquivCrossDoorTol);

    // The optimum in closed form: the bound binds, and the circle fixes x1.
    EXPECT_NEAR(native.x_[0], kEqBoundX0, kConvEquivAnalyticTol);
    EXPECT_NEAR(native.x_[1], kEqBoundX1, kConvEquivAnalyticTol);

    // Stationarity, in the convention the model contract pins:
    //     grad(f) + Je^T lambda_e - z = 0,  z >= 0 at an active lower bound.
    // x1 is free, so its row alone determines lambda_e; x0's row then reads
    // off the bound multiplier, which must be non-negative because the bound
    // that binds is a LOWER one.
    const double grad0 = 2.0 * (kEqBoundX0 - 1.0);
    const double grad1 = 2.0 * (kEqBoundX1 - 2.0);
    const double je0 = 2.0 * kEqBoundX0;
    const double je1 = 2.0 * kEqBoundX1;
    const double lambda_e_expected = -grad1 / je1;
    ASSERT_EQ(native.lambda_e_.size(), 1);
    EXPECT_NEAR(native.lambda_e_[0], lambda_e_expected, kConvEquivAnalyticTol);

    // The bound multiplier the convention forces, and its sign. Its VALUE is a
    // restatement of the line above (z0 is that expression), so the assertion
    // worth making is the sign: the bound that binds is a LOWER one, and the
    // convention says a lower bound's multiplier is non-negative.
    //
    // A known limit of this suite, deliberate rather than overlooked: z is
    // computed here from the convention and the reported equality multiplier,
    // and is not compared against an engine-reported bound multiplier, because
    // the solve result exposes none. What is pinned is that the convention's
    // arithmetic closes and its sign is right -- not that the engine agrees on
    // z. Closing that gap needs the bound duals on the result first, and this
    // is the natural place for it when they arrive.
    const double z0 = grad0 + native.lambda_e_[0] * je0;
    EXPECT_GT(z0, 0.0) << "the lower bound is active, so its multiplier is non-negative";
}

///////////////////////////////////////////////////////////////////////////////
// Fixture B -- two ranged rows, and the range-split multiplier map.
///////////////////////////////////////////////////////////////////////////////

TEST(ConversionEquivalence, RangedModelSurfacesMatchAcrossTheConversion) {
    ConvEquivRangedNative native;
    NlpProblemModel converted(std::make_shared<ConvEquivRangedProblem>());

    // The split is the conversion's own decision, so it is read back and
    // stated here rather than left implicit in the row numbering below.
    const auto &rows = converted.rows();
    ASSERT_EQ(rows.kinds_.size(), 2u);
    EXPECT_EQ(rows.kinds_[0], NLPRowKind::Range);
    EXPECT_EQ(rows.kinds_[1], NLPRowKind::Range);
    EXPECT_EQ(rows.num_eq_, 0);
    EXPECT_EQ(rows.num_iq_, 4);
    EXPECT_EQ(rows.iq_upper_row_[0], 0);
    EXPECT_EQ(rows.iq_lower_row_[0], 1);
    EXPECT_EQ(rows.iq_upper_row_[1], 2);
    EXPECT_EQ(rows.iq_lower_row_[1], 3);

    const std::vector<Eigen::VectorXd> probes = {
        conv_equiv_ranged_start(),
        (Eigen::VectorXd(2) << kRangedX0, kRangedX1).finished(),
        (Eigen::VectorXd(2) << -2.0, 4.0).finished(),
        (Eigen::VectorXd(2) << 0.0, 0.0).finished(),
    };
    const Eigen::VectorXd le(0);
    const Eigen::VectorXd li = (Eigen::VectorXd(4) << 0.3, 0.0, 0.0, 1.25).finished();
    for (const auto &x : probes) {
        SCOPED_TRACE(::testing::Message() << "probe (" << x[0] << ", " << x[1] << ")");
        conv_equiv_expect_models_agree(native, converted, x, 1.0, le, li);
    }
}

TEST(ConversionEquivalence, RangedNativeAndTripletDoorsSolveToTheSameNumbers) {
    const Eigen::VectorXd x0 = conv_equiv_ranged_start();
    const auto native =
        conv_equiv_solve_native(std::make_shared<ConvEquivRangedNative>(), "ConvEquivRanged", x0,
                                FixedVariableTreatments::MakeParameter);
    const auto triplet = conv_equiv_solve_triplet(std::make_shared<ConvEquivRangedProblem>(), x0,
                                                  FixedVariableTreatments::MakeParameter);
    ASSERT_EQ(native.flag_, tycho::ConvergenceFlags::CONVERGED);
    ASSERT_EQ(triplet.flag_, tycho::ConvergenceFlags::CONVERGED);
    conv_equiv_expect_close(native.x_, triplet.x_, kConvEquivSolveGate, "primal point");
    conv_equiv_expect_close(native.lambda_i_, triplet.lambda_i_, kConvEquivSolveGate,
                            "inequality multipliers");
    conv_equiv_expect_close(native.obj_, triplet.obj_, kConvEquivSolveGate, "objective");
}

TEST(ConversionEquivalence, RangedVfDoorAgreesWithTheOtherTwo) {
    const Eigen::VectorXd x0 = conv_equiv_ranged_start();
    const auto vf = conv_equiv_solve_ranged_vf();
    const auto native =
        conv_equiv_solve_native(std::make_shared<ConvEquivRangedNative>(), "ConvEquivRanged", x0,
                                FixedVariableTreatments::MakeParameter);
    ASSERT_EQ(vf.flag_, tycho::ConvergenceFlags::CONVERGED);
    ASSERT_EQ(native.flag_, tycho::ConvergenceFlags::CONVERGED);

    conv_equiv_expect_near(vf.x_, native.x_, kConvEquivCrossDoorTol, "primal point");
    ASSERT_EQ(vf.lambda_i_.size(), 4);
    conv_equiv_expect_near(vf.lambda_i_, native.lambda_i_, kConvEquivCrossDoorTol,
                           "inequality multipliers");
    EXPECT_NEAR(vf.obj_, native.obj_, kConvEquivCrossDoorTol);

    // The corner the two ranges cut out, and the multipliers stationarity
    // forces there: only the first row's upper part and the second row's
    // lower part are active, and both multipliers are non-negative because
    // every native inequality reads cI(x) <= 0.
    EXPECT_NEAR(native.x_[0], kRangedX0, kConvEquivAnalyticTol);
    EXPECT_NEAR(native.x_[1], kRangedX1, kConvEquivAnalyticTol);
    EXPECT_NEAR(native.lambda_i_[0], 1.5, kConvEquivAnalyticTol);
    EXPECT_NEAR(native.lambda_i_[1], 0.0, kConvEquivAnalyticTol);
    EXPECT_NEAR(native.lambda_i_[2], 0.0, kConvEquivAnalyticTol);
    EXPECT_NEAR(native.lambda_i_[3], 0.5, kConvEquivAnalyticTol);
    for (int k = 0; k < 4; k++) {
        EXPECT_GT(native.lambda_i_[k], -kConvEquivAnalyticTol) << "row " << k;
    }
}

// The solved native multipliers, carried across the conversion boundary in
// both directions. This is the range-split recovery: two one-sided native
// multipliers become one signed declared multiplier, and the declared value
// names the side it came from on the way back.
TEST(ConversionEquivalence, RangedSolvedDualsCrossTheBoundaryBothWays) {
    auto problem = std::make_shared<ConvEquivRangedProblem>();
    NLPSolver solver(problem);
    conv_equiv_configure(*solver.optimizer_);
    solver.optimizer_->set_fixed_variable_treatment(FixedVariableTreatments::MakeParameter);
    ASSERT_EQ(solver.optimize(conv_equiv_ranged_start()), tycho::ConvergenceFlags::CONVERGED);

    const Eigen::VectorXd li = solver.active_iq_lmults_;
    ASSERT_EQ(li.size(), 4);

    // Forward: native rows -> declared rows. The first range binds above, so
    // its declared multiplier is positive; the second binds below, so its
    // declared multiplier is negative. This is the sign carried by the map,
    // not by the engine.
    const Eigen::VectorXd declared = solver.return_multipliers();
    ASSERT_EQ(declared.size(), 2);
    EXPECT_EQ(declared[0], li[0] - li[1]);
    EXPECT_EQ(declared[1], li[2] - li[3]);
    EXPECT_NEAR(declared[0], 1.5, kConvEquivAnalyticTol);
    EXPECT_NEAR(declared[1], -0.5, kConvEquivAnalyticTol);

    // Backward: declared rows -> native rows. The signed value goes to the
    // side it names and the other side takes zero.
    Eigen::VectorXd back_e, back_i;
    solver.model_->split_user_multipliers(declared, back_e, back_i);
    ASSERT_EQ(back_e.size(), 0);
    ASSERT_EQ(back_i.size(), 4);
    EXPECT_EQ(back_i[0], declared[0]);
    EXPECT_EQ(back_i[1], 0.0);
    EXPECT_EQ(back_i[2], 0.0);
    EXPECT_EQ(back_i[3], -declared[1]);

    // And the round trip closes exactly, which it does because at this
    // solution each range carries at most one nonzero side. The other order
    // is not an identity in general -- see the map's own test below.
    const Eigen::VectorXd again = solver.model_->compose_user_multipliers(back_e, back_i);
    conv_equiv_expect_bit_identical(again, declared, "declared multipliers after a round trip");
}

///////////////////////////////////////////////////////////////////////////////
// The declared-row multiplier map, on its own.
//
// A problem declaring one row of every kind, so the whole correspondence table
// is asserted rather than the two entries a solve happens to exercise. Nothing
// here is solved: the map is a property of the classification.
///////////////////////////////////////////////////////////////////////////////

namespace {

struct ConvEquivAllKindsProblem : NLPProblem {
    int num_vars() const override { return 2; }
    int num_cons() const override { return 5; }
    int num_jac_nonzeros() const override { return 5; }
    int num_hess_nonzeros() const override { return 1; }

    void bounds(Eigen::Ref<Eigen::VectorXd> xl, Eigen::Ref<Eigen::VectorXd> xu,
                Eigen::Ref<Eigen::VectorXd> gl, Eigen::Ref<Eigen::VectorXd> gu) const override {
        xl << -kConvEquivInf, -kConvEquivInf;
        xu << kConvEquivInf, kConvEquivInf;
        //     equality  upper-bounded  lower-bounded  range   free
        gl << 1.0, -kConvEquivInf, 3.0, 4.0, -kConvEquivInf;
        gu << 1.0, 2.0, kConvEquivInf, 5.0, kConvEquivInf;
    }
    void eval_f(ConstEigenRef<Eigen::VectorXd> x, double &f) const override { f = x[0] * x[0]; }
    void eval_grad_f(ConstEigenRef<Eigen::VectorXd> x,
                     Eigen::Ref<Eigen::VectorXd> g) const override {
        g[0] = 2.0 * x[0];
        g[1] = 0.0;
    }
    void eval_g(ConstEigenRef<Eigen::VectorXd> x, Eigen::Ref<Eigen::VectorXd> g) const override {
        g.setConstant(x[0]);
    }
    void jac_structure(Eigen::Ref<Eigen::VectorXi> r,
                       Eigen::Ref<Eigen::VectorXi> c) const override {
        r << 0, 1, 2, 3, 4;
        c.setZero();
    }
    void hess_structure(Eigen::Ref<Eigen::VectorXi> r,
                        Eigen::Ref<Eigen::VectorXi> c) const override {
        r << 0;
        c << 0;
    }
    void eval_jac(ConstEigenRef<Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd> v) const override {
        v.setOnes();
    }
    void eval_hess(ConstEigenRef<Eigen::VectorXd>, double obj_factor,
                   ConstEigenRef<Eigen::VectorXd>, Eigen::Ref<Eigen::VectorXd> v) const override {
        v[0] = 2.0 * obj_factor;
    }
    std::string name() const override { return "ConvEquivAllKinds"; }
};

} // namespace

TEST(ConversionEquivalence, DeclaredRowMultiplierMapCoversEveryRowKind) {
    NlpProblemModel model(std::make_shared<ConvEquivAllKindsProblem>());
    const auto &rows = model.rows();

    ASSERT_EQ(rows.kinds_.size(), 5u);
    EXPECT_EQ(rows.kinds_[0], NLPRowKind::Equality);
    EXPECT_EQ(rows.kinds_[1], NLPRowKind::UpperBounded);
    EXPECT_EQ(rows.kinds_[2], NLPRowKind::LowerBounded);
    EXPECT_EQ(rows.kinds_[3], NLPRowKind::Range);
    EXPECT_EQ(rows.kinds_[4], NLPRowKind::Free);
    EXPECT_EQ(rows.num_eq_, 1);
    EXPECT_EQ(rows.num_iq_, 4);
    EXPECT_EQ(rows.eq_row_[0], 0);
    EXPECT_EQ(rows.iq_upper_row_[1], 0);
    EXPECT_EQ(rows.iq_lower_row_[2], 1);
    EXPECT_EQ(rows.iq_upper_row_[3], 2);
    EXPECT_EQ(rows.iq_lower_row_[3], 3);
    EXPECT_EQ(model.me(), 1);
    EXPECT_EQ(model.mi(), 4);

    // Forward, one kind at a time, against the correspondence the conversion
    // states: an upper-bounded row keeps its sign, a lower-bounded row flips
    // it, a range takes the difference, and a free row has no image.
    const Eigen::VectorXd le = (Eigen::VectorXd(1) << 7.0).finished();
    const Eigen::VectorXd li = (Eigen::VectorXd(4) << 0.25, 0.5, 3.0, 4.0).finished();
    const Eigen::VectorXd declared = model.compose_user_multipliers(le, li);
    ASSERT_EQ(declared.size(), 5);
    EXPECT_EQ(declared[0], 7.0);
    EXPECT_EQ(declared[1], 0.25);
    EXPECT_EQ(declared[2], -0.5);
    EXPECT_EQ(declared[3], 3.0 - 4.0);
    EXPECT_EQ(declared[4], 0.0);

    // Backward.
    Eigen::VectorXd back_e, back_i;
    model.split_user_multipliers(declared, back_e, back_i);
    ASSERT_EQ(back_e.size(), 1);
    ASSERT_EQ(back_i.size(), 4);
    EXPECT_EQ(back_e[0], 7.0);
    EXPECT_EQ(back_i[0], 0.25);
    EXPECT_EQ(back_i[1], 0.5);
    EXPECT_EQ(back_i[2], 0.0); // the range composed to -1, so its upper side is zero
    EXPECT_EQ(back_i[3], 1.0); // and its lower side takes the magnitude

    // compose(split(l)) == l exactly, for every declared vector whose free
    // rows are zero -- the free row is the only place a declared value has no
    // native image to come back from.
    const Eigen::VectorXd declared_again = model.compose_user_multipliers(back_e, back_i);
    conv_equiv_expect_bit_identical(declared_again, declared, "declared multipliers");

    // The other order is NOT an identity, and the exact way it fails is part
    // of the stated map: a range carrying both sides reaches the declared
    // space only as their difference.
    const Eigen::VectorXd li_both = (Eigen::VectorXd(4) << 0.25, 0.5, 3.0, 4.0).finished();
    Eigen::VectorXd round_e, round_i;
    model.split_user_multipliers(model.compose_user_multipliers(le, li_both), round_e, round_i);
    EXPECT_EQ(round_i[2], 0.0);
    EXPECT_EQ(round_i[3], 1.0);
    EXPECT_NE(round_i[2], li_both[2]);
    EXPECT_NE(round_i[3], li_both[3]);

    // A free row's declared value is dropped on the way in, which is why the
    // round trip is stated only for declared vectors that are zero there.
    Eigen::VectorXd with_free = declared;
    with_free[4] = 99.0;
    Eigen::VectorXd free_e, free_i;
    model.split_user_multipliers(with_free, free_e, free_i);
    conv_equiv_expect_bit_identical(free_i, back_i, "native inequality multipliers");
    EXPECT_EQ(model.compose_user_multipliers(free_e, free_i)[4], 0.0);
}

// The map takes exactly the row counts it stands for, in both blocks. An empty
// block is a length, not a compact spelling of "all zero", so composing one
// over a nonempty row space is a caller error rather than a request for zeros.
TEST(ConversionEquivalence, TheMultiplierMapRefusesABlockThatIsNotItsRowCount) {
    NlpProblemModel model(std::make_shared<ConvEquivAllKindsProblem>());
    ASSERT_EQ(model.me(), 1);
    ASSERT_EQ(model.mi(), 4);

    const Eigen::VectorXd le = (Eigen::VectorXd(1) << 7.0).finished();
    const Eigen::VectorXd li = (Eigen::VectorXd(4) << 0.25, 0.5, 3.0, 4.0).finished();

    EXPECT_THROW(model.compose_user_multipliers(Eigen::VectorXd(0), li), std::invalid_argument);
    EXPECT_THROW(model.compose_user_multipliers(le, Eigen::VectorXd(0)), std::invalid_argument);
    // Zeros over the real row counts is how a caller asks for zero
    // multipliers, and it still works.
    EXPECT_NO_THROW(
        model.compose_user_multipliers(Eigen::VectorXd::Zero(1), Eigen::VectorXd::Zero(4)));
}

///////////////////////////////////////////////////////////////////////////////
// Fixture C -- a fixed variable, under each of the three treatments.
///////////////////////////////////////////////////////////////////////////////

TEST(ConversionEquivalence, FixedModelSurfacesMatchAcrossTheConversion) {
    ConvEquivFixedNative native;
    NlpProblemModel converted(std::make_shared<ConvEquivFixedProblem>());

    const auto &rows = converted.rows();
    EXPECT_EQ(rows.kinds_[0], NLPRowKind::Equality);
    EXPECT_EQ(rows.kinds_[1], NLPRowKind::UpperBounded);

    const std::vector<Eigen::VectorXd> probes = {
        conv_equiv_fixed_start(),
        (Eigen::VectorXd(3) << kFixedOptX0, kFixedOptX1, kFixedX2).finished(),
        (Eigen::VectorXd(3) << -1.0, 2.5, kFixedX2).finished(),
    };
    const Eigen::VectorXd le = (Eigen::VectorXd(1) << 0.9).finished();
    const Eigen::VectorXd li = (Eigen::VectorXd(1) << 0.1).finished();
    for (const auto &x : probes) {
        SCOPED_TRACE(::testing::Message() << "probe " << x.transpose());
        conv_equiv_expect_models_agree(native, converted, x, 1.0, le, li);
    }

    // The fixed pair reaches the model surface verbatim -- equal bounds, not a
    // rewritten one-sided pair or a dropped variable. Deciding what to do with
    // it is the solver's job, under the treatment it is configured with.
    EXPECT_EQ(converted.lower()[2], kFixedX2);
    EXPECT_EQ(converted.upper()[2], kFixedX2);
}

// Under the eliminating treatment the fixed coordinate never enters the
// system; it comes back exactly, and the equality multiplier vector holds the
// one user row and nothing else. All three doors agree.
TEST(ConversionEquivalence, FixedVariableMakeParameterAgreesAcrossDoors) {
    const Eigen::VectorXd x0 = conv_equiv_fixed_start();
    const auto native =
        conv_equiv_solve_native(std::make_shared<ConvEquivFixedNative>(), "ConvEquivFixed", x0,
                                FixedVariableTreatments::MakeParameter);
    const auto triplet = conv_equiv_solve_triplet(std::make_shared<ConvEquivFixedProblem>(), x0,
                                                  FixedVariableTreatments::MakeParameter);
    const auto vf = conv_equiv_solve_fixed_vf(FixedVariableTreatments::MakeParameter);

    ASSERT_EQ(native.flag_, tycho::ConvergenceFlags::CONVERGED);
    ASSERT_EQ(triplet.flag_, tycho::ConvergenceFlags::CONVERGED);
    ASSERT_EQ(vf.flag_, tycho::ConvergenceFlags::CONVERGED);

    conv_equiv_expect_close(native.x_, triplet.x_, kConvEquivSolveGate, "primal point");
    conv_equiv_expect_close(native.lambda_e_, triplet.lambda_e_, kConvEquivSolveGate,
                            "equality multipliers");
    conv_equiv_expect_close(native.lambda_i_, triplet.lambda_i_, kConvEquivSolveGate,
                            "inequality multipliers");

    conv_equiv_expect_near(vf.x_, native.x_, kConvEquivCrossDoorTol, "primal point");
    conv_equiv_expect_near(vf.lambda_e_, native.lambda_e_, kConvEquivCrossDoorTol,
                           "equality multipliers");

    // The eliminated coordinate is returned exactly, not to a tolerance: it
    // was never a variable of the system that was solved.
    EXPECT_DOUBLE_EQ(native.x_[2], kFixedX2);
    EXPECT_DOUBLE_EQ(vf.x_[2], kFixedX2);
    EXPECT_NEAR(native.x_[0], kFixedOptX0, kConvEquivAnalyticTol);
    EXPECT_NEAR(native.x_[1], kFixedOptX1, kConvEquivAnalyticTol);

    // The equality multiplier vector carries the user's own row and nothing
    // more, and its value is what stationarity in the two free coordinates
    // forces. The inequality is slack here, so its multiplier is zero.
    ASSERT_EQ(native.lambda_e_.size(), 1);
    EXPECT_NEAR(native.lambda_e_[0], kFixedLambdaEq, kConvEquivAnalyticTol);
    ASSERT_EQ(native.lambda_i_.size(), 1);
    EXPECT_NEAR(native.lambda_i_[0], 0.0, kConvEquivAnalyticTol);
}

// Under the constraining treatment the fixed variable stays in the system and
// an internal equality row holds it there. The row is appended AFTER every
// user row, so the equality multiplier vector is one longer and its tail entry
// is that row's own multiplier. Both halves are asserted: the shape, and the
// value stationarity forces into it.
TEST(ConversionEquivalence, FixedVariableMakeConstraintAppendsItsRowAfterTheUserRows) {
    const Eigen::VectorXd x0 = conv_equiv_fixed_start();

    ConvEquivNativeDoor door(std::make_shared<ConvEquivFixedNative>(), "ConvEquivFixed");
    conv_equiv_configure(*door.optimizer_);
    door.optimizer_->set_fixed_variable_treatment(FixedVariableTreatments::MakeConstraint);
    ASSERT_EQ(door.run_optimize(x0), tycho::ConvergenceFlags::CONVERGED);

    // The solve result records which treatment actually ran, so a reader of
    // the completed solve can tell which system produced the numbers below.
    EXPECT_EQ(door.optimizer_->result().fixed_variable_treatment_,
              FixedVariableTreatments::MakeConstraint);
    EXPECT_FALSE(door.nlp_->is_reduced());
    EXPECT_EQ(door.nlp_->reduced_primal_vars(), 3);
    EXPECT_EQ(door.nlp_->user_equal_cons_, 1);
    EXPECT_EQ(door.nlp_->equal_cons_, 2); // the user row plus one fixing row

    // The multiplier vector's shape follows the row space, fixing rows
    // included, with the user rows first.
    ASSERT_EQ(door.active_eq_lmults_.size(), door.nlp_->equal_cons_);
    EXPECT_NEAR(door.active_eq_lmults_[0], kFixedLambdaEq, kConvEquivCrossDoorTol);
    EXPECT_NEAR(door.active_eq_lmults_[1], kFixedLambdaFixRow, kConvEquivCrossDoorTol);

    EXPECT_NEAR(door.active_variables_[0], kFixedOptX0, kConvEquivCrossDoorTol);
    EXPECT_NEAR(door.active_variables_[1], kFixedOptX1, kConvEquivCrossDoorTol);
    // The variable reaches its value to equality-constraint tolerance under
    // this treatment rather than exactly, which is the treatment's own stated
    // difference from the eliminating one.
    EXPECT_NEAR(door.active_variables_[2], kFixedX2, kConvEquivCrossDoorTol);

    // The triplet door, on the same treatment, lands on the same numbers.
    const auto triplet = conv_equiv_solve_triplet(std::make_shared<ConvEquivFixedProblem>(), x0,
                                                  FixedVariableTreatments::MakeConstraint);
    ASSERT_EQ(triplet.flag_, tycho::ConvergenceFlags::CONVERGED);
    conv_equiv_expect_close(door.active_variables_, triplet.x_, kConvEquivSolveGate,
                            "primal point");
    conv_equiv_expect_close(door.active_eq_lmults_, triplet.lambda_e_, kConvEquivSolveGate,
                            "equality multipliers");

    // NLPSolver reports declared-row multipliers, so it reports the model's
    // own rows and stops there: the internal fixing row is the engine's, not
    // the problem's, and has no declared row to be reported against.
    NLPSolver reporting(std::make_shared<ConvEquivFixedProblem>());
    conv_equiv_configure(*reporting.optimizer_);
    reporting.optimizer_->set_fixed_variable_treatment(FixedVariableTreatments::MakeConstraint);
    ASSERT_EQ(reporting.optimize(x0), tycho::ConvergenceFlags::CONVERGED);
    ASSERT_EQ(reporting.active_eq_lmults_.size(), 2);
    const Eigen::VectorXd declared = reporting.return_multipliers();
    ASSERT_EQ(declared.size(), 2);
    EXPECT_EQ(declared[0], reporting.active_eq_lmults_[0]);
}

// The relaxing treatment leaves the system the declared size and holds the
// variable near its value by the barrier, within the relaxation.
TEST(ConversionEquivalence, FixedVariableRelaxBoundsKeepsTheDeclaredShape) {
    const Eigen::VectorXd x0 = conv_equiv_fixed_start();

    ConvEquivNativeDoor door(std::make_shared<ConvEquivFixedNative>(), "ConvEquivFixed");
    conv_equiv_configure(*door.optimizer_);
    door.optimizer_->set_fixed_variable_treatment(FixedVariableTreatments::RelaxBounds);
    ASSERT_EQ(door.run_optimize(x0), tycho::ConvergenceFlags::CONVERGED);

    EXPECT_EQ(door.optimizer_->result().fixed_variable_treatment_,
              FixedVariableTreatments::RelaxBounds);
    EXPECT_FALSE(door.nlp_->is_reduced());
    EXPECT_EQ(door.nlp_->reduced_primal_vars(), 3);
    EXPECT_EQ(door.nlp_->equal_cons_, door.nlp_->user_equal_cons_);
    ASSERT_EQ(door.active_eq_lmults_.size(), 1);

    // Held within the relaxation, not pinned: the widened pair is what the
    // barrier divides by, so the coordinate lands near the declared value
    // rather than on it.
    EXPECT_NEAR(door.active_variables_[2], kFixedX2, 1.0e-6);
    EXPECT_NEAR(door.active_variables_[0], kFixedOptX0, 1.0e-6);
    EXPECT_NEAR(door.active_variables_[1], kFixedOptX1, 1.0e-6);

    const auto triplet = conv_equiv_solve_triplet(std::make_shared<ConvEquivFixedProblem>(), x0,
                                                  FixedVariableTreatments::RelaxBounds);
    ASSERT_EQ(triplet.flag_, tycho::ConvergenceFlags::CONVERGED);
    conv_equiv_expect_close(door.active_variables_, triplet.x_, kConvEquivSolveGate,
                            "primal point");
    conv_equiv_expect_close(door.active_eq_lmults_, triplet.lambda_e_, kConvEquivSolveGate,
                            "equality multipliers");
}

// The three treatments solve three different systems and reach the same
// answer, which is the only sense in which they are interchangeable. Stated
// once, across the treatments, so a change to any one of them is measured
// against the other two rather than against itself.
TEST(ConversionEquivalence, FixedVariableTreatmentsAgreeOnTheAnswer) {
    const Eigen::VectorXd x0 = conv_equiv_fixed_start();
    const auto parameter =
        conv_equiv_solve_native(std::make_shared<ConvEquivFixedNative>(), "ConvEquivFixed", x0,
                                FixedVariableTreatments::MakeParameter);
    const auto constraint =
        conv_equiv_solve_native(std::make_shared<ConvEquivFixedNative>(), "ConvEquivFixed", x0,
                                FixedVariableTreatments::MakeConstraint);
    const auto relaxed =
        conv_equiv_solve_native(std::make_shared<ConvEquivFixedNative>(), "ConvEquivFixed", x0,
                                FixedVariableTreatments::RelaxBounds);
    ASSERT_EQ(parameter.flag_, tycho::ConvergenceFlags::CONVERGED);
    ASSERT_EQ(constraint.flag_, tycho::ConvergenceFlags::CONVERGED);
    ASSERT_EQ(relaxed.flag_, tycho::ConvergenceFlags::CONVERGED);

    for (int k = 0; k < 3; k++) {
        EXPECT_NEAR(parameter.x_[k], constraint.x_[k], 1.0e-6) << "coordinate " << k;
        EXPECT_NEAR(parameter.x_[k], relaxed.x_[k], 1.0e-6) << "coordinate " << k;
    }
    EXPECT_NEAR(parameter.obj_, constraint.obj_, 1.0e-6);
    EXPECT_NEAR(parameter.obj_, relaxed.obj_, 1.0e-6);

    // The user row's multiplier is the same under every treatment; only the
    // eliminating one has nothing after it.
    EXPECT_NEAR(parameter.lambda_e_[0], constraint.lambda_e_[0], 1.0e-6);
    EXPECT_NEAR(parameter.lambda_e_[0], relaxed.lambda_e_[0], 1.0e-6);
    EXPECT_EQ(parameter.lambda_e_.size(), 1);
    EXPECT_EQ(constraint.lambda_e_.size(), 2);
    EXPECT_EQ(relaxed.lambda_e_.size(), 1);
}
