///////////////////////////////////////////////////////////////////////////////
// ParsedInput correctness tests (VF_REVIEW 1.12)
//
// ParsedInput gathers a wrapped function's inputs from an outer vector via
// an index map (varlocs_). Two bugs are exercised here:
//   1. varlocs_ was never validated (empty / out-of-bounds / size-mismatch),
//      which is an out-of-bounds scatter/gather in release builds.
//   2. Duplicate indices in varlocs_ overwrote instead of summed the
//      Jacobian-column / gradient / Hessian scatter contributions, so a
//      variable appearing twice in the wrapped function's input silently
//      lost one of its contributions.
///////////////////////////////////////////////////////////////////////////////

#include <tycho/tycho.h>
#include "test_utils.h"
#include "vf_test_utils.h"
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

namespace {

using Inner = GenericFunction<-1, -1>;
using PI = ParsedInput<Inner, -1, -1>;

Inner make_inner_product() {
    auto args2 = Arguments<2>();
    return Inner(args2.coeff<0>() * args2.coeff<1>());
}

} // namespace

class ParsedInputTest : public VectorFunctionFixture {};

///////////////////////////////////////////////////////////////////////////////
// Duplicate-index accumulation
///////////////////////////////////////////////////////////////////////////////

TEST_F(ParsedInputTest, DuplicateIndicesAccumulate) {
    // inner(a, b) = a * b, with both a and b mapped to the same outer index
    // (2) of a 3-vector -> f(x) = x2 * x2 = x2^2.
    //
    // Pre-fix, the non-contiguous scatter used '=' rather than '+=', so the
    // second write to jx.col(2) / adjgrad[2] / adjhess(2,2) clobbered the
    // first instead of accumulating both contributions. That makes the
    // Jacobian/gradient read as d/dx2 = x2 (only one of the two chain-rule
    // terms) instead of the correct 2*x2, and the Hessian read as a stray
    // single off-diagonal inner term instead of the correct 2.
    Inner inner = make_inner_product();
    Eigen::VectorXi vlocs(2);
    vlocs << 2, 2;
    PI f(inner, vlocs, 3);

    EXPECT_EQ(f.input_rows(), 3);
    EXPECT_EQ(f.output_rows(), 1);

    Eigen::VectorXd x = deterministic_random_vector(3, 61, 0.5, 2.0);

    // f(x) = x2^2
    Eigen::VectorXd fx(1);
    fx.setZero();
    f.compute(x, fx);
    EXPECT_NEAR(fx[0], x[2] * x[2], 1e-12);

    verify_jacobian_fd(f, x, 1e-5); // d/dx2 = 2*x2; pre-fix gives x2

    Eigen::VectorXd lm = deterministic_random_vector(1, 62, -1.0, 1.0);
    verify_adjoint_consistency(f, x, lm);
    verify_adjoint_hessian_fd(f, x, lm, 1e-4); // d2/dx2^2 = 2; pre-fix gives a stray value
}

///////////////////////////////////////////////////////////////////////////////
// Constructor validation
///////////////////////////////////////////////////////////////////////////////

TEST_F(ParsedInputTest, ConstructorValidationOutOfBounds) {
    Inner inner = make_inner_product();
    Eigen::VectorXi vlocs(2);
    vlocs << 0, 5; // 5 is outside [0, 3)
    EXPECT_THROW(PI(inner, vlocs, 3), std::invalid_argument);
}

TEST_F(ParsedInputTest, ConstructorValidationNegative) {
    Inner inner = make_inner_product();
    Eigen::VectorXi vlocs(2);
    vlocs << -1, 0;
    EXPECT_THROW(PI(inner, vlocs, 3), std::invalid_argument);
}

TEST_F(ParsedInputTest, ConstructorValidationSizeMismatch) {
    Inner inner = make_inner_product();
    Eigen::VectorXi vlocs(3);
    vlocs << 0, 1, 2; // inner takes 2 inputs, not 3
    EXPECT_THROW(PI(inner, vlocs, 3), std::invalid_argument);
}

TEST_F(ParsedInputTest, ConstructorValidationEmpty) {
    Inner inner = make_inner_product();
    Eigen::VectorXi vlocs(0);
    EXPECT_THROW(PI(inner, vlocs, 3), std::invalid_argument);
}
