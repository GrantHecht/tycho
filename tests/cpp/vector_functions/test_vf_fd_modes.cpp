///////////////////////////////////////////////////////////////////////////////
// Finite-difference derivative-mode correctness tests
//
// Covers VF_REVIEW 1.5 (FD step vectors must self-heal on set_io_rows —
// dynamic-size functions like LambdaFunction/ADFun never manually re-set
// their steps after construction, so the mode-constructor's size-0 step
// vector was read out of bounds), VF_REVIEW 3.6 (FD steps must be relative
// to the input magnitude, not a fixed absolute value that underflows the
// representable spacing at large scale), and VF_REVIEW 3.8 (nested
// forward-over-forward Hessian FD needs a coarser default step than a
// single-layer Jacobian FD).
///////////////////////////////////////////////////////////////////////////////

#include "test_utils.h"
#include "vf_test_utils.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace tycho;
using namespace TychoTest;

namespace {

///////////////////////////////////////////////////////////////////////////////
// Static-size FD-mode helper functions
///////////////////////////////////////////////////////////////////////////////

/// f(x) = x^2, IR=OR=1, forward-FD Jacobian / forward-FD Hessian.
struct SquareFDStatic : VectorFunction<SquareFDStatic, 1, 1, DenseDerivativeMode::FDiffFwd,
                                       DenseDerivativeMode::FDiffFwd> {
    using Base = VectorFunction<SquareFDStatic, 1, 1, DenseDerivativeMode::FDiffFwd,
                                DenseDerivativeMode::FDiffFwd>;
    VF_TYPE_ALIASES(Base)

    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        VecRef<OutType> fx = fx_.const_cast_derived();
        fx[0] = x[0] * x[0];
    }
};

/// f(x) = x^3, IR=OR=1, forward-FD Jacobian / forward-FD (nested) Hessian.
struct CubeFDStatic : VectorFunction<CubeFDStatic, 1, 1, DenseDerivativeMode::FDiffFwd,
                                     DenseDerivativeMode::FDiffFwd> {
    using Base = VectorFunction<CubeFDStatic, 1, 1, DenseDerivativeMode::FDiffFwd,
                                DenseDerivativeMode::FDiffFwd>;
    VF_TYPE_ALIASES(Base)

    template <class InType, class OutType>
    inline void compute_impl(CVecRef<InType> x, CVecRef<OutType> fx_) const {
        VecRef<OutType> fx = fx_.const_cast_derived();
        fx[0] = x[0] * x[0] * x[0];
    }
};

} // namespace

class VFFDModes : public VectorFunctionFixture {};

///////////////////////////////////////////////////////////////////////////////
// VF_REVIEW 1.5 — dynamic-size FD step vectors must self-heal
///////////////////////////////////////////////////////////////////////////////

TEST_F(VFFDModes, DynamicSizeFDStepsSelfHeal) {
    // LambdaFunction<-1, -1, ...> has no compile-time size: its FD-mode base
    // constructors run with input_rows() == 0 and size the step vector to 0
    // before set_io_rows() (called from LambdaFunction's own constructor)
    // establishes the real size. LambdaFunction never manually re-sets the
    // step vector afterward, so this exercises the self-heal path directly.
    auto square_fn = [](const auto * /*self*/, const auto &x, auto &fx) {
        fx = x.array().square();
    };
    using FnT = decltype(square_fn);
    LambdaFunction<-1, -1, FnT> f(InputOutputSize<-1, -1>{5, 5}, square_fn);

    Eigen::VectorXd x = deterministic_random_vector(5, 7, 0.5, 2.0);
    verify_jacobian_fd(f, x, 1e-4);
}

///////////////////////////////////////////////////////////////////////////////
// VF_REVIEW 3.6 — FD steps must be relative to the input scale
///////////////////////////////////////////////////////////////////////////////

TEST_F(VFFDModes, RelativeStepAccurateAtLargeScale) {
    // A fixed absolute step of 1e-7 loses accuracy as |x| grows: cancellation
    // in f(x+h)-f(x) costs ~2.4e-4 relative error at x=1e6 and, once h falls
    // below the representable spacing of x (~2.4e-7 at x=1e9), the perturbed
    // point rounds by a full ULP the nominal divisor doesn't match, costing
    // ~28% relative error. A relative (max(1,|x|)-scaled) step with a
    // realized-step divisor stays at ~5e-8 relative error at both scales.
    SquareFDStatic f;
    Eigen::VectorXd x(1);
    Eigen::VectorXd fx(1);
    Eigen::MatrixXd jx(1, 1);

    x << 1.0e6; // old absolute step: ~2.4e-4 rel err — fails the 1e-5 bound
    fx.setZero();
    jx.setZero();
    f.compute_jacobian(x, fx, jx);
    EXPECT_NEAR(jx(0, 0), 2.0e6, 2.0e6 * 1e-5);

    x << 1.0e9; // old absolute step: ~28% rel err — catastrophic
    fx.setZero();
    jx.setZero();
    f.compute_jacobian(x, fx, jx);
    EXPECT_NEAR(jx(0, 0), 2.0e9, 2.0e9 * 1e-3);
}

///////////////////////////////////////////////////////////////////////////////
// VF_REVIEW 3.8 — nested FD-of-FD adjoint Hessian accuracy at moderate scale
///////////////////////////////////////////////////////////////////////////////

TEST_F(VFFDModes, NestedHessianAccurateAtModerateScale) {
    // f(x) = x^3, lambda*f'' = 6*lambda*x. Exercise the forward-over-forward
    // Hessian FD path (adjoint gradient re-differenced by forward FD) at a
    // moderate scale where the old fixed 1e-7 Hessian step is already too
    // fine relative to the gradient's own FD noise floor.
    CubeFDStatic f;
    Eigen::VectorXd x(1);
    x << 100.0;
    Eigen::VectorXd lm(1);
    lm << 1.0;

    Eigen::MatrixXd adjhess = f.adjointhessian(x, lm);
    double expected = 6.0 * lm[0] * x[0];
    EXPECT_NEAR(adjhess(0, 0), expected, std::abs(expected) * 1e-2);
}
