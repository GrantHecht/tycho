// =============================================================================
// Tycho — Phase 4 compile-fail probe.
// A non-templated compute_impl paired with <AutodiffFwd, AutodiffFwd> MUST
// fail to compile: AutodiffFwd's dual-number arithmetic requires a templated
// compute_impl that can be instantiated with dual<double>.  This file is
// EXCLUDE_FROM_ALL — only built indirectly by the matching CTest case via
// the run_build_must_fail.cmake wrapper.
// =============================================================================

#include <tycho/tycho.h>

struct NonTemplateBrachAutodiff
    : tycho::vf::VectorFunction<NonTemplateBrachAutodiff, 5, 3,
                                tycho::vf::DenseDerivativeMode::AutodiffFwd,
                                tycho::vf::DenseDerivativeMode::AutodiffFwd> {
    inline void compute_impl(
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>> x,
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> fx) const {
        fx[0] = x[2];
        fx[1] = -x[2];
        fx[2] = x[4];
    }
};

int main() {
    NonTemplateBrachAutodiff f;
    Eigen::Matrix<double, 5, 1> x;
    x.setZero();
    Eigen::Matrix<double, 3, 1> fx;
    fx.setZero();
    Eigen::Matrix<double, 3, 5> jx;
    jx.setZero();
    // This call instantiates DenseFirstDerivatives<..., AutodiffFwd>::
    // compute_jacobian_impl, which calls compute(xdual, fdual) with
    // dual<double>-typed Eigen vectors.  The non-templated compute_impl
    // above cannot accept those types — substitution failure.
    f.compute_jacobian(x, fx, jx);
    return 0;
}
