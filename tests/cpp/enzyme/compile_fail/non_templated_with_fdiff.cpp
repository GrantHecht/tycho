// =============================================================================
// Compile-fail probe (Phase 4 constraint #1):
//
// A non-templated compute_impl is supported only when the Jacobian mode is
// EnzymeAD.  FDiffCentArray (and any FD Jacobian mode) calls compute() with
// an Eigen::Array-typed Scalar for SuperScalar dispatch, which a concrete-
// double signature cannot accept.
//
// This file MUST FAIL TO COMPILE.  The wired-up ctest entry passes when
// cmake --build returns non-zero AND the EXPECTED_REGEX matches the
// resulting diagnostic.
// =============================================================================

#include <tycho/tycho.h>

namespace tycho_compile_fail_probes {

struct ProbeNonTemplatedFDiff
    : tycho::vf::VectorFunction<ProbeNonTemplatedFDiff, 5, 3,
                                tycho::vf::DenseDerivativeMode::FDiffCentArray,
                                tycho::vf::DenseDerivativeMode::FDiffFwd> {
    inline void compute_impl(
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>> x,
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> fx) const {
        fx[0] = x[0];
        fx[1] = x[1];
        fx[2] = x[2];
    }
};

// Force instantiation of the FDiffCentArray Jacobian path which dispatches
// compute() with Eigen::Array<double, W, 1> Scalar.
void force_instantiation() {
    ProbeNonTemplatedFDiff f;
    Eigen::Matrix<double, 5, 1> x = Eigen::Matrix<double, 5, 1>::Zero();
    Eigen::Matrix<double, 3, 1> fx = Eigen::Matrix<double, 3, 1>::Zero();
    Eigen::Matrix<double, 3, 5> jx = Eigen::Matrix<double, 3, 5>::Zero();
    f.compute_jacobian(x, fx, jx);
}

} // namespace tycho_compile_fail_probes
