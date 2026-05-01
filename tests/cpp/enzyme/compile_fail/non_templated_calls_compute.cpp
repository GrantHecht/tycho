// =============================================================================
// Compile-fail probe (Phase 4 constraint #3):
//
// Calling .compute() directly on a non-templated VF is a separate signature
// mismatch.  The .compute() entry point dispatches to compute_impl with an
// Eigen::Array-typed Scalar (or wraps it via gather_input), which a concrete-
// double signature cannot accept.  Use the templated form if .compute() is
// needed directly.
//
// This file MUST FAIL TO COMPILE.
// =============================================================================

#include <tycho/tycho.h>

namespace tycho_compile_fail_probes {

struct ProbeNonTemplatedCallsCompute
    : tycho::vf::VectorFunction<ProbeNonTemplatedCallsCompute, 5, 3,
                                tycho::vf::DenseDerivativeMode::EnzymeAD,
                                tycho::vf::DenseDerivativeMode::EnzymeAD> {
    inline void compute_impl(
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>> x,
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> fx) const {
        fx[0] = x[0];
        fx[1] = x[1];
        fx[2] = x[2];
    }
};

// Force instantiation of the .compute() direct call.
void force_instantiation() {
    ProbeNonTemplatedCallsCompute f;
    Eigen::Matrix<double, 5, 1> x = Eigen::Matrix<double, 5, 1>::Zero();
    Eigen::Matrix<double, 3, 1> fx = Eigen::Matrix<double, 3, 1>::Zero();
    f.compute(x, fx);
}

} // namespace tycho_compile_fail_probes
