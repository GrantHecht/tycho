// =============================================================================
// Compile-fail probe (Phase 4 constraint #2):
//
// A non-templated compute_impl cannot be marked `is_vectorizable = true`.
// SuperScalar dispatch's VectorImpl path requires a templated compute_impl
// that accepts Eigen::Matrix<Eigen::Array<double, W, 1>, ...> Scalar; a
// concrete-double signature cannot accept it.
//
// This file MUST FAIL TO COMPILE.
// =============================================================================

#include <tycho/tycho.h>

namespace tycho_compile_fail_probes {

struct ProbeNonTemplatedVectorizable
    : tycho::vf::VectorFunction<ProbeNonTemplatedVectorizable, 5, 3,
                                tycho::vf::DenseDerivativeMode::EnzymeAD,
                                tycho::vf::DenseDerivativeMode::EnzymeAD> {
    static constexpr bool is_vectorizable = true;

    inline void compute_impl(
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>> x,
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> fx) const {
        fx[0] = x[0];
        fx[1] = x[1];
        fx[2] = x[2];
    }
};

// Force instantiation of the SuperScalar dispatch path.
void force_instantiation() {
    using SS = Eigen::Array<double, 4, 1>;
    ProbeNonTemplatedVectorizable f;
    Eigen::Matrix<SS, 5, 1> x = Eigen::Matrix<SS, 5, 1>::Zero();
    Eigen::Matrix<SS, 3, 1> fx = Eigen::Matrix<SS, 3, 1>::Zero();
    Eigen::Matrix<SS, 3, 5> jx = Eigen::Matrix<SS, 3, 5>::Zero();
    f.compute_jacobian(x, fx, jx);
}

} // namespace tycho_compile_fail_probes
