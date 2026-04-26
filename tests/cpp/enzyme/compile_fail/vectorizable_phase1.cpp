// =============================================================================
// Tycho — Copyright 2026-present Grant R. Hecht. Apache 2.0.
//
// Compile-fail probe: opting an EnzymeAD VectorFunction into Vectorizable
// (i.e. is_vectorizable = true and a SuperScalar / Eigen::Array Scalar) must
// trigger the Phase 1 SuperScalar static_assert in
// include/tycho/detail/vf/derivatives/dense_enzyme.h.
//
// This translation unit is *expected* to fail to compile.  CTest invokes
// `cmake --build` on this target with WILL_FAIL TRUE, so a successful build
// would mark the test as failing.
// =============================================================================
#include <tycho/tycho.h>

struct VectorizableEnzymeBrach
    : tycho::vf::VectorFunction<VectorizableEnzymeBrach, 5, 3,
                                tycho::vf::DenseDerivativeMode::EnzymeAD,
                                tycho::vf::DenseDerivativeMode::AutodiffFwd> {
    static constexpr bool is_vectorizable = true;

    template <class InType, class OutType>
    inline void compute_impl(tycho::vf::CVecRef<InType> x,
                             tycho::vf::CVecRef<OutType> fx_) const {
        tycho::vf::VecRef<OutType> fx = fx_.const_cast_derived();
        using Scalar = typename InType::Scalar;
        using std::cos;
        using std::sin;
        const Scalar v = x[2];
        const Scalar theta = x[4];
        fx[0] = sin(theta) * v;
        fx[1] = -cos(theta) * v;
        fx[2] = Scalar(32.2) * cos(theta);
    }
};

int main() {
    VectorizableEnzymeBrach f;
    using SS = Eigen::Array<double, 4, 1>;
    Eigen::Matrix<SS, 5, 1> x;
    x.setZero();
    Eigen::Matrix<SS, 3, 1> fx;
    fx.setZero();
    Eigen::Matrix<SS, 3, 5> jx;
    jx.setZero();
    // The next call must trigger the Phase 1 static_assert because Scalar=SS
    // (an Eigen::Array, hence IsSuperScalar) but EnzymeAD requires
    // !IsSuperScalar in compute_jacobian_impl.
    f.compute_jacobian(x, fx, jx);
    return 0;
}
